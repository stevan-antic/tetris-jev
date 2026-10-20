"""pytest suite for selectable brain profiles (set_brain) + level selector
(set_level): snapshot profile listing, brain switching, ChatBrain round
trips (offline, via monkeypatched brains._post), and level resets — all
against the real engine. No test ever touches the network."""

import asyncio

import pytest

import backend.main as main_mod
import backend.players.brains as brains
from backend.engine.game import GRAVITY_MS, Game
from backend.main import Session
from backend.players.llm import pick_best_option


# ----------------------------------------------------------------- helpers

def board_rows(snap):
    return snap["board"]


async def drive_one_decision(session, max_ticks=100):
    """Tick the session until the active piece's decision task has applied
    its commands (ghost set). The brain must answer instantly (mocked)."""
    for _ in range(max_ticks):
        session.tick()
        task = session._decision_task
        if task is not None and not task.done():
            await task
        if session._ghost is not None:
            return True
    return False


def spy_placements(monkeypatch):
    """Wrap backend.main.enumerate_placements so each call's option list is
    captured (the decision path for each piece, in order)."""
    original = main_mod.enumerate_placements
    captured = []

    def spy(game):
        opts = original(game)
        captured.append(opts)
        return opts

    monkeypatch.setattr(main_mod, "enumerate_placements", spy)
    return captured


# ---------------------------------------------------------------- profiles

class TestSnapshotProfiles:
    def test_lists_six_selectable_profiles_copilot_unavailable(self):
        s = Session()
        snap = s.build_snapshot()
        profiles = snap["profiles"]
        assert isinstance(profiles, list)
        by_id = {p["id"]: p["available"] for p in profiles}
        expected = {
            "local-anthropic",
            "local-copilot",
            "local-gemini",
            "local-unsloth_studio",
            "local-unsloth_studio_grinate",
            "typesafe_classifier_jev",
        }
        assert set(by_id) == expected
        assert by_id["local-copilot"] is False
        for pid in expected - {"local-copilot"}:
            assert by_id[pid] is True
        # sorted by id
        ids = [p["id"] for p in profiles]
        assert ids == sorted(ids)

    def test_default_brain_is_local_gemini(self):
        s = Session()
        assert s.build_snapshot()["brain"] == "local-gemini"

    def test_brain_and_profiles_present_in_human_mode_too(self):
        s = Session(mode="human")
        snap = s.build_snapshot()
        assert "brain" in snap and "profiles" in snap
        assert len(snap["profiles"]) == 6


# --------------------------------------------------------------- set_brain

class TestSetBrain:
    def test_switch_to_local_anthropic_acks(self):
        s = Session()
        out = s.handle_client_message({"type": "set_brain", "profile": "local-anthropic"})
        assert out == [{"type": "ack"}]
        assert s.build_snapshot()["brain"] == "local-anthropic"

    def test_switch_back_to_gemini_acks(self):
        s = Session()
        s.handle_client_message({"type": "set_brain", "profile": "local-anthropic"})
        out = s.handle_client_message({"type": "set_brain", "profile": "local-gemini"})
        assert out == [{"type": "ack"}]
        assert s.build_snapshot()["brain"] == "local-gemini"

    def test_unknown_profile_no_ack_brain_unchanged(self):
        s = Session()
        assert s.handle_client_message({"type": "set_brain", "profile": "no-such"}) == []
        assert s.handle_client_message({"type": "set_brain"}) == []
        assert s.handle_client_message({"type": "set_brain", "profile": 42}) == []
        assert s.build_snapshot()["brain"] == "local-gemini"

    def test_copilot_unavailable_no_ack_brain_unchanged(self):
        s = Session()
        assert s.handle_client_message({"type": "set_brain", "profile": "local-copilot"}) == []
        assert s.build_snapshot()["brain"] == "local-gemini"


# --------------------------------------------------- ChatBrain decisions

def jev_chat_session():
    """Session in jev mode on the default (ChatBrain) profile, with the
    decision pause removed so mocked round trips land instantly."""
    s = Session(mode="jev")
    s.jev_pause = 0.0
    return s


def chat_response(text):
    return {"choices": [{"message": {"content": text}}]}


class TestChatBrainDecision:
    def test_valid_option_lands_on_its_preview(self, monkeypatch):
        captured = spy_placements(monkeypatch)

        async def fake_post(url, headers, payload, timeout_s):
            # the brain asks about the most recently enumerated piece; answer
            # with the FIRST option of that list.
            chosen = captured[-1][0]
            return chat_response(chosen.id)

        monkeypatch.setattr(brains, "_post", fake_post)

        async def scenario():
            s = jev_chat_session()
            assert await drive_one_decision(s) is True
            s.tick()  # one more tick applies the buffered command sequence
            chosen = captured[0][0]
            assert not chosen.hold
            assert s.game.board.serialize() == chosen.preview

        asyncio.run(scenario())

    def test_garbage_reply_falls_back_to_heuristic(self, monkeypatch):
        captured = spy_placements(monkeypatch)

        async def garbage(*a, **k):
            return chat_response("banana")  # no option id in the reply

        monkeypatch.setattr(brains, "_post", garbage)

        async def scenario():
            s = jev_chat_session()
            assert await drive_one_decision(s) is True
            s.tick()
            best = pick_best_option(captured[0])
            assert best is not None
            assert s.game.board.serialize() == best.preview

        asyncio.run(scenario())

    def test_timeout_reply_falls_back_to_heuristic(self, monkeypatch):
        captured = spy_placements(monkeypatch)

        async def boom(*a, **k):
            raise TimeoutError("simulated timeout")

        monkeypatch.setattr(brains, "_post", boom)

        async def scenario():
            s = jev_chat_session()
            assert await drive_one_decision(s) is True
            s.tick()
            best = pick_best_option(captured[0])
            assert best is not None
            assert s.game.board.serialize() == best.preview

        asyncio.run(scenario())


# --------------------------------------------------------------- set_level

class TestSetLevel:
    def test_set_level_7_acks_and_resets(self):
        s = Session()
        # dirty the board so the reset is observable
        s.game.board.grid[-1][0] = "J"
        assert not s.game.board.is_empty()
        out = s.handle_client_message({"type": "set_level", "level": 7})
        assert out == [{"type": "ack"}]
        snap = s.build_snapshot()
        assert snap["level"] == 7
        assert snap["lines"] == 0
        assert s.game.board.is_empty()          # fresh board
        assert s.game.start_level == 7
        # engine gravity table: level 7 => 190 ms/row (direct engine call)
        assert GRAVITY_MS[7] == 190
        assert s.game._gravity_ms() == 190

    def test_set_level_1_default_gravity(self):
        s = Session()
        out = s.handle_client_message({"type": "set_level", "level": 1})
        assert out == [{"type": "ack"}]
        assert s.build_snapshot()["level"] == 1
        assert s.game._gravity_ms() == GRAVITY_MS[1] == 1000

    def test_set_level_invalid_no_ack(self):
        s = Session()
        level7 = s.game.start_level
        for payload in (
            {"type": "set_level", "level": 11},
            {"type": "set_level", "level": 0},
            {"type": "set_level", "level": "fast"},
            {"type": "set_level", "level": True},  # bool is not a level
            {"type": "set_level"},
        ):
            assert s.handle_client_message(payload) == []
        assert s.game.start_level == level7  # game untouched


class TestSetLevelDuringDecision:
    def test_cancel_in_flight_decision_cleanly(self):
        async def scenario():
            s = Session(mode="jev")
            s.tick()  # spawns the first piece, starts its decision task
            task = s._decision_task
            assert task is not None and not task.done()
            out = s.handle_client_message({"type": "set_level", "level": 5})
            assert out == [{"type": "ack"}]
            # the handler cancelled the task and cleared the ghost
            assert s._decision_task is None
            assert s._ghost is None
            assert s._last_decision_serial is None
            # the old task dies with CancelledError — its commands never land
            with pytest.raises(asyncio.CancelledError):
                await task
            snap = s.build_snapshot()
            assert snap["ghost"] is None
            assert snap["level"] == 5
            assert s.game.board.is_empty()  # no piece locked from the old decision
            # a fresh game has an active piece but no new decision task yet
            assert s.game.cur is not None

        asyncio.run(scenario())
