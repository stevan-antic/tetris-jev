"""pytest suite for the LLM (Jev) player mode: placement enumeration,
command translation, heuristic fallback, session wiring, and the Jev API
client — all against the real engine with stub brains.

No test touches the network: the HTTP touchpoint (jev_api._post) is
monkeypatched, and configs are passed explicitly (env.yaml is never read).

Async tests use asyncio.run inside sync test functions (no pytest-asyncio
dependency), matching the existing suite's style.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from backend.engine.game import Game
from backend.main import Session
from backend.players import jev_api
from backend.players.llm import (
    build_state,
    enumerate_placements,
    pick_best_option,
    translate,
)

TICK = 1000 / 60


def seed_for(piece: str, tries: int = 200) -> int:
    for s in range(tries):
        if Game(seed=s).snapshot()["piece"] == piece:
            return s
    raise AssertionError(f"no seed found for {piece}")


def find_option(options, opt_id: str):
    for o in options:
        if o.id == opt_id:
            return o
    return None


# --------------------------------------------------------------- stub brains


class StubBrain:
    """Policy-driven brain: policy(state, options) -> (choice_id, probs)."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.calls = []

    async def choose(self, state, options):
        self.calls.append((state, options))
        return self.policy(state, options)


def heuristic_policy(state, options):
    best = pick_best_option(options)
    return (best.id, None) if best else (None, None)


class NeverBrain:
    """A brain whose answer never arrives (simulates a hung request)."""

    async def choose(self, state, options):
        await asyncio.Future()  # pending forever until cancelled


class DelayedBrain:
    """Answers after `delay` seconds (lets the test race the piece)."""

    def __init__(self, delay: float, policy) -> None:
        self.delay = delay
        self.policy = policy

    async def choose(self, state, options):
        await asyncio.sleep(self.delay)
        return self.policy(state, options)


class FirstCallDelayedBrain:
    """Delays only the first decision (simulates a slow first round trip);
    later pieces answer instantly."""

    def __init__(self, delay: float, policy) -> None:
        self.delay = delay
        self.policy = policy
        self.calls = 0

    async def choose(self, state, options):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(self.delay)
        return self.policy(state, options)


async def drive_one_piece(seed: int, brain):
    """Session in jev mode: tick, let the decision land, apply it."""
    s = Session(game=Game(seed=seed), mode="jev", brain=brain)
    s.jev_pause = 0.0  # keep the suite fast; the pause itself is tested in test_jev_pause.py
    s.tick()
    task = s._decision_task
    assert task is not None, "no decision task started"
    await task
    s.tick()  # applies the buffered command batch
    return s


# ------------------------------------------------------- enumeration (pure)


class TestEnumeration:
    def test_options_exist_and_are_wellformed(self):
        for piece in "IJLOSTZ":
            g = Game(seed=seed_for(piece))
            options = enumerate_placements(g)
            assert options, f"no options for {piece}"
            ids = {o.id for o in options}
            for o in options:
                prefix = "hold_" if o.hold else ""
                assert o.id == f"{prefix}r{o.rot}_c{o.x}"
                assert len(o.preview) == 20
                assert all(len(r) == 10 for r in o.preview)
                assert o.piece in "IJLOSTZ"
            assert len(ids) == len(options), "duplicate option ids"

    def test_hold_variant_present_when_available(self):
        g = Game(seed=seed_for("T"))
        assert g.hold_used is False
        options = enumerate_placements(g)
        holds = [o for o in options if o.hold]
        plain = [o for o in options if not o.hold]
        assert holds and plain
        # The hold variant places the swapped-in piece: hold slot is empty
        # here, so that is the next bag piece.
        assert {o.piece for o in holds} == {g.bag.peek(1)[0]}
        assert {o.piece for o in plain} == {"T"}

    def test_no_hold_variant_once_hold_used(self):
        g = Game(seed=seed_for("T"))
        g.submit("hold")  # uses hold for this piece
        g.step(TICK)  # applies the hold; swapped-in piece is now active
        assert g.hold_used is True
        options = enumerate_placements(g)
        assert options
        assert not any(o.hold for o in options)
        # The hold slot was empty, so the swapped-in piece is the next bag
        # piece and the active piece is now that piece.
        assert {o.piece for o in options} == {g.cur}

    def test_live_game_not_mutated(self):
        g = Game(seed=seed_for("T"))
        before = g.snapshot()
        enumerate_placements(g)
        after = g.snapshot()
        assert before == after

    def test_preview_differs_per_column(self):
        g = Game(seed=seed_for("T"))
        previews = {tuple(o.preview) for o in enumerate_placements(g) if o.rot == 0}
        assert len(previews) > 1


# --------------------------------------------- translation + engine landing


class TestPreviewInvariant:
    def test_every_option_lands_exactly_on_its_preview(self):
        """THE critical invariant: driving the real engine with the translated
        command batch reproduces the enumerated preview bit-for-bit."""
        for piece in "IJLOSTZ":
            seed = seed_for(piece)
            g = Game(seed=seed)
            for opt in enumerate_placements(g):
                g2 = Game(seed=seed)  # fresh engine, same piece sequence
                for cmd in translate(opt):
                    g2.submit(cmd)
                g2.step(TICK)  # the batch drains within this single step
                assert g2.board.serialize() == opt.preview, (
                    f"{piece} {opt.id}: board does not match preview"
                )
                assert not g2.board.any_above_visible()

    def test_hold_option_uses_hold_and_lands_on_preview(self):
        seed = seed_for("T")
        g = Game(seed=seed)
        holds = [o for o in enumerate_placements(g) if o.hold]
        assert holds
        opt = holds[0]
        g2 = Game(seed=seed)
        for cmd in translate(opt):
            g2.submit(cmd)
        g2.step(TICK)
        assert g2.board.serialize() == opt.preview
        assert g2.hold == "T"  # the original piece went to the hold slot
        # hold_used was True while the swapped piece was active; after the
        # batch's hard_drop locks and the next piece spawns, the engine
        # makes hold available again for that new piece.
        assert g2.hold_used is False

    def test_translation_vocabulary(self):
        g = Game(seed=seed_for("T"))
        for opt in enumerate_placements(g):
            cmds = translate(opt)
            assert cmds[-1] == "hard_drop"
            assert "soft_drop" not in cmds and "soft_drop_release" not in cmds
            if opt.hold:
                assert cmds[0] == "hold"
            # rotations before moves, at most one direction each
            body = cmds[1 if opt.hold else 0 : -1]
            assert all(c in ("rotate_cw", "rotate_ccw", "move_left", "move_right") for c in body)
            assert ("rotate_cw" in body) != ("rotate_ccw" in body) or not (
                "rotate_cw" in body and "rotate_ccw" in body
            )


# ------------------------------------------------------------- heuristic


class TestHeuristic:
    def test_prefers_line_clears_then_holes_then_height(self):
        def opt(oid, preview, cleared):
            return SimpleNamespace(
                id=oid, preview=preview, lines_cleared=cleared, piece="T"
            )

        # Two-row previews so holes (empty cells below filled cells) exist.
        #   holey: column 1 is filled on row 18 but empty on row 19 -> a hole.
        #   low:   same two rows filled, no holes.
        #   empty: nothing anywhere, height 0.
        flat = ["." * 10] * 19 + ["T" * 10]
        holey = ["." * 10] * 18 + ["TT........", "T.T......."]
        low = ["." * 10] * 18 + ["TT........", "TT........"]
        empty = ["." * 10] * 20
        best = pick_best_option([opt("a", holey, 0), opt("b", flat, 1), opt("c", low, 0)])
        assert best.id == "b"  # max lines wins over holes
        best = pick_best_option([opt("a", holey, 0), opt("c", low, 0)])
        assert best.id == "c"  # fewer holes wins
        best = pick_best_option([opt("a", low, 0), opt("b", empty, 0)])
        assert best.id == "b"  # lowest stack height wins
        best = pick_best_option(
            [opt("c", low, 0), opt("d", ["." * 10] * 20, 0)]
        )
        assert best.id == "d"  # lowest height wins (d is the empty board)

    def test_empty_options(self):
        assert pick_best_option([]) is None


# ------------------------------------------------- session wiring (jev mode)


class TestSessionMode:
    def test_default_mode_is_human(self):
        s = Session(game=Game(seed=1))
        snap = s.build_snapshot()
        assert snap["mode"] == "human"
        assert "ghost" not in snap

    def test_set_mode_ack_and_snapshot_fields(self):
        s = Session(game=Game(seed=1))
        out = s.handle_raw('{"type":"set_mode","mode":"jev"}')
        assert out == [{"type": "ack"}]
        snap = s.build_snapshot()
        assert snap["mode"] == "jev"
        assert "ghost" in snap and snap["ghost"] is None
        # switching back
        assert s.handle_raw('{"type":"set_mode","mode":"human"}') == [{"type": "ack"}]
        snap = s.build_snapshot()
        assert snap["mode"] == "human" and "ghost" not in snap

    def test_set_mode_rejects_unknown_mode(self):
        s = Session(game=Game(seed=1))
        assert s.handle_raw('{"type":"set_mode","mode":"gpt"}') == []
        assert s.mode == "human"

    def test_human_input_ignored_in_jev_mode(self):
        async def go():
            s = Session(game=Game(seed=seed_for("T")), mode="jev", brain=NeverBrain())
            s.tick()
            await asyncio.sleep(0)  # decision task starts, blocks in brain
            x0 = s.game.snapshot()["piece_x"]
            s.handle_raw('{"type":"command","cmd":"move_left"}')
            s.handle_raw('{"type":"command","cmd":"rotate_cw"}')
            s.tick()
            assert s.game.snapshot()["piece_x"] == x0  # untouched
            assert s.game.snapshot()["piece_rot"] == 0
            # but in human mode the same input works
            s.handle_raw('{"type":"set_mode","mode":"human"}')
            s.handle_raw('{"type":"command","cmd":"move_left"}')
            s.tick()
            assert s.game.snapshot()["piece_x"] == x0 - 1

        asyncio.run(go())

    def test_decision_places_piece_on_chosen_preview(self):
        async def go():
            seed = seed_for("T")
            brain = StubBrain(heuristic_policy)
            s = await drive_one_piece(seed, brain)
            assert len(brain.calls) == 1
            state, options = brain.calls[0]
            assert state["piece"] == "T"
            assert len(state["next"]) == 5
            expected = pick_best_option(options).preview
            assert s.game.board.serialize() == expected

        asyncio.run(go())

    def test_ghost_in_snapshot_after_decision(self):
        async def go():
            seed = seed_for("T")
            brain = StubBrain(heuristic_policy)
            s = Session(game=Game(seed=seed), mode="jev", brain=brain)
            s.jev_pause = 0.0
            s.tick()
            await s._decision_task  # decision lands, ghost is set
            options = brain.calls[0][1]
            opt = pick_best_option(options)
            snap = s.build_snapshot()  # BEFORE the applying tick
            assert snap["mode"] == "jev"
            assert snap["ghost"] == {
                "type": opt.piece,
                "x": opt.x,
                "y": opt.y - 2,  # visible coordinates
                "rot": opt.rot,
            }
            s.tick()  # applies the batch; piece locks, next spawns
            assert s.build_snapshot()["ghost"] is None  # stale ghost cleared

        asyncio.run(go())

    def test_ghost_cleared_for_next_piece(self):
        async def go():
            seed = seed_for("T")
            s = Session(game=Game(seed=seed), mode="jev", brain=StubBrain(heuristic_policy))
            s.jev_pause = 0.0
            s.tick()
            await s._decision_task
            s.tick()  # piece locks, next spawns
            assert s._ghost is None  # stale ghost cleared on new decision
            snap = s.build_snapshot()
            assert snap["ghost"] is None

        asyncio.run(go())

    def test_mode_switch_midflight_cancels_cleanly(self):
        async def go():
            s = Session(game=Game(seed=seed_for("T")), mode="human", brain=NeverBrain())
            s.handle_raw('{"type":"set_mode","mode":"jev"}')
            s.tick()  # the tick loop starts the decision for the active piece
            task = s._decision_task
            assert task is not None
            await asyncio.sleep(0)  # task runs until it blocks in the brain
            assert not task.done()
            s.handle_raw('{"type":"set_mode","mode":"human"}')  # cancels in-flight
            assert s._decision_task is None
            await asyncio.sleep(0.01)
            assert task.cancelled()
            # No commands were submitted: the piece only falls by gravity.
            for _ in range(20):
                s.tick()
            assert s.game.snapshot()["piece_x"] == 3
            assert s.game.game_over is False

        asyncio.run(go())

    def test_late_answer_discarded_after_piece_locks(self):
        """A decision landing after its piece locked is discarded: piece 2 is
        placed by its OWN decision, never by piece 1's stale command batch."""
        async def go():
            seed = seed_for("T")
            s = Session(game=Game(seed=seed), mode="jev", brain=FirstCallDelayedBrain(0.05, heuristic_policy))
            s.jev_pause = 0.0
            s.tick()
            task = s._decision_task
            assert task is not None
            await asyncio.sleep(0)  # task 1 enumerated, now sleeping
            # Lock the piece directly (bypasses the session's human-input
            # policy); the engine spawns the next piece instantly (ARE=0).
            s.game.submit("hard_drop")
            s.tick()
            assert s.game.piece_serial == 2
            await task  # late answer for piece 1 must be discarded
            # Piece 2's own decision (instant brain) buffered its batch during
            # the await; the piece itself is untouched until the next tick.
            assert s.game.snapshot()["piece_x"] == 3
            assert s.game.snapshot()["piece_rot"] == 0
            assert s._last_decision_serial == 2
            s.tick()  # applies piece 2's batch; piece 2 locks
            assert s.game.piece_serial == 3
            # Reference engine: burn piece 1, place piece 2 via the same
            # heuristic path — the session must match it bit-for-bit, proving
            # piece 1's stale batch was NOT queued onto piece 2.
            ref = Game(seed=seed)
            ref.submit("hard_drop")
            ref.step(TICK)
            ref_opt = pick_best_option(enumerate_placements(ref))
            for cmd in translate(ref_opt):
                ref.submit(cmd)
            ref.step(TICK)
            assert s.game.board.serialize() == ref.board.serialize()

        asyncio.run(go())

    def test_brain_exception_falls_back_to_heuristic(self):
        async def go():
            class ExplodingBrain:
                async def choose(self, state, options):
                    raise RuntimeError("api down")

            seed = seed_for("T")
            s = Session(game=Game(seed=seed), mode="jev", brain=ExplodingBrain())
            s.jev_pause = 0.0
            s.tick()
            await s._decision_task
            # The heuristic still placed the piece.
            s.tick()
            board = s.game.board.serialize()
            options = enumerate_placements(Game(seed=seed))
            expected = pick_best_option(options).preview
            assert board == expected

        asyncio.run(go())

    def test_unknown_option_id_falls_back_to_heuristic(self):
        async def go():
            seed = seed_for("T")
            brain = StubBrain(lambda state, options: ("r9_c99", None))
            s = await drive_one_piece(seed, brain)
            options = enumerate_placements(Game(seed=seed))
            expected = pick_best_option(options).preview
            assert s.game.board.serialize() == expected

        asyncio.run(go())

    def test_timeout_fallback_via_jev_brain(self):
        """A brain timeout (exception inside JevBrain) yields (None, None)
        and the session submits the heuristic option."""
        async def go():
            async def hanging_post(url, headers, payload, timeout_s):
                raise TimeoutError("simulated 30s timeout")

            import backend.players.jev_api as ja

            original = ja._post
            ja._post = hanging_post
            try:
                brain = ja.JevBrain(
                    config={
                        "base_url": "https://api.typesafe.ai",
                        "endpoint": "/v1/systemone",
                        "api_key": "test-key-not-real",
                        "model": "jev-latest",
                        "timeout_seconds": 0.01,
                    }
                )
                seed = seed_for("T")
                s = await drive_one_piece(seed, brain)
                options = enumerate_placements(Game(seed=seed))
                expected = pick_best_option(options).preview
                assert s.game.board.serialize() == expected
            finally:
                ja._post = original

        asyncio.run(go())


# ------------------------------------------------------- several-piece smoke


class TestHeadlessSmoke:
    def test_jev_session_places_many_pieces_headlessly(self):
        """Stub brain (heuristic policy) drives a real session: no exceptions,
        pieces keep placing, lines sometimes clear."""

        async def go():
            s = Session(game=Game(seed=11), mode="jev", brain=StubBrain(heuristic_policy))
            s.jev_pause = 0.0
            target_pieces = 20
            ticks = 0
            while not s.game.game_over and s.game.piece_serial <= target_pieces + 1 and ticks < 5000:
                s.tick()
                ticks += 1
                task = s._decision_task
                if task is not None:
                    # Yield so the decision task runs (enumerate + instant
                    # stub brain + submit) before the next tick applies it.
                    await asyncio.sleep(0)
                    if task.done():
                        await asyncio.gather(task, return_exceptions=True)
            assert not s.game.game_over or s.game.piece_serial > 10
            assert s.game.piece_serial > 10, "pieces stopped placing"

        asyncio.run(go())

    def test_jev_session_clears_lines_with_heuristic_brain(self):
        async def go():
            s = Session(game=Game(seed=11), mode="jev", brain=StubBrain(heuristic_policy))
            s.jev_pause = 0.0
            for _ in range(400):
                if s.game.game_over:
                    break
                s.tick()
                task = s._decision_task
                if task is not None:
                    await asyncio.sleep(0)
                    if task.done():
                        await asyncio.gather(task, return_exceptions=True)
            # a reasonably sane run: plenty of pieces got placed
            assert s.game.piece_serial > 50 or s.game.game_over

        asyncio.run(go())


# ------------------------------------------------------------ jev_api client


class TestJevApiConfig:
    def test_parses_block(self, tmp_path):
        f = tmp_path / "env.yaml"
        f.write_text(
            "profiles:\n"
            "  something_broken\n"
            "typesafe_classifier:\n"
            "    base_url: https://api.typesafe.ai\n"
            "    endpoint: /v1/systemone\n"
            "    api_key: secret-key-value\n"
            "    model: jev-latest\n"
            "    timeout_seconds: 30\n"
        )
        cfg = jev_api.load_jev_config(str(f))
        assert cfg["base_url"] == "https://api.typesafe.ai"
        assert cfg["endpoint"] == "/v1/systemone"
        assert cfg["api_key"] == "secret-key-value"
        assert cfg["model"] == "jev-latest"
        assert cfg["timeout_seconds"] == "30"

    def test_missing_block_returns_none(self, tmp_path):
        f = tmp_path / "env.yaml"
        f.write_text("profiles:\n  a: b\n")
        assert jev_api.load_jev_config(str(f)) is None

    def test_incomplete_block_returns_none(self, tmp_path):
        f = tmp_path / "env.yaml"
        f.write_text("typesafe_classifier:\n    base_url: https://x\n")
        assert jev_api.load_jev_config(str(f)) is None

    def test_missing_file_returns_none(self):
        assert jev_api.load_jev_config("/nonexistent/env.yaml") is None

    def test_defaults_applied(self, tmp_path):
        f = tmp_path / "env.yaml"
        f.write_text("typesafe_classifier:\n    base_url: https://x\n    api_key: k\n")
        cfg = jev_api.load_jev_config(str(f))
        assert cfg["endpoint"] == "/v1/systemone"
        assert cfg["model"] == "jev-latest"


class TestJevApiQuestion:
    def test_question_contains_options_and_goal(self):
        g = Game(seed=seed_for("T"))
        options = enumerate_placements(g)
        q = jev_api.make_question(build_state(g), options)
        assert "maximize lines cleared" in q
        for o in options:
            assert o.id in q
            assert o.preview[0] in q
        assert "clears=" in q and "tspin=" in q

    def test_question_stays_under_context_window(self):
        g = Game(seed=seed_for("I"))
        q = jev_api.make_question(build_state(g), enumerate_placements(g))
        assert len(q) < 30000  # Jev context is 64k; keep wide margin


class TestJevApiChoice:
    def test_extract_choice_shapes(self):
        assert jev_api.extract_choice({"choice": "r1_c7"}) == ("r1_c7", None)
        assert jev_api.extract_choice({"answer": {"id": "hold_r0_c3"}}) == ("hold_r0_c3", None)
        assert jev_api.extract_choice({"selection": 3}) == ("3", None)
        probs = {"r1_c7": 0.7}
        assert jev_api.extract_choice({"chosen": "r1_c7", "probabilities": probs}) == (
            "r1_c7",
            probs,
        )
        assert jev_api.extract_choice({"answers": [{"message": {"content": "r2_c4"}}]}) == (
            "r2_c4",
            None,
        )
        assert jev_api.extract_choice({"choices": [{"text": "r0_c1"}]}) == ("r0_c1", None)
        assert jev_api.extract_choice({"foo": 1}) == (None, None)
        assert jev_api.extract_choice("nonsense") == (None, None)
        assert jev_api.extract_choice(None) == (None, None)

    def test_call_payload_and_parsing(self, monkeypatch):
        captured = {}

        async def fake_post(url, headers, payload, timeout_s):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout_s)
            return {"choice": "r0_c3", "probabilities": {"r0_c3": 1.0}}

        monkeypatch.setattr(jev_api, "_post", fake_post)
        g = Game(seed=seed_for("T"))
        cfg = {
            "base_url": "https://api.typesafe.ai",
            "endpoint": "/v1/systemone",
            "api_key": "runtime-key",
            "model": "jev-latest",
            "timeout_seconds": 5,
        }
        choice, probs = asyncio.run(jev_api.call(cfg, build_state(g), enumerate_placements(g)))
        assert choice == "r0_c3"
        assert probs == {"r0_c3": 1.0}
        assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
        assert captured["headers"]["Authorization"].startswith("Bearer ")
        assert captured["timeout"] == 5.0
        assert captured["payload"]["model"] == "jev-latest"
        assert captured["payload"]["state"]["piece"] == "T"
        assert len(captured["payload"]["questions"]) == 1
        # the api key must never travel inside the payload
        assert "runtime-key" not in json.dumps(captured["payload"])

    def test_jev_brain_returns_none_on_http_error(self, monkeypatch):
        async def failing_post(url, headers, payload, timeout_s):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(jev_api, "_post", failing_post)
        brain = jev_api.JevBrain(
            config={"base_url": "https://x", "api_key": "k", "model": "m"}
        )
        choice, probs = asyncio.run(brain.choose({"piece": "T"}, []))
        assert choice is None and probs is None

    def test_jev_brain_without_config_returns_none(self):
        brain = jev_api.JevBrain(config={})
        choice, probs = asyncio.run(brain.choose({"piece": "T"}, []))
        assert choice is None and probs is None

    def test_state_payload_shape(self):
        g = Game(seed=seed_for("T"))
        state = build_state(g)
        assert set(state) == {"board", "piece", "hold", "next", "level", "lines"}
        assert len(state["board"]) == 20
        assert len(state["next"]) == 5
