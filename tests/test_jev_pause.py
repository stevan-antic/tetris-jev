"""pytest suite for the configurable pause between Jev's decisions.

Wire contract: client message {"type":"set_jev_pause","seconds":<number>},
clamped to [0, 30], non-numeric/missing -> ignored with no ack; every
snapshot carries a top-level "jev_pause" key in seconds (default 1.0).

Behavior: the decision task sleeps jev_pause seconds BEFORE enumerating, so
during the pause no ghost is shown and no commands are submitted. A pause
change mid-decision only applies to the NEXT piece (the in-flight task is
left alone).

Reuses the stub-brain fixtures from test_llm_stub.py — no test touches the
network. Async tests use asyncio.run inside sync test functions, matching
the existing suite's style.
"""

import asyncio

from backend.engine.game import Game
from backend.main import Session
from backend.players.llm import enumerate_placements, pick_best_option

from tests.test_llm_stub import StubBrain, heuristic_policy, seed_for


class TestJevPause:
    def test_default_pause_is_one_second(self):
        s = Session(game=Game(seed=1))
        assert s.jev_pause == 1.0
        # The key is present in EVERY snapshot, human mode too, so the
        # frontend control always shows the authoritative value.
        assert s.build_snapshot()["jev_pause"] == 1.0
        s.handle_raw('{"type":"set_mode","mode":"jev"}')
        assert s.build_snapshot()["jev_pause"] == 1.0

    def test_set_jev_pause_acks_and_updates_snapshot(self):
        s = Session(game=Game(seed=1))
        out = s.handle_raw('{"type":"set_jev_pause","seconds":2.5}')
        assert out == [{"type": "ack"}]
        assert s.jev_pause == 2.5
        assert s.build_snapshot()["jev_pause"] == 2.5

    def test_set_jev_pause_clamps_to_range(self):
        s = Session(game=Game(seed=1))
        assert s.handle_raw('{"type":"set_jev_pause","seconds":100}') == [{"type": "ack"}]
        assert s.jev_pause == 30
        assert s.build_snapshot()["jev_pause"] == 30
        assert s.handle_raw('{"type":"set_jev_pause","seconds":-5}') == [{"type": "ack"}]
        assert s.jev_pause == 0
        assert s.build_snapshot()["jev_pause"] == 0

    def test_set_jev_pause_ignores_non_numeric(self):
        s = Session(game=Game(seed=1))
        before = s.jev_pause
        snap_before = s.build_snapshot()["jev_pause"]
        # Non-numeric, missing, and bool payloads are all ignored entirely:
        # no ack, value unchanged, snapshot unchanged.
        assert s.handle_raw('{"type":"set_jev_pause","seconds":"fast"}') == []
        assert s.handle_raw('{"type":"set_jev_pause"}') == []
        assert s.handle_raw('{"type":"set_jev_pause","seconds":true}') == []
        assert s.jev_pause == before
        assert s.build_snapshot()["jev_pause"] == snap_before

    def test_pause_delays_the_decision(self):
        async def go():
            brain = StubBrain(heuristic_policy)
            s = Session(game=Game(seed=seed_for("T")), mode="jev", brain=brain)
            s.jev_pause = 0.05
            s.tick()
            task = s._decision_task
            assert task is not None
            await asyncio.sleep(0.01)  # 50ms pause -> task is still sleeping
            assert not task.done()
            assert brain.calls == []   # brain not yet consulted
            assert s._ghost is None    # no ghost while Jev "thinks"
            assert s.game.snapshot()["piece_x"] == 3  # only gravity moved it
            await task                 # pause elapses, decision lands
            assert len(brain.calls) == 1
            assert s._ghost is not None
            s.tick()  # applies the buffered batch
            expected = pick_best_option(enumerate_placements(Game(seed=seed_for("T")))).preview
            assert s.game.board.serialize() == expected

        asyncio.run(go())

    def test_zero_pause_decides_immediately(self):
        async def go():
            brain = StubBrain(heuristic_policy)
            s = Session(game=Game(seed=seed_for("T")), mode="jev", brain=brain)
            s.jev_pause = 0
            s.tick()
            task = s._decision_task
            # One millisecond is far below the 50ms used in the delay test:
            # the decision lands with no timer of its own.
            await asyncio.sleep(0.001)
            assert task.done()
            assert len(brain.calls) == 1

        asyncio.run(go())

    def test_set_jev_pause_midflight_leaves_decision_undisturbed(self):
        async def go():
            seed = seed_for("T")
            brain = StubBrain(heuristic_policy)
            s = Session(game=Game(seed=seed), mode="jev", brain=brain)
            s.jev_pause = 0.05
            s.tick()
            task = s._decision_task
            await asyncio.sleep(0.01)  # mid-pause
            # The change only applies to the NEXT piece; the in-flight task
            # is not cancelled or restarted.
            assert s.handle_raw('{"type":"set_jev_pause","seconds":0.3}') == [{"type": "ack"}]
            assert s.jev_pause == 0.3
            assert not task.cancelled()
            await task
            assert len(brain.calls) == 1  # exactly one decision for the piece
            s.tick()  # applies the batch exactly once
            expected = pick_best_option(enumerate_placements(Game(seed=seed))).preview
            assert s.game.board.serialize() == expected

        asyncio.run(go())

    def test_cancel_interrupts_a_long_pause(self):
        """Mode switch during the pause cancels the task quickly (the sleep
        does not have to elapse) and no commands were submitted."""
        async def go():
            s = Session(game=Game(seed=seed_for("T")), mode="jev", brain=StubBrain(heuristic_policy))
            s.jev_pause = 30  # would take half a minute if cancel missed
            s.tick()
            task = s._decision_task
            assert task is not None
            await asyncio.sleep(0)  # task is now inside the 30s sleep
            s.handle_raw('{"type":"set_mode","mode":"human"}')
            await asyncio.sleep(0.01)
            assert task.cancelled()
            x0 = s.game.snapshot()["piece_x"]
            s.handle_raw('{"type":"command","cmd":"move_left"}')
            for _ in range(20):
                s.tick()
            assert s.game.snapshot()["piece_x"] == x0 - 1  # human input now works

        asyncio.run(go())
