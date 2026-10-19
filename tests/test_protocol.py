"""pytest suite for the WebSocket layer (backend/main.py).

Covers: seq monotonicity per session, command ingestion in the frontend wire
format (plus protocol.py's canonical shape), soft-drop press/release phases,
resync/ack replies, malformed-input tolerance, the end-to-end socket flow via
fastapi.testclient.TestClient, and the game-over notification path.

Runs from the repo root: .venv/bin/python -m pytest tests/ -q
"""

import time

from fastapi.testclient import TestClient

import backend.main as M
from backend.engine.game import Game
from backend.main import Session


def find_seed(piece: str, tries: int = 30) -> int:
    """Search for a seed whose first dealt piece is `piece`."""
    for s in range(tries):
        if Game(seed=s).snapshot()["piece"] == piece:
            return s
    raise AssertionError(f"no seed found for {piece} in {tries} tries")


T_SEED = find_seed("T")


def terminal_game() -> Game:
    """A real engine top-out: fill every visible row, force a lock-out.

    Mirrors the engine's terminal state (game_over True, no active piece)
    with the same direct board manipulation test_engine.py uses.
    """
    g = Game(seed=T_SEED)
    g.step(5)
    for r in range(2, 22):
        for x in range(10):
            g.board.lock_cells(((x, r),), "J")
    g.submit("hard_drop")
    g.step(5)
    assert g.snapshot()["game_over"] is True
    return g


# ---------------------------------------------------------------- seq numbers


class TestSeqMonotonic:
    def test_tick_seqs_strictly_increasing_from_one(self):
        s = Session(game=Game(seed=T_SEED))
        seqs = [s.tick()[0]["seq"] for _ in range(5)]
        assert seqs[0] == 1
        assert all(b > a for a, b in zip(seqs, seqs[1:]))

    def test_resync_snapshot_shares_the_counter(self):
        s = Session(game=Game(seed=T_SEED))
        last = s.build_snapshot()["seq"]
        out = s.handle_raw('{"type":"resync"}')
        assert out[1]["seq"] > last  # one shared, strictly monotonic counter


# ------------------------------------------------------------------ commands


class TestSessionCommands:
    def test_move_left_changes_piece_x(self):
        s = Session(game=Game(seed=T_SEED))
        first = s.build_snapshot()
        assert first["piece"]["x"] == 3  # spawn column
        s.handle_raw('{"type":"command","cmd":"move_left","ts":1}')
        snap = s.tick()[0]
        assert snap["piece"]["x"] == 2
        # protocol.py's canonical {"command": ...} shape works too
        s.handle_raw('{"type":"command","command":"move_left","ts":2}')
        assert s.tick()[0]["piece"]["x"] == 1

    def test_piece_y_is_visible_coordinate(self):
        s = Session(game=Game(seed=T_SEED))
        snap = s.build_snapshot()
        engine_y = s.game.snapshot()["piece_y"]
        assert snap["piece"]["y"] == engine_y - 2  # board strips 2 hidden rows

    def test_hard_drop_changes_board(self):
        s = Session(game=Game(seed=T_SEED))
        before = s.game.snapshot()["board"]
        assert all(row == "." * 10 for row in before)
        s.handle_raw('{"type":"command","cmd":"hard_drop","ts":1}')
        s.tick()
        after = s.game.snapshot()["board"]
        assert any(ch != "." for row in after for ch in row)

    def test_soft_drop_phase_mapping(self):
        s = Session(game=Game(seed=T_SEED))
        s.handle_raw('{"type":"command","cmd":"soft_drop","phase":"down","ts":1}')
        s.tick()
        assert s.game.snapshot()["soft_drop"] is True
        s.handle_raw('{"type":"command","cmd":"soft_drop","phase":"up","ts":2}')
        s.tick()
        assert s.game.snapshot()["soft_drop"] is False
        # release with no prior down: harmless no-op, must not crash
        s.handle_raw('{"type":"command","cmd":"soft_drop","phase":"up","ts":3}')
        s.tick()
        assert s.game.snapshot()["soft_drop"] is False
        # canonical release shape
        s.handle_raw('{"type":"command","command":"soft_drop","release":true}')
        s.tick()
        assert s.game.snapshot()["soft_drop"] is False
        # canonical press shape (no release flag)
        s.handle_raw('{"type":"command","command":"soft_drop"}')
        s.tick()
        assert s.game.snapshot()["soft_drop"] is True


# ----------------------------------------------------------- ack / resync


class TestAckResync:
    def test_ack_reply(self):
        s = Session(game=Game(seed=T_SEED))
        assert s.handle_raw('{"type":"ack","seq":7}') == [{"type": "ack"}]

    def test_resync_roundtrip(self):
        s = Session(game=Game(seed=T_SEED))
        last = s.build_snapshot()["seq"]
        out = s.handle_raw('{"type":"resync"}')
        assert len(out) == 2
        assert out[0] == {"type": "ack"}
        snap = out[1]
        assert snap["type"] == "snapshot"
        assert snap["seq"] > last
        assert len(snap["board"]) == 20  # full board


# ------------------------------------------------------------- malformed input


class TestMalformed:
    def test_malformed_ignored_and_session_recovers(self):
        s = Session(game=Game(seed=T_SEED))
        for bad in (
            "not json",
            "[]",
            '{"nope":1}',
            '{"type":"bogus"}',
            '{"type":"command","cmd":"jump"}',
        ):
            assert s.handle_raw(bad) == []
        # session still fully functional afterwards
        s.handle_raw('{"type":"command","cmd":"move_left","ts":9}')
        assert s.tick()[0]["piece"]["x"] == 2


# ---------------------------------------------------------------- game over


class TestSessionGameOver:
    def test_game_over_notification_then_quiet(self):
        s = Session(game=terminal_game())
        out = s.tick()
        assert [m["type"] for m in out] == ["game_over", "snapshot"]
        assert out[0]["seq"] == 1
        assert {"score", "lines", "level"} <= set(out[0])
        assert out[1]["game_over"] is True
        assert out[1]["seq"] == out[0]["seq"] + 1
        # sim ticks stop after the notification...
        assert s.tick() == []
        # ...but commands are drained/ignored and the socket semantics live on
        s.handle_raw('{"type":"command","cmd":"move_left","ts":1}')
        assert s.tick() == []
        assert s.handle_raw('{"type":"ack","seq":5}') == [{"type": "ack"}]
        resync = s.handle_raw('{"type":"resync"}')
        assert resync[0] == {"type": "ack"}
        assert resync[1]["type"] == "snapshot"
        assert resync[1]["seq"] > out[1]["seq"]


# ----------------------------------------------------------------- end to end


class TestE2E:
    def test_websocket_flow(self, monkeypatch):
        monkeypatch.setattr(M, "create_game", lambda: Game(seed=T_SEED))
        client = TestClient(M.app)
        with client.websocket_connect("/ws") as ws:
            seqs = []
            for _ in range(5):
                msg = ws.receive_json()
                assert msg["type"] == "snapshot"
                seqs.append(msg["seq"])
            assert seqs[0] == 1  # initial snapshot
            assert all(b > a for a, b in zip(seqs, seqs[1:]))
            last_seq = seqs[-1]

            # move_left applies on the next tick; piece.x is stable over the
            # poll (level-1 gravity = ~1 row/s)
            ws.send_json({"type": "command", "cmd": "move_left", "ts": 1})
            deadline = time.monotonic() + 2.0
            moved = False
            while time.monotonic() < deadline:
                msg = ws.receive_json()
                piece = msg.get("piece") if msg.get("type") == "snapshot" else None
                if piece is not None:
                    last_seq = msg["seq"]
                    if piece.get("x") == 2:
                        moved = True
                        break
            assert moved, "piece never reached x == 2 after move_left"

            # malformed text is ignored; an ack gets an ack reply
            ws.send_text("garbage{ not json")
            ws.send_json({"type": "ack", "seq": 99})
            deadline = time.monotonic() + 2.0
            acked = False
            while time.monotonic() < deadline:
                msg = ws.receive_json()
                if msg.get("type") == "ack":
                    acked = True
                    break
            assert acked, "no ack reply after ack"

            # resync: ack then a fresh full snapshot with a new seq
            ws.send_json({"type": "resync"})
            deadline = time.monotonic() + 2.0
            snap = None
            while time.monotonic() < deadline:
                msg = ws.receive_json()
                if msg.get("type") == "ack":
                    snap = ws.receive_json()  # resync reply is [ack, snapshot]
                    break
            assert snap is not None, "no resync snapshot"
            assert snap["type"] == "snapshot"
            assert snap["seq"] > last_seq
            assert len(snap["board"]) == 20

    def test_websocket_game_over(self, monkeypatch):
        monkeypatch.setattr(M, "create_game", terminal_game)
        client = TestClient(M.app)
        with client.websocket_connect("/ws") as ws:
            over = None
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                msg = ws.receive_json()
                if msg.get("type") == "game_over":
                    over = msg
                    break
            assert over is not None, "no game_over message"
            assert {"score", "lines", "level"} <= set(over)
            final = ws.receive_json()
            assert final["type"] == "snapshot"
            assert final["game_over"] is True
            assert final["seq"] > over["seq"]
