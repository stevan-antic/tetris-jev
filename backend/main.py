"""WebSocket server for the Tetris app (phase 1, human mode).

Design:
  - FastAPI app. The frontend is served as static files at "/" and the game
    socket lives at "/ws" (the socket route is registered BEFORE the mount so
    it wins routing).
  - One Session per WebSocket connection. The Session owns the authoritative
    Game and a SeqCounter; every snapshot (initial, per-tick, resync, final)
    draws its seq from that one counter, so the client's seq-gap detection
    sees a strictly monotonic stream per connection.
  - Fixed-step 60Hz simulation: each tick advances the engine by
    TICK_MS = 1000/60 and pushes exactly ONE snapshot. A per-tick snapshot is
    the simplest smooth cadence (the client renders what arrives) and keeps
    input latency at one tick.
  - A connection runs three asyncio tasks sharing ONE outbound queue: a
    ticker (sim + snapshots; stops after the game-over notification, socket
    stays open), a receiver (client commands/acks/resync) and a sender
    (serialize + send). On disconnect or any error all tasks are cancelled
    and the session is dropped — a reconnect is a brand-new game.

Wire contract (ground truth: frontend/game.js): client commands carry
{"type":"command","cmd":...,"ts":...} with soft drop using phase:"down"/"up";
protocol.py's canonical {"type":"command","command":...,"release":true}
shape is tolerated as well. Malformed/unknown input is ignored silently.
"""

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from . import protocol
from .engine.board import HIDDEN_ROWS
from .engine.game import Game

TICK_MS = 1000 / 60  # fixed 60Hz simulation step (one snapshot per tick)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="tetris-jev")


def create_game() -> Game:
    """Fresh game for a new connection. Module-level so tests can monkeypatch
    backend.main.create_game (e.g. to force a game-over terminal state)."""
    return Game()


class Session:
    """Game state + protocol bookkeeping for one connection.

    Deliberately socket-free: tests drive it with plain dicts/strings.
    """

    def __init__(self, game: Game | None = None) -> None:
        self.game = game if game is not None else create_game()
        self.seq = protocol.SeqCounter()
        self.game_over_notified = False

    def build_snapshot(self) -> dict:
        """Full snapshot from the engine state, seq from the shared counter."""
        state = self.game.snapshot()
        # The engine exposes the piece as flat fields (piece/piece_x/piece_y/
        # piece_rot); the wire contract wants a "piece" object whose x/y are
        # the bounding box top-left in VISIBLE coordinates. The board array
        # is grid[HIDDEN_ROWS:], so visible y = engine py - HIDDEN_ROWS.
        piece = state.pop("piece")
        x = state.pop("piece_x")
        y = state.pop("piece_y")
        rot = state.pop("piece_rot")
        state["piece"] = (
            {"type": piece, "x": x, "y": y - HIDDEN_ROWS, "rot": rot}
            if piece is not None
            else None
        )
        return protocol.build_snapshot(self.seq.next(), state)

    def handle_client_message(self, msg: dict) -> list[dict]:
        """Dispatch one parsed client message; returns outgoing messages."""
        if not isinstance(msg, dict):
            return []
        mtype = msg.get("type")
        if mtype == "command":
            name = msg.get("cmd")  # frontend wire format
            if not isinstance(name, str):
                name = msg.get("command")  # protocol.py canonical format
            if not isinstance(name, str):
                return []
            # soft drop release: frontend sends phase:"up", protocol.py uses
            # release:true; soft_drop with neither means press. A release
            # with no prior down is a harmless engine-side no-op.
            if name == "soft_drop" and (
                msg.get("phase") == "up" or msg.get("release") is True
            ):
                name = "soft_drop_release"
            try:
                self.game.submit(name)  # buffered; applied on the next tick
            except ValueError:
                return []  # not in the engine vocabulary — ignored
            return []
        if mtype == "ack":
            return [{"type": "ack"}]
        if mtype == "resync":
            # Client detected a gap / reconnected: ack plus a fresh full
            # snapshot with the next seq from the same counter.
            return [{"type": "ack"}, self.build_snapshot()]
        return []

    def handle_raw(self, raw: str | bytes) -> list[dict]:
        """Parse raw client text; malformed/unknown input is ignored."""
        try:
            msg = protocol.parse_message(raw)
        except ValueError:
            return []
        return self.handle_client_message(msg)

    def _game_over_messages(self) -> list[dict]:
        state = self.game.snapshot()
        over = protocol.make_game_over(
            self.seq.next(), state["score"], state["lines"], state["level"]
        )
        return [over, self.build_snapshot()]  # final snapshot, fresh seq

    def tick(self) -> list[dict]:
        """One fixed sim step + snapshot; after game over, nothing."""
        if self.game.game_over:
            if self.game_over_notified:
                return []  # sim stopped after the notification
            self.game_over_notified = True
            return self._game_over_messages()
        self.game.step(TICK_MS)
        if self.game.game_over:  # top-out caused by this very tick
            self.game_over_notified = True
            return self._game_over_messages()
        return [self.build_snapshot()]


async def _ticker(session: Session, out: "asyncio.Queue") -> None:
    """Fixed-step sim loop paced by absolute deadline (next_t += TICK_MS) so
    the loop stays near 60Hz even when a tick runs late; a fixed dt is fine
    if it occasionally slips."""
    loop = asyncio.get_running_loop()
    next_t = loop.time()
    while True:
        for msg in session.tick():
            out.put_nowait(msg)
        if session.game_over_notified:
            return  # sim stops; receiver/sender keep the socket alive
        next_t += TICK_MS
        delay = next_t - loop.time()
        await asyncio.sleep(delay if delay > 0 else 0.0)


async def _receiver(ws: WebSocket, session: Session, out: "asyncio.Queue") -> None:
    while True:
        raw = await ws.receive_text()
        for msg in session.handle_raw(raw):
            out.put_nowait(msg)


async def _sender(ws: WebSocket, out: "asyncio.Queue") -> None:
    while True:
        msg = await out.get()
        await ws.send_text(protocol.encode_message(msg))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(game=create_game())
    out: asyncio.Queue = asyncio.Queue()
    # Initial snapshot so the client has state before the first tick; it
    # draws seq 1 from the session counter (strictly monotonic stream).
    await out.put(session.build_snapshot())
    tasks = [
        asyncio.create_task(_ticker(session, out)),
        asyncio.create_task(_receiver(ws, session, out)),
        asyncio.create_task(_sender(ws, out)),
    ]
    try:
        await asyncio.gather(*tasks)
    except WebSocketDisconnect:
        pass
    finally:
        # Any disconnect/error tears everything down; the session is dropped.
        for task in tasks:
            task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            # TestClient's teardown cancels the app's scope right after the
            # disconnect; that cancellation can land inside this reap await.
            # The child tasks are already cancelled and will finish on their
            # own, so the reap being interrupted is harmless.
            pass


# Mounted after the /ws route so the socket endpoint wins routing.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
