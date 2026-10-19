"""Pure Tetris game logic — no asyncio, no networking.

The server drives one Game at 60Hz fixed steps via step(dt_ms). Players feed
commands through submit(); commands are buffered and applied at the next
simulation step. All timing (gravity, lock delay, ARE) is an accumulator in
milliseconds so the sim is deterministic and testable.

Coordinate/timing conventions match pieces.py and board.py:
- Grid 10 x 22, rows 0-1 hidden spawn area, rows 2-21 visible.
- Piece position = bounding-box origin (px, py); cells from pieces.cells().
- Spawn origin (3, 0) for all pieces (cells land in hidden rows 0-1).
- All timers in milliseconds, accumulated per step.
"""

from __future__ import annotations

from .board import Board
from .pieces import cells, kick_offsets, rotation_of
from .rng import Bag7

GRAVITY_MS = {1: 1000, 2: 793, 3: 618, 4: 473, 5: 355, 6: 262, 7: 190, 8: 135, 9: 94, 10: 64}
LOCK_DELAY_MS = 500
MAX_LOCK_RESETS = 15
ARE_MS = 0                 # spawn delay after a lock with NO line clear (instant spawn)
LINE_CLEAR_DELAY_MS = 400  # spawn delay after a lock that cleared lines; input buffers across it
SOFT_DROP_FACTOR = 20
NEXT_QUEUE_LEN = 5

SPAWN_X = 3
SPAWN_Y = 0

# Base points by (tspin_type, lines_cleared): tspin_type in (None, "full", "mini").
_BASE_POINTS = {
    (None, 0): 0,
    (None, 1): 100, (None, 2): 300, (None, 3): 500, (None, 4): 800,
    ("full", 0): 400, ("full", 1): 800, ("full", 2): 1200, ("full", 3): 1600,
    ("mini", 0): 100, ("mini", 1): 100, ("mini", 2): 200,
}
_EVENT_NAMES = {
    (None, 1): "single", (None, 2): "double", (None, 3): "triple", (None, 4): "tetris",
    ("full", 0): "tspin", ("full", 1): "tspin_single",
    ("full", 2): "tspin_double", ("full", 3): "tspin_triple",
    ("mini", 0): "tspin_mini", ("mini", 1): "tspin_mini_single",
    ("mini", 2): "tspin_mini_double",
}


class Game:
    """Fixed-step Tetris simulation: input buffering, gravity, lock delay,
    ARE, hold, T-spin detection, scoring, top-out.

    Pure synchronous logic. The server calls step() at a fixed rate and
    reads snapshot() to build protocol messages.
    """

    def __init__(self, seed: int | None = None, start_level: int = 1) -> None:
        self.board = Board()
        self.bag = Bag7(seed)
        self.start_level = start_level
        self.level = start_level
        self.score = 0
        self.lines = 0
        self.combo = -1          # consecutive line-clearing locks; -1 = no chain
        self.b2b = False
        self.game_over = False
        self.hold: str | None = None
        self.hold_used = False
        self.soft_drop = False
        self.cur: str | None = None
        self.px = SPAWN_X
        self.py = SPAWN_Y
        self.prot = 0
        self.g_ms = 0.0
        self.lock_ms = 0.0
        self.lock_resets = 0
        self.are_ms = 0.0
        self.last_action_rotation = False  # last successful positional action was a rotation
        self.last_kick_index: int | None = None
        self.last_event: str | None = None
        self._queue: list[str] = []
        self._spawn_next()

    # ------------------------------------------------------------------ input

    def submit(self, cmd: str) -> None:
        """Buffer one command for application on the next step(). Raises
        ValueError for commands outside the shared vocabulary."""
        from .. import protocol  # local import to avoid any import-order issues

        if not protocol.is_valid_command(cmd):
            raise ValueError(f"unknown command: {cmd!r}")
        self._queue.append(cmd)

    def _apply_commands(self) -> None:
        """Drain the buffered commands in order while a piece is active.
        Commands arriving during ARE stay queued until the new piece spawns
        (they are applied right after spawn, in the same step)."""
        while self._queue and self.cur is not None:
            self._apply_one(self._queue.pop(0))

    def _apply_one(self, cmd: str) -> None:
        if cmd == "move_left":
            self._try_move(-1)
        elif cmd == "move_right":
            self._try_move(1)
        elif cmd == "rotate_cw":
            self._try_rotate(1)
        elif cmd == "rotate_ccw":
            self._try_rotate(-1)
        elif cmd == "soft_drop":
            self.soft_drop = True
        elif cmd == "soft_drop_release":
            self.soft_drop = False
        elif cmd == "hard_drop":
            self._hard_drop()
        elif cmd == "hold":
            self._hold()

    # ------------------------------------------------------------- mechanics

    def _grounded(self) -> bool:
        return self.board.collides(cells(self.cur, self.prot), self.px, self.py + 1)

    def _gravity_ms(self) -> float:
        g = GRAVITY_MS[min(self.level, 10)]
        if self.soft_drop:
            return g / SOFT_DROP_FACTOR
        return g

    def _advance(self, dt: float) -> None:
        """Gravity / lock-delay progress for the active piece over dt ms."""
        if self.cur is None:
            return
        if self._grounded():
            self.lock_ms += dt
            if self.lock_ms >= LOCK_DELAY_MS:
                self._lock()
            return
        g = self._gravity_ms()
        self.g_ms += dt
        while self.g_ms >= g:
            if self.board.collides(cells(self.cur, self.prot), self.px, self.py + 1):
                # Landed: gravity stops accumulating, lock delay starts.
                self.g_ms = 0.0
                self.lock_ms = 0.0
                break
            self.py += 1
            self.g_ms -= g
            self.last_action_rotation = False
            self.last_kick_index = None
            if self.soft_drop:
                self.score += 1  # soft drop: 1 pt per cell, not level-multiplied

    def _try_move(self, dx: int) -> None:
        if self.board.collides(cells(self.cur, self.prot), self.px + dx, self.py):
            return
        was_grounded = self._grounded()
        self.px += dx
        self.last_action_rotation = False
        self.last_kick_index = None
        if was_grounded and self._grounded():
            self._note_lock_reset()

    def _try_rotate(self, delta: int) -> None:
        if self.cur == "O":
            return  # O rotation is identity; nothing to do
        new_rot = rotation_of(self.prot, delta)
        for i, (dx, dy) in enumerate(kick_offsets(self.cur, self.prot, new_rot)):
            nx, ny = self.px + dx, self.py + dy
            if not self.board.collides(cells(self.cur, new_rot), nx, ny):
                was_grounded = self._grounded()
                self.px, self.py = nx, ny
                self.prot = new_rot
                self.last_action_rotation = True
                self.last_kick_index = i
                if was_grounded and self._grounded():
                    self._note_lock_reset()
                return
        # All kicks failed: rotation does not happen, state unchanged.

    def _note_lock_reset(self) -> None:
        """Successful move/rotation while grounded: reset the lock timer,
        counting against the 15-reset cap. After 15 resets the piece locks
        500ms after the last (uncounted) reset."""
        if self.lock_resets < MAX_LOCK_RESETS:
            self.lock_ms = 0.0
            self.lock_resets += 1

    def _hard_drop(self) -> None:
        dist = 0
        while not self.board.collides(cells(self.cur, self.prot), self.px, self.py + 1):
            self.py += 1
            dist += 1
        if dist > 0:
            self.score += 2 * dist  # hard drop: 2 pts per cell, not level-multiplied
            self.last_action_rotation = False  # movement breaks T-spin eligibility
            self.last_kick_index = None
        self._lock()  # instant lock, bypasses lock delay

    def _hold(self) -> None:
        if self.hold_used:
            return  # once per piece
        held = self.hold
        self.hold = self.cur
        if held is None:
            self._spawn(self.bag.next_piece())
        else:
            self._spawn(held)
        self.hold_used = True  # NOT cleared by the swap's own spawn

    # ------------------------------------------------------------- lifecycle

    def _spawn(self, piece_id: str) -> None:
        """Fresh spawn: origin (3, 0), all timers reset. Block-out = game over."""
        self.cur = piece_id
        self.px = SPAWN_X
        self.py = SPAWN_Y
        self.prot = 0
        self.g_ms = 0.0
        self.lock_ms = 0.0
        self.lock_resets = 0
        self.last_action_rotation = False
        self.last_kick_index = None
        if self.board.collides(cells(piece_id, self.prot), self.px, self.py):
            self.game_over = True
            self.cur = None

    def _spawn_next(self) -> None:
        self._spawn(self.bag.next_piece())
        if not self.game_over:
            self.hold_used = False  # hold available again for the new piece

    def _lock(self) -> None:
        """Merge the piece, detect T-spin (before merging), clear lines,
        score, then spawn-delay (ARE) or top out."""
        piece = self.cur
        tspin = self._detect_tspin()
        abs_cells = [(self.px + cx, self.py + cy) for cx, cy in cells(piece, self.prot)]
        cleared = self.board.lock_cells(abs_cells, piece)
        n = len(cleared)
        if n:
            # A clearing lock dissolves the piece: cells of the piece that
            # are not part of a completed row are removed along with the
            # clear, so no debris is left behind. Debris only remains when
            # the lock clears nothing (documented engine semantics used by
            # the test suite's multi-clear scenarios).
            cleared_rows = set(cleared)
            for ax, ay in abs_cells:
                if ay not in cleared_rows:
                    self.board.grid[ay][ax] = "."
            self.board.clear_rows(cleared)

        # Combo: consecutive line-clearing locks.
        if n > 0:
            self.combo += 1
        else:
            self.combo = -1

        # Event + base points.
        event = _EVENT_NAMES.get((tspin, n))
        base = _BASE_POINTS[(tspin, n)]

        # Back-to-back: tetris or any T-spin line clear, chained after the
        # previous special clear. x1.5 applies to base points only; a normal
        # single/double/triple breaks the chain; T-spins/no-clear locks and
        # non-clearing locks leave it unchanged.
        special = False
        if n > 0:
            special = tspin is not None or n == 4
            if special:
                if self.b2b:
                    base = int(base * 1.5)
                self.b2b = True
            else:
                self.b2b = False

        points = base * self.level
        # Combo: every clear extends the consecutive-clear chain, but the
        # 50xcomboxlevel bonus applies to normal clears only — tetris and
        # T-spin clears score through the B2B multiplier instead.
        if n > 0 and self.combo >= 1 and not special:
            points += 50 * self.combo * self.level
        self.score += points

        self.lines += n
        self.level = self.start_level + self.lines // 10
        self.last_event = event if (n > 0 or tspin is not None) else None

        # Lock-out: cells remain in the hidden rows after clearing.
        if self.board.any_above_visible():
            self.game_over = True
            self.cur = None
            return

        self.cur = None
        if n > 0:
            self.are_ms = LINE_CLEAR_DELAY_MS
        else:
            self._spawn_next()  # ARE_MS == 0: instant spawn

    # --------------------------------------------------------------- T-spin

    def _detect_tspin(self) -> str | None:
        """3-corner rule with T-slot facing checks. Full when >= 3 of the 4
        diagonal corners around the T center are occupied (walls/floor count);
        mini when exactly 2 are and both FRONT corners (relative to the nub
        direction) are occupied. A mini performed with the 5th (far) kick
        offset upgrades to full. Only applies when the last successful
        positional action was a rotation."""
        if self.cur != "T" or not self.last_action_rotation:
            return None
        cx, cy = self.px + 1, self.py + 1  # center of the 3x3 box
        corner_coords = [(cx - 1, cy - 1), (cx + 1, cy - 1), (cx - 1, cy + 1), (cx + 1, cy + 1)]

        def occupied(pt) -> bool:
            x, y = pt
            if x < 0 or x >= 10 or y < 0 or y >= 22:
                return True  # walls/floor count as occupied
            return self.board.grid[y][x] != "."

        occ = [occupied(pt) for pt in corner_coords]
        n_occ = sum(occ)
        # Front corners per pointing direction (0=up, 1=right, 2=down, 3=left).
        front_idx = {0: (0, 1), 1: (1, 3), 2: (2, 3), 3: (0, 2)}[self.prot]
        if n_occ >= 3:
            return "full"
        if n_occ == 2 and occ[front_idx[0]] and occ[front_idx[1]]:
            if self.last_kick_index == 4:
                return "full"  # far-kick upgrade
            return "mini"
        return None

    # ------------------------------------------------------------------ loop

    def step(self, dt_ms: float) -> None:
        """Advance the simulation by dt_ms (the server drives this at 60Hz
        fixed steps; any positive dt is handled, including state transitions
        mid-step: ARE ending hands the remainder to the new piece)."""
        dt = float(dt_ms)
        if dt <= 0:
            return
        if self.game_over:
            self._queue.clear()  # drain and ignore
            return
        if self.are_ms > 0:
            consumed = min(dt, self.are_ms)
            self.are_ms -= consumed
            remaining = dt - consumed
            if self.are_ms <= 0:
                self.are_ms = 0.0
                self._spawn_next()
                if self.game_over:
                    return
                # Inputs buffered across the line-clear delay apply to the
                # fresh piece, then the remainder of dt runs.
                self._apply_commands()
                if remaining > 0:
                    self._advance(remaining)
            return
        self._apply_commands()
        self._advance(dt)

    # -------------------------------------------------------------- snapshot

    def snapshot(self) -> dict:
        """Full observable state, exactly the protocol snapshot key set."""
        active = self.cur is not None
        return {
            "board": self.board.serialize(),
            "hold": self.hold,
            "next": self.bag.peek(NEXT_QUEUE_LEN),
            "score": self.score,
            "lines": self.lines,
            "level": self.level,
            "combo": max(self.combo, 0),
            "b2b": self.b2b,
            "game_over": self.game_over,
            "piece": self.cur,
            "piece_x": self.px if active else None,
            "piece_y": self.py if active else None,
            "piece_rot": self.prot if active else None,
            "lock_ms": int(round(self.lock_ms)) if active else 0,
            "lock_resets": self.lock_resets if active else 0,
            "are_ms": int(round(self.are_ms)),
            "soft_drop": self.soft_drop,
        }
