"""LLM player: placement enumeration and command translation.

Pure logic — no networking here (see jev_api.py for the API client).
Imports only the engine; the engine never imports this module.

Enumeration model
-----------------
For the active piece we enumerate every placement reachable by the exact
command path the player will use: rotation (0-3, no wall kicks — a rotation
that requires a kick at the piece's current position is skipped so the
translated command sequence is guaranteed to reproduce the preview) ->
horizontal moves -> hard drop, starting from the piece's LIVE position.
Gravity runs during the decision pause/brain call, so the piece is generally
no longer at the spawn row when the commands are applied; the reachability
and landing checks therefore use the live (px, py), not the spawn state.

Because enumeration simulates exactly the command sequence that translate()
emits, executing the translated commands reproduces the preview board
bit-for-bit (the critical invariant asserted in tests).

Preview = the 20 visible board rows after the hard-drop lock + line clears,
in the same wire format the protocol uses.
"""

from __future__ import annotations

from typing import List, Optional

from ..engine.board import WIDTH, TOTAL_ROWS, Board
from ..engine.pieces import cells


class Placement:
    """One enumerated placement option."""

    __slots__ = ("id", "piece", "x", "y", "rot", "hold", "preview",
                 "lines_cleared", "tspin")

    def __init__(self, id: str, piece: str, x: int, y: int, rot: int,
                 hold: bool, preview: List[str], lines_cleared: int,
                 tspin: bool = False) -> None:
        self.id = id
        self.piece = piece        # the piece that gets placed (swapped-in for hold)
        self.x = x                # final bounding-box x (engine coords)
        self.y = y                # final bounding-box y (engine coords)
        self.rot = rot            # final rotation state
        self.hold = hold          # True if this is the hold variant
        self.preview = preview    # 20 visible rows after lock + clears
        self.lines_cleared = lines_cleared
        self.tspin = tspin


def _copy_board(board: Board) -> Board:
    """Shallow copy of a Board sharing no mutable state with the original."""
    nb = Board()
    nb.grid = [row[:] for row in board.grid]
    return nb


def _tspin_3corner(board: Board, x: int, y: int) -> bool:
    """3-corner T-spin rule for a T piece whose bounding box is at (x, y):
    at least 3 of the 4 box corners must be occupied (walls/floor count)."""
    occupied = 0
    for cx, cy in ((x, y), (x + 2, y), (x, y + 2), (x + 2, y + 2)):
        if cx < 0 or cx >= WIDTH or cy < 0 or cy >= TOTAL_ROWS:
            occupied += 1
        elif board.grid[cy][cx] != ".":
            occupied += 1
    return occupied >= 3


def enumerate_placements(game) -> List[Placement]:
    """All reachable placements for the game's active piece, plus the hold
    variant when hold is still available for this piece."""
    if game.cur is None:
        return []

    variants = [(game.cur, False)]
    if not game.hold_used:
        # When the hold slot is empty, holding swaps in the NEXT bag piece;
        # peek() does not consume, so the bag state is unchanged.
        swapped = game.hold if game.hold is not None else game.bag.peek(1)[0]
        variants.append((swapped, True))

    # The piece's LIVE position: gravity runs during the decision pause and
    # the brain call, so the piece is no longer at the spawn row by the time
    # translate()'s commands are applied. Every reachability check below must
    # use this row — checking y=0 (the old behavior) wrongly accepted paths
    # the stack blocks lower down, and the engine then silently refused the
    # horizontal moves, so the piece landed short of the target column.
    px, py = game.px, game.py

    options: List[Placement] = []
    for piece, is_hold in variants:
        for rot in range(4):
            shape = cells(piece, rot)
            # Only rotations that fit at the piece's CURRENT position WITHOUT
            # a wall kick are enumerated — this guarantees the preview
            # invariant (translate() rotates in place, so a kick would shift
            # the piece and break it).
            if game.board.collides(shape, px, py):
                continue
            for x in range(WIDTH):
                # The piece must be able to travel horizontally from its
                # current x to x at its current row y.
                dx = x - px
                if dx > 0:
                    blocked = any(
                        game.board.collides(shape, sx, py)
                        for sx in range(px + 1, x + 1)
                    )
                elif dx < 0:
                    blocked = any(
                        game.board.collides(shape, sx, py)
                        for sx in range(px - 1, x - 1, -1)
                    )
                else:
                    blocked = False
                if blocked:
                    continue
                # Hard-drop simulation: find the landing y from the piece's
                # CURRENT row (it has already fallen below the spawn row).
                y = py
                while not game.board.collides(shape, x, y + 1):
                    y += 1
                # Lock on a copy, clear the completed rows.
                sim_board = _copy_board(game.board)
                abs_cells = [(x + cx, y + cy) for cx, cy in shape]
                cleared_rows = sim_board.lock_cells(abs_cells, piece)
                if cleared_rows:
                    sim_board.clear_rows(cleared_rows)
                preview = sim_board.serialize()
                # T-spin: only when the last action before the drop is the
                # rotation itself (dx == 0) — a horizontal move after the
                # rotation breaks T-spin eligibility, matching the engine.
                tspin = False
                if piece == "T" and rot != 0 and dx == 0:
                    tspin = _tspin_3corner(game.board, x, y)
                prefix = "hold_" if is_hold else ""
                options.append(Placement(
                    id=f"{prefix}r{rot}_c{x}",
                    piece=piece,
                    x=x, y=y, rot=rot,
                    hold=is_hold,
                    preview=preview,
                    lines_cleared=len(cleared_rows),
                    tspin=tspin,
                ))
    return options


def translate(option: Placement) -> List[str]:
    """Placement -> command sequence through the shared vocabulary:
    [hold] + rotations (shortest path) + move_left/right + hard_drop."""
    cmds: List[str] = []
    if option.hold:
        cmds.append("hold")
    if option.rot == 3:
        cmds.append("rotate_ccw")          # 1 step beats 3 CW
    elif option.rot > 0:
        cmds.extend(["rotate_cw"] * option.rot)
    dx = option.x - 3
    if dx > 0:
        cmds.extend(["move_right"] * dx)
    elif dx < 0:
        cmds.extend(["move_left"] * (-dx))
    cmds.append("hard_drop")
    return cmds


def _holes(preview: List[str]) -> int:
    """Empty cells below each column's topmost filled cell."""
    holes = 0
    for x in range(WIDTH):
        seen = False
        for row in preview:
            if row[x] != ".":
                seen = True
            elif seen:
                holes += 1
    return holes


def _max_height(preview: List[str]) -> int:
    """Tallest column height in visible rows."""
    for i, row in enumerate(preview):
        if any(c != "." for c in row):
            return len(preview) - i
    return 0


def pick_best_option(options: List[Placement]) -> Optional[Placement]:
    """Heuristic fallback: max lines cleared, then fewest holes, then lowest
    stack height. Deterministic; used whenever Jev fails or answers garbage."""
    if not options:
        return None
    return min(
        options,
        key=lambda o: (-o.lines_cleared, _holes(o.preview), _max_height(o.preview)),
    )


def build_state(game) -> dict:
    """Current game state for the API payload (pre-placement board)."""
    return {
        "board": game.board.serialize(),
        "piece": game.cur,
        "hold": game.hold,
        "next": game.bag.peek(5),
        "level": game.level,
        "lines": game.lines,
    }
