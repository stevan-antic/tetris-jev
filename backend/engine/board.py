"""Playfield: 10-wide grid, 22 rows (2 hidden spawn rows on top + 20 visible).

Coordinate convention (matches pieces.py): x increases right (col 0 = left),
y increases DOWN (row 0 = top). The grid is 22 rows tall; rows 0-1 are the
hidden spawn area above the 20-row visible field (render layer strips the
hidden rows; game.snapshot() exposes only grid[2:] via serialize()).

Cells passed to collides()/lock_cells() are the box-relative offsets from
pieces.cells(piece, rot) plus the box origin (x, y) of the piece; the
absolute cell position is (x + cx, y + cy).
"""

WIDTH = 10
TOTAL_ROWS = 22
HIDDEN_ROWS = 2


class Board:
    """Minimal, pure playfield: occupancy, collision, lock, line clears.

    No timers, no RNG, no scoring. Grid stores '.' for empty cells and the
    piece letter for locked cells.
    """

    def __init__(self) -> None:
        self.grid: list[list[str]] = [["."] * WIDTH for _ in range(TOTAL_ROWS)]

    def collides(self, cells, x: int, y: int) -> bool:
        """True if any box-relative cell (cx, cy) placed at box origin (x, y)
        — i.e. absolute position (x + cx, y + cy) — falls outside cols 0-9,
        outside rows 0-21, or onto an occupied grid cell."""
        for cx, cy in cells:
            ax = x + cx
            ay = y + cy
            if ax < 0 or ax >= WIDTH:
                return True
            if ay < 0 or ay >= TOTAL_ROWS:
                return True
            if self.grid[ay][ax] != ".":
                return True
        return False

    def lock_cells(self, cells, piece_id: str) -> list[int]:
        """Merge ABSOLUTE cells — already-offset (x + cx, y + cy) positions —
        into the grid. Returns the sorted list of ABSOLUTE row indices that
        became completely filled. Filled rows are NOT removed here — the
        caller decides when to clear.

        For the box-relative convention (cells straight from pieces.cells()
        plus the piece origin), use lock_cells_at().
        """
        return self._merge(cells, piece_id)

    def lock_cells_at(self, cells, x: int, y: int, piece_id: str) -> list[int]:
        """Box-relative variant: merge ``cells`` (offsets from
        pieces.cells()) at box origin (x, y). Returns sorted absolute row
        indices that became complete."""
        return self._merge([(x + cx, y + cy) for cx, cy in cells], piece_id)

    def _merge(self, absolute_cells, piece_id: str) -> list[int]:
        rows_touched = set()
        for ax, ay in absolute_cells:
            self.grid[ay][ax] = piece_id
            rows_touched.add(ay)
        cleared: list[int] = []
        for row in sorted(rows_touched):
            if all(self.grid[row][cx] != "." for cx in range(WIDTH)):
                cleared.append(row)
        return cleared

    def clear_rows(self, rows: list[int]) -> None:
        """Remove the given absolute rows (they must be full), shifting
        everything above down and inserting empty rows on top."""
        rows_set = set(rows)
        new_grid = [row for idx, row in enumerate(self.grid) if idx not in rows_set]
        while len(new_grid) < TOTAL_ROWS:
            new_grid.insert(0, ["."] * WIDTH)
        self.grid = new_grid

    def serialize(self) -> list[str]:
        """The 20 VISIBLE rows (grid[HIDDEN_ROWS:]) as strings, '.' empty or
        the piece letter. This exact format is the WS protocol board format."""
        return ["".join(row) for row in self.grid[HIDDEN_ROWS:]]

    def any_above_visible(self) -> bool:
        """True if any occupied cell sits in the hidden rows 0-1 (lock-out)."""
        for row in self.grid[:HIDDEN_ROWS]:
            if any(c != "." for c in row):
                return True
        return False

    def is_empty(self) -> bool:
        """True if every cell in the entire grid (hidden + visible) is empty."""
        return all(c == "." for row in self.grid for c in row)
