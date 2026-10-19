"""Tetromino definitions, SRS rotation states, and SRS wall-kick tables.

Coordinate convention: x increases right, y increases DOWN (screen-like).
A rotation state is a tuple of (x, y) cell offsets relative to the piece
bounding-box top-left. Rotation states are indexed 0=spawn, 1=R (CW from
spawn), 2=180, 3=L (CCW).

The four rotation states of each piece are derived programmatically by
clockwise rotation inside the piece's bounding box: (x, y) -> (N-1-y, x)
for an N x N box. JLSTZ live in a 3x3 box, I in a 4x4 box. The O piece is
the exception: it occupies a 2x2 block, so rotation is identity and all
four states are identical.

The wall-kick tables below are the published SRS tables converted to this
y-down convention (dy = -(published dy)). Antisymmetry of each table (the
set of negated offsets for transition (a, b) equals the offset set for
(b, a)) is asserted at import time.
"""

PIECE_IDS = ("I", "J", "L", "O", "S", "T", "Z")

COLORS = {
    "I": "#00FFFF",  # cyan
    "J": "#0000FF",  # blue
    "L": "#FF7F00",  # orange
    "O": "#FFFF00",  # yellow
    "S": "#00FF00",  # green
    "T": "#800080",  # purple
    "Z": "#FF0000",  # red
}

# Spawn-state cells (y-down), box-origin relative.
_SPAWN = {
    "T": ((1, 0), (0, 1), (1, 1), (2, 1)),  # nub up, bar bottom
    "J": ((0, 0), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 0), (0, 1), (1, 1), (2, 1)),
    "S": ((1, 0), (2, 0), (0, 1), (1, 1)),
    "Z": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "I": ((0, 1), (1, 1), (2, 1), (3, 1)),  # 4x4 box
    "O": ((1, 0), (2, 0), (1, 1), (2, 1)),  # 3x3 box, all states identical
}

_BOX_SIZE = {"I": 4}  # default 3 for everything except I


def _cw(cells, n):
    """Clockwise rotation of a cell set inside an n x n box (y-down)."""
    return tuple((n - 1 - y, x) for x, y in cells)


def _build_states():
    states = {}
    for piece, spawn in _SPAWN.items():
        if piece == "O":
            rots = [spawn] * 4  # rotation is identity for O
        else:
            n = _BOX_SIZE.get(piece, 3)
            rots = [spawn]
            for _ in range(3):
                rots.append(_cw(rots[-1], n))
        states[piece] = tuple(tuple(sorted(s)) for s in rots)
    return states


_STATES = _build_states()

# Published SRS wall-kick tables, converted to y-down (dy = -(published dy)).
# Keyed by (from_rotation, to_rotation) for all 8 directed transitions.
KICKS_JLSTZ = {
    (0, 1): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (1, 0): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (1, 2): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (2, 1): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (2, 3): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (3, 2): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (3, 0): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (0, 3): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
}

KICKS_I = {
    (0, 1): ((0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)),
    (1, 0): ((0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)),
    (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)),
    (2, 1): ((0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)),
    (2, 3): ((0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)),
    (3, 2): ((0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)),
    (3, 0): ((0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)),
    (0, 3): ((0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)),
}

_KICKS_O = ((0, 0),)

# Import-time sanity check: both tables are antisymmetric as sets.
for _tbl in (KICKS_JLSTZ, KICKS_I):
    assert len(_tbl) == 8
    for _key, _offs in _tbl.items():
        _a, _b = _key
        assert len(_offs) == 5 and _offs[0] == (0, 0), _key
        assert {(-dx, -dy) for dx, dy in _offs} == set(_tbl[(_b, _a)]), _key


def cells(piece: str, rotation: int) -> tuple[tuple[int, int], ...]:
    """Cells (x, y) of `piece` in rotation state `rotation` (0..3),
    box-origin relative."""
    try:
        states = _STATES[piece]
    except KeyError:
        raise ValueError(f"unknown piece: {piece!r}") from None
    return states[rotation % 4]


def kick_offsets(piece: str, from_rotation: int, to_rotation: int) -> tuple[tuple[int, int], ...]:
    """SRS wall-kick offsets to try in order for rotating
    from_rotation -> to_rotation. Offsets are (dx, dy) in the same y-down
    convention as cells() (dy positive = down). O always returns ((0, 0),).
    Invalid transitions raise ValueError."""
    if piece == "O":
        return _KICKS_O
    if piece == "I":
        table = KICKS_I
    elif piece in ("J", "L", "S", "T", "Z"):
        table = KICKS_JLSTZ
    else:
        raise ValueError(f"unknown piece: {piece!r}")
    try:
        return table[(from_rotation, to_rotation)]
    except KeyError:
        raise ValueError(
            f"invalid rotation transition for {piece}: "
            f"{from_rotation} -> {to_rotation}"
        ) from None


def rotation_of(rotation: int, delta: int) -> int:
    """(rotation + delta) % 4 helper."""
    return (rotation + delta) % 4
