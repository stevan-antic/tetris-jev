"""WebSocket message types and serialization for the tetris-jev backend.

Framework-agnostic: plain JSON dicts, no FastAPI/asyncio imports, so the
server task can build on this and tests can round-trip it directly.

Message vocabulary (the ``type`` field):

- ``snapshot``       full game state, carries a monotonically increasing seq
- ``game_over``      terminal state notification with final score/lines/level
- ``ack``            client -> server: acknowledges latest snapshot seq
- ``resync``         client -> server: seq gap detected, requests full snapshot
- ``command``        client -> server: a player command (optionally timestamped)
- ``option_request`` server -> client: LLM-mode placement options
- ``option_choice``  client -> server: LLM-mode chosen option index
"""

from __future__ import annotations

import json

# Shared command vocabulary shared by human and LLM players. ``soft_drop`` is
# the press half of a hold-state pair; ``soft_drop_release`` is the release.
COMMANDS = (
    "move_left",
    "move_right",
    "rotate_cw",
    "rotate_ccw",
    "soft_drop",
    "soft_drop_release",
    "hard_drop",
    "hold",
)

# All message types on the wire.
MESSAGE_TYPES = frozenset(
    ("snapshot", "game_over", "ack", "resync", "command", "option_request", "option_choice")
)

# Board serialization alphabet: '.' empty, otherwise a locked piece id.
_BOARD_CHARS = frozenset("IJLOSTZ.")
_BOARD_ROWS = 20
_BOARD_COLS = 10


def is_valid_command(name: str) -> bool:
    """Return True if ``name`` is a known command in the shared vocabulary."""
    return name in COMMANDS


class SeqCounter:
    """Monotonically increasing sequence numbers for snapshots.

    Every snapshot carries a seq; clients ack the latest and send ``resync``
    when they detect a gap. Starts at 0, first issued value is ``start + 1``.
    """

    def __init__(self, start: int = 0) -> None:
        self._next = start

    def next(self) -> int:
        """Issue the next sequence number and advance the counter."""
        self._next += 1
        return self._next

    @property
    def current(self) -> int:
        """The last sequence number issued (initially ``start``)."""
        return self._next


def _validate_board(board: list) -> None:
    """Raise ValueError unless board is 20 rows of 10-char strings in the alphabet."""
    if not isinstance(board, list) or len(board) != _BOARD_ROWS:
        raise ValueError(
            f"board must be a list of {_BOARD_ROWS} rows, got {type(board).__name__}"
        )
    for i, row in enumerate(board):
        if not isinstance(row, str) or len(row) != _BOARD_COLS:
            raise ValueError(
                f"board row {i} must be a {_BOARD_COLS}-char string, got {row!r}"
            )
        bad = set(row) - _BOARD_CHARS
        if bad:
            raise ValueError(f"board row {i} has invalid chars: {sorted(bad)}")


def build_snapshot(seq: int, state: dict) -> dict:
    """Build a ``snapshot`` message from the engine's state dict.

    ``state`` keys are passed through verbatim; the fixed contract is:
    board (20 x 10 strings), hold, next, score, lines, level, combo, b2b,
    game_over, piece, piece_x, piece_y, piece_rot, lock_ms, lock_resets,
    are_ms, soft_drop. The piece position fields may be None when no piece
    is active.
    """
    if not isinstance(seq, int) or isinstance(seq, bool):
        raise ValueError(f"seq must be an int, got {seq!r}")
    _validate_board(state["board"])
    msg = {"type": "snapshot", "seq": seq}
    msg.update(state)
    return msg


def make_ack(seq: int) -> dict:
    """Build a client ``ack`` message for the latest received snapshot seq."""
    return {"type": "ack", "seq": seq}


def make_resync(seq: int) -> dict:
    """Build a client ``resync`` message carrying the last seq it has."""
    return {"type": "resync", "seq": seq}


def make_game_over(seq: int, score: int, lines: int, level: int) -> dict:
    """Build a ``game_over`` message with final stats."""
    return {
        "type": "game_over",
        "seq": seq,
        "score": score,
        "lines": lines,
        "level": level,
    }


def make_command(
    command: str, ts: float | None = None, release: bool = False
) -> dict:
    """Build a ``command`` message.

    ``ts`` is an optional client timestamp; ``release=True`` marks the
    release half of a hold-state pair (soft_drop press/release).
    """
    if not is_valid_command(command):
        raise ValueError(f"unknown command: {command!r}")
    msg: dict = {"type": "command", "command": command, "release": release}
    if ts is not None:
        msg["ts"] = ts
    return msg


def make_option_request(seq: int, options: list) -> dict:
    """Build an ``option_request`` message (LLM mode): placement options."""
    return {"type": "option_request", "seq": seq, "options": options}


def make_option_choice(option: int) -> dict:
    """Build an ``option_choice`` message (LLM mode): chosen option index."""
    if not isinstance(option, int) or isinstance(option, bool):
        raise ValueError(f"option must be an int index, got {option!r}")
    return {"type": "option_choice", "option": option}


def encode_message(msg: dict) -> str:
    """Serialize a message dict to compact JSON."""
    return json.dumps(msg, separators=(",", ":"))


def parse_message(raw: str | bytes) -> dict:
    """Parse raw JSON into a message dict; ValueError on any malformed input."""
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc
    if not isinstance(msg, dict):
        raise ValueError(f"message must be a JSON object, got {type(msg).__name__}")
    if "type" not in msg:
        raise ValueError("message is missing the 'type' field")
    if msg["type"] not in MESSAGE_TYPES:
        raise ValueError(f"unknown message type: {msg['type']!r}")
    return msg


def parse_command(msg: dict) -> tuple[str, bool]:
    """Extract (command, release) from a ``command`` message.

    Raises ValueError if the message is not a command, the command name is
    missing/not a string, or it is not in the shared vocabulary.
    """
    if msg.get("type") != "command":
        raise ValueError(f"not a command message: {msg.get('type')!r}")
    command = msg.get("command")
    if not isinstance(command, str) or not is_valid_command(command):
        raise ValueError(f"unknown command: {command!r}")
    return command, bool(msg.get("release", False))
