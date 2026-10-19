"""Player controller interface for the tetris-jev engine.

Human (WebSocket relay) and LLM players subclass :class:`PlayerController`.
The game engine exposes ``submit(command)``; a player's job is to decide
which command to issue and when, then push it through the same command
path the server wires into ``game.submit()``.

Command vocabulary (strings), shared across player modes:

- ``move_left`` / ``move_right``: shift the active piece one column
- ``rotate_cw`` / ``rotate_ccw``: rotate with SRS wall kicks
- ``soft_drop``: press half of a hold-state pair (20x gravity while held)
- ``soft_drop_release``: release half; ends the held soft-drop state
- ``hard_drop``: instant drop and lock
- ``hold``: swap active piece with the hold slot (once per piece)
"""

from __future__ import annotations


class PlayerController:
    """Interface for player implementations feeding the game engine.

    Subclasses decide *which* command to issue and *when*; the engine keeps
    simulation truth (collision, timing, scoring). Commands submitted
    between simulation steps are buffered and applied on the next step.
    """

    def submit(self, command: str) -> None:
        """Submit one command for application on the next simulation step."""
        raise NotImplementedError

    def reset(self) -> None:
        """Optional hook: clear per-game internal state. Default no-op."""
