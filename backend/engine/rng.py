"""7-bag randomizer: shuffles IJLOSTZ, deals pieces, refills when empty."""

import random
from collections import deque

from .pieces import PIECE_IDS


class Bag7:
    """7-bag randomizer: shuffles IJLOSTZ, deals pieces, refills when empty.

    Bag7() with no seed is not reproducible. Bag7(seed) produces an
    identical piece sequence for the same seed (all shuffling uses only the
    instance's own random.Random).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._queue: deque[str] = deque()

    def _refill(self) -> None:
        bag = list(PIECE_IDS)
        self._rng.shuffle(bag)
        # Prepend: pieces are dealt LIFO (from the right), so a new bag must
        # go to the LEFT of any remainder. That keeps the deal order
        # independent of when a refill happens (peek may refill early).
        self._queue.extendleft(reversed(bag))

    def next_piece(self) -> str:
        if not self._queue:
            self._refill()
        return self._queue.pop()

    def peek(self, n: int) -> list[str]:
        """Next n pieces without consuming, in deal order. Tops up from fresh
        bags as needed (looking across bag boundaries does not change deal
        order). With LIFO dealing the next piece is the queue's last item,
        so deal order is the reversed tail."""
        while len(self._queue) < n:
            self._refill()
        return [self._queue[-1 - i] for i in range(n)]
