"""First-come admission to the batch slots, with bounded overtaking."""
from __future__ import annotations

import itertools

OVERTAKE_MAX = 4        # times a waiting request may be passed by later ones before it must go next


class Waiter:
    __slots__ = ("ticket", "skipped")

    def __init__(self, ticket: int):
        self.ticket, self.skipped = ticket, 0


class AdmissionQueue:
    """Requests waiting for a free slot, oldest first. The oldest takes the next free slot; a later one may go
    first only when `hit` says its prompt resumes a slot's held prefix, and not past the oldest's overtake budget.
    All calls are made under the caller's lock."""

    def __init__(self, overtake_max: int = OVERTAKE_MAX):
        self.overtake_max, self.waiters, self._tickets = overtake_max, [], itertools.count()

    def join(self) -> Waiter:
        w = Waiter(next(self._tickets))
        self.waiters.append(w)
        return w

    def leave(self, w: Waiter) -> None:
        if w in self.waiters:
            self.waiters.remove(w)

    def may_go(self, w: Waiter, hit: bool = False) -> bool:
        """Whether `w` may take a free slot now; a granted overtake is charged to the oldest waiter."""
        head = self.waiters[0] if self.waiters else w
        if w is head:
            return True
        if hit and head.skipped < self.overtake_max:
            head.skipped += 1
            return True
        return False

    def __len__(self) -> int:
        return len(self.waiters)
