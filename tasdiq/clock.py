"""Simulated clock shared by the store, the rail, the surface and the monitors."""

from __future__ import annotations

import threading


class SimClock:
    """Integer simulated time in seconds, protected by a lock so that concurrent executors read a consistent value."""

    def __init__(self, start: int = 0) -> None:
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError("start must be a non-negative integer number of seconds")
        self._t = start
        self._lock = threading.Lock()

    def now(self) -> int:
        """Return the current simulated time in seconds."""
        with self._lock:
            return self._t

    def advance(self, seconds: int) -> int:
        """Move the clock forward by a non-negative number of seconds and return the new time."""
        if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 0:
            raise ValueError("seconds must be a non-negative integer")
        with self._lock:
            self._t += seconds
            return self._t

    def set(self, t: int) -> int:
        """Set the clock to an absolute non-negative time and return it."""
        if not isinstance(t, int) or isinstance(t, bool) or t < 0:
            raise ValueError("t must be a non-negative integer number of seconds")
        with self._lock:
            self._t = t
            return self._t
