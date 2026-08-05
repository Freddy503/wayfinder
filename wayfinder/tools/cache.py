"""Disk cache for every outbound call.

This is load-bearing, not an optimisation. Evals re-run the same twenty trips
over and over; without a cache each run pays for the same geocoding and the
same searches, and — worse — a changed answer could come from the *data* rather
than from the change you were testing. Pinning the data means a delta between
two experiments is attributable to the thing you actually varied.

Cache keys are content-addressed: namespace plus a hash of the arguments. Clear
it with `rm -rf .cache/` when you want fresh data.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

CACHE_DIR = Path(os.environ.get("WAYFINDER_CACHE_DIR", ".cache")).resolve()

T = TypeVar("T")


def _key(namespace: str, payload: Any) -> Path:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(f"{namespace}\x00{blob}".encode()).hexdigest()[:32]
    return CACHE_DIR / namespace / f"{digest}.json"


def cached(namespace: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Memoise a JSON-serialisable function call to disk."""

    def decorate(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            path = _key(namespace, {"args": args, "kwargs": kwargs})
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    # An unreadable entry is a miss, not a failure. Raising
                    # here would kill a run over a corrupt file that we can
                    # simply fetch again and overwrite.
                    path.unlink(missing_ok=True)

            result = fn(*args, **kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write, then rename. A plain `write_text` that is interrupted —
            # the process killed, the disk full — leaves a half-written file
            # that looks like a valid cache hit forever after.
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(tmp, path)
            return result

        return wrapper

    return decorate


class RateLimiter:
    """Minimum spacing between calls, shared across threads.

    Nominatim's usage policy caps you at one request a second. Subagents run
    concurrently, so the guard has to be process-wide rather than per-caller.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._interval = min_interval_seconds
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last = time.monotonic()
