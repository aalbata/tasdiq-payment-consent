"""Worker planning, resumable progress and process pools for the runners.

The number of worker processes follows the memory policy of the protocol: workers =
min(max_workers, floor((available_ram_gb - 1.0) / per_worker_gb)), never above twelve and
never such that llama threads per worker times workers exceed twelve. Each worker process
loads at most one model (mmap-shared weights) and reports its peak resident set size
through psutil. Once the first work unit of a pool completes, its measured peak resident
set size is logged and the worker count is reduced for the remaining units whenever the
projected total (workers times that peak) would exceed RAM_LIMIT_FRACTION of the installed
memory. Completed work is keyed so that a restarted run skips it, so a reduction only
changes the scheduling of the units still pending.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import psutil

MAX_WORKERS_HARD: int = 12
MAX_TOTAL_THREADS: int = 12
RESERVED_RAM_GB: float = 1.0
RAM_LIMIT_FRACTION: float = 0.7
"""Share of the installed memory that the projected total of the workers may not exceed."""


@dataclass(frozen=True)
class WorkerPlan:
    """Chosen worker count with the facts it was derived from."""

    available_ram_gb: float
    total_ram_gb: float
    per_worker_gb: float
    max_workers: int
    n_threads: int
    workers: int

    @property
    def total_threads(self) -> int:
        return self.workers * self.n_threads

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["total_threads"] = self.total_threads
        return out


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        value = getter(key, default)
        return default if value is None else value
    return getattr(cfg, key, default)


def available_ram_gb() -> tuple[float, float]:
    """Return (available, total) system memory in gibibytes."""
    memory = psutil.virtual_memory()
    return memory.available / 2**30, memory.total / 2**30


def plan_workers(cfg: Any, log: Callable[[str], None] | None = print) -> WorkerPlan:
    """Choose the number of worker processes from the available memory and the configuration.

    Reads max_workers (default 2), n_threads (default 3) and per_worker_gb (default 0.9) from
    cfg (a mapping or an object with get). The result is at least one worker, at most
    max_workers, at most twelve, and small enough that workers times n_threads stays within
    twelve threads. The plan is logged through log when given.
    """
    max_workers = int(_cfg(cfg, "max_workers", 2))
    n_threads = max(1, int(_cfg(cfg, "n_threads", 3)))
    per_worker_gb = float(_cfg(cfg, "per_worker_gb", 0.9))
    if per_worker_gb <= 0:
        raise ValueError("per_worker_gb must be positive")
    available, total = available_ram_gb()
    by_memory = int(math.floor((available - RESERVED_RAM_GB) / per_worker_gb))
    workers = min(max_workers, max(1, by_memory), MAX_WORKERS_HARD, max(1, MAX_TOTAL_THREADS // n_threads))
    workers = max(1, workers)
    plan = WorkerPlan(
        available_ram_gb=round(available, 3),
        total_ram_gb=round(total, 3),
        per_worker_gb=per_worker_gb,
        max_workers=max_workers,
        n_threads=n_threads,
        workers=workers,
    )
    if log is not None:
        log(
            f"workers: available RAM {plan.available_ram_gb:.2f} GiB of {plan.total_ram_gb:.2f} GiB, "
            f"{plan.per_worker_gb} GiB per worker, chosen {plan.workers} worker(s) x {plan.n_threads} thread(s)"
        )
    return plan


def workers_within_memory(workers: int, peak_rss_bytes: int, total_ram_gb: float, limit_fraction: float = RAM_LIMIT_FRACTION) -> int:
    """Largest worker count, at most ``workers``, whose projected total stays within the memory limit.

    The projected total is the worker count times the measured peak resident set size of
    one worker; the limit is ``limit_fraction`` of the installed memory. The result is at
    least one worker; a non-positive peak leaves the count unchanged.
    """
    workers = max(1, int(workers))
    if peak_rss_bytes <= 0 or total_ram_gb <= 0:
        return workers
    budget = float(total_ram_gb) * float(limit_fraction) * 2**30
    return max(1, min(workers, int(math.floor(budget / float(peak_rss_bytes)))))


def log_measured_peak(plan: WorkerPlan, workers: int, peak_rss_bytes: int, log: Callable[[str], None] | None = None) -> int:
    """Log the measured peak of the first completed unit and return the worker count for the remaining units."""
    kept = workers_within_memory(workers, peak_rss_bytes, plan.total_ram_gb)
    if log is not None:
        peak_gb = peak_rss_bytes / 2**30
        budget = plan.total_ram_gb * RAM_LIMIT_FRACTION
        action = f"keeping {workers} worker(s)" if kept == workers else f"reducing to {kept} worker(s) for the remaining units"
        log(
            f"workers: first completed unit peak RSS {peak_gb:.2f} GiB; projected {workers} x {peak_gb:.2f} = "
            f"{workers * peak_gb:.2f} GiB against a limit of {budget:.2f} GiB ({RAM_LIMIT_FRACTION:.0%} of {plan.total_ram_gb:.2f} GiB); {action}"
        )
    return kept


def peak_rss_bytes(process: psutil.Process | None = None) -> int:
    """Return the peak resident set size of a process in bytes (the current value where no peak is kept)."""
    proc = process if process is not None else psutil.Process()
    info = proc.memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is None:
        peak = getattr(info, "rss", 0)
    return int(peak)


def current_rss_bytes(process: psutil.Process | None = None) -> int:
    """Return the current resident set size of a process in bytes."""
    proc = process if process is not None else psutil.Process()
    return int(proc.memory_info().rss)


class Progress:
    """Set of completed work keys persisted as one JSON line per key.

    A key is any string that identifies one unit of work, for example
    "task|arm|seed". Keys are appended as they complete, so a restarted run reads the
    file and skips what is already done.
    """

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path is not None else None
        self._done: set[str] = set()
        if self.path is not None and self.path.is_file():
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        self._done.add(str(json.loads(line)["key"]))

    def __contains__(self, key: str) -> bool:
        return key in self._done

    def __len__(self) -> int:
        return len(self._done)

    @property
    def done(self) -> frozenset[str]:
        return frozenset(self._done)

    def mark(self, key: str, **detail: Any) -> None:
        """Record a key as completed."""
        if key in self._done:
            return
        self._done.add(key)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"key": key, **detail}, ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")

    def pending(self, keys: Iterable[str]) -> list[str]:
        """Return the keys not yet completed, in the given order."""
        return [k for k in keys if k not in self._done]


def make_key(*parts: Any) -> str:
    """Join the parts of a work key with a vertical bar."""
    return "|".join(str(p) for p in parts)


_WORKER_STATE: dict[str, Any] = {}


def worker_state() -> dict[str, Any]:
    """Return the per-process state dictionary filled by a pool initializer (for example the loaded model)."""
    return _WORKER_STATE


def _initialize(initializer: Callable[..., Any] | None, initargs: tuple[Any, ...]) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE["pid"] = os.getpid()
    if initializer is not None:
        result = initializer(*initargs)
        if isinstance(result, Mapping):
            _WORKER_STATE.update(result)


def _call(payload: tuple[Callable[..., Any], Any]) -> dict[str, Any]:
    fn, item = payload
    result = fn(item)
    if not isinstance(result, dict):
        result = {"result": result}
    result.setdefault("worker", str(os.getpid()))
    result.setdefault("peak_rss_bytes", peak_rss_bytes())
    return result


def _call_indexed(payload: tuple[int, Callable[..., Any], Any]) -> tuple[int, dict[str, Any]]:
    index, fn, item = payload
    return index, _call((fn, item))


def run_in_workers(
    fn: Callable[[Any], dict[str, Any]],
    items: Sequence[Any],
    workers: int,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
    plan: WorkerPlan | None = None,
    log: Callable[[str], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Apply fn to every item and yield the result mappings as they complete.

    With one worker everything runs in the calling process (the initializer first); with more,
    a spawn-context pool runs the initializer once per process and fn in the workers. Every
    result carries worker (the process id) and peak_rss_bytes (the process peak resident set).
    Results of one worker arrive in order; across workers the order follows completion.

    When a plan is given and the pool has more than one worker, the peak resident set size
    of the first completed unit is logged and the worker count for the remaining units is
    reduced when the projected total would exceed RAM_LIMIT_FRACTION of the installed
    memory: the pool is closed and a smaller pool takes over the units not yet yielded.
    Units in flight when the pool closes are run again by the smaller pool.
    """
    if workers <= 1 or len(items) <= 1:
        _initialize(initializer, initargs)
        for item in items:
            yield _call((fn, item))
        return
    context = multiprocessing.get_context("spawn")
    pending: dict[int, Any] = dict(enumerate(items))
    measured = plan is None
    while pending:
        current = max(1, int(workers))
        if current <= 1:
            _initialize(initializer, initargs)
            for index in list(pending):
                item = pending.pop(index)
                yield _call((fn, item))
            return
        with context.Pool(processes=current, initializer=_initialize, initargs=(initializer, initargs)) as pool:
            payloads = [(index, fn, item) for index, item in pending.items()]
            restart = False
            for index, result in pool.imap_unordered(_call_indexed, payloads):
                pending.pop(index, None)
                yield result
                if not measured:
                    measured = True
                    kept = log_measured_peak(plan, current, int(result.get("peak_rss_bytes") or 0), log)
                    if kept < current and pending:
                        workers = kept
                        restart = True
                        break
            if not restart:
                pending.clear()


def chunk(items: Sequence[Any], n_chunks: int) -> list[list[Any]]:
    """Split a sequence into at most n_chunks contiguous, nearly equal lists (no empty lists)."""
    n_chunks = max(1, min(int(n_chunks), len(items))) if items else 1
    size = math.ceil(len(items) / n_chunks) if items else 0
    return [list(items[i : i + size]) for i in range(0, len(items), size)] if items else []
