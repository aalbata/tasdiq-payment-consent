"""Worker planning: the memory policy, the measured-peak reduction and the process pool."""

from __future__ import annotations

from typing import Any

import psutil

from tasdiq.runners import workers as workers_mod
from tasdiq.runners.workers import RAM_LIMIT_FRACTION, WorkerPlan, log_measured_peak, plan_workers, run_in_workers, workers_within_memory

GIB = 2**30


def _plan(total_gb: float = 16.0, workers: int = 4) -> WorkerPlan:
    return WorkerPlan(available_ram_gb=8.0, total_ram_gb=total_gb, per_worker_gb=1.2, max_workers=4, n_threads=3, workers=workers)


def test_workers_within_memory_hand_computed() -> None:
    # 16 GiB installed, limit 70 percent = 11.2 GiB: four workers of 2.07 GiB (8.28 GiB) fit; four of 3.9 GiB do not.
    assert workers_within_memory(4, int(2.07 * GIB), 16.0) == 4
    assert workers_within_memory(4, int(3.9 * GIB), 16.0) == 2
    assert workers_within_memory(4, int(12 * GIB), 16.0) == 1
    assert workers_within_memory(4, 0, 16.0) == 4
    assert workers_within_memory(1, int(12 * GIB), 16.0) == 1
    assert RAM_LIMIT_FRACTION == 0.7


def test_log_measured_peak_reports_projection_and_decision() -> None:
    lines: list[str] = []
    kept = log_measured_peak(_plan(), 4, int(2.07 * GIB), lines.append)
    assert kept == 4 and len(lines) == 1
    assert "2.07 GiB" in lines[0] and "keeping 4 worker(s)" in lines[0] and "70%" in lines[0]
    lines.clear()
    kept = log_measured_peak(_plan(), 4, int(6 * GIB), lines.append)
    assert kept == 1 and "reducing to 1 worker(s)" in lines[0]


def test_plan_workers_reads_per_worker_gb(monkeypatch: Any) -> None:
    monkeypatch.setattr(workers_mod, "available_ram_gb", lambda: (5.8, 16.0))
    plan = plan_workers({"max_workers": 4, "n_threads": 3, "per_worker_gb": 1.2}, log=None)
    assert plan.workers == 4 and plan.per_worker_gb == 1.2
    monkeypatch.setattr(workers_mod, "available_ram_gb", lambda: (3.3, 16.0))
    assert plan_workers({"max_workers": 4, "n_threads": 3, "per_worker_gb": 1.2}, log=None).workers == 1


def _heavy_unit(item: int) -> dict[str, Any]:
    """Work unit that reports a peak resident set size equal to the installed memory."""
    return {"item": item, "peak_rss_bytes": int(psutil.virtual_memory().total)}


def test_run_in_workers_reduces_the_pool_after_the_first_unit() -> None:
    plan = plan_workers({"max_workers": 2, "n_threads": 1, "per_worker_gb": 0.1}, log=None)
    lines: list[str] = []
    results = list(run_in_workers(_heavy_unit, list(range(6)), 2, plan=plan, log=lines.append))
    assert sorted(r["item"] for r in results) == list(range(6))
    assert len(lines) == 1 and "reducing to 1 worker(s)" in lines[0]
    later = {r["worker"] for r in results[1:]}
    assert len(later) == 1
