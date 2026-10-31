"""Runners: configuration, worker planning and the trace, capture and live tracks.

The track modules are imported on first attribute access so that importing the package
does not pull in the analysis layer or the model runtime.
"""

from typing import Any

from tasdiq.runners.config import ModelSpec, RunConfig, load_config, make_config
from tasdiq.runners.workers import Progress, WorkerPlan, plan_workers, run_in_workers

_LAZY: dict[str, tuple[str, str]] = {
    "run_trace": ("tasdiq.runners.trace", "run_trace"),
    "premise_check": ("tasdiq.runners.trace", "premise_check"),
    "measured_classes": ("tasdiq.runners.trace", "measured_classes"),
    "run_capture": ("tasdiq.runners.capture", "run_capture"),
    "run_live": ("tasdiq.runners.live", "run_live"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(name)
    module = __import__(target[0], fromlist=[target[1]])
    return getattr(module, target[1])


__all__ = [
    "ModelSpec",
    "Progress",
    "RunConfig",
    "WorkerPlan",
    "load_config",
    "make_config",
    "measured_classes",
    "plan_workers",
    "premise_check",
    "run_capture",
    "run_in_workers",
    "run_live",
    "run_trace",
]
