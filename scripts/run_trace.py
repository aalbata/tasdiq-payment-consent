"""Run the trace track for a configuration (deterministic monitors, no model inference)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from tasdiq.runners.cli import run_or_count  # noqa: E402
from tasdiq.runners.trace import run_trace, trace_counts  # noqa: E402
from tasdiq.surface import READERS  # noqa: E402


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Trace-track options: the simulated reader and the number of worker processes."""
    parser.add_argument("--user-model", choices=READERS, default=None, help="simulated reader of the confirmation surface (replaces user_model of the configuration; the inattentive reader is the sensitivity run)")
    parser.add_argument("--workers", type=int, default=None, help="worker processes (replaces the planned number)")


def overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Configuration overrides from the trace-track options."""
    return {"user_model": args.user_model, "trace_workers": args.workers}


def main(argv: list[str] | None = None) -> int:
    return run_or_count("Run the trace track: every variant through every arm, window and revision rule.", run_trace, trace_counts, argv, extra=add_arguments, overrides=overrides)


if __name__ == "__main__":
    raise SystemExit(main())
