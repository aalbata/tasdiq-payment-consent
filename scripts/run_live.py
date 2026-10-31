"""Run the live track for a configuration (a local model drives the agent against the monitor)."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from tasdiq.runners.cli import run_or_count  # noqa: E402
from tasdiq.runners.live import live_counts, run_live  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return run_or_count("Run the live track: every live task through every model, arm, language and seed.", run_live, live_counts, argv)


if __name__ == "__main__":
    raise SystemExit(main())
