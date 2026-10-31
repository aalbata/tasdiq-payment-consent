"""Run the capture track for a configuration (local model inference on the CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from tasdiq.runners.capture import capture_counts, run_capture  # noqa: E402
from tasdiq.runners.cli import run_or_count  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return run_or_count("Run the capture track: every variant through every capture pipeline and model.", run_capture, capture_counts, argv)


if __name__ == "__main__":
    raise SystemExit(main())
