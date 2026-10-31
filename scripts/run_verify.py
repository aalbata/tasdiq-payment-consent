"""Run the bounded explicit-state exploration of the consent state machine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from tasdiq.runners.cli import run_or_count  # noqa: E402
from tasdiq.verify.explore import run_verify, verify_counts  # noqa: E402


def _add_depth(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--depth", type=int, default=None, help="depth bound (default from the configuration)")


def main(argv: list[str] | None = None) -> int:
    return run_or_count(
        "Explore the consent state machine under every revision rule up to a depth bound.",
        run_verify,
        verify_counts,
        argv,
        extra=_add_depth,
        overrides=lambda args: {"explore_depth": args.depth},
    )


if __name__ == "__main__":
    raise SystemExit(main())
