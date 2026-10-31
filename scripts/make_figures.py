"""Draw the figures from the claim ledger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from tasdiq.analysis.figures import make_figures  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw every figure (tasdiq_fig1 to tasdiq_fig6, PDF and PNG) from results/ledger.json and results/manifest.json.")
    parser.add_argument("--results", default="results", help="results directory holding ledger.json (relative to the code root)")
    parser.add_argument("--out", default="figures", help="output directory, for example ../manuscript/figures (relative to the code root)")
    args = parser.parse_args(argv)
    for path in make_figures(_resolve(args.results), _resolve(args.out)):
        print(path)
    return 0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else CODE_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
