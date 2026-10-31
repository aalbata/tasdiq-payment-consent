"""Fill the LaTeX table templates from the claim ledger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from tasdiq.analysis.tables import fill_tables, write_templates  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fill the LaTeX table templates (placeholders <<claim_id>> and <<claim_id:ci>>) from results/ledger.json; missing templates are generated from the ledger first.")
    parser.add_argument("--results", default="results", help="results directory holding ledger.json and manifest.json (relative to the code root)")
    parser.add_argument("--templates", default="tables", help="template directory (relative to the code root)")
    parser.add_argument("--out", default="tables_filled", help="output directory, for example ../manuscript/tables (relative to the code root)")
    parser.add_argument("--refresh-templates", action="store_true", help="regenerate every template from the ledger before filling, replacing existing ones")
    args = parser.parse_args(argv)
    results, templates, out = _resolve(args.results), _resolve(args.templates), _resolve(args.out)
    if args.refresh_templates:
        for path in write_templates(results, templates, overwrite=True):
            print(f"template: {path}")
    for path in fill_tables(results, templates, out):
        print(path)
    return 0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else CODE_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
