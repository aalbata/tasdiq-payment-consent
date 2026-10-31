"""Fill README.md from README.template.md and the claim ledger.

The template holds placeholders written as a claim identifier between a pair of double
angle brackets, optionally followed by a colon and one of ``ci`` (the interval as
``[lo, hi]``), ``n`` (the sample size) or ``count`` (the numerator). Every placeholder is
replaced from results/ledger.json; the script asserts that each placeholder occurs exactly
once in the template, that every placeholder names a claim in the ledger and that none
survives in the output. Fractions are printed as percentages with one decimal, counts as
integers, seconds with three significant figures, bytes as gigabytes with two decimals and
text values unchanged. A claim without a value or interval is printed as ``n/a``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from tasdiq.analysis.tables import PLACEHOLDER  # noqa: E402

MISSING = "n/a"


def format_number(value: Any, unit: str) -> str:
    """The displayed text of a value under the rounding rule of its unit."""
    if value is None:
        return MISSING
    if isinstance(value, str):
        return value
    if unit == "fraction":
        return f"{100.0 * float(value):.1f}"
    if unit == "count":
        return f"{int(round(float(value)))}"
    if unit == "seconds":
        return f"{float(value):.3g}"
    if unit == "bytes":
        return f"{float(value) / 1e9:.2f}"
    if unit in ("text", "verdict"):
        return str(value)
    return f"{float(value):.3g}"


def format_field(claim: Mapping[str, Any], field: str | None) -> str:
    """The displayed text of a placeholder: the value, or its ``ci``, ``n`` or ``count``."""
    unit = str(claim.get("unit"))
    if field is None:
        return format_number(claim.get("value"), unit)
    if field == "ci":
        ci = claim.get("ci")
        if not ci or ci.get("lo") is None or ci.get("hi") is None:
            return MISSING
        return f"[{format_number(ci['lo'], unit)}, {format_number(ci['hi'], unit)}]"
    value = claim.get(field)
    return MISSING if value is None else f"{int(value)}"


def fill_text(text: str, claims: Mapping[str, Mapping[str, Any]], name: str = "template") -> tuple[str, list[str]]:
    """Replace every placeholder of ``text`` exactly once and return the text with the placeholders replaced.

    Raises ``AssertionError`` when a placeholder occurs more than once, ``KeyError`` when a
    placeholder names a claim absent from the ledger, and ``AssertionError`` when a double
    angle bracket survives in the output.
    """
    occurrences = Counter(m.group(0) for m in PLACEHOLDER.finditer(text))
    repeated = sorted(k for k, v in occurrences.items() if v != 1)
    assert not repeated, f"{name}: placeholders occurring more than once: {repeated}"
    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(text)} - set(claims))
    if missing:
        raise KeyError(f"{name}: placeholders without a ledger claim: {missing}")
    filled = PLACEHOLDER.sub(lambda m: format_field(claims[m.group(1)], m.group(2)), text)
    assert "<<" not in filled and ">>" not in filled, f"{name}: a placeholder remains after filling"
    return filled, list(occurrences)


def fill_readme(template: Path, ledger: Path, out: Path) -> list[str]:
    """Fill ``template`` from ``ledger`` into ``out`` and return the placeholders replaced."""
    if not ledger.is_file():
        raise FileNotFoundError(f"no claim ledger at {ledger}; build the ledger before filling the README")
    claims = json.loads(ledger.read_text(encoding="utf-8"))
    filled, placeholders = fill_text(template.read_text(encoding="utf-8"), claims, template.name)
    out.write_text(filled, encoding="utf-8", newline="\n")
    return placeholders


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else CODE_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fill README.md from README.template.md and results/ledger.json; every placeholder must occur exactly once and name a ledger claim.")
    parser.add_argument("--template", default="README.template.md", help="template file (relative to the code root)")
    parser.add_argument("--ledger", default="results/ledger.json", help="claim ledger (relative to the code root)")
    parser.add_argument("--out", default="README.md", help="output file (relative to the code root)")
    args = parser.parse_args(argv)
    placeholders = fill_readme(_resolve(args.template), _resolve(args.ledger), _resolve(args.out))
    print(f"{args.out}: {len(placeholders)} placeholders filled from {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
