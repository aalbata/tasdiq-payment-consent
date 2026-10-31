"""Build the benchmark files from the authored seeds.

Writes data/benchmark.jsonl, scenarios.jsonl, splits.json, hashes.json, review_set.csv,
adjudication_sheet.csv, annotation_guide.md and build_report.json, then prints the seed
counts per family, the variants per language, the dev and test seed counts, the
near-duplicate report and the review-set size.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from tasdiq.bench.export import export_benchmark  # noqa: E402
from tasdiq.bench.render import build_all_variants  # noqa: E402
from tasdiq.bench.schema import load_seeds  # noqa: E402
from tasdiq.runners.cli import build_parser  # noqa: E402
from tasdiq.runners.config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("Build the benchmark files from the seed scenarios.")
    parser.add_argument("--review-per-lang", type=int, default=40)
    parser.add_argument("--strict-content", action="store_true", help="fail when the corpus content requirements are not met")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    seeds = load_seeds(cfg.seeds_dir)
    result = export_benchmark(
        seeds,
        out_dir=cfg.data_dir,
        dev_fraction=float(cfg.get("dev_fraction", 0.25)),
        split_seed=int(cfg.get("split_seed", 20261006)),
        review_per_lang=args.review_per_lang,
        strict_content=args.strict_content,
    )
    main_seeds = [s for s in seeds if not s.dialect_extra]
    variants = build_all_variants(main_seeds)
    per_family = Counter(s.family for s in main_seeds)
    per_lang = Counter(v.lang for v in variants)
    per_kind = Counter(v.kind for v in variants)
    split_counts = Counter(result.splits.values())
    print(f"seeds: {result.n_seeds} total, {result.n_main_seeds} in the main set, {result.n_seeds - result.n_main_seeds} dialect_extra")
    print("seeds per family: " + ", ".join(f"{k} {v}" for k, v in sorted(per_family.items())))
    print(f"variants: {result.n_variants}; per language: " + ", ".join(f"{k} {v}" for k, v in sorted(per_lang.items())) + "; per kind: " + ", ".join(f"{k} {v}" for k, v in sorted(per_kind.items())))
    print(f"splits: dev {split_counts.get('dev', 0)} seeds, test {split_counts.get('test', 0)} seeds; excluded variants {result.n_excluded}")
    cross = sum(1 for d in result.duplicates if d.cross_seed)
    print(f"near-duplicate pairs: {len(result.duplicates)} ({cross} across seeds)")
    for dup in result.duplicates[:20]:
        print(f"  {dup.to_dict()}")
    print(f"review set rows: {result.review_rows}")
    for issue in result.content_issues:
        print(f"content: {issue}")
    print(f"written to {result.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
