"""Build the claim ledger (results/ledger.json) and its manifest from the run directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from tasdiq.analysis.ledger import TRACKS, build_ledger, write_ledger  # noqa: E402
from tasdiq.runners.config import load_config  # noqa: E402


def parse_runs(values: list[str] | None) -> dict[str, list[str]] | None:
    """Turn ``--run track=run_id`` arguments into a track to run-id mapping."""
    if not values:
        return None
    runs: dict[str, list[str]] = {}
    for item in values:
        track, _, run_id = item.partition("=")
        if track not in TRACKS or not run_id:
            raise SystemExit(f"--run expects <track>=<run_id> with track in {TRACKS}: {item!r}")
        runs.setdefault(track, []).append(run_id)
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute every reported number from the run directories under results/ and write results/ledger.json and results/manifest.json.")
    parser.add_argument("--config", default="configs/full.yaml", help="YAML configuration naming results_dir and data_dir (default configs/full.yaml)")
    parser.add_argument("--run", action="append", default=None, metavar="TRACK=RUN_ID", help="restrict a track to one run directory; repeatable; default reads every run directory of every track (a ledger_runs mapping in the configuration has the same effect)")
    parser.add_argument("--n-boot", type=int, default=2000, help="cluster bootstrap resamples (default 2000)")
    parser.add_argument("--seed", type=int, default=20261006, help="base seed of the bootstrap and permutation generators")
    parser.add_argument("--n-perm", type=int, default=10000, help="permutations of each permutation test (default 10000)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, root=CODE_ROOT)
    runs = parse_runs(args.run)
    if runs is None and isinstance(cfg.get("ledger_runs"), dict):
        runs = {str(k): [str(v) for v in vs] for k, vs in cfg.get("ledger_runs").items()}
    ledger, manifest = build_ledger(cfg.results_dir, cfg.data_dir, runs=runs, n_boot=args.n_boot, seed=args.seed, root=CODE_ROOT, log=print, n_perm=args.n_perm)
    ledger_path, manifest_path = write_ledger(cfg.results_dir, ledger, manifest)
    print(f"ledger: {len(ledger)} claims written to {ledger_path}")
    print(f"manifest: {manifest_path}")
    for cid in sorted(c for c in ledger if c.endswith("_verdict")):
        claim = ledger.get(cid)
        print(f"{cid}: {claim.value}; {claim.description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
