"""Command-line argument handling shared by the scripts.

Every script takes --config (a YAML file under configs/), and optionally --split (dev or
test, replacing the configured splits), --limit (the maximum number of work units),
--resume/--no-resume (skip or redo work units a previous run completed), --run-id,
--model (restrict the capture and live tracks to one configured model) and --dry-run
(print the number of work units per track and model and exit without running).
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Mapping, Sequence

from tasdiq.runners.config import RunConfig, load_config


def build_parser(description: str) -> argparse.ArgumentParser:
    """Return the argument parser common to the run scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML configuration file, relative to the code root")
    parser.add_argument("--split", choices=("dev", "test"), default=None, help="run one split only")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of work units")
    parser.add_argument("--run-id", default=None, help="name of the run directory under results/<track>/")
    parser.add_argument("--model", default=None, help="run the work units of this configured model only (capture and live tracks)")
    parser.add_argument("--dry-run", action="store_true", help="print the number of work units per track and model and exit")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", default=True, help="skip completed work units (default)")
    resume.add_argument("--no-resume", dest="resume", action="store_false", help="redo every work unit")
    return parser


def config_from_args(args: argparse.Namespace, extra: dict[str, Any] | None = None) -> RunConfig:
    """Load the configuration named by the arguments with the command-line overrides applied."""
    overrides: dict[str, Any] = {
        "splits": [args.split] if getattr(args, "split", None) else None,
        "limit": getattr(args, "limit", None),
        "resume": getattr(args, "resume", True),
        "run_id": getattr(args, "run_id", None),
        "model_filter": getattr(args, "model", None),
    }
    overrides.update(extra or {})
    return load_config(args.config, overrides=overrides)


def print_counts(counts: Mapping[str, Any], out: Callable[[str], None] = print) -> None:
    """Print a dry-run count mapping: one summary line, one line per model, then the JSON document."""
    track = counts.get("track")
    out(f"{track}: run {counts.get('run_id')}: {counts.get('items')} work units, {counts.get('pending')} pending")
    for name, entry in (counts.get("models") or {}).items():
        detail = ", ".join(f"{k} {v}" for k, v in sorted((entry.get("pipelines") or entry.get("arms") or {}).items()))
        out(f"{track}: model {name}: {entry.get('items')} work units, {entry.get('pending')} pending" + (f" ({detail})" if detail else ""))
    out(json.dumps(counts, ensure_ascii=False, sort_keys=True, indent=2, default=str))


def run_or_count(description: str, run: Callable[[RunConfig], Any], count: Callable[[RunConfig], Mapping[str, Any]], argv: Sequence[str] | None = None, extra: Callable[[argparse.ArgumentParser], None] | None = None, overrides: Callable[[argparse.Namespace], dict[str, Any]] | None = None) -> int:
    """Parse the common arguments and either run the track or, with --dry-run, print its counts.

    extra adds script-specific arguments to the parser and overrides turns the parsed
    namespace into configuration overrides.
    """
    parser = build_parser(description)
    if extra is not None:
        extra(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args, overrides(args) if overrides is not None else None)
    if args.dry_run:
        print_counts(count(cfg))
        return 0
    run(cfg)
    return 0


def parse_run_args(description: str, argv: Sequence[str] | None = None) -> RunConfig:
    """Parse the common arguments and return the loaded configuration."""
    parser = build_parser(description)
    return config_from_args(parser.parse_args(argv))
