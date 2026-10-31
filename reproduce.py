"""Reproduce the study's results.

Fast path (default): regenerate the aggregates of every shipped run under results/ from its
raw JSON-lines records, rerun the bounded exploration (no model needed; under the verify run
id that ledger_runs names, when the configuration names exactly one), build the claim
ledger (results/ledger.json and results/manifest.json) from the shipped records, and then
draw the figures into figures/ and fill the tables into tables_filled/. Nothing is
downloaded and no model is loaded.

Full path (--full): build the benchmark from the seeds, run the exploration, the trace
track (and the trace sensitivity runs the configuration names under
trace_sensitivity_runs, such as the inattentive-reader run), the capture track and the
live track for the given configuration, then take the fast path. The capture and live tracks need the model files named in the configuration in
the local Hugging Face cache (or in models_dir) and the opus-mt-ar-en weights for the
translate_mt pipeline.

Usage:
    python reproduce.py --config configs/full.yaml
    python reproduce.py --config configs/full.yaml --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))

from tasdiq.runners.config import load_config, read_jsonl, write_json  # noqa: E402


def regenerate_aggregates(results_dir: Path, log=print) -> list[Path]:
    """Recompute aggregates.json for every run directory under results/<track>/ from its records."""
    from tasdiq.runners.capture import aggregate_captures
    from tasdiq.runners.live import aggregate_episodes
    from tasdiq.runners.trace import aggregate_sessions

    written: list[Path] = []
    for track, filename, aggregate in (("trace", "sessions.jsonl", aggregate_sessions), ("capture", "captures.jsonl", aggregate_captures), ("live", "episodes.jsonl", aggregate_episodes)):
        track_dir = results_dir / track
        if not track_dir.is_dir():
            continue
        for run_dir in sorted(p for p in track_dir.iterdir() if p.is_dir()):
            records = read_jsonl(run_dir / filename)
            if not records:
                continue
            path = write_json(run_dir / "aggregates.json", aggregate(records))
            written.append(path)
            log(f"aggregates: {track}/{run_dir.name}: {len(records)} records")
    return written


def verify_config(cfg, runs):
    """The configuration the exploration runs under: the single verify run named in ledger_runs, else cfg itself."""
    from tasdiq.runners.config import make_config

    named = list((runs or {}).get("verify") or [])
    if len(named) != 1 or str(named[0]) == cfg.run_id:
        return cfg
    return make_config(cfg.values, source=cfg.source, root=CODE_ROOT, overrides={"run_id": str(named[0])})


TIMING_KEYS: tuple[str, ...] = ("elapsed_seconds", "started", "ended", "peak_rss_bytes")
"""Timing and memory fields of a verify run that a rerun keeps; the rerun's own values go under reproduce_<key>."""


def _read_json(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _keep_timing(original: dict | None, rerun: dict) -> dict:
    """Return rerun with the original's timing fields restored and its own moved under reproduce_<key>."""
    out = dict(rerun)
    for key in TIMING_KEYS:
        if key in out:
            out[f"reproduce_{key}"] = out[key]
        if original is not None and key in original:
            out[key] = original[key]
    return out


def rerun_verify(cfg, log=print) -> Path:
    """Rerun the exploration of a verify run and keep the shipped run's timing fields.

    The manifest and explore.json of an existing run keep their elapsed_seconds, started,
    ended and peak_rss_bytes (and the per-exploration elapsed_seconds, matched by arm and
    rule); the rerun's own values are written under reproduce_<key>. A run directory that
    did not exist before is written as a fresh run.
    """
    from tasdiq.verify.explore import run_verify

    run_dir = cfg.results_dir / "verify" / cfg.run_id
    manifest_before = _read_json(run_dir / "manifest.json")
    explore_before = _read_json(run_dir / "explore.json")
    run_verify(cfg, log)
    if manifest_before is None and explore_before is None:
        return run_dir
    manifest = _read_json(run_dir / "manifest.json") or {}
    write_json(run_dir / "manifest.json", _keep_timing(manifest_before, manifest))
    explore = _read_json(run_dir / "explore.json") or {}
    kept = _keep_timing(explore_before, explore)
    earlier = {(r.get("arm"), r.get("rule")): r for r in ((explore_before or {}).get("results") or [])}
    results = []
    for result in kept.get("results") or []:
        previous = earlier.get((result.get("arm"), result.get("rule")))
        results.append(_keep_timing({"elapsed_seconds": previous["elapsed_seconds"]} if previous and "elapsed_seconds" in previous else None, result))
    kept["results"] = results
    write_json(run_dir / "explore.json", kept)
    return run_dir


def fast_path(cfg, log=print, n_boot: int = 2000) -> None:
    from tasdiq.analysis.figures import make_figures
    from tasdiq.analysis.ledger import build_ledger, write_ledger
    from tasdiq.analysis.tables import fill_tables

    results_dir = cfg.results_dir
    if not results_dir.is_dir():
        log(f"no results directory at {results_dir}; run with --full to produce the results")
    else:
        regenerate_aggregates(results_dir, log)
    runs = cfg.get("ledger_runs") if isinstance(cfg.get("ledger_runs"), dict) else None
    rerun_verify(verify_config(cfg, runs), log)
    ledger, manifest = build_ledger(results_dir, cfg.data_dir, runs=runs, n_boot=n_boot, root=CODE_ROOT, log=log)
    ledger_path, manifest_path = write_ledger(results_dir, ledger, manifest)
    log(f"ledger: {len(ledger)} claims in {ledger_path}; runs listed in {manifest_path}")
    for name, fn in (("figures", lambda: make_figures(results_dir, CODE_ROOT / "figures")), ("tables", lambda: fill_tables(results_dir, CODE_ROOT / "tables", CODE_ROOT / "tables_filled"))):
        paths = fn()
        log(f"{name}: {len(paths)} files written")


def full_path(cfg, log=print, n_boot: int = 2000) -> None:
    import subprocess

    from tasdiq.runners.capture import run_capture
    from tasdiq.runners.live import run_live
    from tasdiq.runners.trace import run_trace
    from tasdiq.verify.explore import run_verify

    subprocess.run([sys.executable, str(CODE_ROOT / "scripts" / "build_benchmark.py"), "--config", str(cfg.source)], check=True, cwd=str(CODE_ROOT))
    run_verify(cfg, log)
    run_trace(cfg, log)
    for sensitivity in sensitivity_configs(cfg):
        run_trace(sensitivity, log)
    run_capture(cfg, log)
    run_live(cfg, log)
    fast_path(cfg, log, n_boot=n_boot)


def sensitivity_configs(cfg) -> list:
    """Configurations of the trace sensitivity runs named by ``trace_sensitivity_runs``.

    The configuration key maps a run id to the settings that replace those of the main
    configuration for that run (for example ``user_model: inattentive`` for the
    inattentive-reader run); every other setting is shared with the main run.
    """
    from tasdiq.runners.config import make_config

    named = cfg.get("trace_sensitivity_runs")
    if not isinstance(named, dict):
        return []
    out = []
    for run_id, settings in named.items():
        overrides = {"run_id": str(run_id)}
        overrides.update(dict(settings or {}))
        out.append(make_config(cfg.values, source=cfg.source, root=CODE_ROOT, overrides=overrides))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the results from the shipped records (default) or from scratch (--full).")
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--full", action="store_true", help="run every track from the seeds and the model files")
    parser.add_argument("--n-boot", type=int, default=2000, help="cluster bootstrap resamples for the ledger intervals (default 2000)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, root=CODE_ROOT)
    if args.full:
        full_path(cfg, n_boot=args.n_boot)
    else:
        fast_path(cfg, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
