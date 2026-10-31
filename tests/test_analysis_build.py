"""Ledger building on toy run directories, figure rendering and table filling."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tasdiq.analysis import figures as figures_mod
from tasdiq.analysis import ledger as ledger_mod
from tasdiq.analysis import tables as tables_mod

CODE_ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 40
N_PERM = 200

LANGS = ("en", "ar")
SEEDS = {"s1": "revision", "s2": "revision", "s3": "expiry_stale", "s4": "ordinary_valid"}
ARMS = (("B0", "none"), ("B3", "none"), ("B4", "none"), ("Tasdiq", "invalidate"), ("Tasdiq", "narrowing"))
WINDOWS = (60, 600)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def _manifest(run_dir: Path, run_id: str, peak: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "config": {"config_hash": "c" * 64, "code_hash": "d" * 64}, "peak_rss_bytes": peak, "elapsed_seconds": 1.0}), encoding="utf-8")


def _session(seed: str, lang: str, arm: str, rule: str, window: int) -> dict[str, Any]:
    family = SEEDS[seed]
    r: dict[str, Any] = {
        "seed_id": seed, "scenario_id": seed, "variant_id": f"{seed}.t0.{lang}.base", "pair_id": f"{seed}.t0",
        "family": family, "lang": lang, "kind": "base", "phenomena": ["digit_script"], "arm": arm, "rule": rule,
        "window_s": window, "mode": "oracle", "completed": True, "completed_confirmed": True, "violations": [],
        "intent_violations": [], "executions": 1, "execution_attempts": 1, "confirmations": 1,
        "unnecessary_confirmations": 0, "unnecessary_confirmations_narrowing": 0, "forced_reconfirmations": 0,
        "stale_executions": 0, "duplicates": 0, "false_blocks": 0, "authorised_requests": 1,
        "capture_class_events": 0, "unconfirmed_executions": 0, "unconfirmed_intended": 0, "declined_presentations": 0,
        "monitor_seconds": [0.0001 * (1 + window % 7), 0.0002, 0.0003 * (1 + len(seed))], "monitor_calls": 3,
    }
    if family == "revision":
        if arm in ("B0", "B3"):
            r["intent_violations"] = ["superseded"]
            r["violations"] = ["superseded"]
            r["completed"] = False
        elif arm == "B4":
            r["intent_violations"] = ["superseded"]
            r["unnecessary_confirmations_narrowing"] = 1
            r["confirmations"] = 2
        elif rule == "invalidate":
            r["unnecessary_confirmations_narrowing"] = 1
            r["confirmations"] = 2
        else:
            r["unconfirmed_executions"] = 1
            r["unconfirmed_intended"] = 1
    if family == "expiry_stale":
        if arm == "B0" and window == 60:
            r["stale_executions"] = 1
        if arm == "Tasdiq" and window == 60:
            r["forced_reconfirmations"] = 1
            r["confirmations"] = 2
    if family == "ordinary_valid" and arm == "B3":
        r["intent_violations"] = ["substitution"]
        r["violations"] = ["mismatch:payee_id"]
        r["executions"] = 2
    if arm == "B0" and lang == "ar":
        r["capture_class_events"] = 1 if family == "ordinary_valid" else 0
    return r


def _capture_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intended = {"payee_id": "P-1", "amount_minor": 25000, "currency": "AED"}
    wrong = {"payee_id": "P-1", "amount_minor": 2500, "currency": "AED"}
    for pipeline in ("direct", "noncanon", "translate", "translate_mt"):
        for seed in SEEDS:
            for lang in LANGS:
                members = [("base", f"{seed}.t0.{lang}.base", None), ("paraphrase", f"{seed}.t0.{lang}.p1", None), ("edit", f"{seed}.t0.{lang}.e1", f"{seed}.t0.{lang}.base")]
                for kind, vid, base in members:
                    ok = True
                    if lang == "ar" and pipeline == "noncanon":
                        ok = False
                    if lang == "ar" and pipeline == "translate" and seed in ("s1", "s2"):
                        ok = False
                    if lang == "ar" and pipeline == "translate_mt" and seed == "s1":
                        ok = False
                    if lang == "ar" and pipeline == "direct" and seed == "s1" and kind == "paraphrase":
                        ok = False
                    produced = dict(intended) if ok else dict(wrong)
                    if kind == "edit":
                        produced = {"payee_id": "P-2", "amount_minor": 25000, "currency": "AED"}
                        target = dict(produced)
                    else:
                        target = dict(intended)
                    rows.append({
                        "variant_id": vid, "seed_id": seed, "pair_id": f"{seed}.t0", "family": SEEDS[seed], "lang": lang, "kind": kind,
                        "base_variant_id": base, "pipeline": pipeline, "model": "toy-1b", "intended": target, "produced": produced,
                        "ask_field": None, "ambiguous_field": None, "parse_failure": None, "phenomena": ["digit_script", "currency_words"],
                        "timings": {"model": 0.5 + 0.01 * len(rows), "canonicalise": 0.001, "gate": 0.0001, "translate": 0.2 if pipeline in ("translate", "translate_mt") else 0.0},
                        "peak_rss_bytes": 1_500_000_000,
                    })
    return rows


def _live_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    tasks = (("t_rev1", "revision"), ("t_rev2", "revision"), ("t_can", "cancellation"), ("t_ctl", "ordinary_valid"))
    for task, family in tasks:
        for lang in LANGS:
            for arm in ("PromptOnly", "Tasdiq"):
                eid = f"{task}|toy-1b|{arm}|{lang}|1"
                reproposed: Any = None if family == "ordinary_valid" else (arm == "PromptOnly" and (lang == "ar" or task == "t_rev1"))
                unsafe = 1 if (arm == "PromptOnly" and reproposed) else 0
                episodes.append({
                    "episode_id": eid, "task_id": task, "family": family, "lang": lang, "arm": arm, "model": "toy-1b", "seed": 1,
                    "valid": True, "parse_failures": 0, "steps": 4, "reproposed_superseded": reproposed,
                    "duplicated_current": bool(arm == "PromptOnly" and lang == "ar" and family == "revision"),
                    "completed_intended": not (arm == "PromptOnly" and unsafe), "unsafe_executions": unsafe,
                    "unconfirmed_executions": unsafe, "unconfirmed_intended": 0, "peak_rss_bytes": 1_800_000_000, "model_seconds": 4.0,
                })
                for i in range(4):
                    steps.append({"episode_id": eid, "model": "toy-1b", "model_seconds": 1.0 + 0.1 * i, "decisions": []})
    # One dev-split episode (a harness-tuning task) with every outcome set, which the ledger must leave out.
    dev_id = "t_dev|toy-1b|PromptOnly|en|1"
    episodes.append({"episode_id": dev_id, "task_id": "t_dev", "family": "revision", "lang": "en", "arm": "PromptOnly", "model": "toy-1b", "seed": 1, "split": "dev",
                     "valid": True, "parse_failures": 3, "steps": 9, "reproposed_superseded": True, "duplicated_current": True, "completed_intended": False,
                     "unsafe_executions": 5, "unconfirmed_executions": 5, "unconfirmed_intended": 0, "peak_rss_bytes": 9_000_000_000, "model_seconds": 900.0})
    steps.append({"episode_id": dev_id, "model": "toy-1b", "model_seconds": 500.0, "decisions": [{"hook": "execute", "monitor_seconds": 50.0}]})
    return episodes, steps


@pytest.fixture(scope="module")
def toy_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("toy")
    results = root / "results"
    data = root / "data"
    trace = results / "trace" / "toy_trace"
    sessions = [_session(seed, lang, arm, rule, w) for seed in SEEDS for lang in LANGS for arm, rule in ARMS for w in WINDOWS]
    _write_jsonl(trace / "sessions.jsonl", sessions)
    _manifest(trace, "toy_trace", 100_000_000)
    capture = results / "capture" / "toy_capture"
    _write_jsonl(capture / "captures.jsonl", _capture_rows())
    _manifest(capture, "toy_capture", 1_500_000_000)
    live = results / "live" / "toy_live"
    episodes, steps = _live_rows()
    _write_jsonl(live / "episodes.jsonl", episodes)
    _write_jsonl(live / "steps.jsonl", steps)
    _manifest(live, "toy_live", 1_800_000_000)
    verify = results / "verify" / "toy_verify"
    verify.mkdir(parents=True)
    (verify / "explore.json").write_text(json.dumps({"results": [
        {"arm": "Tasdiq", "rule": "invalidate", "depth": 4, "states": 50, "transitions": 90, "n_violations": 0, "n_intent_departures": 0, "executions_max": 1, "complete": True},
        {"arm": "TasdiqNonAtomic", "rule": "invalidate", "depth": 4, "states": 70, "transitions": 120, "n_violations": 3, "n_intent_departures": 1, "executions_max": 2, "complete": False},
    ]}), encoding="utf-8")
    _manifest(verify, "toy_verify", 0)
    _write_jsonl(data / "scenarios.jsonl", [{"seed_id": s, "family": f, "dialect_extra": False} for s, f in SEEDS.items()])
    variants = []
    for seed, fam in SEEDS.items():
        for lang in LANGS + ("mix",):
            for kind in ("base", "paraphrase", "edit"):
                variants.append({"variant_id": f"{seed}.t0.{lang}.{kind}", "seed_id": seed, "family": fam, "lang": lang, "kind": kind, "split": "dev" if seed == "s1" else "test", "review_status": "unreviewed", "dialect_extra": False})
    _write_jsonl(data / "benchmark.jsonl", variants)
    (data / "review_set.csv").write_text("variant_id,label_status\ns1.t0.ar.base,unreviewed\ns2.t0.ar.base,unreviewed\n", encoding="utf-8")
    (data / "build_report.json").write_text(json.dumps({"n_near_duplicate_pairs": 2, "n_dialect_extra_seeds": 0}), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def built(toy_root: Path) -> tuple[ledger_mod.Ledger, dict[str, Any]]:
    ledger, manifest = ledger_mod.build_ledger(toy_root / "results", toy_root / "data", n_boot=N_BOOT, seed=7, root=toy_root, n_perm=N_PERM)
    ledger_mod.write_ledger(toy_root / "results", ledger, manifest)
    return ledger, manifest


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def test_ledger_claim_ids_and_hand_checked_values(built: tuple[ledger_mod.Ledger, dict[str, Any]]) -> None:
    ledger, manifest = built
    for cid in ("cap_invariance_ar_direct_toy", "cap_sensitivity_ar_direct_toy", "enf_intent_violations_B4_w600", "enf_intent_violations_B4_revision_w600",
                "attr_capture_class_B0_ar_w600", "rev_curve_narrowing_unnecessary_reconf", "rev_curve_invalidate_unnecessary_reconf",
                "win_curve_600_stale_rate", "win_curve_60_forced_reconf_rate", "den_context_change_60", "den_end_to_end_invalid_600",
                "live_reproposal_toy_ar_tasdiq", "live_reproposal_toy_all_promptonly", "h1_invariance_diff_en_minus_ar_direct_toy",
                "h2_premise_intent_violations_B4", "h2_capture_error_reduction_direct_vs_noncanon_toy", "h3_unnecessary_reconf_diff_invalidate_minus_narrowing",
                "h2_reduction_diff_direct_minus_translate_toy", "h2_reduction_diff_direct_minus_translate_mt_toy", "enf_unnecessary_reconf_narrowing_sessions_Tasdiq_invalidate_w600",
                "live_executions_toy_all_promptonly", "live_confirmations_toy_all_tasdiq",
                "h5_reproposal_toy", "sec_reproposal_lang_diff_ar_minus_en_toy_promptonly", "lat_model_direct_toy_median", "lat_model_live_toy_p95",
                "mem_peak_rss_bytes_capture", "ver_violations_TasdiqNonAtomic_invalidate_d4", "bench_seeds_total", "bench_variants_ar_edit",
                "bench_seeds_dev", "bench_review_rows", "cal_window_s_B4", "cal_windows_swept",
                "lat_monitor_trace_median", "lat_monitor_trace_p95", "lat_monitor_trace_n"):
        assert cid in ledger, cid
    # Monitor hook timings of the trace track: three timed calls per session, read from monitor_seconds.
    n_sessions = len(SEEDS) * len(LANGS) * len(ARMS) * len(WINDOWS)
    assert ledger.get("lat_monitor_trace_n").value == 3 * n_sessions
    assert 0.0001 <= ledger.get("lat_monitor_trace_median").value <= 0.0015
    assert any(e["token"] == "monitor_trace" and e["stage"] == "monitor" and e["pipeline"] == "trace" for e in manifest["groups"]["latency"])
    # B4 executes the superseded record on every revision session: 2 seeds x 2 languages.
    assert ledger.get("enf_intent_violations_B4_revision_w600").value == 4
    assert ledger.get("enf_intent_violations_B4_w600").value == 4
    # B4's superseded class is a measured class, so the premise count (over both windows) is zero with an exact bound.
    assert ledger.get("h2_premise_intent_violations_B4").value == 0
    assert ledger.get("h2_premise_measured_B4_superseded").value == 8
    assert ledger.get("h2_premise_intent_violation_rate_B4").ci["method"] == "clopper_pearson"
    # Invalidate re-confirms every revision session; narrowing none: paired difference of 1.0.
    diff = ledger.get("h3_unnecessary_reconf_diff_invalidate_minus_narrowing")
    assert diff.value == pytest.approx(1.0)
    assert diff.ci["method"] == "cluster_bootstrap" and diff.clusters == 2
    assert ledger.get("h3_narrowing_intent_violation_rate").count == 0
    # Window curves: forced re-confirmations at 60 s only; B0 shows the context change at 60 s.
    assert ledger.get("win_curve_60_forced_reconf_rate").value == pytest.approx(1.0)
    assert ledger.get("win_curve_600_forced_reconf_rate").value == 0.0
    assert ledger.get("den_context_change_60").value == pytest.approx(1.0)
    # Live: PromptOnly re-proposes on 3 of 6 eligible valid episodes (t_rev1 en and ar, t_rev2 ar, t_can ar).
    rep = ledger.get("h5_reproposal_toy")
    assert rep.count == 4 and rep.n == 6
    assert ledger.get("h5_unsafe_executions_behind_monitor_toy").value == 0
    sec = ledger.get("sec_reproposal_lang_diff_ar_minus_en_toy_promptonly")
    assert sec.p_value is not None and sec.p_holm is not None and sec.p_holm >= sec.p_value
    # Capture: Arabic direct invariance fails on one paraphrase of four pairs (edits excluded, base is the reference).
    inv = ledger.get("cap_invariance_ar_direct_toy")
    assert inv.n == 8 and inv.count == 7
    assert ledger.get("cap_invariance_en_direct_toy").value == 1.0
    red = ledger.get("h2_capture_error_reduction_direct_vs_noncanon_toy")
    assert red.value is not None and red.value > 0.5
    # Both translate-then-bind comparators are paired with the direct pipeline on the members all three pipelines share.
    same = ledger.get("h2_reduction_diff_direct_minus_translate_toy")
    mt = ledger.get("h2_reduction_diff_direct_minus_translate_mt_toy")
    assert same.n == mt.n and same.computed_by == mt.computed_by == "tasdiq.analysis.stats.paired_cluster_diff"
    assert same.value > mt.value > 0
    # The narrowing label is read on every arm; invalidate re-confirms the revision sessions under it.
    assert ledger.get("enf_unnecessary_reconf_narrowing_sessions_Tasdiq_invalidate_w600").count == 4
    assert ledger.get("enf_unnecessary_reconf_narrowing_sessions_Tasdiq_narrowing_w600").count == 0
    # Latency and memory, verification, benchmark composition.
    assert ledger.get("lat_model_direct_toy_n").value == 24
    assert ledger.get("mem_peak_rss_bytes_live").value == 1_800_000_000
    # The dev-split episode is excluded from every live claim: counts, the step latency and the monitor latency.
    assert ledger.get("live_n_toy_all_promptonly").value == 8 and ledger.get("live_n_toy_en_promptonly").value == 4
    assert ledger.get("lat_model_live_toy_max").value < 2.0 and ledger.get("lat_model_live_toy_n").value == 64
    assert "lat_monitor_live_n" not in ledger
    assert ledger.get("ver_violations_TasdiqNonAtomic_invalidate_d4").value == 3
    assert ledger.get("bench_seeds_total").value == 4 and ledger.get("bench_variants_ar_edit").value == 4
    assert ledger.get("bench_seeds_dev").value == 1 and ledger.get("bench_review_rows").value == 2
    # Verdicts carry one of the three outcomes and the manifest names every run.
    for k in range(1, 6):
        assert ledger.get(f"h{k}_verdict").value in ("supports", "refutes", "inconclusive")
    assert ledger.get("h3_verdict").value == "supports"
    assert ledger.get("h4_verdict").value == "inconclusive"
    assert {(r["track"], r["run_id"]) for r in manifest["runs"]} == {("trace", "toy_trace"), ("capture", "toy_capture"), ("live", "toy_live"), ("verify", "toy_verify")}
    assert manifest["groups"]["trace"]["primary_window"] == 600
    assert manifest["n_boot"] == N_BOOT
    for claim in ledger.claims.values():
        assert claim.description
        assert claim.source_run is not None or claim.unit == "verdict"


def test_ledger_round_trips_and_runs_are_selectable(toy_root: Path, built: tuple[ledger_mod.Ledger, dict[str, Any]]) -> None:
    ledger, _ = built
    back = ledger_mod.Ledger.read(toy_root / "results" / "ledger.json")
    assert back == ledger
    only_trace, manifest = ledger_mod.build_ledger(toy_root / "results", runs={"trace": ["toy_trace"], "capture": [], "live": [], "verify": []}, n_boot=10, seed=1, root=toy_root)
    assert "enf_intent_violations_B4_w600" in only_trace
    assert "cap_invariance_ar_direct_toy" not in only_trace
    assert only_trace.get("h1_verdict").value == "inconclusive"
    assert [r["track"] for r in manifest["runs"]] == ["trace"]
    with pytest.raises(FileNotFoundError):
        ledger_mod.build_ledger(toy_root / "results", runs={"trace": ["absent"]}, n_boot=10, seed=1, root=toy_root)


def test_builder_tokens_and_seeds() -> None:
    assert ledger_mod.model_slug("qwen2.5-1.5b") == "qwen25"
    assert ledger_mod.model_slug("qwen3-4b") == "qwen3"
    assert ledger_mod.arm_token("Tasdiq", "narrowing") == "Tasdiq_narrowing"
    assert ledger_mod.arm_token("B4", None) == "B4" and ledger_mod.arm_token("B4", "none") == "B4"
    assert ledger_mod.curve_token("Tasdiq", "cap_settle") == "cap_settle"
    assert ledger_mod.curve_token("TasdiqNonAtomic", "invalidate") is None
    assert ledger_mod.window_token(600) == "w600" and ledger_mod.window_token(None) == "wnone"
    assert ledger_mod.claim_seed(1, "a") != ledger_mod.claim_seed(1, "b")
    assert ledger_mod.claim_seed(1, "a") == ledger_mod.claim_seed(1, "a")


def test_ledger_is_deterministic(toy_root: Path) -> None:
    a, _ = ledger_mod.build_ledger(toy_root / "results", toy_root / "data", n_boot=20, seed=3, root=toy_root)
    b, _ = ledger_mod.build_ledger(toy_root / "results", toy_root / "data", n_boot=20, seed=3, root=toy_root)
    assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_figures_render_from_toy_ledger(toy_root: Path, built: tuple[ledger_mod.Ledger, dict[str, Any]]) -> None:
    out = toy_root / "figures"
    paths = figures_mod.make_figures(toy_root / "results", out)
    assert len(paths) == 12
    for path in paths:
        assert path.is_file() and path.stat().st_size > 0, path
    names = sorted(p.name for p in paths)
    assert names == sorted([f"tasdiq_fig{i}.{ext}" for i in range(1, 7) for ext in ("pdf", "png")])


def test_figures_degrade_gracefully_without_tracks(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    led = ledger_mod.Ledger(results / "ledger.json")
    led.add("h1_verdict", "inconclusive", "verdict")
    led.write()
    paths = figures_mod.make_figures(results, tmp_path / "figures")
    assert len(paths) == 12 and all(p.stat().st_size > 0 for p in paths)
    with pytest.raises(FileNotFoundError):
        figures_mod.make_figures(tmp_path / "nowhere", tmp_path / "figures")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_fill_tables_from_toy_ledger(toy_root: Path, built: tuple[ledger_mod.Ledger, dict[str, Any]]) -> None:
    templates = toy_root / "tables"
    out = toy_root / "tables_filled"
    written = tables_mod.fill_tables(toy_root / "results", templates, out)
    assert sorted(p.stem for p in written) == sorted(tables_mod.TABLE_NAMES)
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert "<<" not in text and ">>" not in text
        assert "\\begin{tabular}" in text and "\\bottomrule" in text
    capture = (out / "tasdiq_tab_capture.tex").read_text(encoding="utf-8")
    assert "87.5 [" in capture
    enforcement = (out / "tasdiq_tab_enforcement_counts.tex").read_text(encoding="utf-8")
    assert "revision & 4 & 4 & 4 & 0 & 0" in enforcement
    rates = (out / "tasdiq_tab_enforcement.tex").read_text(encoding="utf-8")
    assert "scenario runs (count) &" in rates and "T-inv" in rates
    cost = (out / "tasdiq_tab_cost.tex").read_text(encoding="utf-8")
    assert "live & & & & 1.80" in cost
    # Existing templates are kept; refreshing rewrites them.
    marker = templates / "tasdiq_tab_cost.tex"
    marker.write_text("% kept\n<<bench_seeds_total>>\n", encoding="utf-8")
    assert tables_mod.write_templates(toy_root / "results", templates) == []
    assert tables_mod.fill_tables(toy_root / "results", templates, out)
    assert (out / "tasdiq_tab_cost.tex").read_text(encoding="utf-8") == "% kept\n4\n"
    assert marker in tables_mod.write_templates(toy_root / "results", templates, overwrite=True)


def test_fill_text_asserts_placeholders_both_ways() -> None:
    claims = {"a": {"value": 0.66666, "unit": "fraction", "ci": {"lo": 0.5, "hi": 0.75}, "n": 9, "count": 6}, "b": {"value": 3, "unit": "count"}, "s": {"value": 0.000234, "unit": "seconds"}, "m": {"value": 2_064_789_504, "unit": "bytes"}, "t": {"value": "cap_settle", "unit": "text"}, "e": {"value": None, "unit": "fraction"}}
    filled = tables_mod.fill_text("x <<a>> <<a:ci>> <<a:n>> <<a:count>> <<b>> <<s>> <<m>> <<t>> <<e>>", claims)
    assert filled == "x 66.7 [50.0, 75.0] 9 6 3 0.000234 2.06 cap\\_settle n/a"
    with pytest.raises(AssertionError):
        tables_mod.fill_text("<<a>> and <<a>>", claims)
    with pytest.raises(KeyError):
        tables_mod.fill_text("<<a>> <<missing>>", claims)
    assert tables_mod.fill_text("plain text", claims) == "plain text"
    assert tables_mod.format_ci(claims["b"]) == "n/a"


def test_build_ledger_script_shows_help_and_runs_on_a_config(toy_root: Path) -> None:
    proc = subprocess.run([sys.executable, str(CODE_ROOT / "scripts" / "build_ledger.py"), "--help"], capture_output=True, text=True, cwd=str(CODE_ROOT))
    assert proc.returncode == 0, proc.stderr
    cfg = toy_root / "toy.yaml"
    cfg.write_text(f"run_id: toy\nresults_dir: {(toy_root / 'results').as_posix()}\ndata_dir: {(toy_root / 'data').as_posix()}\nledger_runs:\n  trace: [toy_trace]\n  capture: [toy_capture]\n  live: [toy_live]\n  verify: [toy_verify]\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, str(CODE_ROOT / "scripts" / "build_ledger.py"), "--config", str(cfg), "--n-boot", "10", "--n-perm", "50"], capture_output=True, text=True, cwd=str(CODE_ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "h3_verdict: supports" in proc.stdout
    manifest = json.loads((toy_root / "results" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_boot"] == 10 and manifest["n_perm"] == 50


# ---------------------------------------------------------------------------
# Record order, Holm families and the sensitivity reader
# ---------------------------------------------------------------------------


def _shuffled_copy(toy_root: Path, target: Path, seed: int) -> None:
    """Copy the toy results and data with every record file written in a shuffled order."""
    import random
    import shutil

    shutil.copytree(toy_root / "results", target / "results")
    shutil.copytree(toy_root / "data", target / "data")
    rng = random.Random(seed)
    for name in ("trace/toy_trace/sessions.jsonl", "capture/toy_capture/captures.jsonl", "live/toy_live/episodes.jsonl", "live/toy_live/steps.jsonl"):
        path = target / "results" / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rng.shuffle(rows)
        _write_jsonl(path, rows)
    for name in ("ledger.json", "manifest.json"):
        (target / "results" / name).unlink(missing_ok=True)


def test_intervals_do_not_depend_on_record_order(toy_root: Path, built: tuple[ledger_mod.Ledger, dict[str, Any]], tmp_path: Path) -> None:
    ledger, _ = built
    shuffled_root = tmp_path / "shuffled"
    _shuffled_copy(toy_root, shuffled_root, seed=5)
    shuffled, manifest = ledger_mod.build_ledger(shuffled_root / "results", shuffled_root / "data", n_boot=N_BOOT, seed=7, root=shuffled_root, n_perm=N_PERM)
    assert shuffled.to_dict() == ledger.to_dict()
    assert manifest["secondary_family"] and manifest["exploratory_family"]
    sessions = [json.loads(line) for line in (shuffled_root / "results" / "trace" / "toy_trace" / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ordered = ledger_mod.ordered_records(sessions, "trace")
    keys = [tuple(str(r.get(k)) for k in ("variant_id", "arm", "window_s", "rule")) for r in ordered]
    assert len(set(keys)) == len(keys)
    assert [r["variant_id"] for r in ordered] == sorted(r["variant_id"] for r in ordered)


def test_holm_families_are_pre_stated_and_adjusted_separately(built: tuple[ledger_mod.Ledger, dict[str, Any]]) -> None:
    from tasdiq.analysis import stats

    ledger, manifest = built
    secondary = manifest["secondary_family"]
    assert sorted(secondary) == ["h1_invariance_diff_en_minus_ar_direct_toy", "h3_unnecessary_reconf_diff_invalidate_minus_narrowing", "h5_completion_diff_change_minus_control_toy"]
    exploratory = manifest["exploratory_family"]
    assert exploratory and all(cid.startswith("sec_") for cid in exploratory)
    assert not set(secondary) & set(exploratory)
    for family in (secondary, exploratory):
        p_values = [ledger.get(cid).p_value for cid in family]
        assert all(p is not None and 0.0 < p <= 1.0 for p in p_values)
        assert [ledger.get(cid).p_holm for cid in family] == pytest.approx(stats.holm(p_values))
    assert manifest["n_perm"] == N_PERM
    # Interval rules without a p-value: B4 minus narrowing, the window curves and every other claim.
    assert ledger.get("h3_unnecessary_reconf_diff_B4_minus_narrowing").p_value is None
    assert ledger.get("win_curve_600_stale_rate").p_value is None
    # Invalidate re-confirms every revision session and narrowing none, in both seeds: the smallest p-value of 2 clusters.
    assert ledger.get("h3_unnecessary_reconf_diff_invalidate_minus_narrowing").p_value == pytest.approx((1 + 2 * N_PERM / 4) / (1 + N_PERM), abs=0.15)
    with_p = sorted(cid for cid, c in ledger.claims.items() if c.p_value is not None)
    assert with_p == sorted(secondary + exploratory)


def test_sensitivity_reader_claims_carry_their_token_and_leave_the_primary_claims_unchanged(toy_root: Path, built: tuple[ledger_mod.Ledger, dict[str, Any]], tmp_path: Path) -> None:
    import shutil

    ledger, _ = built
    root = tmp_path / "readers"
    shutil.copytree(toy_root / "results", root / "results")
    shutil.copytree(toy_root / "data", root / "data")
    sessions = [json.loads(line) for line in (root / "results" / "trace" / "toy_trace" / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    inattentive = []
    for r in sessions:
        r2 = dict(r)
        r2["reader"] = "inattentive"
        if r2["arm"] == "B4" and r2["family"] == "revision":
            r2["completed"] = False
            r2["declined_presentations"] = 1
        inattentive.append(r2)
    _write_jsonl(root / "results" / "trace" / "toy_trace_inattentive" / "sessions.jsonl", inattentive)
    _manifest(root / "results" / "trace" / "toy_trace_inattentive", "toy_trace_inattentive", 90_000_000)
    both, manifest = ledger_mod.build_ledger(root / "results", root / "data", runs={"trace": ["toy_trace", "toy_trace_inattentive"], "capture": ["toy_capture"], "live": ["toy_live"], "verify": ["toy_verify"]}, n_boot=N_BOOT, seed=7, root=root, n_perm=N_PERM)
    assert manifest["groups"]["trace"]["readers"] == ["exact", "inattentive"]
    assert manifest["groups"]["trace"]["primary_reader"] == "exact"
    # Every primary claim is unchanged by the second run.
    for cid, claim in ledger.claims.items():
        if claim.source_run != "toy_trace" and claim.unit != "verdict":
            continue
        assert both.get(cid) == claim, cid
    # The sensitivity run carries its own token and its differences against the primary reader.
    assert both.get("enf_intent_violations_reader_inattentive_B4_w600").value == 4
    assert both.get("enf_completion_reader_inattentive_B4_w600").value == pytest.approx(0.5)
    assert both.get("enf_completion_reader_inattentive_B4_revision_w600").value == 0.0
    assert both.get("enf_declined_presentations_reader_inattentive_B4_w600").value == 4
    assert both.get("enf_declined_presentations_B4_w600").value == 0
    diff = both.get("sens_reader_inattentive_minus_exact_completion_B4_w600")
    assert diff.value == pytest.approx(-0.5) and diff.n == 8 and diff.clusters == 4 and diff.p_value is None
    assert both.get("sens_reader_inattentive_minus_exact_declined_presentation_sessions_B4_w600").value == pytest.approx(0.5)
    assert both.get("sens_reader_inattentive_minus_exact_intent_violation_sessions_B4_w600").value == 0.0
    assert both.get("sens_reader_inattentive_minus_exact_completion_Tasdiq_invalidate_w600").value == 0.0
    assert "attr_capture_class_reader_inattentive_B4_all_w600" not in both
    assert "enf_sessions_reader_inattentive_B0_w60" in both
