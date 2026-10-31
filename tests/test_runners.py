"""Trace and capture runners on the fixture seed and the authored seeds (no model inference)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tasdiq.capture.model import ModelResult
from tasdiq.reference import RACE_INTERLEAVINGS
from tasdiq.runners.capture import context_for_variant, english_text_for, enumerate_capture_items, run_capture
from tasdiq.runners.config import code_hash, load_config, make_config, read_jsonl
from tasdiq.runners.trace import (
    arm_kwargs,
    edit_is_inquiry,
    enumerate_items,
    intent_timeline_for,
    label_window_for,
    load_benchmark,
    measured_classes,
    premise_check,
    run_item,
    run_trace,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
SEEDS = CODE_ROOT / "tasdiq" / "bench" / "seeds"


class ExtractionRunner:
    """Scripted model that answers every extraction call with one fixed RawExtraction JSON."""

    def __init__(self, values: Any = None) -> None:
        self.context: dict[str, Any] = {}
        self.calls = 0

    def identity(self) -> dict[str, Any]:
        return {"model_file": "scripted", "model_sha256": "0" * 64, "n_threads": 1, "seed": 1}

    def chat(self, messages, tools=None, json_schema=None, max_tokens=256, temperature=0.0, seed=None) -> ModelResult:
        self.calls += 1
        text = json.dumps({"action_type": "payment", "payee_mention": "Omar Khalid", "amount_text": "250 dirhams", "currency_text": None, "negated": False, "refers_to_previous": False, "uncertain_fields": []})
        return ModelResult(text=text, raw_text=text, tool_calls=[], prompt_tokens=100, completion_tokens=40, seconds=0.01, finish_reason="stop", kind="chat")

    def translate(self, text: str, src_lang: str, dst_lang: str) -> ModelResult:
        return ModelResult(text=text, raw_text=text, tool_calls=[], prompt_tokens=50, completion_tokens=20, seconds=0.01, finish_reason="stop", kind="translate")


def _cfg(tmp_path: Path, **values: Any):
    base = {
        "results_dir": str(tmp_path / "results"),
        "data_dir": str(tmp_path / "data"),
        "seeds_dir": str(SEEDS),
        "seed_ids": ["ord_001", "rev_005"],
        "splits": ["dev", "test"],
        "languages": ["en", "ar"],
        "trace_kinds": ["base"],
        "arms": ["B0", "B2", "B4", "B7", "Tasdiq"],
        "arm_params": {"B2": {"per_tx_cap_minor": 500000, "daily_cap_minor": 1000000, "allowlist": "scenario_payees"}, "B7": {"initial_policy": "first_intended_payee_with_amount_cap"}},
        "windows": [600],
        "revision_rules": ["invalidate"],
        "trace_workers": 1,
        "max_workers": 1,
        "run_id": "unit",
        "resume": True,
    }
    base.update(values)
    return make_config(base)


def test_configs_load_and_carry_hashes() -> None:
    for name in ("smoke", "pilot", "full"):
        cfg = load_config(CODE_ROOT / "configs" / f"{name}.yaml")
        assert cfg.run_id == name
        assert len(cfg.config_hash) == 64 and cfg.code_hash == code_hash()
        assert cfg.models and all(m.expected_sha256 for m in cfg.models)
        assert cfg.seeds_dir.is_dir() and cfg.path("live_tasks_file").is_file()
    full = load_config(CODE_ROOT / "configs" / "full.yaml")
    subset = full.model("qwen3-4b").subset
    assert subset["capture_pipelines"] == ["direct", "translate_mt"] and subset["capture_seed_count"] == 25
    assert subset["live_seeds"] == [1] and subset["live_temperature"] == 0.0 and subset["live_splits"] == ["test"]
    assert full.get("capture_splits") == ["test"] and full.get("live_splits") == ["test"]
    assert full.get("capture_seed_subsets") == {"noncanon": 0.3333, "translate": 0.3333}
    assert full.get("capture_pipeline_members")["english_reference"] == {"languages": ["ar", "mix"], "kinds": ["base"]}
    assert full.get("live_seeds") == [1, 2, 3] and full.get("live_temperature") == 0.3 and full.get("explore_depth") == 8
    assert len(full.get("windows")) == 7 and len(full.get("arms")) == 14


def test_trace_runner_writes_sessions_and_resumes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    specs, variants, splits = load_benchmark(cfg)
    items = enumerate_items(cfg, specs, variants, splits)
    assert len(items) == 2 * 2 * 5 or len(items) == 3 * 2 * 5
    run_dir = run_trace(cfg, log=None)
    sessions = read_jsonl(run_dir / "sessions.jsonl")
    assert len(sessions) == len(items)
    assert {s["arm"] for s in sessions} == {"B0", "B2", "B4", "B7", "Tasdiq"}
    calibrated = [s for s in sessions if s["arm"] in ("B2", "B7") and s["seed_id"] == "ord_001"]
    assert calibrated and all(s["completed"] and s["intent_violations"] == [] for s in calibrated)
    by_seed = {spec.seed_id: spec for spec in specs}
    assert all(s["arm_config"]["allowlist"] == sorted(by_seed[s["seed_id"]].payees) for s in sessions if s["arm"] == "B2")
    assert all(s["arm_config"]["initial_policy"]["payees"] == ["P-114"] for s in sessions if s["arm"] == "B7" and s["seed_id"] == "ord_001")
    assert all(s["label_window_s"] == 600 and s["arm_window_s"] is None for s in sessions if s["arm"] in ("B0", "B2", "B7"))
    assert all(s["label_window_s"] == 600 and s["arm_window_s"] == 600 for s in sessions if s["arm"] in ("B4", "Tasdiq"))
    assert (run_dir / "aggregates.json").is_file() and (run_dir / "manifest.json").is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_sessions_written"] == len(items) and manifest["config"]["code_hash"] == cfg.code_hash
    premise = premise_check(sessions)
    assert premise["Tasdiq"] == 0
    ordinary = [s for s in sessions if s["seed_id"] == "ord_001"]
    assert all(s["intent_violations"] == [] for s in ordinary if s["arm"] in ("B4", "Tasdiq"))
    assert all(s["completed"] for s in ordinary)
    run_trace(cfg, log=None)
    assert len(read_jsonl(run_dir / "sessions.jsonl")) == len(items)


def test_trace_runner_without_resume_starts_from_an_empty_run_directory(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, seed_ids=["ord_001"], languages=["en"], arms=["B4", "Tasdiq"], resume=False, run_id="redo")
    specs, variants, splits = load_benchmark(cfg)
    items = enumerate_items(cfg, specs, variants, splits)
    run_dir = run_trace(cfg, log=None)
    (run_dir / "done.jsonl").write_text('{"key": "stale"}\n', encoding="utf-8")
    run_trace(cfg, log=None)
    sessions = read_jsonl(run_dir / "sessions.jsonl")
    assert len(sessions) == len(items) and len({s["variant_id"] + s["arm"] for s in sessions}) == len(items)
    assert not (run_dir / "done.jsonl").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_sessions_written"] == len(items)
    assert len(read_jsonl(run_dir / "B4.jsonl")) > 0


def test_capture_runner_writes_rows_with_scripted_model(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, models=[{"name": "scripted", "file": "scripted.gguf", "tracks": ["capture"]}], capture_pipelines=["direct", "english_reference"], capture_kinds=["base", "edit"], seed_ids=["ord_001"])
    specs, variants, splits = load_benchmark(cfg)
    items = enumerate_capture_items(cfg, specs, variants, splits)
    assert items and all(i.model == "scripted" for i in items)
    assert not any(i.pipeline == "english_reference" and i.variant_id.endswith("en.base") for i in items)
    run_dir = run_capture(cfg, log=None, runner_factory=ExtractionRunner)
    rows = read_jsonl(run_dir / "captures.jsonl")
    assert len(rows) == len(items)
    assert all("correct" in r and "produced" in r and "intended" in r for r in rows)
    english_base = [r for r in rows if r["variant_id"].endswith("en.base") and r["pipeline"] == "direct"]
    assert english_base and english_base[0]["produced"]["payee_id"] == "P-114" and english_base[0]["correct"] is True
    assert (run_dir / "scripted__direct.jsonl").is_file() and (run_dir / "aggregates.json").is_file()
    aggregates = json.loads((run_dir / "aggregates.json").read_text(encoding="utf-8"))
    assert aggregates["n_rows"] == len(rows)
    run_capture(cfg, log=None, runner_factory=ExtractionRunner)
    assert len(read_jsonl(run_dir / "captures.jsonl")) == len(items)


def test_scripts_import_and_show_help() -> None:
    import subprocess
    import sys

    for script in ("build_benchmark", "run_trace", "run_capture", "run_live", "run_verify", "make_figures", "fill_tables"):
        proc = subprocess.run([sys.executable, str(CODE_ROOT / "scripts" / f"{script}.py"), "--help"], capture_output=True, text=True, cwd=str(CODE_ROOT))
        assert proc.returncode == 0, proc.stderr
    proc = subprocess.run([sys.executable, str(CODE_ROOT / "reproduce.py"), "--help"], capture_output=True, text=True, cwd=str(CODE_ROOT))
    assert proc.returncode == 0, proc.stderr


# ----------------------------------------------------------------------------------------
# Race accounting, label windows, calibration rules and the premise check
# ----------------------------------------------------------------------------------------


def _race_record(tmp_path: Path, arm: str, rule: str | None = None) -> dict[str, Any]:
    cfg = _cfg(tmp_path, seed_ids=["rep_005"], splits=["test"], languages=["en"], arms=[arm], revision_rules=[rule or "invalidate"], arm_params={})
    specs, variants, splits = load_benchmark(cfg)
    items = [it for it in enumerate_items(cfg, specs, variants, splits) if it.arm == arm]
    assert len(items) == 1
    by_seed = {s.seed_id: s for s in specs}
    by_variant = {v.variant_id: v for v in variants}
    record, _ = run_item(cfg, by_seed["rep_005"], by_variant[items[0].variant_id], items[0], "test", {})
    return record


@pytest.mark.parametrize("arm", ["B4", "B5", "Tasdiq"])
def test_race_credits_each_executor_with_its_own_debit_for_atomic_arms(tmp_path: Path, arm: str) -> None:
    record = _race_record(tmp_path, arm)
    assert set(record["interleaving_labels"]) == set(RACE_INTERLEAVINGS)
    for order, labels in record["interleaving_labels"].items():
        assert labels["executions"] == 1, order
        assert labels["intent_violations"] == [], order
        assert labels["completed"] is True, order
    assert record["intent_violations"] == [] and record["duplicates"] == 0 and record["rail_executions"] == 1
    assert record["any_intent_violation"] is False


def test_race_still_catches_the_non_atomic_ablation(tmp_path: Path) -> None:
    record = _race_record(tmp_path, "TasdiqNonAtomic")
    labels = record["interleaving_labels"]
    assert labels["A-check,B-check,A-commit,B-commit"]["executions"] == 2
    assert labels["B-check,A-check,B-commit,A-commit"]["executions"] == 2
    assert labels["A-check,A-commit,B-check,B-commit"]["executions"] == 1
    assert labels["B-check,B-commit,A-check,A-commit"]["executions"] == 1
    assert record["intent_violations"] == ["duplicate"] and record["any_intent_violation"] is True


def test_fixed_window_arms_are_labelled_with_their_own_window(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert label_window_for("B3", cfg) == 300
    assert label_window_for("B0", cfg) == 600 and label_window_for("B7", cfg) == 600
    pinned = _cfg(tmp_path, arm_params={"B3": {"window_s": 120}}, label_window_s=900)
    assert label_window_for("B3", pinned) == 120 and label_window_for("B0", pinned) == 900
    specs, variants, splits = load_benchmark(_cfg(tmp_path, arms=["B3", "B0"], arm_params={}))
    items = enumerate_items(_cfg(tmp_path, arms=["B3", "B0"], arm_params={}), specs, variants, splits)
    assert {it.window_s for it in items if it.arm == "B3"} == {300}
    assert {it.window_s for it in items if it.arm == "B0"} == {600}


def test_arm_params_may_not_pin_a_swept_parameter(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    specs, _, _ = load_benchmark(cfg)
    spec = specs[0]
    with pytest.raises(ValueError):
        arm_kwargs("B4", 600, None, spec, _cfg(tmp_path, arm_params={"B4": {"window_s": 60}}))
    with pytest.raises(ValueError):
        arm_kwargs("B5", 600, None, spec, _cfg(tmp_path, arm_params={"B5": {"interval_s": 60}}))
    with pytest.raises(ValueError):
        arm_kwargs("Tasdiq", 600, "invalidate", spec, _cfg(tmp_path, arm_params={"Tasdiq": {"revision_rule": "narrowing"}}))
    assert arm_kwargs("B3", 600, None, spec, _cfg(tmp_path, arm_params={"B3": {"window_s": 120}})) == {"window_s": 120}


def test_calibration_rules_read_the_scenario_only(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    specs, _, _ = load_benchmark(cfg)
    spec = next(s for s in specs if s.seed_id == "ord_001")
    b2 = arm_kwargs("B2", None, None, spec, cfg)
    assert b2["allowlist"] == list(spec.payees) and b2["per_tx_cap_minor"] == 500000 and b2["daily_cap_minor"] == 1000000
    b7 = arm_kwargs("B7", None, None, spec, cfg)
    first = next(t.intended for t in spec.turns if t.speaker == "user")
    assert b7["initial_policy"] == {"payees": [first.payee_id], "cap_minor": first.amount_minor, "currency": first.currency}
    with pytest.raises(ValueError):
        arm_kwargs("B7", None, None, spec, _cfg(tmp_path, arm_params={"B7": {"initial_policy": "scenario_payees"}}))


def test_measured_classes_combine_arm_and_rule_entries() -> None:
    assert measured_classes("B4") == {"cancelled", "superseded", "lapsed"}
    assert measured_classes("B4", "none") == {"cancelled", "superseded", "lapsed"}
    assert measured_classes("B5", "none") == {"lapsed"}
    assert measured_classes("Tasdiq", "invalidate") == {"lapsed"}
    assert measured_classes("Tasdiq", "narrowing") == {"lapsed"}
    assert measured_classes("Tasdiq", "cap_settle") == {"lapsed", "superseded", "amount_change"}
    assert measured_classes("B0") == frozenset() and measured_classes("B0", "none") == frozenset()


def test_premise_check_separates_measured_classes() -> None:
    sessions = [
        {"arm": "Tasdiq", "rule": "invalidate", "mode": "oracle", "intent_violations": ["lapsed"]},
        {"arm": "Tasdiq", "rule": "narrowing", "mode": "oracle", "intent_violations": ["superseded", "lapsed", "amount_change"]},
        {"arm": "Tasdiq", "rule": "cap_settle", "mode": "oracle", "intent_violations": ["superseded", "superseded", "amount_change", "lapsed", "substitution"]},
        {"arm": "B4", "rule": "none", "mode": "oracle", "intent_violations": ["cancelled", "superseded", "substitution", "lapsed"]},
        {"arm": "B5", "rule": "none", "mode": "oracle", "intent_violations": ["lapsed", "duplicate"]},
        {"arm": "B4", "rule": "none", "mode": "captured", "intent_violations": ["substitution"]},
    ]
    premise = premise_check(sessions)
    assert premise["Tasdiq"] == 2 and premise["B4"] == 1 and premise["Tasdiq/cap_settle"] == 1
    assert premise["measured"] == {
        "Tasdiq": {"lapsed": 2},
        "Tasdiq/cap_settle": {"superseded": 2, "amount_change": 1, "lapsed": 1},
        "B4": {"cancelled": 1, "superseded": 1, "lapsed": 1},
    }
    assert "B5" not in premise
    with_b5 = premise_check(sessions, arms=("B4", "B5", "Tasdiq"))
    assert with_b5["B5"] == 1 and with_b5["measured"]["B5"] == {"lapsed": 1}


@pytest.mark.parametrize("edit_id", ["rev_002_e5", "rev_009_e7"])
def test_inquiry_edit_at_a_revision_turn_leaves_the_earlier_intent_in_force(tmp_path: Path, edit_id: str) -> None:
    # rev_002 e5 asks whether the amount could be raised (inquiry versus approval undecidable) and rev_009 e7 asks whether
    # the discount was applied: neither withdraws the earlier confirmed request, so its execution is INTENDED and AUTHORISED.
    seed_id = edit_id[:7]
    cfg = _cfg(tmp_path, seed_ids=[seed_id], languages=["en"], trace_kinds=["edit"], arms=["B4", "Tasdiq"], run_id=f"inq-{edit_id}")
    specs, variants, splits = load_benchmark(cfg)
    items = [it for it in enumerate_items(cfg, specs, variants, splits) if it.variant_id.endswith(edit_id)]
    assert len(items) == 2
    variant = next(v for v in variants if v.variant_id == items[0].variant_id)
    assert edit_is_inquiry(variant)
    earlier = 80000 if seed_id == "rev_002" else 380000
    assert intent_timeline_for(specs[0], variant).intended_at(10_000).amount_minor == earlier
    for item in items:
        record, lines = run_item(cfg, specs[0], variant, item, "test", {})
        executes = _execute_lines(lines)
        assert [(t, amount, verdict) for t, amount, verdict in executes] == [(29 if seed_id == "rev_002" else 44, earlier, "ALLOW")]
        assert record["intent_violations"] == [] and record["violations"] == []
        assert record["executions"] == 1 and record["completed"] and record["completed_confirmed"]


# ----------------------------------------------------------------------------------------
# Capture context: the English member of the same edit, and the previous record source
# ----------------------------------------------------------------------------------------


def test_english_reference_uses_the_english_member_of_the_same_edit_or_paraphrase(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, seed_ids=["ord_001"], trace_kinds=["base", "paraphrase", "edit"])
    specs, variants, _ = load_benchmark(cfg)
    spec = specs[0]
    turn = spec.turn(0)
    for variant in variants:
        if variant.turn_idx != 0:
            continue
        english = english_text_for(spec, variant)
        context = context_for_variant(spec, variant)
        assert context.english_text == english
        if variant.kind == "base":
            assert english == turn.text.en
        elif variant.kind == "paraphrase":
            assert english == turn.paraphrases.get("en")[variant.paraphrase_idx - 1]
        else:
            edit = next(e for e in turn.edits if e.edit_id == variant.edit_id)
            assert english == edit.text.en
            if variant.lang != "en":
                assert english != turn.text.en


def test_capture_rows_record_the_previous_record_source(tmp_path: Path) -> None:
    for source in ("oracle", "captured"):
        cfg = _cfg(tmp_path / source, models=[{"name": "scripted", "file": "scripted.gguf", "tracks": ["capture"]}], capture_pipelines=["direct"], capture_kinds=["base", "edit"], seed_ids=["ord_001", "rev_005", "can_005"], previous_record_source=source)
        run_dir = run_capture(cfg, log=None, runner_factory=ExtractionRunner)
        rows = read_jsonl(run_dir / "captures.jsonl")
        assert rows and all(r["previous_record_source"] == source for r in rows)
        assert {r["intended_kind"] for r in rows} == {"record", "no_record"}
        later = [r for r in rows if r["turn_idx"] > 0]
        assert later
        by_id = {(r["variant_id"], r["pipeline"], r["model"]): r for r in rows}
        found_captured = False
        for r in later:
            if source == "oracle":
                assert r["previous_record"] is not None and str(r["previous_record"]["payee_id"]).startswith("P-")
                continue
            expected = None
            for idx in range(r["turn_idx"] - 1, -1, -1):
                earlier = by_id.get((f"{r['seed_id']}.t{idx}.{r['lang']}.base", r["pipeline"], r["model"]))
                if earlier is not None and earlier["outcome_kind"] == "record":
                    expected = {k: earlier["produced"][k] for k in ("payee_id", "amount_minor", "currency", "total_max_minor")}
                    break
            assert r["previous_record"] == expected
            found_captured = found_captured or expected is not None
        if source == "captured":
            assert found_captured


TIMED_HOOKS = {"consent_request", "execute", "user_event"}


def test_trace_sessions_carry_monitor_hook_timings(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, seed_ids=["rev_005"], arms=["B0", "Tasdiq"], run_id="timed")
    run_dir = run_trace(cfg, log=None)
    sessions = read_jsonl(run_dir / "sessions.jsonl")
    assert sessions
    for s in sessions:
        assert isinstance(s["monitor_seconds"], list) and s["monitor_calls"] == len(s["monitor_seconds"])
        assert s["monitor_calls"] >= 1 and all(isinstance(v, float) and v >= 0.0 for v in s["monitor_seconds"])
    for arm in ("B0", "Tasdiq"):
        lines = [line for line in read_jsonl(run_dir / f"{arm}.jsonl") if line.get("hook") in TIMED_HOOKS]
        assert lines and all(isinstance(line.get("monitor_seconds"), float) and line["monitor_seconds"] >= 0.0 for line in lines)
    # The timings are wall-clock only: the labels are those of an untimed run.
    assert {s["completed"] for s in sessions if s["arm"] == "Tasdiq"} == {True}


def test_race_executors_are_timed_without_their_parked_interval(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, seed_ids=["rep_005"], arms=["Tasdiq"], run_id="race")
    specs, variants, splits = load_benchmark(cfg)
    items = [it for it in enumerate_items(cfg, specs, variants, splits) if it.arm == "Tasdiq"]
    by_seed = {s.seed_id: s for s in specs}
    by_variant = {v.variant_id: v for v in variants}
    record, lines = run_item(cfg, by_seed["rep_005"], by_variant[items[0].variant_id], items[0], "test", {})
    assert record["monitor_calls"] == len(record["monitor_seconds"]) >= 3
    assert all(0.0 <= v < 5.0 for v in record["monitor_seconds"])
    executes = [line for line in lines if line.get("hook") == "execute"]
    assert executes and all(isinstance(line.get("monitor_seconds"), float) for line in executes)


# ----------------------------------------------------------------------------------------
# Turn-keyed record map, seed subsets, member restrictions, model filter and dry-run counts
# ----------------------------------------------------------------------------------------


def _execute_lines(lines: list[dict[str, Any]]) -> list[tuple[int, int, str | None]]:
    return [(int(line["t"]), int((line.get("request") or {}).get("amount_minor")), line.get("verdict")) for line in lines if line.get("hook") == "execute"]


@pytest.mark.parametrize("rule", ["cap_settle", "invalidate", "narrowing"])
def test_edit_on_one_turn_leaves_executions_scripted_for_another_turn_unchanged(tmp_path: Path, rule: str) -> None:
    # tot_001 states (P-171, 1,000 USD) at turn 0 and again at turn 2 (only the cap changes); edit e5 lowers turn 2 to 900 USD.
    cfg = _cfg(tmp_path, seed_ids=["tot_001"], languages=["en"], trace_kinds=["edit"], arms=["Tasdiq", "B0"], revision_rules=[rule], run_id=f"tot-{rule}")
    specs, variants, splits = load_benchmark(cfg)
    items = [it for it in enumerate_items(cfg, specs, variants, splits) if it.variant_id.endswith("tot_001_e5")]
    assert len(items) == 2
    by_seed = {s.seed_id: s for s in specs}
    by_variant = {v.variant_id: v for v in variants}
    for item in items:
        record, lines = run_item(cfg, by_seed["tot_001"], by_variant[item.variant_id], item, "test", {})
        executes = _execute_lines(lines)
        assert [(t, amount) for t, amount, _ in executes] == [(34, 100000), (66, 90000)]
        if item.arm == "Tasdiq":
            assert executes[0][2] == "DENY" and executes[1][2] == "ALLOW"
            assert record["intent_violations"] == [] and record["completed"]
        else:
            assert executes[0][2] == "ALLOW" and "cap_exceeded" in record["intent_violations"]


@pytest.mark.parametrize("rule", ["invalidate", "narrowing", "cap_settle"])
def test_revision_back_to_an_earlier_tuple_keeps_later_revisions_effective(tmp_path: Path, rule: str) -> None:
    # rev_009 edit e4 revises turn 1 back to the turn-0 tuple (4,200 AED); the revision at turn 2 (4,000 AED) must still
    # reach the record in force, so that the stale 4,200 AED submission at t=44 is denied and the 4,000 AED one executes.
    cfg = _cfg(tmp_path, seed_ids=["rev_009"], languages=["en"], trace_kinds=["edit"], arms=["Tasdiq"], revision_rules=[rule], run_id=f"rev9-{rule}")
    specs, variants, splits = load_benchmark(cfg)
    items = [it for it in enumerate_items(cfg, specs, variants, splits) if it.variant_id.endswith("rev_009_e4")]
    assert len(items) == 1
    record, lines = run_item(cfg, specs[0], next(v for v in variants if v.variant_id == items[0].variant_id), items[0], "test", {})
    executes = _execute_lines(lines)
    assert [(t, amount) for t, amount, _ in executes] == [(44, 420000), (61, 400000)]
    assert executes[0][2] == "DENY" and executes[1][2] == "ALLOW"
    assert record["intent_violations"] == [] and record["completed"]


def test_edit_on_the_first_turn_rewrites_only_that_turn(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, seed_ids=["tot_001"], languages=["en"], trace_kinds=["edit"], arms=["B0"], run_id="tot-e1")
    specs, variants, splits = load_benchmark(cfg)
    items = [it for it in enumerate_items(cfg, specs, variants, splits) if it.variant_id.endswith("tot_001_e1")]
    assert len(items) == 1
    _, lines = run_item(cfg, specs[0], next(v for v in variants if v.variant_id == items[0].variant_id), items[0], "test", {})
    assert [(t, amount) for t, amount, _ in _execute_lines(lines)] == [(34, 120000), (66, 100000)]


def test_trace_counts_and_excluded_variants(tmp_path: Path) -> None:
    from tasdiq.runners.trace import excluded_variants, trace_counts

    cfg = _cfg(tmp_path, seed_ids=["ord_001", "ord_002"], languages=["en"], trace_kinds=["base"], arms=["B0"], run_id="counts")
    assert excluded_variants(cfg) == frozenset()
    counts = trace_counts(cfg)
    assert counts["track"] == "trace" and counts["items"] == 2 and counts["pending"] == 2 and counts["arms"] == {"B0": 2}
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "splits.json").write_text(json.dumps({"seed_splits": {}, "excluded_variants": ["ord_002.t0.en.base"]}), encoding="utf-8")
    assert excluded_variants(cfg) == frozenset({"ord_002.t0.en.base"})
    assert trace_counts(cfg)["items"] == 1
    run_dir = run_trace(cfg, log=None)
    sessions = read_jsonl(run_dir / "sessions.jsonl")
    assert [s["variant_id"] for s in sessions] == ["ord_001.t0.en.base"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_excluded_variants"] == 1
    assert trace_counts(cfg)["pending"] == 0


def test_capture_items_follow_subsets_members_and_the_model_filter(tmp_path: Path) -> None:
    from tasdiq.runners.capture import capture_counts, capture_seed_subsets, filter_by_model

    seed_ids = ["ord_001", "ord_002", "ord_003", "rev_001", "rev_002", "rev_003", "can_001", "can_002"]
    models = [
        {"name": "small", "file": "small.gguf", "tracks": ["capture"]},
        {"name": "large", "file": "large.gguf", "tracks": ["capture"], "subset": {"capture_pipelines": ["direct", "translate_mt"], "capture_seed_count": 3}},
    ]
    cfg = _cfg(
        tmp_path,
        models=models,
        seed_ids=seed_ids,
        splits=["dev"],
        capture_splits=["dev", "test"],
        languages=["en", "ar", "mix"],
        capture_pipelines=["direct", "noncanon", "translate", "translate_mt", "english_reference"],
        capture_kinds=["base", "edit"],
        capture_pipeline_members={"translate": {"languages": ["ar", "mix"]}, "translate_mt": {"languages": ["ar", "mix"]}, "english_reference": {"languages": ["ar", "mix"], "kinds": ["base"]}},
        capture_seed_subsets={"noncanon": 0.375, "translate": 0.375},
    )
    specs, variants, splits = load_benchmark(cfg)
    subsets = capture_seed_subsets(cfg, specs, splits)
    assert len(subsets["pipelines"]["noncanon"]) == 3 and subsets["pipelines"]["noncanon"] == subsets["pipelines"]["translate"]
    assert subsets["models"]["large"] == subsets["pipelines"]["noncanon"]
    assert len({next(s.family for s in specs if s.seed_id == sid) for sid in subsets["models"]["large"]}) == 3
    items = enumerate_capture_items(cfg, specs, variants, splits)
    small = [it for it in items if it.model == "small"]
    large = [it for it in items if it.model == "large"]
    by_variant = {v.variant_id: v for v in variants}
    assert {it.pipeline for it in large} == {"direct", "translate_mt"} and {it.seed_id for it in large} == set(subsets["models"]["large"])
    assert all(by_variant[it.variant_id].lang != "en" for it in items if it.pipeline in ("translate", "translate_mt", "english_reference"))
    assert all(by_variant[it.variant_id].kind == "base" for it in items if it.pipeline == "english_reference")
    assert {it.seed_id for it in small if it.pipeline in ("noncanon", "translate")} == set(subsets["pipelines"]["noncanon"])
    assert {it.seed_id for it in small if it.pipeline == "direct"} == set(seed_ids)
    assert len(small) + len(large) == len(items) and len(filter_by_model(items, make_config({**cfg.values, "model_filter": "large"}))) == len(large)
    with pytest.raises(KeyError):
        filter_by_model(items, make_config({**cfg.values, "model_filter": "absent"}))
    counts = capture_counts(cfg)
    assert counts["items"] == len(items) and counts["models"]["large"]["items"] == len(large) and counts["seed_subsets"]["models"]["large"] == 3
    only_test = _cfg(tmp_path, models=models[:1], seed_ids=seed_ids, splits=["dev"], capture_splits=["test"], capture_pipelines=["direct"], capture_kinds=["base"], languages=["en"])
    specs2, variants2, splits2 = load_benchmark(only_test)
    assert {splits2[it.seed_id] for it in enumerate_capture_items(only_test, specs2, variants2, splits2)} == {"test"}


def test_run_capture_honours_the_model_filter_and_writes_the_subsets(tmp_path: Path) -> None:
    models = [{"name": "scripted", "file": "scripted.gguf", "tracks": ["capture"]}, {"name": "other", "file": "other.gguf", "tracks": ["capture"]}]
    cfg = _cfg(tmp_path, models=models, capture_pipelines=["direct"], capture_kinds=["base"], seed_ids=["ord_001"], languages=["en"], model_filter="scripted", capture_seed_subsets={"direct": 1.0})
    run_dir = run_capture(cfg, log=None, runner_factory=ExtractionRunner)
    rows = read_jsonl(run_dir / "captures.jsonl")
    assert rows and {r["model"] for r in rows} == {"scripted"}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_filter"] == "scripted" and manifest["n_items"] == 2 * len(rows) and manifest["n_items_selected"] == len(rows)
    assert manifest["seed_subsets"]["pipelines"]["direct"] == ["ord_001"] and manifest["capture_splits"] == ["dev", "test"]


def test_scripts_dry_run_prints_counts_without_writing(tmp_path: Path) -> None:
    import subprocess
    import sys

    import yaml

    values = {
        "results_dir": str(tmp_path / "results"),
        "data_dir": str(tmp_path / "data"),
        "seeds_dir": str(SEEDS),
        "seed_ids": ["ord_001"],
        "splits": ["dev", "test"],
        "languages": ["en"],
        "trace_kinds": ["base"],
        "arms": ["B0"],
        "windows": [600],
        "capture_pipelines": ["direct"],
        "capture_kinds": ["base"],
        "models": [{"name": "scripted", "file": "scripted.gguf", "tracks": ["capture", "live"]}],
        "live_tasks": ["live_ord_001"],
        "live_langs": ["en"],
        "verify_arms": ["Tasdiq"],
        "revision_rules": ["invalidate"],
        "explore_depth": 2,
        "run_id": "dry",
    }
    config_path = tmp_path / "dry.yaml"
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    expected = {"run_trace": 1, "run_capture": 1, "run_live": 2, "run_verify": 1}
    for script, n in expected.items():
        proc = subprocess.run([sys.executable, str(CODE_ROOT / "scripts" / f"{script}.py"), "--config", str(config_path), "--dry-run"], capture_output=True, text=True, cwd=str(CODE_ROOT))
        assert proc.returncode == 0, proc.stderr
        assert f"{n} work units" in proc.stdout.splitlines()[0], proc.stdout
    proc = subprocess.run([sys.executable, str(CODE_ROOT / "scripts" / "run_capture.py"), "--config", str(config_path), "--dry-run", "--model", "scripted"], capture_output=True, text=True, cwd=str(CODE_ROOT))
    assert proc.returncode == 0 and "model scripted: 1 work units" in proc.stdout
    assert not (tmp_path / "results").exists()


def test_reproduce_rerun_keeps_the_shipped_verify_timing(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("reproduce_module", CODE_ROOT / "reproduce.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    from tasdiq.verify.explore import run_verify

    cfg = make_config({"results_dir": str(tmp_path / "results"), "explore_depth": 2, "revision_rules": ["invalidate"], "verify_arms": ["Tasdiq"], "windows": [600], "run_id": "shipped"})
    run_dir = run_verify(cfg, log=None)
    manifest_path, explore_path = run_dir / "manifest.json", run_dir / "explore.json"
    shipped = json.loads(manifest_path.read_text(encoding="utf-8"))
    shipped["elapsed_seconds"], shipped["started"], shipped["ended"], shipped["peak_rss_bytes"] = 430.0, 1.0, 431.0, 12345
    manifest_path.write_text(json.dumps(shipped), encoding="utf-8")
    explore_shipped = json.loads(explore_path.read_text(encoding="utf-8"))
    explore_shipped["elapsed_seconds"] = 274.8
    explore_shipped["results"][0]["elapsed_seconds"] = 22.5
    explore_path.write_text(json.dumps(explore_shipped), encoding="utf-8")
    module.rerun_verify(cfg, log=lambda _: None)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (manifest["elapsed_seconds"], manifest["started"], manifest["ended"], manifest["peak_rss_bytes"]) == (430.0, 1.0, 431.0, 12345)
    assert manifest["reproduce_elapsed_seconds"] >= 0.0 and manifest["reproduce_peak_rss_bytes"] > 0 and manifest["reproduce_ended"] >= manifest["reproduce_started"]
    explore_doc = json.loads(explore_path.read_text(encoding="utf-8"))
    assert explore_doc["elapsed_seconds"] == 274.8 and explore_doc["reproduce_elapsed_seconds"] >= 0.0
    assert explore_doc["results"][0]["elapsed_seconds"] == 22.5 and explore_doc["results"][0]["reproduce_elapsed_seconds"] >= 0.0
    assert explore_doc["results"][0]["states"] == explore_shipped["results"][0]["states"]
    fresh = make_config({**cfg.values, "run_id": "fresh"})
    fresh_dir = module.rerun_verify(fresh, log=lambda _: None)
    assert "reproduce_elapsed_seconds" not in json.loads((fresh_dir / "manifest.json").read_text(encoding="utf-8"))


def test_sensitivity_runs_derive_from_the_main_configuration_and_the_trace_script_takes_the_reader(tmp_path: Path) -> None:
    import importlib.util
    import subprocess
    import sys

    import yaml

    spec = importlib.util.spec_from_file_location("reproduce_module", CODE_ROOT / "reproduce.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    values = {"results_dir": str(tmp_path / "results"), "run_id": "main", "user_model": "exact", "windows": [600], "revision_rules": ["invalidate"], "arms": ["B4"], "seed_ids": ["rev_003"], "languages": ["en"], "trace_sensitivity_runs": {"main_inattentive": {"user_model": "inattentive"}}}
    cfg = make_config(values)
    assert module.sensitivity_configs(make_config({k: v for k, v in values.items() if k != "trace_sensitivity_runs"})) == []
    derived = module.sensitivity_configs(cfg)
    assert [c.run_id for c in derived] == ["main_inattentive"]
    assert derived[0].get("user_model") == "inattentive" and derived[0].get("arms") == ["B4"]
    assert cfg.get("user_model") == "exact" and derived[0].config_hash != cfg.config_hash
    config_path = tmp_path / "main.yaml"
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(CODE_ROOT / "scripts" / "run_trace.py"), "--config", str(config_path), "--dry-run", "--run-id", "main_inattentive", "--user-model", "inattentive", "--workers", "1"], capture_output=True, text=True, cwd=str(CODE_ROOT))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[0].startswith("trace: run main_inattentive:")
    proc = subprocess.run([sys.executable, str(CODE_ROOT / "scripts" / "run_trace.py"), "--config", str(config_path), "--dry-run", "--user-model", "careless"], capture_output=True, text=True, cwd=str(CODE_ROOT))
    assert proc.returncode != 0

def _consent_lines(lines: list[dict[str, Any]]) -> list[tuple[int, str | None, str | None]]:
    return [(int(line["t"]), line.get("verdict"), line.get("reason")) for line in lines if line.get("hook") == "consent_request"]


def test_restatement_edit_executes_once_and_the_second_presentation_is_declined(tmp_path: Path) -> None:
    # rev_003 edit e1 changes turn 0 to Tariq Nour, so the revision at t=22 restates the record that executes at t=26;
    # the reader declines the re-confirmation at t=35 and the second submission at t=41 finds no consent.
    cfg = _cfg(tmp_path, seed_ids=["rev_003"], languages=["en"], trace_kinds=["edit"], arms=["B4"], run_id="rev3-e1")
    specs, variants, splits = load_benchmark(cfg)
    items = [it for it in enumerate_items(cfg, specs, variants, splits) if it.variant_id.endswith("rev_003_e1")]
    assert len(items) == 1
    record, lines = run_item(cfg, specs[0], next(v for v in variants if v.variant_id == items[0].variant_id), items[0], "test", {})
    executes = _execute_lines(lines)
    assert [(t, amount) for t, amount, _ in executes] == [(26, 35000), (41, 35000)]
    assert executes[0][2] == "ALLOW" and executes[1][2] == "DENY"
    assert _consent_lines(lines) == [(10, "ALLOW", "confirmed"), (35, "DENY", "already_executed")]
    assert record["rail_executions"] == 1 and record["executions"] == 1
    assert record["intent_violations"] == [] and record["completed"]


def test_additional_payment_of_the_same_record_is_confirmed_and_executed_twice(tmp_path: Path) -> None:
    # ref_004 intends the same record twice (an additional request at t=300); each confirmation is approved and the
    # lost acknowledgement and the retry at t=335 are replayed under the key of the second payment.
    cfg = _cfg(tmp_path, seed_ids=["ref_004"], languages=["en"], trace_kinds=["base"], arms=["B4"], run_id="ref4")
    specs, variants, splits = load_benchmark(cfg)
    items = enumerate_items(cfg, specs, variants, splits)
    assert [it.variant_id for it in items] == ["ref_004.t0.en.base", "ref_004.t1.en.base"]
    key = ("P-230", 90000, "AED")
    for item in items:
        variant = next(v for v in variants if v.variant_id == item.variant_id)
        timeline = intent_timeline_for(specs[0], variant)
        assert timeline.required == {key: 2} and timeline.intended_count(key, 300) == 2
        unbounded = intent_timeline_for(specs[0], variant)
        unbounded.required = None
        assert unbounded.intended_count(key, 299) == 1 and unbounded.intended_count(key, 300) == 2
        record, lines = run_item(cfg, specs[0], variant, item, "test", {})
        assert _consent_lines(lines) == [(10, "ALLOW", "confirmed"), (312, "ALLOW", "confirmed")]
        assert [(t, verdict) for t, _, verdict in _execute_lines(lines)] == [(18, "ALLOW"), (320, "ALLOW"), (320, "ALLOW"), (335, "ALLOW")]
        assert record["rail_executions"] == 2 and record["intent_violations"] == [] and record["completed"]
