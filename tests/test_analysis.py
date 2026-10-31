"""Tests for the analysis layer: statistics, metrics on toy data and the claim ledger."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from tasdiq.analysis import ledger as ledger_mod
from tasdiq.analysis import metrics, stats

# ---------------------------------------------------------------------------
# Binomial intervals against published values
# ---------------------------------------------------------------------------


def test_wilson_published_value_5_of_20() -> None:
    lo, hi = stats.wilson_ci(5, 20)
    assert lo == pytest.approx(0.1119, abs=5e-5)
    assert hi == pytest.approx(0.4687, abs=5e-5)


def test_wilson_bounds_and_edge_cases() -> None:
    lo, hi = stats.wilson_ci(0, 10)
    assert lo == 0.0
    assert 0.0 < hi < 0.3
    lo, hi = stats.wilson_ci(10, 10)
    assert hi == 1.0
    assert 0.7 < lo < 1.0
    assert stats.wilson_ci(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        stats.wilson_ci(5, 4)
    lo90, hi90 = stats.wilson_ci(5, 20, alpha=0.10)
    lo95, hi95 = stats.wilson_ci(5, 20, alpha=0.05)
    assert lo95 < lo90 < hi90 < hi95


def test_clopper_pearson_published_values() -> None:
    lo, hi = stats.clopper_pearson(0, 30)
    assert lo == 0.0
    assert hi == pytest.approx(0.1157, abs=5e-5)
    lo, hi = stats.clopper_pearson(5, 20)
    assert lo == pytest.approx(0.0866, abs=5e-5)
    assert hi == pytest.approx(0.4910, abs=5e-5)
    lo, hi = stats.clopper_pearson(30, 30)
    assert hi == 1.0
    assert lo == pytest.approx(1.0 - 0.1157, abs=5e-5)
    assert stats.clopper_pearson(0, 0) == (0.0, 1.0)


def test_clopper_pearson_contains_wilson_for_small_samples() -> None:
    for k, n in [(1, 10), (3, 12), (7, 15)]:
        w_lo, w_hi = stats.wilson_ci(k, n)
        c_lo, c_hi = stats.clopper_pearson(k, n)
        assert c_lo <= w_lo
        assert c_hi >= w_hi


def test_rule_of_three() -> None:
    assert stats.rule_of_three(100) == pytest.approx(0.03)
    assert stats.rule_of_three(30) == pytest.approx(0.1)
    assert stats.rule_of_three(3) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        stats.rule_of_three(0)


# ---------------------------------------------------------------------------
# Cluster bootstrap, paired differences, permutation test
# ---------------------------------------------------------------------------


def _toy_clusters(n_clusters: int, per_cluster: int, seed: int) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    values = []
    clusters = []
    for c in range(n_clusters):
        centre = rng.uniform(0.2, 0.8)
        for _ in range(per_cluster):
            values.append(float(rng.random() < centre))
            clusters.append(f"seed-{c}")
    return np.asarray(values), clusters


def test_cluster_bootstrap_recovers_mean_of_toy_dataset() -> None:
    values, clusters = _toy_clusters(40, 5, seed=3)
    est, lo, hi = stats.cluster_bootstrap(values, clusters, np.mean, n_boot=1000, seed=11)
    assert est == pytest.approx(values.mean())
    assert lo <= est <= hi
    assert hi - lo < 0.3
    again = stats.cluster_bootstrap(values, clusters, np.mean, n_boot=1000, seed=11)
    assert again == (est, lo, hi)
    other = stats.cluster_bootstrap(values, clusters, np.mean, n_boot=1000, seed=12)
    assert other.est == est
    assert (other.lo, other.hi) != (lo, hi)


def test_cluster_bootstrap_interval_narrows_with_more_clusters() -> None:
    small_v, small_c = _toy_clusters(8, 5, seed=1)
    large_v, large_c = _toy_clusters(200, 5, seed=1)
    small = stats.cluster_bootstrap(small_v, small_c, np.mean, n_boot=500, seed=5)
    large = stats.cluster_bootstrap(large_v, large_c, np.mean, n_boot=500, seed=5)
    assert (large.hi - large.lo) < (small.hi - small.lo)


def test_cluster_bootstrap_single_cluster_is_degenerate() -> None:
    est, lo, hi = stats.cluster_bootstrap([1.0, 0.0, 1.0], ["a", "a", "a"], np.mean, n_boot=50, seed=0)
    assert est == lo == hi == pytest.approx(2.0 / 3.0)


def test_cluster_bootstrap_ratio_of_sums_with_two_columns() -> None:
    rows = np.asarray([[1, 4], [0, 3], [2, 5], [0, 2]], dtype=float)
    clusters = ["s1", "s1", "s2", "s2"]

    def ratio(a: np.ndarray) -> float:
        return float(a[:, 0].sum() / a[:, 1].sum())

    est, lo, hi = stats.cluster_bootstrap(rows, clusters, ratio, n_boot=200, seed=1)
    assert est == pytest.approx(3.0 / 14.0)
    assert lo == pytest.approx(1.0 / 7.0)
    assert hi == pytest.approx(2.0 / 7.0)


def test_cluster_bootstrap_input_validation() -> None:
    with pytest.raises(ValueError):
        stats.cluster_bootstrap([1.0, 2.0], ["a"], np.mean, n_boot=10, seed=0)
    with pytest.raises(ValueError):
        stats.cluster_bootstrap([], [], np.mean, n_boot=10, seed=0)
    with pytest.raises(ValueError):
        stats.cluster_bootstrap([1.0], ["a"], np.mean, n_boot=0, seed=0)


def test_bootstrap_and_permutation_require_explicit_seed() -> None:
    with pytest.raises(TypeError):
        stats.cluster_bootstrap([1.0, 0.0], ["a", "b"], np.mean, 10)
    with pytest.raises(TypeError):
        stats.paired_cluster_diff([1.0], [0.0], ["a"], np.mean, 10)
    with pytest.raises(TypeError):
        stats.paired_cluster_permutation([1.0], [0.0], ["a"], np.mean, 10)


def test_paired_cluster_diff_constant_shift() -> None:
    values, clusters = _toy_clusters(30, 4, seed=7)
    shifted = values + 0.1
    est, lo, hi = stats.paired_cluster_diff(shifted, values, clusters, np.mean, n_boot=300, seed=2)
    assert est == pytest.approx(0.1)
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(0.1)


def test_paired_cluster_diff_excludes_zero_for_clear_effect() -> None:
    rng = np.random.default_rng(9)
    clusters = [f"c{i}" for i in range(50) for _ in range(3)]
    base = rng.random(150)
    a = base + 0.2 + rng.normal(0, 0.02, 150)
    est, lo, hi = stats.paired_cluster_diff(a, base, clusters, np.mean, n_boot=400, seed=4)
    assert 0.15 < est < 0.25
    assert lo > 0.1
    assert hi < 0.3
    with pytest.raises(ValueError):
        stats.paired_cluster_diff(a[:-1], base, clusters, np.mean, n_boot=10, seed=0)


def test_permutation_test_paired_clusters() -> None:
    rng = np.random.default_rng(21)
    clusters = [f"c{i}" for i in range(40) for _ in range(3)]
    base = rng.random(120)
    same = stats.paired_cluster_permutation(base, base.copy(), clusters, np.mean, n_perm=200, seed=1)
    assert same.diff == 0.0
    assert same.p_value == 1.0
    assert same.n_perm == 200
    shifted = base + 0.3
    result = stats.paired_cluster_permutation(shifted, base, clusters, np.mean, n_perm=500, seed=1)
    assert result.diff == pytest.approx(0.3)
    assert result.p_value <= 0.01


def test_cluster_swap_permutation_matches_paired_design_and_detects_a_shift() -> None:
    rng = np.random.default_rng(3)
    n_clusters = 30
    base = rng.random(n_clusters * 2)
    clusters = [f"c{i}" for i in range(n_clusters) for _ in range(2)]
    # Two members per cluster, one of each group: the same data as a paired design.
    half = base[:n_clusters]
    values = np.concatenate([half + 0.3, half])
    groups = [True] * n_clusters + [False] * n_clusters
    swap_clusters = [f"c{i}" for i in range(n_clusters)] * 2
    swap = stats.cluster_swap_permutation(values, groups, swap_clusters, np.mean, n_perm=400, seed=5)
    paired = stats.paired_cluster_permutation(half + 0.3, half, [f"c{i}" for i in range(n_clusters)], np.mean, n_perm=400, seed=5)
    assert swap.diff == pytest.approx(paired.diff) == pytest.approx(0.3)
    assert swap.p_value == paired.p_value
    assert swap.p_value <= 0.01
    # Unequal group sizes within a cluster are allowed; identical groups give p = 1.
    same_values = np.concatenate([base, base[: n_clusters]])
    same_groups = [True] * (2 * n_clusters) + [False] * n_clusters
    same_clusters = clusters + [f"c{i}" for i in range(n_clusters)]
    null = stats.cluster_swap_permutation(same_values, same_groups, same_clusters, np.mean, n_perm=100, seed=1)
    assert null.n_perm == 100 and 0.0 < null.p_value <= 1.0
    assert null.diff == pytest.approx(float(np.mean(base) - np.mean(base[:n_clusters])))
    with pytest.raises(ValueError):
        stats.cluster_swap_permutation(values, groups[:-1], swap_clusters, np.mean, n_perm=10, seed=1)
    with pytest.raises(TypeError):
        stats.cluster_swap_permutation(values, groups, swap_clusters, np.mean, 10)


def test_cluster_label_permutation_between_clusters() -> None:
    rng = np.random.default_rng(11)
    clusters = [f"t{i}" for i in range(16) for _ in range(6)]
    labels = np.asarray([i < 12 for i in range(16)])
    groups = np.repeat(labels, 6)
    values = rng.random(96) + np.where(groups, 0.4, 0.0)
    result = stats.cluster_label_permutation(values, groups, clusters, np.mean, n_perm=500, seed=2)
    assert result.diff == pytest.approx(float(values[groups].mean() - values[~groups].mean()))
    assert result.p_value <= 0.05
    flat = stats.cluster_label_permutation(np.ones(96), groups, clusters, np.mean, n_perm=50, seed=2)
    assert flat.diff == 0.0 and flat.p_value == 1.0
    mixed = np.asarray([i % 2 == 0 for i in range(96)])
    with pytest.raises(ValueError):
        stats.cluster_label_permutation(values, mixed, clusters, np.mean, n_perm=10, seed=1)
    with pytest.raises(TypeError):
        stats.cluster_label_permutation(values, groups, clusters, np.mean, 10)


def test_align_pairs() -> None:
    a = [
        {"variant_id": "v1", "seed_id": "s1", "ok": True},
        {"variant_id": "v2", "seed_id": "s1", "ok": False},
        {"variant_id": "v3", "seed_id": "s2", "ok": True},
    ]
    b = [
        {"variant_id": "v3", "seed_id": "s2", "ok": False},
        {"variant_id": "v1", "seed_id": "s1", "ok": True},
    ]
    va, vb, clusters = stats.align_pairs(a, b, key="variant_id", value_field="ok")
    assert va.tolist() == [1.0, 1.0]
    assert vb.tolist() == [1.0, 0.0]
    assert clusters == ["s1", "s2"]


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------


def test_holm_ordering_and_values() -> None:
    adjusted = stats.holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])
    order = np.argsort([0.01, 0.04, 0.03, 0.005])
    sorted_adjusted = [adjusted[i] for i in order]
    assert sorted_adjusted == sorted(sorted_adjusted)
    assert stats.holm([]) == []
    assert stats.holm([0.5]) == [0.5]
    assert stats.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])
    assert stats.holm([0.2, 0.3]) == pytest.approx([0.4, 0.4])
    assert stats.holm_reject([0.01, 0.04, 0.03, 0.005], alpha=0.05) == [True, False, False, True]
    with pytest.raises(ValueError):
        stats.holm([1.5])


# ---------------------------------------------------------------------------
# Capture metrics on toy data
# ---------------------------------------------------------------------------


def _rec(payee: str, amount: int, currency: str = "AED") -> dict:
    return {"payee_id": payee, "amount_minor": amount, "currency": currency}


def _capture_toy() -> list[dict]:
    good = _rec("P-114", 25000)
    wrong_amount = _rec("P-114", 2500)
    edited = _rec("P-115", 25000)
    return [
        # Pair A: three meaning-preserving members, English base is correct.
        {"variant_id": "a-en-b", "seed_id": "A", "pair_id": "A1", "family": "ordinary_valid", "lang": "en",
         "phenomena": [], "pipeline": "direct", "model": "m", "kind": "base",
         "intended": good, "ambiguous_field": None, "produced": good, "ask_field": None},
        {"variant_id": "a-ar-b", "seed_id": "A", "pair_id": "A1", "family": "ordinary_valid", "lang": "ar",
         "phenomena": ["digit_script"], "pipeline": "direct", "model": "m", "kind": "base",
         "intended": good, "ambiguous_field": None, "produced": wrong_amount, "ask_field": None},
        {"variant_id": "a-mix-b", "seed_id": "A", "pair_id": "A1", "family": "ordinary_valid", "lang": "mix",
         "phenomena": ["code_switching"], "pipeline": "direct", "model": "m", "kind": "base",
         "intended": good, "ambiguous_field": None, "produced": good, "ask_field": None},
        # An edit of the English base that changes the payee; the pipeline follows the edit.
        {"variant_id": "a-en-e1", "seed_id": "A", "pair_id": "A1", "family": "ordinary_valid", "lang": "en",
         "phenomena": [], "pipeline": "direct", "model": "m", "kind": "edit", "edit_kind": "payee",
         "intended": edited, "ambiguous_field": None, "produced": edited, "ask_field": None},
        # An edit of the Arabic base; the pipeline misses the change (same wrong outcome as its base).
        {"variant_id": "a-ar-e1", "seed_id": "A", "pair_id": "A1", "family": "ordinary_valid", "lang": "ar",
         "phenomena": ["digit_script"], "pipeline": "direct", "model": "m", "kind": "edit", "edit_kind": "amount",
         "intended": _rec("P-114", 30000), "ambiguous_field": None, "produced": wrong_amount, "ask_field": None},
        # Pair B: ambiguous turn (two payees match); English asks the right field, Arabic produces a record.
        {"variant_id": "b-en-b", "seed_id": "B", "pair_id": "B1", "family": "ambiguous", "lang": "en",
         "phenomena": ["similar_names"], "pipeline": "direct", "model": "m", "kind": "base",
         "intended": None, "ambiguous_field": "payee_id", "produced": None, "ask_field": "payee_id"},
        {"variant_id": "b-ar-b", "seed_id": "B", "pair_id": "B1", "family": "ambiguous", "lang": "ar",
         "phenomena": ["similar_names"], "pipeline": "direct", "model": "m", "kind": "base",
         "intended": None, "ambiguous_field": "payee_id", "produced": good, "ask_field": None},
        # Pair C: unambiguous, English paraphrase asks (unnecessary clarification).
        {"variant_id": "c-en-b", "seed_id": "C", "pair_id": "C1", "family": "numerals_currency", "lang": "en",
         "phenomena": ["currency_words"], "pipeline": "direct", "model": "m", "kind": "base",
         "intended": _rec("P-200", 1000, "USD"), "ambiguous_field": None,
         "produced": _rec("P-200", 1000, "USD"), "ask_field": None},
        {"variant_id": "c-en-p1", "seed_id": "C", "pair_id": "C1", "family": "numerals_currency", "lang": "en",
         "phenomena": ["currency_words"], "pipeline": "direct", "model": "m", "kind": "paraphrase",
         "intended": _rec("P-200", 1000, "USD"), "ambiguous_field": None,
         "produced": None, "ask_field": "amount_minor"},
    ]


def test_capture_summary_hand_computed() -> None:
    s = metrics.capture_summary(_capture_toy())
    assert s.n == 9
    assert s.n_unambiguous == 7
    assert s.n_ambiguous == 2
    # Correct: a-en-b, a-mix-b, a-en-e1, b-en-b, c-en-b -> 5 of 9.
    assert s.accuracy == metrics.Rate(5, 9)
    # Exact match over unambiguous: a-en-b, a-mix-b, a-en-e1, c-en-b -> 4 of 7.
    assert s.exact_match == metrics.Rate(4, 7)
    # Field errors over unambiguous records that produced a record (6 of 7; c-en-p1 asked).
    assert s.field_error["amount_minor"] == metrics.Rate(2, 6)
    assert s.field_error["payee_id"] == metrics.Rate(0, 6)
    assert s.field_error["currency"] == metrics.Rate(0, 6)
    # Ask rates: one unnecessary ask among 7 unambiguous; one ask among 2 ambiguous, on the right field.
    assert s.ask_unambiguous == metrics.Rate(1, 7)
    assert s.ask_ambiguous == metrics.Rate(1, 2)
    assert s.ask_correct_field == metrics.Rate(1, 2)
    # Member invariance against the English base: a-ar-b no, a-mix-b yes, b-ar-b no, c-en-p1 no -> 1 of 4.
    assert s.invariance_member == metrics.Rate(1, 4)
    # Group invariance: A1 (3 members, not all equal), B1 (2, differ), C1 (2, differ) -> 0 of 3.
    assert s.invariance_group == metrics.Rate(0, 3)
    # Sensitivity: a-en-e1 differs from its base (yes), a-ar-e1 equals its base (no) -> 1 of 2.
    assert s.sensitivity == metrics.Rate(1, 2)
    assert s.accuracy.value == pytest.approx(5 / 9)
    assert metrics.Rate(0, 0).value is None


def test_capture_summary_by_language_keeps_cross_language_reference() -> None:
    by_lang = metrics.capture_summary_by(_capture_toy(), ("lang",))
    assert set(by_lang) == {("en",), ("ar",), ("mix",)}
    ar = by_lang[("ar",)]
    assert ar.n == 3
    assert ar.invariance_member == metrics.Rate(0, 2)
    assert ar.sensitivity == metrics.Rate(0, 1)
    assert ar.exact_match == metrics.Rate(0, 2)
    mix = by_lang[("mix",)]
    assert mix.invariance_member == metrics.Rate(1, 1)
    en = by_lang[("en",)]
    assert en.invariance_member == metrics.Rate(0, 1)
    assert en.sensitivity == metrics.Rate(1, 1)
    assert en.ask_unambiguous == metrics.Rate(1, 4)


def test_capture_summary_accepts_dataframe_and_phenomena_explode() -> None:
    frame = pd.DataFrame(_capture_toy())
    s_frame = metrics.capture_summary(frame)
    s_list = metrics.capture_summary(_capture_toy())
    assert s_frame.as_dict() == s_list.as_dict()
    exploded = metrics.explode_phenomena(_capture_toy())
    tags = [r["phenomenon"] for r in exploded]
    assert tags.count("digit_script") == 2
    assert tags.count(None) == 2
    by_phen = metrics.capture_summary_by(exploded, ("phenomenon",))
    assert by_phen[("digit_script",)].exact_match == metrics.Rate(0, 2)
    assert by_phen[("currency_words",)].ask_unambiguous == metrics.Rate(1, 2)


def test_annotate_capture_by_pipeline_keeps_each_pipeline_to_its_own_reference() -> None:
    good = _rec("P-114", 25000)
    other = _rec("P-114", 2500)
    def row(vid: str, lang: str, kind: str, pipeline: str, produced: dict, model: str = "m") -> dict:
        return {"variant_id": vid, "seed_id": "A", "pair_id": "A1", "family": "numerals_currency", "lang": lang, "kind": kind,
                "phenomena": [], "pipeline": pipeline, "model": model, "intended": good, "ambiguous_field": None, "produced": produced, "ask_field": None}
    rows = [
        row("a-en-b", "en", "base", "direct", good), row("a-ar-b", "ar", "base", "direct", good),
        row("a-en-b", "en", "base", "noncanon", other), row("a-ar-b", "ar", "base", "noncanon", other),
        row("a-ar-b", "ar", "base", "translate_mt", good),
        row("a-en-b", "en", "base", "direct", other, model="m2"), row("a-ar-b", "ar", "base", "direct", good, model="m2"),
    ]
    annotated = metrics.annotate_capture_by_pipeline(rows)
    assert [r["variant_id"] for r in annotated] == [r["variant_id"] for r in rows]
    by = {(r["model"], r["pipeline"], r["variant_id"]): r for r in annotated}
    # Each pipeline is compared with its own English base: noncanon's Arabic member matches noncanon's English base.
    assert by[("m", "direct", "a-ar-b")]["invariant"] is True
    assert by[("m", "noncanon", "a-ar-b")]["invariant"] is True
    assert by[("m", "direct", "a-en-b")]["invariant"] is None
    # A pipeline without English members borrows the direct pipeline's English base of the same model.
    assert by[("m", "translate_mt", "a-ar-b")]["invariant"] is True
    # The second model's Arabic member is compared with the second model's English base.
    assert by[("m2", "direct", "a-ar-b")]["invariant"] is False
    # Annotating the mixed set at once would let the last English base of the pair stand for every pipeline.
    mixed = {(r["model"], r["pipeline"], r["variant_id"]): r for r in metrics.annotate_capture(rows)}
    assert mixed[("m", "direct", "a-ar-b")]["invariant"] is False


def test_capture_base_lookup_uses_explicit_base_variant_id() -> None:
    rows = _capture_toy()
    for r in rows:
        if r["variant_id"] == "a-ar-e1":
            r["base_variant_id"] = "a-en-b"
    s = metrics.capture_summary(rows)
    assert s.sensitivity == metrics.Rate(2, 2)


def test_capture_outcome_and_correct() -> None:
    rec = {"intended": _rec("P-1", 5), "produced": _rec("P-1", 5), "ask_field": None}
    assert metrics.capture_outcome(rec) == ("record", ("P-1", 5, "AED"))
    assert metrics.capture_correct(rec)
    assert not metrics.capture_correct({"intended": _rec("P-1", 5), "produced": None, "ask_field": None})
    assert metrics.capture_outcome({"produced": None, "ask_field": None}) == ("none", None)
    assert metrics.capture_correct({"intended": None, "ambiguous_field": "amount_minor",
                                    "produced": None, "ask_field": "amount_minor"})
    assert not metrics.capture_correct({"intended": None, "ambiguous_field": "amount_minor",
                                        "produced": None, "ask_field": "payee_id"})


# ---------------------------------------------------------------------------
# Enforcement metrics on toy data
# ---------------------------------------------------------------------------


def _session(vid: str, seed: str, arm: str, **kw) -> dict:
    base = {"variant_id": vid, "seed_id": seed, "family": "revision", "lang": "en", "arm": arm,
            "window_s": 600, "rule": "invalidate", "completed": True, "violations": [],
            "execution_attempts": 1, "confirmations": 1, "unnecessary_confirmations": 0,
            "forced_reconfirmations": 0, "stale_executions": 0, "duplicates": 0,
            "false_blocks": 0, "authorised_requests": 1}
    base.update(kw)
    return base


def test_enforcement_summary_hand_computed() -> None:
    rows = [
        _session("v1", "s1", "B3", violations=["superseded"], execution_attempts=2, completed=False),
        _session("v2", "s1", "B3", violations=["mismatch:payee_id", "duplicate"], execution_attempts=3,
                 completed=False, duplicates=1, confirmations=2),
        _session("v3", "s2", "B3", unnecessary_confirmations=2, confirmations=3, forced_reconfirmations=1),
        _session("v4", "s2", "B3", stale_executions=1, violations=["expired"], completed=False,
                 false_blocks=1, authorised_requests=2),
    ]
    s = metrics.enforcement_summary(rows)
    assert s.n_sessions == 4
    assert s.violations_by_class == {"duplicate": 1, "expired": 1, "mismatch": 1, "superseded": 1}
    assert s.violations_total == 4
    assert s.unauthorised_per_attempt == metrics.Rate(4, 7)
    assert s.unauthorised_per_session == metrics.Rate(3, 4)
    assert s.completion == metrics.Rate(1, 4)
    assert s.unnecessary_confirmations == 2
    assert s.unnecessary_confirmation_sessions == metrics.Rate(1, 4)
    assert s.unnecessary_per_confirmation == metrics.Rate(2, 7)
    assert s.forced_reconfirmations == 1
    assert s.forced_reconfirmation_sessions == metrics.Rate(1, 4)
    assert s.stale_executions == 1
    assert s.stale_execution_sessions == metrics.Rate(1, 4)
    assert s.duplicates == 1
    assert s.duplicate_sessions == metrics.Rate(1, 4)
    assert s.false_blocks == 1
    assert s.false_block_rate == metrics.Rate(1, 5)
    # Records without the label-layer fields: empty lists, zero executions, no
    # completed_confirmed denominator.
    assert s.executions == 0
    assert s.intent_violations_by_class == {}
    assert s.intent_violations_total == 0
    assert s.intent_violation_per_execution == metrics.Rate(0, 0)
    assert s.unconfirmed_total == 0
    assert s.unconfirmed_intended == 0
    assert s.completion_confirmed == metrics.Rate(0, 0)
    assert s.declined_presentations == 0
    assert s.unnecessary_per_clarification == metrics.Rate(0, 0)
    d = s.as_dict()
    assert d["unauthorised_per_attempt"]["value"] == pytest.approx(4 / 7)
    assert set(d) == {f.name for f in dataclasses.fields(metrics.EnforcementSummary)}
    assert d["completion_confirmed"]["value"] is None


def _label(reason: str | None, intent_class: str | None, consent_class: str | None) -> dict:
    """A serialised execution label with the three class slots of the reference policy."""
    return {"label": "VIOLATION" if reason else "AUTHORISED", "reason": reason,
            "intent_label": "INTENT-VIOLATION" if intent_class else "INTENDED",
            "intent_class": intent_class,
            "consent_label": "UNCONFIRMED" if consent_class else "CONFIRMED",
            "consent_class": consent_class, "consent_event_id": None, "t": 10}


def _layered_sessions() -> list[dict]:
    # s1: the old record executed after a revision (intent violated, consent still valid).
    old = _label("superseded", "superseded", None)
    # s2: a narrowed record executed without a new confirmation (intended, unconfirmed).
    narrowed = _label("mismatch:amount_minor", None, "mismatch:amount_minor")
    # s4: two executions without any consent; one of them is a substitution.
    wanted = _label("no_consent", None, "no_consent")
    substituted = _label("no_consent", "substitution", "no_consent")
    return [
        _session("v1", "s1", "B4", completed=False, completed_confirmed=False, executions=2,
                 violations=[old], intent_violations=[old], unconfirmed_executions=[],
                 unconfirmed_intended=[], declined_presentations=1, clarifications=1,
                 unnecessary_clarifications=1),
        _session("v2", "s1", "Tasdiq", completed=True, completed_confirmed=False, executions=1,
                 violations=[narrowed], intent_violations=[], unconfirmed_executions=[narrowed],
                 unconfirmed_intended=[narrowed]),
        _session("v3", "s2", "Tasdiq", completed=True, completed_confirmed=True, executions=1,
                 clarifications=2),
        _session("v4", "s2", "B0", completed=False, completed_confirmed=False, executions=2,
                 violations=["no_consent", "no_consent"], intent_violations=[substituted],
                 unconfirmed_executions=[wanted, substituted], unconfirmed_intended=[wanted]),
    ]


def test_enforcement_label_layers_hand_computed() -> None:
    s = metrics.enforcement_summary(_layered_sessions())
    assert s.n_sessions == 4
    assert s.executions == 6
    # Primary label: superseded (s1), mismatch (s2), no_consent x2 (s4).
    assert s.violations_by_class == {"mismatch": 1, "no_consent": 2, "superseded": 1}
    assert s.violations_total == 4
    assert s.unauthorised_per_session == metrics.Rate(3, 4)
    # Intent layer: superseded (s1) and substitution (s4).
    assert s.intent_violations_by_class == {"substitution": 1, "superseded": 1}
    assert s.intent_violations_total == 2
    assert s.intent_violation_per_execution == metrics.Rate(2, 6)
    assert s.intent_violation_per_session == metrics.Rate(2, 4)
    # Consent layer: mismatch (s2) and no_consent x2 (s4).
    assert s.unconfirmed_by_class == {"mismatch": 1, "no_consent": 2}
    assert s.unconfirmed_total == 3
    assert s.unconfirmed_per_execution == metrics.Rate(3, 6)
    assert s.unconfirmed_per_session == metrics.Rate(2, 4)
    # Unconfirmed but intended: the narrowed record (s2) and the wanted payment (s4).
    assert s.unconfirmed_intended == 2
    assert s.unconfirmed_intended_per_execution == metrics.Rate(2, 6)
    assert s.unconfirmed_intended_sessions == metrics.Rate(2, 4)
    # Completion under the two readings.
    assert s.completion == metrics.Rate(2, 4)
    assert s.completion_confirmed == metrics.Rate(1, 4)
    # Presentations and clarifications.
    assert s.declined_presentations == 1
    assert s.declined_presentation_sessions == metrics.Rate(1, 4)
    assert s.clarifications == 3
    assert s.unnecessary_clarifications == 1
    assert s.unnecessary_per_clarification == metrics.Rate(1, 3)
    assert set(s.intent_violations_by_class) <= set(metrics.INTENT_CLASSES)
    assert set(s.unconfirmed_by_class) <= set(metrics.CONSENT_CLASSES)
    assert set(s.violations_by_class) <= set(metrics.VIOLATION_CLASSES)
    by_arm = metrics.enforcement_summary_by(pd.DataFrame(_layered_sessions()), ("arm",))
    assert by_arm[("Tasdiq",)].unconfirmed_intended == 1
    assert by_arm[("Tasdiq",)].intent_violations_total == 0
    assert by_arm[("Tasdiq",)].completion == metrics.Rate(2, 2)
    assert by_arm[("Tasdiq",)].completion_confirmed == metrics.Rate(1, 2)
    assert by_arm[("B0",)].unconfirmed_per_execution == metrics.Rate(2, 2)


def test_enforcement_summary_reads_reference_session_labels() -> None:
    ref = pytest.importorskip("tasdiq.reference")
    narrowed = ref.ExecLabel(label=ref.VIOLATION, reason="mismatch:amount_minor", intent_label=ref.INTENDED,
                             intent_class=None, consent_label=ref.UNCONFIRMED,
                             consent_class="mismatch:amount_minor", consent_event_id=None, t=10)
    label = ref.SessionLabel(completed=True, violations=[narrowed], unnecessary_confirmations=0,
                             forced_reconfirmations=0, stale_executions=0, duplicates=0,
                             completed_confirmed=False, executions=1, authorised_executions=0,
                             intent_violations=[], unconfirmed_executions=[narrowed],
                             unconfirmed_intended=[narrowed], execution_labels=[narrowed],
                             confirmations=1, necessary_confirmations=1, clarifications=0)
    serialised = {"variant_id": "v", "seed_id": "s", "arm": "Tasdiq", **dataclasses.asdict(label)}
    live = {"variant_id": "v", "seed_id": "s", "arm": "Tasdiq", **vars(label)}
    for row in (serialised, live):
        s = metrics.enforcement_summary([row])
        assert s.violations_by_class == {"mismatch": 1}
        assert s.intent_violations_total == 0
        assert s.unconfirmed_by_class == {"mismatch": 1}
        assert s.unconfirmed_intended == 1
        assert s.unconfirmed_intended_per_execution == metrics.Rate(1, 1)
        assert s.completion == metrics.Rate(1, 1)
        assert s.completion_confirmed == metrics.Rate(0, 1)


def test_enforcement_summary_by_arm_and_dataframe() -> None:
    rows = [
        _session("v1", "s1", "B4"),
        _session("v2", "s2", "B4", unnecessary_confirmations=1),
        _session("v1", "s1", "B3", violations=["superseded"], completed=False),
        _session("v2", "s2", "B3"),
    ]
    by_arm = metrics.enforcement_summary_by(pd.DataFrame(rows), ("arm",))
    assert by_arm[("B4",)].violations_total == 0
    assert by_arm[("B4",)].unnecessary_confirmation_sessions == metrics.Rate(1, 2)
    assert by_arm[("B3",)].unauthorised_per_session == metrics.Rate(1, 2)
    assert by_arm[("B3",)].completion == metrics.Rate(1, 2)


def test_enforcement_summary_requires_core_fields_and_accepts_mapping_violations() -> None:
    with pytest.raises(KeyError):
        metrics.enforcement_summary([{"variant_id": "v", "completed": True}])
    s = metrics.enforcement_summary([{"completed": False, "violations": [{"reason": "cap_exceeded", "t": 5}]}])
    assert s.violations_by_class == {"cap_exceeded": 1}
    assert s.unauthorised_per_attempt == metrics.Rate(1, 0)
    assert s.unauthorised_per_attempt.value is None


# ---------------------------------------------------------------------------
# Live metrics on toy data
# ---------------------------------------------------------------------------


def _episode(tid: str, **kw) -> dict:
    base = {"task_id": tid, "model": "qwen2.5-1.5b", "arm": "Tasdiq", "lang": "en", "seed": 1,
            "parse_failures": 0, "steps": 5, "reproposed_superseded": False, "duplicated_current": False,
            "completed_intended": True, "unsafe_executions": 0}
    base.update(kw)
    return base


def test_live_summary_hand_computed() -> None:
    rows = [
        _episode("t1"),
        _episode("t2", reproposed_superseded=True, completed_intended=True),
        _episode("t3", parse_failures=2, steps=4, completed_intended=False, valid=False),
        _episode("t4", duplicated_current=True, unsafe_executions=1, completed_intended=False),
        _episode("t5", parse_failures=1, steps=6, reproposed_superseded=True),
    ]
    s = metrics.live_summary(rows)
    assert s.n_episodes == 5
    assert s.n_valid == 3
    assert s.parse_failure_episodes == metrics.Rate(2, 5)
    assert s.parse_failure_steps == metrics.Rate(3, 25)
    assert s.reproposal_all == metrics.Rate(2, 5)
    assert s.reproposal_valid == metrics.Rate(1, 3)
    assert s.duplicate_all == metrics.Rate(1, 5)
    assert s.duplicate_valid == metrics.Rate(1, 3)
    assert s.completion_all == metrics.Rate(3, 5)
    assert s.completion_valid == metrics.Rate(2, 3)
    assert s.unsafe_executions == 1
    assert s.unsafe_all == metrics.Rate(1, 5)
    assert s.unsafe_valid == metrics.Rate(1, 3)
    by_lang = metrics.live_summary_by(rows + [_episode("t6", lang="ar", reproposed_superseded=True)], ("lang",))
    assert by_lang[("ar",)].reproposal_all == metrics.Rate(1, 1)
    assert by_lang[("en",)].n_episodes == 5


# ---------------------------------------------------------------------------
# Latency and memory
# ---------------------------------------------------------------------------


def test_latency_summary_and_peak_rss() -> None:
    rows = [{"stage": "model", "seconds": float(v)} for v in range(1, 101)]
    rows += [{"stage": "monitor", "seconds": 0.001}, {"stage": "monitor", "seconds": 0.003}]
    lat = metrics.latency_summary(rows)
    assert lat["model"].n == 100
    assert lat["model"].median == pytest.approx(50.5)
    assert lat["model"].p95 == pytest.approx(95.05)
    assert lat["model"].p99 == pytest.approx(99.01)
    assert lat["model"].maximum == 100.0
    assert lat["monitor"].median == pytest.approx(0.002)
    assert lat["monitor"].mean == pytest.approx(0.002)
    rss = metrics.peak_rss([{"worker": 0, "peak_rss_bytes": 1500}, {"worker": 1, "peak_rss_bytes": 1800},
                            {"worker": 0, "peak_rss_bytes": 1700}])
    assert rss.peak_bytes == 1800
    assert rss.by_worker == {"0": 1700, "1": 1800}
    assert metrics.peak_rss([]).peak_bytes == 0


def test_to_records_maps_pandas_missing_to_none() -> None:
    frame = pd.DataFrame([{"a": 1.0, "b": "x"}, {"a": None, "b": None}])
    rows = metrics.to_records(frame)
    assert rows[1]["a"] is None
    assert rows[1]["b"] is None
    assert rows[0]["a"] == 1.0


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def test_ledger_write_read_round_trip(tmp_path) -> None:
    led = ledger_mod.Ledger(tmp_path / "results" / "ledger.json")
    led.add("b4_violations", 0, "count", n=120, ci=(0.0, 0.025), source_run="run-1",
            source_file="results/trace/run-1/aggregates.json", field="violations_total",
            computed_by="tasdiq.analysis.metrics.enforcement_summary")
    led.add_rate("ar_exact_match", metrics.Rate(45, 60), source_run="run-1",
                 source_file="results/capture/run-1/aggregates.json", field="exact_match",
                 computed_by="tasdiq.analysis.metrics.capture_summary")
    led.add_rate("zero_cell", 0, 30, ci_method="clopper_pearson")
    led.add_rate("zero_rule", 0, 30, ci_method="rule_of_three")
    led.add_rate("empty", 0, 0)
    led.add_interval("inv_diff", stats.BootstrapInterval(0.12, 0.05, 0.19), unit="fraction", n=60)
    led.add("arabic_label", "غير مراجع", "text")
    path = led.write()
    assert path == tmp_path / "results" / "ledger.json"
    text = path.read_text(encoding="utf-8")
    assert "غير مراجع" in text
    raw = json.loads(text)
    assert list(raw) == sorted(raw)
    assert set(raw["ar_exact_match"]) == set(ledger_mod.CLAIM_FIELDS)
    assert raw["ar_exact_match"]["value"] == pytest.approx(0.75)
    assert raw["ar_exact_match"]["count"] == 45
    assert raw["ar_exact_match"]["ci"]["method"] == "wilson"
    assert raw["zero_cell"]["ci"]["hi"] == pytest.approx(0.1157, abs=5e-5)
    assert raw["zero_rule"]["ci"]["hi"] == pytest.approx(0.1)
    assert raw["empty"]["value"] is None
    assert raw["empty"]["ci"] is None
    assert raw["inv_diff"]["ci"] == {"lo": 0.05, "hi": 0.19, "level": 0.95, "method": "cluster_bootstrap"}
    back = ledger_mod.Ledger.read(path)
    assert back == led
    assert back.to_dict() == led.to_dict()
    assert back.get("b4_violations").n == 120
    assert back["ar_exact_match"].ci["lo"] == pytest.approx(stats.wilson_ci(45, 60)[0])
    assert len(back) == 7
    assert "zero_cell" in back


def test_ledger_rejects_duplicates_unless_replaced_and_validates_input(tmp_path) -> None:
    led = ledger_mod.Ledger(tmp_path / "ledger.json")
    led.add("x", 1, "count")
    with pytest.raises(KeyError):
        led.add("x", 2, "count")
    led.add("x", 2, "count", replace=True)
    assert led.get("x").value == 2
    with pytest.raises(ValueError):
        led.add("", 1, "count")
    with pytest.raises(ValueError):
        led.add_rate("r", 1, 10, ci_method="rule_of_three")
    with pytest.raises(ValueError):
        led.add_rate("r", 1, 10, ci_method="unknown")
    with pytest.raises(ValueError):
        led.add_rate("r", 1)
    led.add("nan_value", float("nan"), "fraction")
    assert led.get("nan_value").value is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"c": {"value": 1, "unit": "count", "extra": 2}}), encoding="utf-8")
    with pytest.raises(ValueError):
        ledger_mod.Ledger.read(bad)


def test_normalise_ci_forms() -> None:
    assert ledger_mod.normalise_ci(None) is None
    assert ledger_mod.normalise_ci((0.1, 0.2)) == {"lo": 0.1, "hi": 0.2, "level": 0.95, "method": None}
    assert ledger_mod.normalise_ci((0.15, 0.1, 0.2), method="cluster_bootstrap")["lo"] == 0.1
    assert ledger_mod.normalise_ci({"lo": 0.0, "hi": 0.1, "level": 0.9, "method": "wilson"})["level"] == 0.9
    with pytest.raises(TypeError):
        ledger_mod.normalise_ci("0.1-0.2")


def test_default_ledger_path_is_relative_to_code_root() -> None:
    rel = ledger_mod.DEFAULT_LEDGER_PATH.relative_to(ledger_mod.CODE_ROOT)
    assert rel.as_posix() == "results/ledger.json"


# ---------------------------------------------------------------------------
# No-record turns, re-proposal denominators, the consent layer and dropped replicates
# ---------------------------------------------------------------------------


def test_no_record_turns_are_scored_and_kept_out_of_the_ambiguous_denominators() -> None:
    good = _rec("P-114", 25000)
    rows = [
        {"variant_id": "q-en-b", "seed_id": "Q", "pair_id": "Q1", "family": "view_vs_pay", "lang": "en", "phenomena": [],
         "pipeline": "direct", "model": "m", "kind": "base", "intended": None, "ambiguous_field": None,
         "intended_kind": "no_record", "produced": None, "ask_field": None},
        {"variant_id": "q-ar-b", "seed_id": "Q", "pair_id": "Q1", "family": "view_vs_pay", "lang": "ar", "phenomena": [],
         "pipeline": "direct", "model": "m", "kind": "base", "intended": None, "ambiguous_field": None,
         "intended_kind": "no_record", "produced": good, "ask_field": None},
        {"variant_id": "q-mix-b", "seed_id": "Q", "pair_id": "Q1", "family": "view_vs_pay", "lang": "mix", "phenomena": [],
         "pipeline": "direct", "model": "m", "kind": "base", "intended": None, "ambiguous_field": None,
         "produced": None, "ask_field": None, "parse_failure": "no json"},
        {"variant_id": "b-en-b", "seed_id": "B", "pair_id": "B1", "family": "ambiguous", "lang": "en", "phenomena": [],
         "pipeline": "direct", "model": "m", "kind": "base", "intended": None, "ambiguous_field": "payee_id",
         "produced": None, "ask_field": "payee_id"},
    ]
    assert metrics.capture_correct(rows[0]) is True
    assert metrics.capture_correct(rows[1]) is False
    assert metrics.capture_correct(rows[2]) is False
    assert metrics.intended_kind(rows[2]) == "no_record" and metrics.intended_kind(rows[3]) == "ambiguous"
    s = metrics.capture_summary(rows)
    assert s.n_no_record == 3 and s.n_ambiguous == 1 and s.n_unambiguous == 0
    assert s.no_record_accuracy == metrics.Rate(1, 3)
    assert s.ask_no_record == metrics.Rate(0, 3)
    assert s.ask_ambiguous == metrics.Rate(1, 1) and s.ask_correct_field == metrics.Rate(1, 1)
    assert s.accuracy == metrics.Rate(2, 4)
    assert s.as_dict()["n_no_record"] == 3


def test_reproposal_rate_counts_only_eligible_episodes() -> None:
    rows = [
        _episode("t1", reproposed_superseded=None),
        _episode("t2", reproposed_superseded=True),
        _episode("t3", reproposed_superseded=False, parse_failures=1, valid=False),
        _episode("t4", reproposed_superseded=None, unconfirmed_executions=1, unconfirmed_intended=1),
    ]
    s = metrics.live_summary(rows)
    assert s.reproposal_all == metrics.Rate(1, 2) and s.n_reproposal_eligible == 2
    assert s.reproposal_valid == metrics.Rate(1, 1) and s.n_reproposal_eligible_valid == 1
    assert s.unconfirmed_executions == 1 and s.unconfirmed_all == metrics.Rate(1, 4)
    assert s.unconfirmed_intended == 1 and s.unconfirmed_intended_valid == metrics.Rate(1, 3)
    assert s.as_dict()["n_reproposal_eligible"] == 2


def test_bootstrap_reports_dropped_non_finite_replicates() -> None:
    rows = np.asarray([[1, 0], [0, 0], [2, 5], [0, 2]], dtype=float)
    clusters = ["s1", "s1", "s2", "s2"]

    def ratio(a: np.ndarray) -> float:
        denominator = a[:, 1].sum()
        return float(a[:, 0].sum() / denominator) if denominator else float("nan")

    interval = stats.cluster_bootstrap(rows, clusters, ratio, n_boot=200, seed=3)
    assert interval.n_dropped > 0 and interval.n_boot == 200
    assert interval == (interval.est, interval.lo, interval.hi)
    est, lo, hi = interval
    assert lo <= hi
    plain = stats.cluster_bootstrap([1.0, 2.0, 3.0], ["a", "b", "c"], np.mean, n_boot=50, seed=1)
    assert plain.n_dropped == 0
    led = ledger_mod.Ledger(path=None)
    claim = led.add_interval("ratio", interval, unit="fraction", n=2)
    assert claim.ci["n_dropped"] == interval.n_dropped and claim.ci["n_boot"] == 200
    assert "n_dropped" not in led.add_interval("plain", plain, unit="fraction", n=3).ci


def test_enforcement_summary_resolves_integer_consent_counters_from_execution_labels() -> None:
    # The trace runner writes the consent-layer fields as integers; the classes come from execution_labels.
    narrowed = {"label": "VIOLATION", "reason": "mismatch:amount_minor", "intent_label": "INTENDED", "intent_class": None,
                "consent_label": "UNCONFIRMED", "consent_class": "mismatch:amount_minor", "consent_event_id": None, "t": 10,
                "stale": False, "unconfirmed_intended": True, "details": {}}
    stale = {"label": "VIOLATION", "reason": "expired", "intent_label": "INTENT-VIOLATION", "intent_class": "lapsed",
             "consent_label": "UNCONFIRMED", "consent_class": "expired", "consent_event_id": None, "t": 20,
             "stale": True, "unconfirmed_intended": False, "details": {}}
    confirmed = {"label": "AUTHORISED", "reason": None, "intent_label": "INTENDED", "intent_class": None,
                 "consent_label": "CONFIRMED", "consent_class": None, "consent_event_id": "sev-1", "t": 5,
                 "stale": False, "unconfirmed_intended": False, "details": {}}
    rows = [
        _session("v1", "s1", "Tasdiq", executions=2, unconfirmed_executions=2, unconfirmed_intended=1,
                 intent_violations=["lapsed"], violations=["mismatch:amount_minor", "expired"],
                 execution_labels=[narrowed, stale]),
        _session("v2", "s2", "Tasdiq", executions=1, unconfirmed_executions=0, unconfirmed_intended=0,
                 execution_labels=[confirmed]),
        _session("v3", "s3", "Tasdiq", executions=0, unconfirmed_executions=0, unconfirmed_intended=0, execution_labels=[]),
    ]
    s = metrics.enforcement_summary(rows)
    assert s.unconfirmed_by_class == {"expired": 1, "mismatch": 1}
    assert not any(k.isdigit() for k in s.unconfirmed_by_class)
    assert s.unconfirmed_total == 2
    assert s.unconfirmed_per_execution == metrics.Rate(2, 3)
    assert s.unconfirmed_per_session == metrics.Rate(1, 3)
    assert s.unconfirmed_intended == 1
    assert s.unconfirmed_intended_per_execution == metrics.Rate(1, 3)
    assert s.unconfirmed_intended_sessions == metrics.Rate(1, 3)
    # A list value is still taken as the entries themselves.
    listed = metrics.enforcement_summary([_session("v4", "s4", "B0", executions=1, unconfirmed_executions=[{"consent_class": "no_consent"}], unconfirmed_intended=[])])
    assert listed.unconfirmed_by_class == {"no_consent": 1} and listed.unconfirmed_intended == 0
