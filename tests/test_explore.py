"""Bounded exploration of the consent state machine at depth 6.

The Tasdiq arm must satisfy I1 to I7 on every reachable state under every revision rule;
the non-atomic ablation must be caught by the exploration with a counterexample path.
"""

from __future__ import annotations

import json

import pytest

from tasdiq.verify.explore import EVENTS, EXECUTORS, INVARIANTS, RULES, Event, World, explore, explore_rules

DEPTH = 6


def test_initial_world_enables_only_confirm_and_prepare() -> None:
    world = World()
    try:
        labels = [e.label for e in world.enabled()]
        assert labels == ["confirm", "prepare(A)", "prepare(B)"]
        assert world.check_invariants() == []
    finally:
        world.close()


def test_state_hash_is_deterministic_and_path_independent_for_equal_states() -> None:
    first = World()
    second = World()
    try:
        assert first.state_hash() == second.state_hash()
        first.apply(Event("confirm"))
        second.apply(Event("confirm"))
        assert first.state_hash() == second.state_hash()
        second.apply(Event("prepare", "A"))
        assert first.state_hash() != second.state_hash()
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("rule", RULES)
def test_tasdiq_satisfies_invariants_to_depth_six(rule: str) -> None:
    result = explore(rule, DEPTH, "Tasdiq")
    assert result.ok, [v.to_dict() for v in result.violations]
    assert result.states > 100
    assert result.depth_reached == DEPTH
    assert result.executions_max == 1
    assert set(result.events_taken) == set(EVENTS)
    document = json.loads(json.dumps(result.to_dict()))
    assert document["n_violations"] == 0 and document["invariants"] == list(INVARIANTS)


def test_nonatomic_ablation_is_caught_with_a_counterexample() -> None:
    result = explore("invalidate", DEPTH, "TasdiqNonAtomic")
    assert not result.ok
    invariants = {v.invariant for v in result.violations}
    assert "I6" in invariants, invariants
    double = next(v for v in result.violations if v.invariant == "I6")
    assert double.path[0] == "confirm"
    assert {f"prepare({e})" for e in EXECUTORS} <= set(double.path)
    assert result.executions_max >= 2


def test_lost_acknowledgement_retry_returns_the_same_receipt() -> None:
    world = World("invalidate", "Tasdiq")
    try:
        for event in (Event("confirm"), Event("prepare", "A"), Event("lost_ack_retry", "A")):
            world.apply(event)
        assert len(world.rail.executions) == 1
        assert world.i5_failures == []
        retry = world.decisions[-1]
        assert retry["verdict"] == "ALLOW" and retry["reason"] == "duplicate_replayed"
        assert world.check_invariants() == []
    finally:
        world.close()


def test_explore_rules_runs_every_rule() -> None:
    results = explore_rules(3, RULES, "Tasdiq")
    assert [r.rule for r in results] == list(RULES)
    assert all(r.ok for r in results)


# ----------------------------------------------------------------------------------------
# Liveness after a crash, the I7 exercise and arms without a window
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm_name", ["Tasdiq", "B4", "B5"])
def test_restart_after_crash_is_live_for_the_atomic_arms(arm_name: str) -> None:
    world = World("invalidate", arm_name)
    try:
        for event in (Event("confirm"), Event("prepare", "A"), Event("crash_restart", "A")):
            world.apply(event)
        last = world.decisions[-1]
        assert last["admitted"] is True and last["liveness"] == "ok"
        assert last["verdict"] == "ALLOW" and last["reason"] == "ok"
        assert len(world.rail.executions) == 1
        assert world.liveness_failures == [] and world.check_invariants() == []
    finally:
        world.close()


def test_crash_of_a_denied_attempt_is_not_a_liveness_check() -> None:
    world = World("invalidate", "Tasdiq")
    try:
        for event in (Event("confirm"), Event("prepare", "A"), Event("prepare", "B"), Event("crash_restart", "B")):
            world.apply(event)
        last = world.decisions[-1]
        assert last["admitted"] is False and last["liveness"] is None
        assert last["verdict"] == "DENY"
    finally:
        world.close()


def test_exploration_reports_liveness_beside_the_invariants() -> None:
    result = explore("invalidate", 4, "Tasdiq")
    assert result.ok and result.live
    assert result.liveness_checks > 0 and result.liveness_failures == []
    document = result.to_dict()
    assert document["n_liveness_failures"] == 0 and document["liveness_checks"] == result.liveness_checks
    assert "liveness_failures" in document and "liveness" not in {v["invariant"] for v in document["violations"]}


def test_world_builds_arms_without_a_window() -> None:
    world = World("invalidate", "B5")
    try:
        assert world.arm.name == "B5" and world.arm.config["interval_s"] == 600
        world.apply(Event("confirm"))
        world.apply(Event("prepare", "A"))
        world.apply(Event("commit", "A"))
        assert len(world.rail.executions) == 1 and world.check_invariants() == []
    finally:
        world.close()


def test_i7_is_exercised_on_every_record() -> None:
    world = World("invalidate", "Tasdiq")
    try:
        world.apply(Event("confirm"))
        assert world.check_invariants() == []
        original = world.arm.binding_problem

        def by_display_name(session, record, req):
            if session.directory.render_name(record.payee_id, "en") != "Omar Khalid":
                return "field_mismatch:display_name"
            return original(session, record, req)

        world.arm.binding_problem = by_display_name
        failures = world.check_invariants()
        assert ("I7", f"the binding of {world.current_id} changes with the display name") in failures
        world.arm.binding_problem = lambda session, record, req: None
        failures = world.check_invariants()
        assert ("I7", f"the binding of {world.current_id} admits another payee_id") in failures
    finally:
        world.close()


def test_run_verify_manifest_records_wall_clock_and_peak_rss(tmp_path) -> None:
    from tasdiq.runners.config import make_config
    from tasdiq.verify.explore import run_verify, verify_counts

    cfg = make_config({"results_dir": str(tmp_path / "results"), "explore_depth": 2, "revision_rules": ["invalidate"], "verify_arms": ["Tasdiq"], "windows": [600], "run_id": "v"})
    counts = verify_counts(cfg)
    assert counts["items"] == 1 and counts["explorations"] == [{"arm": "Tasdiq", "rule": "invalidate"}] and counts["depth"] == 2
    run_dir = run_verify(cfg, log=None)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    explore_doc = json.loads((run_dir / "explore.json").read_text(encoding="utf-8"))
    assert manifest["peak_rss_bytes"] > 0 and explore_doc["peak_rss_bytes"] == manifest["peak_rss_bytes"]
    assert manifest["elapsed_seconds"] >= 0.0 and manifest["ended"] >= manifest["started"]
    assert abs((manifest["ended"] - manifest["started"]) - manifest["elapsed_seconds"]) < 1e-6
    assert manifest["n_items"] == 1 and manifest["workers"] == 1
