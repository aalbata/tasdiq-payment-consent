"""Tests for the consent store and the idempotency ledger.

Covers every transition of the state machine, the atomicity of try_consume under eight racing threads, and a
hypothesis state machine that checks invariants I1 to I6 of the design draft after every step.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, precondition, rule

from tasdiq.clock import SimClock
from tasdiq.rail import BankRail, FaultPlan, RailError, RailTimeout, Receipt
from tasdiq.records import STATUSES, ConsentRecord, ExecutionRequest, binding_hash, fingerprint
from tasdiq.store import ConsentStore, IdempotencyLedger, InvalidTransition

BASE_FIELDS: dict[str, Any] = dict(
    principal_id="U-1",
    session_id="S-1",
    payee_id="P-114",
    amount_minor=25000,
    currency="AED",
)


def make_draft(store: ConsentStore, **overrides: Any) -> ConsentRecord:
    fields = dict(BASE_FIELDS)
    fields.update(overrides)
    return store.create_draft(fields)


def make_active(store: ConsentStore, window_s: int = 600, **overrides: Any) -> ConsentRecord:
    draft = make_draft(store, **overrides)
    store.present(draft.cr_id)
    return store.confirm(draft.cr_id, "ev-1", "en", window_s)


def make_req(record: ConsentRecord, **overrides: Any) -> ExecutionRequest:
    fields = dict(
        req_id="r-1",
        session_id=record.session_id,
        principal_id=record.principal_id,
        payee_id=record.payee_id,
        amount_minor=record.amount_minor,
        currency=record.currency,
        fee_minor=0,
        consent_ref=record.cr_id,
        idempotency_key="k-1",
        t=0,
    )
    fields.update(overrides)
    return ExecutionRequest(**fields)


def test_create_draft_defaults_and_hash() -> None:
    store = ConsentStore(SimClock(5))
    record = make_draft(store)
    assert record.status == "draft" and record.uses == 0 and record.allowed_uses == 1 and record.revision == 1
    assert record.action_type == "payment" and record.total_max_minor == 25000
    assert record.issued_at is None and record.expires_at is None and record.confirmation_event_id is None
    assert record.cr_id.startswith("cr-") and record.binding_hash == binding_hash(record)
    assert store.get(record.cr_id) == record and store.contains(record.cr_id)
    assert store.history()[0] == {
        "kind": "transition",
        "t": 5,
        "cr_id": record.cr_id,
        "from": None,
        "to": "draft",
        "reason": "create_draft",
        "detail": {"revision": 1},
    }


def test_create_draft_ids_are_unique_and_deterministic() -> None:
    def ids() -> list[str]:
        store = ConsentStore(SimClock(0))
        return [make_draft(store).cr_id for _ in range(5)]

    first = ids()
    assert len(set(first)) == 5 and first == ids()
    store = ConsentStore(SimClock(0))
    given = make_draft(store, cr_id="cr-given")
    assert given.cr_id == "cr-given"
    with pytest.raises(ValueError):
        make_draft(store, cr_id="cr-given")


def test_create_draft_accepts_state_keys_of_a_snapshot() -> None:
    store = ConsentStore(SimClock(0))
    old = make_active(store, total_max_minor=26000)
    fields = old.snapshot()
    del fields["cr_id"]
    fields.update(revision=2, amount_minor=24000, supersedes=old.cr_id)
    new = store.create_draft(fields)
    assert new.status == "draft" and new.uses == 0 and new.revision == 2 and new.supersedes == old.cr_id
    assert new.confirmation_event_id is None and new.surface_hash is None
    assert new.issued_at == old.issued_at and new.expires_at == old.expires_at
    assert new.binding_hash != old.binding_hash


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"amount_minor": 0}, ValueError),
        ({"amount_minor": -1}, ValueError),
        ({"amount_minor": 1.5}, ValueError),
        ({"amount_minor": True}, ValueError),
        ({"amount_minor": "250"}, ValueError),
        ({"currency": ""}, ValueError),
        ({"payee_id": None}, ValueError),
        ({"total_max_minor": 24999}, ValueError),
        ({"total_max_minor": "x"}, TypeError),
        ({"revision": 0}, ValueError),
        ({"allowed_uses": 0}, ValueError),
        ({"action_type": "transfer"}, ValueError),
        ({"issued_at": 10, "expires_at": 10}, ValueError),
        ({"supersedes": 7}, TypeError),
        ({"display_name": "Omar"}, ValueError),
    ],
)
def test_create_draft_rejects_malformed_fields(overrides: dict, error: type) -> None:
    store = ConsentStore(SimClock(0))
    with pytest.raises(error):
        make_draft(store, **overrides)
    assert store.all_records() == []


def test_present_and_confirm_transitions() -> None:
    clock = SimClock(100)
    store = ConsentStore(clock)
    draft = make_draft(store)
    pending = store.present(draft.cr_id)
    assert pending.status == "pending"
    assert store.present(draft.cr_id).status == "pending"
    clock.advance(20)
    active = store.confirm(draft.cr_id, "ev-7", "ar", 600, surface_hash="abc")
    assert active.status == "active" and active.issued_at == 120 and active.expires_at == 720
    assert active.confirmation_event_id == "ev-7" and active.lang_shown == "ar" and active.surface_hash == "abc"
    with pytest.raises(InvalidTransition):
        store.present(draft.cr_id)
    with pytest.raises(InvalidTransition):
        store.confirm(draft.cr_id, "ev-8", "en", 600)
    assert store.active_for("S-1") == [active]
    assert store.active_for("S-2") == []


def test_confirm_from_draft_without_presentation_and_argument_checks() -> None:
    store = ConsentStore(SimClock(0))
    draft = make_draft(store)
    with pytest.raises(ValueError):
        store.confirm(draft.cr_id, "", "en", 600)
    with pytest.raises(ValueError):
        store.confirm(draft.cr_id, "ev-1", "en", 0)
    assert store.confirm(draft.cr_id, "ev-1", "en", 60).status == "active"
    with pytest.raises(KeyError):
        store.get("cr-missing")
    with pytest.raises(KeyError):
        store.present("cr-missing")


def test_get_returns_copies() -> None:
    store = ConsentStore(SimClock(0))
    record = make_active(store)
    copy = store.get(record.cr_id)
    copy.status = "revoked"
    copy.uses = 5
    assert store.get(record.cr_id).status == "active" and store.get(record.cr_id).uses == 0
    record.status = "consumed"
    assert store.get(record.cr_id).status == "active"


def test_revoke_from_each_status() -> None:
    store = ConsentStore(SimClock(0))
    draft = make_draft(store)
    assert store.revoke(draft.cr_id) is False
    assert store.get(draft.cr_id).status == "draft"
    pending = store.present(make_draft(store).cr_id)
    assert store.revoke(pending.cr_id) is True and store.get(pending.cr_id).status == "revoked"
    active = make_active(store)
    assert store.revoke(active.cr_id) is True and store.get(active.cr_id).status == "revoked"
    assert store.revoke(active.cr_id) is False
    assert store.try_consume(active.cr_id, fingerprint(active), 1) == (False, "revoked")
    ignored = [e for e in store.history() if e["kind"] == "ignored"]
    assert [e["reason"] for e in ignored] == ["revoke", "revoke"]


def test_supersede_sets_link_and_blocks_execution() -> None:
    store = ConsentStore(SimClock(0))
    old = make_active(store)
    new = make_draft(store, revision=2, amount_minor=20000)
    assert store.supersede(old.cr_id, new.cr_id) is True
    assert store.get(old.cr_id).status == "superseded"
    assert store.get(new.cr_id).supersedes == old.cr_id
    assert store.supersede(old.cr_id, new.cr_id) is False
    assert store.try_consume(old.cr_id, fingerprint(old), 1) == (False, "superseded")
    with pytest.raises(ValueError):
        store.supersede(old.cr_id, old.cr_id)
    with pytest.raises(KeyError):
        store.supersede(old.cr_id, "cr-missing")
    draft = make_draft(store)
    assert store.supersede(draft.cr_id, new.cr_id) is False
    assert store.get(new.cr_id).supersedes == old.cr_id


def test_sweep_expires_only_active_records_at_or_after_expiry() -> None:
    clock = SimClock(0)
    store = ConsentStore(clock)
    active = make_active(store, window_s=600)
    pending = store.present(make_draft(store).cr_id)
    assert store.sweep(599) == []
    assert store.sweep(600) == [active.cr_id]
    assert store.get(active.cr_id).status == "expired"
    assert store.get(pending.cr_id).status == "pending"
    assert store.sweep(10000) == []
    assert store.try_consume(active.cr_id, fingerprint(active), 10) == (False, "expired")


def test_try_consume_reasons_and_single_use() -> None:
    store = ConsentStore(SimClock(0))
    draft = make_draft(store)
    assert store.try_consume(draft.cr_id, fingerprint(draft), 0) == (False, "no_consent")
    assert store.try_consume("cr-missing", "x", 0) == (False, "no_consent")
    store.present(draft.cr_id)
    assert store.try_consume(draft.cr_id, fingerprint(draft), 0) == (False, "no_consent")
    active = store.confirm(draft.cr_id, "ev-1", "en", 600)
    wrong = fingerprint(make_req(active, amount_minor=25001))
    assert store.try_consume(active.cr_id, wrong, 10) == (False, "field_mismatch:fingerprint")
    assert store.get(active.cr_id).uses == 0
    assert store.try_consume(active.cr_id, fingerprint(active), 600) == (False, "expired")
    assert store.get(active.cr_id).status == "expired"
    again = make_active(store, window_s=600)
    assert store.try_consume(again.cr_id, fingerprint(make_req(again)), 599) == (True, "ok")
    consumed = store.get(again.cr_id)
    assert consumed.status == "consumed" and consumed.uses == 1
    assert store.try_consume(again.cr_id, fingerprint(again), 599) == (False, "consumed")
    assert store.active_for("S-1") == []


def test_try_consume_multi_use_and_release() -> None:
    store = ConsentStore(SimClock(0))
    record = make_active(store, allowed_uses=3)
    fp = fingerprint(record)
    assert store.try_consume(record.cr_id, fp, 1) == (True, "ok")
    assert store.get(record.cr_id).status == "active" and store.get(record.cr_id).uses == 1
    assert store.try_consume(record.cr_id, fp, 2) == (True, "ok")
    assert store.try_consume(record.cr_id, fp, 3) == (True, "ok")
    assert store.get(record.cr_id).status == "consumed" and store.get(record.cr_id).uses == 3
    assert store.try_consume(record.cr_id, fp, 4) == (False, "consumed")
    assert store.release(record.cr_id) is True
    released = store.get(record.cr_id)
    assert released.status == "active" and released.uses == 2
    assert store.try_consume(record.cr_id, fp, 5) == (True, "ok")
    assert store.get(record.cr_id).status == "consumed"
    assert store.release(record.cr_id) and store.release(record.cr_id) and store.release(record.cr_id)
    assert store.get(record.cr_id).uses == 0 and store.get(record.cr_id).status == "active"
    assert store.release(record.cr_id) is False
    draft = make_draft(store)
    assert store.release(draft.cr_id) is False
    store.revoke(record.cr_id)
    assert store.release(record.cr_id) is False


def test_history_records_every_transition_with_time_and_reason() -> None:
    clock = SimClock(10)
    store = ConsentStore(clock)
    record = make_active(store, window_s=100)
    store.try_consume(record.cr_id, "bad", 20)
    store.try_consume(record.cr_id, fingerprint(record), 30)
    store.release(record.cr_id)
    clock.set(200)
    store.sweep(200)
    history = store.history()
    assert [(e["kind"], e["from"], e["to"], e["reason"], e["t"]) for e in history] == [
        ("transition", None, "draft", "create_draft", 10),
        ("transition", "draft", "pending", "present", 10),
        ("transition", "pending", "active", "confirm", 10),
        ("denied", "active", None, "field_mismatch:fingerprint", 20),
        ("transition", "active", "consumed", "consume", 30),
        ("transition", "consumed", "active", "release", 10),
        ("transition", "active", "expired", "sweep", 200),
    ]
    assert history[2]["detail"] == {"event_id": "ev-1", "window_s": 100, "lang": "en"}
    history.clear()
    assert len(store.history()) == 7


def test_ledger_new_in_flight_replay_mismatch_and_fail() -> None:
    clock = SimClock(0)
    ledger = IdempotencyLedger(clock)
    assert ledger.begin("k-1", "fp-a") == ("new", None)
    assert ledger.state_of("k-1") == "in_flight" and len(ledger) == 1
    assert ledger.begin("k-1", "fp-a") == ("in_flight", None)
    assert ledger.begin("k-1", "fp-b") == ("mismatch", None)
    receipt = Receipt("rcpt-1", "P-114", 25000, "AED", 0, 0, "k-1")
    ledger.complete("k-1", receipt)
    assert ledger.state_of("k-1") == "complete"
    state, replayed = ledger.begin("k-1", "fp-a")
    assert state == "replay" and replayed is receipt
    assert ledger.begin("k-1", "fp-b") == ("mismatch", None)
    ledger.fail("k-1")
    assert ledger.state_of("k-1") == "complete"
    assert ledger.begin("k-2", "fp-c") == ("new", None)
    ledger.fail("k-2")
    assert ledger.state_of("k-2") is None
    assert ledger.begin("k-2", "fp-d") == ("new", None)
    with pytest.raises(KeyError):
        ledger.complete("k-9", receipt)
    with pytest.raises(ValueError):
        ledger.begin("", "fp")
    with pytest.raises(ValueError):
        IdempotencyLedger(clock, retention_s=0)


def test_ledger_purge_respects_retention() -> None:
    clock = SimClock(0)
    ledger = IdempotencyLedger(clock, retention_s=100)
    ledger.begin("k-1", "fp")
    ledger.complete("k-1", {"receipt_id": "rcpt-1"})
    clock.set(50)
    ledger.begin("k-2", "fp")
    assert ledger.purge(99) == []
    assert ledger.purge(100) == ["k-1"]
    assert ledger.purge(1000) == []
    assert ledger.state_of("k-2") == "in_flight"
    assert ledger.begin("k-1", "fp") == ("new", None)


def test_lost_acknowledgement_retry_with_same_key_causes_no_second_debit() -> None:
    clock = SimClock(0)
    store = ConsentStore(clock)
    ledger = IdempotencyLedger(clock)
    rail = BankRail(clock, faults=FaultPlan({"r-1": "timeout_after_debit"}))
    record = make_active(store)
    first = make_req(record, req_id="r-1")
    assert ledger.begin("k-1", fingerprint(first)) == ("new", None)
    assert store.try_consume(record.cr_id, fingerprint(first), 1) == (True, "ok")
    with pytest.raises(RailTimeout):
        rail.execute(first)
    assert rail.count() == 1
    retry = make_req(record, req_id="r-2", t=2)
    assert ledger.begin("k-1", fingerprint(retry)) == ("in_flight", None)
    assert ledger.begin("k-1", fingerprint(make_req(record, amount_minor=1))) == ("mismatch", None)
    assert store.try_consume(record.cr_id, fingerprint(retry), 2) == (False, "consumed")
    assert rail.count() == 1
    ledger.complete("k-1", rail.executions[0])
    state, receipt = ledger.begin("k-1", fingerprint(retry))
    assert state == "replay" and receipt is rail.executions[0]
    assert rail.count() == 1


@pytest.mark.parametrize("allowed_uses", [1, 3])
def test_try_consume_race_with_eight_threads(allowed_uses: int) -> None:
    for round_idx in range(25):
        store = ConsentStore(SimClock(0))
        record = make_active(store, allowed_uses=allowed_uses)
        fp = fingerprint(record)
        barrier = threading.Barrier(8)
        results: list[tuple[bool, str]] = []

        def worker() -> None:
            barrier.wait()
            results.append(store.try_consume(record.cr_id, fp, 1))

        threads = [threading.Thread(target=worker, name=f"racer-{round_idx}-{i}") for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 8
        successes = sum(1 for ok, _ in results if ok)
        assert successes == allowed_uses
        assert all(reason == "consumed" for ok, reason in results if not ok)
        after = store.get(record.cr_id)
        assert after.uses == allowed_uses and after.status == "consumed"
        consumes = [e for e in store.history() if e["reason"] == "consume"]
        assert len(consumes) == allowed_uses


class StoreMachine(RuleBasedStateMachine):
    """Random sequences of store, ledger and rail operations that must never violate I1 to I6."""

    records = Bundle("records")

    def __init__(self) -> None:
        super().__init__()
        self.clock = SimClock(0)
        self.store = ConsentStore(self.clock)
        self.ledger = IdempotencyLedger(self.clock)
        self.rail = BankRail(self.clock, faults=FaultPlan())
        self.step = 0
        self.execs: list[dict[str, Any]] = []
        self.terminal_step: dict[str, int] = {}
        self.receipt_by_key: dict[str, Receipt] = {}
        self.event_counter = 0
        self.last: tuple[str, ExecutionRequest] | None = None
        self.ledger_states: Counter[str] = Counter()

    def _observe(self) -> None:
        for record in self.store.all_records():
            if record.status in ("revoked", "expired", "superseded") and record.cr_id not in self.terminal_step:
                self.terminal_step[record.cr_id] = self.step

    @rule(
        target=records,
        amount=st.integers(min_value=1, max_value=5000),
        payee=st.sampled_from(["P-1", "P-2"]),
        uses=st.integers(min_value=1, max_value=3),
        cap_extra=st.integers(min_value=0, max_value=100),
        revision=st.integers(min_value=1, max_value=3),
    )
    def draft(self, amount: int, payee: str, uses: int, cap_extra: int, revision: int) -> str:
        self.step += 1
        record = self.store.create_draft(
            {
                "principal_id": "U",
                "session_id": "S",
                "payee_id": payee,
                "amount_minor": amount,
                "currency": "AED",
                "total_max_minor": amount + cap_extra,
                "allowed_uses": uses,
                "revision": revision,
            }
        )
        self._observe()
        return record.cr_id

    @rule(cr=records)
    def present(self, cr: str) -> None:
        self.step += 1
        try:
            self.store.present(cr)
        except InvalidTransition:
            pass
        self._observe()

    @rule(cr=records, window=st.integers(min_value=1, max_value=120))
    def confirm(self, cr: str, window: int) -> None:
        self.step += 1
        self.event_counter += 1
        try:
            self.store.confirm(cr, f"ev-{self.event_counter}", "en", window)
        except InvalidTransition:
            pass
        self._observe()

    @rule(cr=records, window=st.integers(min_value=30, max_value=240))
    def activate(self, cr: str, window: int) -> None:
        self.step += 1
        if self.store.get(cr).status in ("draft", "pending"):
            self.event_counter += 1
            self.store.present(cr)
            active = self.store.confirm(cr, f"ev-{self.event_counter}", "en", window)
            assert active.status == "active" and active.expires_at == self.clock.now() + window
        self._observe()

    @rule(cr=records)
    def revoke(self, cr: str) -> None:
        self.step += 1
        before = self.store.get(cr).status
        result = self.store.revoke(cr)
        assert result == (before in ("active", "pending"))
        self._observe()

    @rule(old=records, new=records)
    def supersede(self, old: str, new: str) -> None:
        self.step += 1
        if old == new:
            return
        before = self.store.get(old).status
        result = self.store.supersede(old, new)
        assert result == (before in ("active", "pending"))
        self._observe()

    @rule(dt=st.integers(min_value=0, max_value=60))
    def tick(self, dt: int) -> None:
        self.step += 1
        self.clock.advance(dt)
        expired = self.store.sweep(self.clock.now())
        for cr_id in expired:
            assert self.store.get(cr_id).status == "expired"
        self._observe()

    @rule(
        cr=records,
        wrong=st.booleans(),
        key=st.one_of(st.sampled_from(["k-a", "k-b", "k-c"]), st.none()),
        fault=st.sampled_from([None, None, "timeout_before_debit", "timeout_after_debit", "error"]),
    )
    def attempt(self, cr: str, wrong: bool, key: str | None, fault: str | None) -> None:
        self.step += 1
        if key is None:
            key = f"k-{self.step}"
        snap = self.store.get(cr)
        req = ExecutionRequest(
            req_id=f"req-{self.step}",
            session_id="S",
            principal_id="U",
            payee_id=snap.payee_id,
            amount_minor=snap.amount_minor + (1 if wrong else 0),
            currency=snap.currency,
            fee_minor=0,
            consent_ref=cr,
            idempotency_key=key,
            t=self.clock.now(),
        )
        self._submit(cr, req, fault)

    @precondition(lambda self: self.last is not None)
    @rule(same_payload=st.booleans(), fault=st.sampled_from([None, "timeout_after_debit"]))
    def retry_last(self, same_payload: bool, fault: str | None) -> None:
        self.step += 1
        cr, previous = self.last
        req = replace(
            previous,
            req_id=f"req-{self.step}",
            amount_minor=previous.amount_minor if same_payload else previous.amount_minor + 1,
            t=self.clock.now(),
        )
        self._submit(cr, req, fault)

    def _submit(self, cr: str, req: ExecutionRequest, fault: str | None) -> None:
        self.last = (cr, req)
        key = req.idempotency_key
        snap = self.store.get(cr)
        t = req.t
        fp = fingerprint(req)
        wrong = fp != fingerprint(snap)
        debits_before = self.rail.count()
        state, receipt = self.ledger.begin(key, fp)
        self.ledger_states[state] += 1
        if state == "replay":
            assert receipt is self.receipt_by_key[key]
            assert self.rail.count() == debits_before
            self._observe()
            return
        if state in ("in_flight", "mismatch"):
            assert self.rail.count() == debits_before
            self._observe()
            return
        assert state == "new"
        ok, reason = self.store.try_consume(cr, fp, t)
        if not ok:
            self.ledger.fail(key)
            if snap.status in ("draft", "pending"):
                assert reason == "no_consent"
            elif snap.status in ("consumed", "revoked", "expired", "superseded"):
                assert reason == snap.status
            elif t >= snap.expires_at:
                assert reason == "expired"
            elif wrong:
                assert reason == "field_mismatch:fingerprint"
            else:
                raise AssertionError(f"active record in window with matching fingerprint denied: {reason}")
            assert self.rail.count() == debits_before
            self._observe()
            return
        assert snap.status == "active" and t < snap.expires_at and not wrong and snap.uses < snap.allowed_uses
        if fault is not None:
            self.rail.faults.add(req.req_id, fault)
        try:
            receipt = self.rail.execute(req)
        except RailTimeout:
            if fault == "timeout_after_debit":
                self._record_exec(snap, req, self.rail.executions[-1], key)
            else:
                self.store.release(cr)
                self.ledger.fail(key)
            self._observe()
            return
        except RailError:
            self.store.release(cr)
            self.ledger.fail(key)
            self._observe()
            return
        self.ledger.complete(key, receipt)
        self.receipt_by_key[key] = receipt
        self._record_exec(snap, req, receipt, key)
        self._observe()

    def _record_exec(self, snap: ConsentRecord, req: ExecutionRequest, receipt: Receipt, key: str) -> None:
        req_binding = binding_hash(
            {
                "principal_id": req.principal_id,
                "action_type": "payment",
                "payee_id": req.payee_id,
                "amount_minor": req.amount_minor,
                "currency": req.currency,
                "total_max_minor": snap.total_max_minor,
                "revision": snap.revision,
            }
        )
        self.execs.append(
            {
                "cr_id": snap.cr_id,
                "step": self.step,
                "t": req.t,
                "status_before": snap.status,
                "expires_before": snap.expires_at,
                "record_binding": snap.binding_hash,
                "request_binding": req_binding,
                "key": key,
                "receipt": receipt,
            }
        )

    @invariant()
    def store_state_is_consistent(self) -> None:
        for record in self.store.all_records():
            assert record.status in STATUSES
            assert 0 <= record.uses <= record.allowed_uses
            if record.status == "consumed":
                assert record.uses == record.allowed_uses
            if record.status == "active":
                assert record.uses < record.allowed_uses
                assert record.issued_at is not None and record.expires_at is not None
                assert record.confirmation_event_id is not None
            if record.status in ("draft", "pending"):
                assert record.uses == 0 and record.confirmation_event_id is None
            assert record.binding_hash == binding_hash(record)

    @invariant()
    def i1_every_execution_had_an_active_record_with_equal_binding(self) -> None:
        for e in self.execs:
            assert e["status_before"] == "active"
            assert e["t"] < e["expires_before"]
            assert e["request_binding"] == e["record_binding"]

    @invariant()
    def i2_i6_executions_per_record_within_allowed_uses(self) -> None:
        counts = Counter(e["cr_id"] for e in self.execs)
        for cr_id, n in counts.items():
            record = self.store.get(cr_id)
            assert n <= record.allowed_uses
            if record.allowed_uses == 1:
                assert n <= 1

    @invariant()
    def i3_i4_no_execution_after_revocation_expiry_or_superseding(self) -> None:
        for e in self.execs:
            terminal = self.terminal_step.get(e["cr_id"])
            if terminal is not None:
                assert e["step"] < terminal

    @invariant()
    def i5_same_key_same_receipt_no_second_debit(self) -> None:
        assert self.rail.count() == len(self.execs)
        keys = [e["key"] for e in self.execs]
        assert len(keys) == len(set(keys))
        for e in self.execs:
            if e["key"] in self.receipt_by_key:
                assert self.receipt_by_key[e["key"]] is e["receipt"]
        receipt_ids = [r.receipt_id for r in self.rail.executions]
        assert len(receipt_ids) == len(set(receipt_ids))


StoreMachine.TestCase.settings = settings(
    max_examples=80,
    stateful_step_count=60,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
TestStoreMachine = StoreMachine.TestCase


# ----------------------------------------------------------------------------------------
# Deferred revocation, release after expiry, supersede side effects, history copies, ledger input
# ----------------------------------------------------------------------------------------


def test_revoke_of_a_consumed_record_is_deferred_until_release() -> None:
    clock = SimClock(0)
    store = ConsentStore(clock)
    record = make_active(store)
    assert store.try_consume(record.cr_id, fingerprint(make_req(record)), 1) == (True, "ok")
    assert store.revoke(record.cr_id) is False
    assert store.revocation_pending(record.cr_id) is True
    assert store.get(record.cr_id).status == "consumed"
    assert store.history()[-1]["kind"] == "deferred"
    assert store.release(record.cr_id) is True
    after = store.get(record.cr_id)
    assert after.status == "revoked" and after.uses == 0
    assert store.history()[-1]["detail"]["deferred_revocation"] is True
    assert store.revocation_pending(record.cr_id) is False
    assert store.try_consume(record.cr_id, fingerprint(make_req(record)), 2) == (False, "revoked")
    assert store.release(record.cr_id) is False


def test_release_after_expiry_moves_the_record_to_expired() -> None:
    clock = SimClock(0)
    store = ConsentStore(clock)
    record = make_active(store, window_s=600)
    assert store.try_consume(record.cr_id, fingerprint(make_req(record)), 599) == (True, "ok")
    clock.set(700)
    assert store.release(record.cr_id) is True
    assert store.get(record.cr_id).status == "expired"
    assert store.active_for("S-1") == []
    assert store.try_consume(record.cr_id, fingerprint(make_req(record)), 700) == (False, "expired")


def test_supersede_of_a_draft_leaves_the_new_record_untouched() -> None:
    store = ConsentStore(SimClock(0))
    a = make_draft(store)
    b = make_draft(store, amount_minor=20000)
    assert store.supersede(a.cr_id, b.cr_id) is False
    assert store.get(b.cr_id).supersedes is None
    store.present(a.cr_id)
    assert store.supersede(a.cr_id, b.cr_id) is True
    assert store.get(b.cr_id).supersedes == a.cr_id


def test_history_entries_do_not_alias_the_store() -> None:
    store = ConsentStore(SimClock(0))
    make_draft(store)
    first = store.history()
    first[0]["detail"]["revision"] = 999
    assert store.history()[0]["detail"]["revision"] == 1


def test_ledger_begin_validates_the_fingerprint_and_records_the_owner() -> None:
    ledger = IdempotencyLedger(SimClock(0))
    with pytest.raises(ValueError):
        ledger.begin("k1", "")
    with pytest.raises(ValueError):
        ledger.begin("k1", None)  # type: ignore[arg-type]
    assert ledger.begin("k1", "fp", owner="R-1") == ("new", None)
    assert ledger.owner_of("k1") == "R-1" and ledger.owner_of("k2") is None
    assert ledger.begin("k1", "fp", owner="R-2") == ("in_flight", None)
    assert ledger.owner_of("k1") == "R-1"
