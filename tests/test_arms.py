"""Tests of the policy arms on canned sessions.

The sessions use the package's clock, consent store, idempotency ledger, records and payee
directory, together with a tiny in-test rail and confirmation surface that follow the
spec signatures (execute(req) -> receipt with executions as ground truth; present(draft,
lang, intended) -> event approving iff the draft equals the intended record at time t).
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass

import pytest

records = pytest.importorskip("tasdiq.records")
store_mod = pytest.importorskip("tasdiq.store")
clock_mod = pytest.importorskip("tasdiq.clock")
payees_mod = pytest.importorskip("tasdiq.payees")
pytest.importorskip("tasdiq.money")

from tasdiq.policy import arms  # noqa: E402
from tasdiq.policy.base import Session, UserEvent  # noqa: E402

PRINCIPAL = "U-1"
SESSION_ID = "S-1"
OMAR = "P-001"
LINA = "P-002"
TELECOM = "P-003"


@dataclass
class Intended:
    payee_id: str
    amount_minor: int
    currency: str = "AED"
    total_max_minor: int | None = None


class CannedIntent:
    """Piecewise-constant intent timeline exposing intended_at(t)."""

    def __init__(self, initial: Intended | None) -> None:
        self.timeline: list[tuple[int, Intended | None]] = [(0, initial)]

    def set(self, t: int, intended: Intended | None) -> None:
        self.timeline.append((t, intended))
        self.timeline.sort(key=lambda x: x[0])

    def intended_at(self, t: int) -> Intended | None:
        current = None
        for at, rec in self.timeline:
            if at <= t:
                current = rec
        return current


@dataclass
class CannedEvent:
    event_id: str
    t: int
    decision: str
    shown_text: str
    lang: str
    reason: str
    simulated: bool = True


class CannedSurface:
    """Simulated user: approves iff the draft equals the intended record at time t."""

    def __init__(self, clock, directory) -> None:
        self.clock = clock
        self.directory = directory
        self.events: list[CannedEvent] = []

    def present(self, draft, lang, intended):
        t = self.clock.now()
        target = intended.intended_at(t) if intended is not None else None
        reason = "no_intent"
        if target is not None:
            reason = ""
            for f in ("payee_id", "amount_minor", "currency"):
                if getattr(draft, f) != getattr(target, f):
                    reason = f
                    break
            if not reason and target.total_max_minor is not None and draft.total_max_minor > target.total_max_minor:
                reason = "total_max_minor"
        decision = "approved" if (target is not None and not reason) else "declined"
        text = records.render_confirmation(draft, self.directory, lang)
        event = CannedEvent(str(uuid.uuid4()), t, decision, text, lang, reason)
        self.events.append(event)
        return event


@dataclass
class CannedReceipt:
    receipt_id: str
    payee_id: str
    amount_minor: int
    currency: str
    fee_minor: int
    t: int
    idempotency_key: str | None


class RailFault(RuntimeError):
    pass


class CannedRail:
    """Rail with a fault plan keyed by req_id; executions is the ground truth of debits."""

    def __init__(self, clock, faults=None) -> None:
        self.clock = clock
        self.faults = dict(faults or {})
        self.executions: list[CannedReceipt] = []
        self._lock = threading.Lock()

    def quote_fee(self, payee, amount_minor, currency) -> int:
        return 0

    def execute(self, req):
        fault = self.faults.get(req.req_id)
        if fault in ("timeout_before_debit", "error"):
            raise RailFault(fault)
        receipt = CannedReceipt(str(uuid.uuid4()), req.payee_id, req.amount_minor, req.currency, req.fee_minor, self.clock.now(), req.idempotency_key)
        with self._lock:
            self.executions.append(receipt)
        if fault == "timeout_after_debit":
            raise RailFault(fault)
        return receipt


def make_directory():
    P = payees_mod.Payee
    return payees_mod.PayeeDirectory(
        [
            P(payee_id=OMAR, name_en="Omar Khalid", name_ar="عمر خالد", aliases=["3omar"], kind="domestic", iban_masked="AE** **** 1234"),
            P(payee_id=LINA, name_en="Lina Haddad", name_ar="لينا حداد", aliases=[], kind="domestic", iban_masked="AE** **** 5678"),
            P(payee_id=TELECOM, name_en="Gulf Telecom", name_ar="اتصالات الخليج", aliases=[], kind="merchant", iban_masked=""),
        ]
    )


def make_session(intended: Intended | None = Intended(OMAR, 25000, "AED", 26000), faults=None, lang="en"):
    clock = clock_mod.SimClock(0)
    directory = make_directory()
    intent = CannedIntent(intended)
    session = Session(
        session_id=SESSION_ID,
        principal_id=PRINCIPAL,
        store=store_mod.ConsentStore(clock),
        ledger=store_mod.IdempotencyLedger(clock),
        rail=CannedRail(clock, faults),
        surface=CannedSurface(clock, directory),
        directory=directory,
        clock=clock,
        lang=lang,
        intent=intent,
        scenario_id="seed-1",
        variant_id="seed-1-en-base",
    )
    return session, intent


def draft(session, payee=OMAR, amount=25000, currency="AED", cap=None, revision=1, supersedes=None, allowed_uses=1):
    return session.store.create_draft(
        {
            "principal_id": PRINCIPAL,
            "session_id": SESSION_ID,
            "action_type": "payment",
            "payee_id": payee,
            "amount_minor": amount,
            "currency": currency,
            "total_max_minor": amount + 1000 if cap is None else cap,
            "revision": revision,
            "allowed_uses": allowed_uses,
            "supersedes": supersedes,
        }
    )


_counter = {"n": 0}


def request(session, payee=OMAR, amount=25000, currency="AED", fee=0, ref=None, key=None, claim=False, req_id=None, t=None):
    _counter["n"] += 1
    return records.ExecutionRequest(
        req_id=req_id or f"r{_counter['n']}",
        session_id=SESSION_ID,
        principal_id=PRINCIPAL,
        payee_id=payee,
        amount_minor=amount,
        currency=currency,
        fee_minor=fee,
        consent_ref=ref,
        idempotency_key=key,
        agent_claims_user_confirmed=claim,
        t=session.clock.now() if t is None else t,
    )


def confirmed(session, arm, **kwargs):
    """Confirm a draft through the arm and return the active consent id."""
    d = draft(session, **kwargs)
    decision = arm.on_consent_request(session, d, session.clock.now())
    assert decision.verdict == "ALLOW", decision
    return decision.consent_id


def at(session, t):
    session.clock.set(t)
    return t


def test_b0_executes_anything():
    session, _ = make_session()
    arm = arms.NoMonitor()
    d1 = arm.on_execute(session, request(session, payee=LINA, amount=999999, currency="USD"), 0)
    d2 = arm.on_execute(session, request(session, payee="P-404", amount=-5, currency="XXX"), 0)
    assert d1.verdict == "ALLOW" and d2.verdict == "ALLOW"
    assert d1.receipt is not None and d1.details["executed"] is True
    assert len(session.rail.executions) == 2
    assert session.log[-1]["arm"] == "B0"


def test_b1_schema_checks():
    session, _ = make_session()
    arm = arms.SchemaOnly()
    assert arm.on_execute(session, request(session, payee="P-404"), 0).reason == "schema_invalid"
    assert arm.on_execute(session, request(session, amount=0), 0).details["field"] == "amount_minor"
    assert arm.on_execute(session, request(session, currency="XXX"), 0).details["field"] == "currency"
    assert len(session.rail.executions) == 0
    assert arm.on_execute(session, request(session, payee=LINA, amount=1), 0).verdict == "ALLOW"
    assert len(session.rail.executions) == 1


def test_b2_blocks_payee_outside_allowlist_and_missing_key():
    session, _ = make_session()
    arm = arms.StaticRestrictions(allowlist=[OMAR], per_tx_cap_minor=100000, daily_cap_minor=150000)
    assert arm.on_execute(session, request(session, payee=LINA, key="k1"), 0).reason == "not_allowlisted"
    missing = arm.on_execute(session, request(session, payee=OMAR, key=None), 0)
    assert missing.verdict == "DENY" and missing.reason == "schema_invalid" and missing.details["field"] == "idempotency_key"
    assert arm.on_execute(session, request(session, payee=OMAR, amount=100001, key="k2"), 0).reason == "over_cap"
    assert len(session.rail.executions) == 0
    assert arm.on_execute(session, request(session, payee=OMAR, amount=100000, key="k3"), 0).verdict == "ALLOW"
    daily = arm.on_execute(session, request(session, payee=OMAR, amount=60000, key="k4"), 10)
    assert daily.reason == "over_cap" and daily.details["cap"] == "daily"
    assert arm.on_execute(session, request(session, payee=OMAR, amount=60000, key="k5"), 90000).verdict == "ALLOW"
    assert len(session.rail.executions) == 2


def test_b3_admits_substitution_inside_window_and_blocks_after():
    session, _ = make_session()
    arm = arms.SessionElevation(window_s=300)
    cr = confirmed(session, arm)
    assert arm.is_elevated(session, 0)
    inside = arm.on_execute(session, request(session, payee=LINA, ref=cr, key="k1"), at(session, 10))
    assert inside.verdict == "ALLOW" and inside.details["elevated"] is True
    assert [r.payee_id for r in session.rail.executions] == [LINA]
    after = arm.on_execute(session, request(session, payee=LINA, ref=cr, key="k2"), at(session, 400))
    assert after.verdict == "DENY" and after.reason == "session_not_elevated"
    assert after.details["presented"] is True
    assert len(session.rail.executions) == 1
    legit = arm.on_execute(session, request(session, payee=OMAR, key="k3"), at(session, 401))
    assert legit.verdict == "ALLOW" and legit.details["presented"] is True
    assert len(session.rail.executions) == 2


def test_b4_blocks_substitution():
    session, _ = make_session()
    arm = arms.TransactionBound()
    cr = confirmed(session, arm)
    d = arm.on_execute(session, request(session, payee=LINA, ref=cr, key="k1"), 5)
    assert d.verdict == "DENY" and d.reason == "field_mismatch:payee_id"
    d = arm.on_execute(session, request(session, amount=25001, ref=cr, key="k2"), 5)
    assert d.reason == "field_mismatch:amount_minor"
    d = arm.on_execute(session, request(session, currency="USD", ref=cr, key="k3"), 5)
    assert d.reason == "field_mismatch:currency"
    assert session.rail.executions == []
    ok = arm.on_execute(session, request(session, ref=cr, key="k4"), 5)
    assert ok.verdict == "ALLOW" and ok.reason == "ok" and ok.receipt["amount_minor"] == 25000
    assert len(session.rail.executions) == 1


def test_b4_blocks_replayed_consent_and_fee_over_cap():
    session, _ = make_session()
    arm = arms.TransactionBound()
    cr = confirmed(session, arm, cap=25500)
    over = arm.on_execute(session, request(session, ref=cr, fee=600, key="k0"), 1)
    assert over.reason == "cap_exceeded"
    assert arm.on_execute(session, request(session, ref=cr, fee=500, key="k1"), 2).verdict == "ALLOW"
    again = arm.on_execute(session, request(session, ref=cr, key="k2"), 3)
    assert again.verdict == "DENY" and again.reason == "consumed"
    assert len(session.rail.executions) == 1


def test_b4_blocks_stale_revision():
    session, intent = make_session()
    arm = arms.TransactionBound()
    cr1 = confirmed(session, arm)
    intent.set(10, Intended(OMAR, 30000, "AED", 31000))
    at(session, 10)
    d2 = draft(session, amount=30000, revision=2, supersedes=cr1)
    second = arm.on_consent_request(session, d2, 10)
    assert second.verdict == "ALLOW" and second.details["presented"] is True
    assert session.store.get(cr1).status == "superseded"
    stale = arm.on_execute(session, request(session, amount=25000, ref=cr1, key="k1"), 11)
    assert stale.verdict == "DENY" and stale.reason == "superseded"
    assert arm.on_execute(session, request(session, amount=25000, ref=second.consent_id, key="k2"), 11).reason == "field_mismatch:amount_minor"
    assert arm.on_execute(session, request(session, amount=30000, ref=second.consent_id, key="k3"), 11).verdict == "ALLOW"
    assert [r.amount_minor for r in session.rail.executions] == [30000]


def test_b4_blocks_expired_consent():
    session, _ = make_session()
    arm = arms.TransactionBound(window_s=600)
    cr = confirmed(session, arm)
    assert arm.on_execute(session, request(session, ref=cr, key="k1"), at(session, 600)).reason == "expired"
    arm.on_tick(600)
    assert session.store.get(cr).status == "expired"
    assert arm.on_execute(session, request(session, ref=cr, key="k2"), 601).reason == "expired"
    assert session.rail.executions == []
    cr2 = confirmed(session, arm)
    assert arm.on_execute(session, request(session, ref=cr2, key="k3"), at(session, 1199)).verdict == "ALLOW"


def test_b4_blocks_in_flight_duplicate_and_payload_mismatch():
    session, _ = make_session()
    arm = arms.TransactionBound()
    cr = confirmed(session, arm)
    req = request(session, ref=cr, key="k1")
    assert session.ledger.begin("k1", records.fingerprint(req)) == ("new", None)
    d = arm.on_execute(session, req, 1)
    assert d.verdict == "DENY" and d.reason == "duplicate_in_flight"
    session.ledger.fail("k1")
    assert arm.on_execute(session, request(session, ref=cr, key="k1"), 2).verdict == "ALLOW"
    mismatch = arm.on_execute(session, request(session, amount=100, ref=cr, key="k1"), 3)
    assert mismatch.verdict == "DENY" and mismatch.reason == "key_payload_mismatch"
    assert len(session.rail.executions) == 1


def test_b4_returns_stored_receipt_for_identical_retry():
    session, _ = make_session()
    arm = arms.TransactionBound()
    cr = confirmed(session, arm)
    first = arm.on_execute(session, request(session, ref=cr, key="k1"), 1)
    retry = arm.on_execute(session, request(session, ref=cr, key="k1"), 2)
    assert first.verdict == "ALLOW" and retry.verdict == "ALLOW"
    assert retry.reason == "duplicate_replayed" and retry.details["executed"] is False
    assert retry.receipt == first.receipt
    assert len(session.rail.executions) == 1


def test_b4_recovers_from_lost_acknowledgement():
    session, _ = make_session(faults={"lost": "timeout_after_debit", "dead": "timeout_before_debit"})
    arm = arms.TransactionBound()
    cr = confirmed(session, arm)
    lost = arm.on_execute(session, request(session, ref=cr, key="k1", req_id="lost"), 1)
    assert lost.verdict == "ALLOW" and lost.reason == "rail_fault" and lost.receipt is None
    assert lost.details["debit_recorded"] is True
    retry = arm.on_execute(session, request(session, ref=cr, key="k1", req_id="retry"), 2)
    assert retry.reason == "duplicate_replayed" and retry.receipt["amount_minor"] == 25000
    assert len(session.rail.executions) == 1
    cr2 = confirmed(session, arm)
    dead = arm.on_execute(session, request(session, ref=cr2, key="k9", req_id="dead"), 3)
    assert dead.reason == "rail_fault" and dead.details["debit_recorded"] is False
    assert session.store.get(cr2).status == "active"
    assert arm.on_execute(session, request(session, ref=cr2, key="k9", req_id="again"), 4).reason == "ok"
    assert len(session.rail.executions) == 2


def test_b4_ignores_cancellation_but_tasdiq_revokes():
    session, _ = make_session()
    arm = arms.TransactionBound()
    cr = confirmed(session, arm)
    ignored = arm.on_user_event(session, UserEvent(kind="cancel", cr_id=cr), 1)
    assert ignored.reason == "unsupported" and session.store.get(cr).status == "active"
    assert arm.on_execute(session, request(session, ref=cr, key="k1"), 2).verdict == "ALLOW"
    session2, _ = make_session()
    tasdiq = arms.Tasdiq(revision_rule="invalidate")
    cr2 = confirmed(session2, tasdiq)
    revoked = tasdiq.on_user_event(session2, UserEvent(kind="cancel", cr_id=cr2), 1)
    assert revoked.verdict == "ALLOW" and revoked.reason == "revoked"
    denied = tasdiq.on_execute(session2, request(session2, ref=cr2, key="k1"), 2)
    assert denied.verdict == "DENY" and denied.reason == "revoked"
    assert session2.rail.executions == []


def test_b4_inherits_confirmation_for_change_outside_invalidation_fields():
    session, _ = make_session()
    arm = arms.TransactionBound()
    cr1 = confirmed(session, arm, cap=26000)
    d2 = draft(session, cap=25500, revision=2, supersedes=cr1)
    inherited = arm.on_consent_request(session, d2, at(session, 100))
    assert inherited.verdict == "ALLOW" and inherited.reason == "inherited" and inherited.details["presented"] is False
    assert len(session.surface.events) == 1
    new = session.store.get(inherited.consent_id)
    assert new.status == "active" and new.expires_at == 600
    assert session.store.get(cr1).status == "superseded"
    assert arm.on_execute(session, request(session, ref=inherited.consent_id, key="k1"), 101).verdict == "ALLOW"


def test_b4surface_blocks_when_session_language_differs():
    session, _ = make_session(lang="en")
    arm = arms.TransactionBoundSurface()
    cr = confirmed(session, arm)
    assert session.store.get(cr).lang_shown == "en"
    session.lang = "ar"
    switched = arm.on_execute(session, request(session, ref=cr, key="k1"), 1)
    assert switched.verdict == "DENY" and switched.reason == "field_mismatch:surface_hash"
    assert switched.details["lang_now"] == "ar" and switched.details["lang_shown"] == "en"
    session.lang = "en"
    assert arm.on_execute(session, request(session, payee=LINA, ref=cr, key="k2"), 2).reason == "field_mismatch:surface_hash"
    same = arm.on_execute(session, request(session, ref=cr, key="k3"), 3)
    assert same.verdict == "ALLOW" and len(session.rail.executions) == 1


def test_b5_enforces_budget_and_revoke_before_prepare():
    session, _ = make_session(faults={"lost": "timeout_after_debit"})
    arm = arms.LedgerCapLease(budget=1, interval_s=600)
    cr = confirmed(session, arm)
    first = arm.on_execute(session, request(session, ref=cr, key="agent-1"), 1)
    assert first.verdict == "ALLOW" and first.receipt["idempotency_key"] == arm.stable_key(cr, 0)
    assert first.receipt["idempotency_key"] == hashlib.sha256(f"{cr}:0:rail".encode("utf-8")).hexdigest()
    exhausted = arm.on_execute(session, request(session, ref=cr, key="agent-2"), 2)
    assert exhausted.verdict == "DENY" and exhausted.reason == "consumed"
    assert len(session.rail.executions) == 1
    cr2 = confirmed(session, arm)
    revoked = arm.on_user_event(session, UserEvent(kind="cancel", cr_id=cr2), 3)
    assert revoked.verdict == "ALLOW" and revoked.details["slots_revoked"] == 1
    denied = arm.on_execute(session, request(session, ref=cr2, key="agent-3"), 4)
    assert denied.verdict == "DENY" and denied.reason == "revoked"
    assert len(session.rail.executions) == 1
    cr3 = confirmed(session, arm)
    lost = arm.on_execute(session, request(session, ref=cr3, key="agent-4", req_id="lost"), 5)
    assert lost.reason == "rail_fault" and lost.details["debit_recorded"] is True
    assert arm.lease_for(cr3).slots[0].state == "Prepared"
    late_cancel = arm.on_user_event(session, UserEvent(kind="cancel", cr_id=cr3), 6)
    assert late_cancel.details["slots_revoked"] == 0
    recovered = arm.on_execute(session, request(session, ref=cr3, key="agent-5", req_id="retry"), 7)
    assert recovered.verdict == "ALLOW" and recovered.reason == "duplicate_replayed"
    assert recovered.receipt["idempotency_key"] == arm.stable_key(cr3, 0)
    assert arm.lease_for(cr3).slots[0].state == "Committed"
    assert len(session.rail.executions) == 2


def test_b5_revision_requires_new_confirmation():
    session, intent = make_session()
    arm = arms.LedgerCapLease()
    cr = confirmed(session, arm)
    intent.set(1, Intended(OMAR, 20000, "AED", 21000))
    changed = arm.on_user_event(session, UserEvent(kind="revise", cr_id=cr, fields={"amount_minor": 20000}), at(session, 1))
    assert changed.verdict == "ALLOW" and changed.details["presented"] is True
    assert session.store.get(cr).status == "superseded"
    assert arm.on_execute(session, request(session, ref=cr, key="k1"), 2).reason == "superseded"
    assert arm.on_execute(session, request(session, amount=20000, ref=changed.consent_id, key="k2"), 2).verdict == "ALLOW"


def test_b6_reuses_approval_without_expiry():
    session, _ = make_session()
    arm = arms.PerCallApproval()
    confirmed(session, arm)
    assert len(session.surface.events) == 1
    later = arm.on_execute(session, request(session, key="k1"), at(session, 10**7))
    assert later.verdict == "ALLOW" and later.details["reused"] is True and later.details["presented"] is False
    twice = arm.on_execute(session, request(session, key="k2"), 10**7 + 1)
    assert twice.verdict == "ALLOW" and len(session.rail.executions) == 2
    assert len(session.surface.events) == 1
    other = arm.on_execute(session, request(session, payee=LINA, key="k3"), 10**7 + 2)
    assert other.verdict == "DENY" and other.details["presented"] is True
    assert arm.on_user_event(session, UserEvent(kind="cancel"), 10**7 + 3).reason == "unsupported"
    assert len(session.rail.executions) == 2


def test_b7_auto_applies_narrowing_and_asks_on_expansion():
    session, intent = make_session()
    arm = arms.ProgentNarrowing(initial_policy={"payees": [OMAR], "cap_minor": 30000, "currency": "AED"})
    assert arm.on_execute(session, request(session, amount=25000), 0).verdict == "ALLOW"
    assert arm.on_execute(session, request(session, payee=LINA), 0).reason == "not_allowlisted"
    narrowed = arm.on_user_event(session, UserEvent(kind="revise", fields={"amount_minor": 20000}), 1)
    assert narrowed.verdict == "ALLOW" and narrowed.reason == "narrowing" and narrowed.details["presented"] is False
    assert arm.current_policy().cap_minor == 20000
    assert arm.on_execute(session, request(session, amount=25000), 2).reason == "over_cap"
    assert len(session.surface.events) == 0
    intent.set(3, Intended(LINA, 20000, "AED", 20000))
    expanded = arm.on_user_event(session, UserEvent(kind="revise", fields={"payee_id": LINA}), at(session, 3))
    assert expanded.verdict == "ALLOW" and expanded.reason == "policy_expansion" and expanded.details["presented"] is True
    assert len(session.surface.events) == 1
    assert LINA in arm.current_policy().payees
    assert arm.on_execute(session, request(session, payee=LINA, amount=20000), 4).verdict == "ALLOW"
    intent.set(5, Intended(LINA, 20000, "AED", 20000))
    refused = arm.on_user_event(session, UserEvent(kind="revise", fields={"amount_minor": 90000}), at(session, 5))
    assert refused.verdict == "DENY" and refused.reason == "policy_expansion"
    assert arm.current_policy().cap_minor == 20000
    cancelled = arm.on_user_event(session, UserEvent(kind="cancel"), 6)
    assert cancelled.reason == "narrowing" and arm.on_execute(session, request(session, payee=LINA, amount=1), 7).reason == "not_allowlisted"


def test_tasdiq_narrowing_activates_lower_amount_without_confirmation():
    session, intent = make_session()
    arm = arms.Tasdiq(revision_rule="narrowing")
    cr = confirmed(session, arm, cap=26000)
    intent.set(1, Intended(OMAR, 20000, "AED", 21000))
    changed = arm.on_user_event(session, UserEvent(kind="revise", cr_id=cr, fields={"amount_minor": 20000}), at(session, 1))
    assert changed.verdict == "ALLOW" and changed.reason == "narrowing" and changed.details["presented"] is False
    assert len(session.surface.events) == 1
    new = session.store.get(changed.consent_id)
    assert new.status == "active" and new.revision == 2 and new.amount_minor == 20000 and new.expires_at == 600
    assert new.confirmation_event_id == session.store.get(cr).confirmation_event_id
    assert session.store.get(cr).status == "superseded"
    old_amount = arm.on_execute(session, request(session, amount=25000, ref=cr, key="k1"), 2)
    assert old_amount.verdict == "DENY" and old_amount.reason == "superseded"
    assert arm.on_execute(session, request(session, amount=20000, ref=cr, key="k2"), 3).verdict == "ALLOW"
    assert [r.amount_minor for r in session.rail.executions] == [20000]
    intent.set(4, Intended(OMAR, 30000, "AED", 31000))
    at(session, 4)
    cr3 = confirmed(session, arm, amount=30000, cap=31000)
    raised = arm.on_user_event(session, UserEvent(kind="revise", cr_id=cr3, fields={"amount_minor": 40000}), at(session, 4))
    assert raised.details["presented"] is True and raised.verdict == "DENY"
    assert session.store.get(cr3).status == "superseded"


def test_tasdiq_invalidate_supersedes_immediately():
    session, intent = make_session()
    arm = arms.Tasdiq(revision_rule="invalidate")
    cr = confirmed(session, arm)
    lower = arm.on_user_event(session, UserEvent(kind="revise", cr_id=cr, fields={"amount_minor": 20000}), 1)
    assert lower.details["presented"] is True and lower.verdict == "DENY"
    assert session.store.get(cr).status == "superseded"
    assert arm.on_execute(session, request(session, ref=cr, key="k1"), 2).reason == "superseded"
    assert arm.on_execute(session, request(session, amount=20000, key="k2"), 2).reason == "no_consent"
    assert session.rail.executions == []
    intent.set(3, Intended(OMAR, 20000, "AED", 21000))
    at(session, 3)
    cr2 = confirmed(session, arm, amount=20000, cap=21000)
    intent.set(4, Intended(OMAR, 18000, "AED", 19000))
    changed = arm.on_user_event(session, UserEvent(kind="revise", cr_id=cr2, fields={"amount_minor": 18000}), at(session, 4))
    assert changed.verdict == "ALLOW" and changed.details["presented"] is True
    assert session.store.get(changed.consent_id).revision == 2
    assert arm.on_execute(session, request(session, amount=18000, ref=changed.consent_id, key="k3"), 5).verdict == "ALLOW"


def test_tasdiq_cap_settle_settles_once_under_the_cap():
    session, _ = make_session()
    arm = arms.Tasdiq(revision_rule="cap_settle")
    cr = confirmed(session, arm, amount=25000, cap=26000)
    same = arm.on_user_event(session, UserEvent(kind="revise", cr_id=cr, fields={"amount_minor": 25000}), 1)
    assert same.verdict == "ALLOW" and same.reason == "cap_settle" and same.details["record_changed"] is False
    assert session.store.get(cr).status == "active"
    within = arm.on_user_event(session, UserEvent(kind="revise", cr_id=cr, fields={"amount_minor": 20000}), 1)
    assert within.verdict == "ALLOW" and within.reason == "cap_settle" and within.details["presented"] is False
    assert within.details["record_changed"] is True and within.details["supersedes"] == cr
    narrowed = session.store.get(within.consent_id)
    old = session.store.get(cr)
    assert old.status == "superseded"
    assert narrowed.status == "active" and narrowed.amount_minor == 20000 and narrowed.total_max_minor == 21000
    assert narrowed.revision == 2 and narrowed.confirmation_event_id == old.confirmation_event_id
    over = arm.on_execute(session, request(session, amount=25000, ref=narrowed.cr_id, key="k0"), 2)
    assert over.reason == "cap_exceeded"
    withdrawn = arm.on_execute(session, request(session, amount=25000, ref=cr, key="k4"), 2)
    assert withdrawn.verdict == "DENY" and withdrawn.reason == "superseded"
    assert arm.on_execute(session, request(session, payee=LINA, amount=20000, ref=narrowed.cr_id, key="k1"), 2).reason == "field_mismatch:payee_id"
    settled = arm.on_execute(session, request(session, amount=15000, ref=cr, key="k2"), 2)
    assert settled.verdict == "ALLOW" and settled.receipt["amount_minor"] == 15000
    again = arm.on_execute(session, request(session, amount=20000, ref=narrowed.cr_id, key="k3"), 3)
    assert again.verdict == "DENY" and again.reason == "consumed"
    assert len(session.rail.executions) == 1
    above = arm.on_user_event(session, UserEvent(kind="revise", cr_id=narrowed.cr_id, fields={"amount_minor": 50000}), 4)
    assert above.verdict == "DENY" and above.reason == "consumed"


def test_trust_agent_executes_on_claim():
    session, _ = make_session()
    arm = arms.TrustAgent()
    claimed = arm.on_execute(session, request(session, payee=LINA, claim=True, key="k1"), 0)
    assert claimed.verdict == "ALLOW" and claimed.reason == "trust_claim"
    honest = arm.on_execute(session, request(session, payee=LINA, claim=False, key="k2"), 0)
    assert honest.verdict == "DENY" and honest.reason == "no_consent"
    assert len(session.rail.executions) == 1


def test_ask_always_asks_even_with_valid_consent():
    session, _ = make_session()
    arm = arms.AskAlways()
    cr = confirmed(session, arm)
    assert len(session.surface.events) == 1
    d = arm.on_execute(session, request(session, ref=cr, key="k1"), 1)
    assert d.verdict == "ALLOW" and d.details["presented"] is True and d.details["fresh_consent_id"] != cr
    assert len(session.surface.events) == 2
    assert session.store.get(cr).status == "active"
    wrong = arm.on_execute(session, request(session, payee=LINA, ref=cr, key="k2"), 2)
    assert wrong.verdict == "DENY" and wrong.reason == "field_mismatch:payee_id"
    assert len(session.rail.executions) == 1


def test_refuse_all_denies():
    session, _ = make_session()
    arm = arms.RefuseAll()
    asked = arm.on_consent_request(session, draft(session), 0)
    assert asked.verdict == "DENY" and asked.reason == "refuse_all" and len(session.surface.events) == 0
    d = arm.on_execute(session, request(session, key="k1"), 0)
    assert d.verdict == "DENY" and d.reason == "refuse_all"
    assert session.rail.executions == []


def test_every_hook_writes_one_log_line_with_consent_snapshot():
    session, _ = make_session()
    arm = arms.Tasdiq(revision_rule="narrowing")
    cr = confirmed(session, arm)
    arm.on_execute(session, request(session, ref=cr, key="k1"), 1)
    arm.on_user_event(session, UserEvent(kind="cancel", cr_id=cr), 2)
    assert [line["hook"] for line in session.log] == ["consent_request", "execute", "user_event"]
    for line in session.log:
        for key in ("t", "arm", "config_hash", "scenario_id", "variant_id", "verdict", "reason", "consent_id", "details"):
            assert key in line
        assert line["arm"] == "Tasdiq" and line["config_hash"] == arm.config_hash
    assert session.log[1]["consent_state"]["status"] == "consumed"
    assert session.log[1]["request"]["idempotency_key"] == "k1"
    assert session.log[2]["event"]["kind"] == "cancel"


ALL_ARMS = ("B0", "B1", "B2", "B3", "B4", "B4Surface", "B5", "B6", "B7", "Tasdiq", "TasdiqNonAtomic", "TrustAgent", "AskAlways", "RefuseAll", "PromptOnly")


def test_registry_builds_every_arm():
    for name in ALL_ARMS:
        arm = arms.make_arm(name)
        assert arm.name == name and isinstance(arm.config, dict)
    for rule in arms.Tasdiq.RULES:
        assert arms.make_arm("Tasdiq", revision_rule=rule).config["revision_rule"] == rule
        assert arms.make_arm("TasdiqNonAtomic", revision_rule=rule).config["revision_rule"] == rule
    with pytest.raises(ValueError):
        arms.Tasdiq(revision_rule="other")
    assert arms.make_arm("PromptOnly").config["prompt_instruction"] is True
    assert arms.Tasdiq.atomic_gate and arms.TransactionBound.atomic_gate and arms.LedgerCapLease.atomic_gate
    assert not arms.TasdiqNonAtomic.atomic_gate and not arms.NoMonitor.atomic_gate


def test_tasdiq_derives_key_from_record_and_slot():
    session, _ = make_session(faults={"lost": "timeout_after_debit", "dead": "timeout_before_debit"})
    arm = arms.Tasdiq(revision_rule="invalidate")
    cr = confirmed(session, arm)
    first = arm.on_execute(session, request(session, ref=cr, key="agent-1"), 1)
    assert first.verdict == "ALLOW" and first.reason == "ok"
    assert first.receipt["idempotency_key"] == arms.Tasdiq.stable_key(cr, 0) == hashlib.sha256(f"{cr}:0:rail".encode("utf-8")).hexdigest()
    assert first.details["slot"] == 0 and first.details["agent_key"] == "agent-1"
    assert session.rail.executions[0].idempotency_key == arms.Tasdiq.stable_key(cr, 0)
    same_payload = arm.on_execute(session, request(session, ref=cr, key="agent-2"), 2)
    assert same_payload.verdict == "ALLOW" and same_payload.reason == "duplicate_replayed"
    assert same_payload.receipt == first.receipt and same_payload.details["executed"] is False
    other_payload = arm.on_execute(session, request(session, amount=100, ref=cr, key="agent-3"), 3)
    assert other_payload.verdict == "DENY" and other_payload.reason == "consumed"
    assert len(session.rail.executions) == 1
    cr2 = confirmed(session, arm)
    lost = arm.on_execute(session, request(session, ref=cr2, key="agent-4", req_id="lost"), 4)
    assert lost.reason == "rail_fault" and lost.details["debit_recorded"] is True and lost.receipt is None
    assert arm.unacknowledged_slots(cr2) == [0]
    recovered = arm.on_execute(session, request(session, ref=cr2, key="agent-5", req_id="retry"), 5)
    assert recovered.verdict == "ALLOW" and recovered.reason == "duplicate_replayed"
    assert recovered.receipt["idempotency_key"] == arms.Tasdiq.stable_key(cr2, 0)
    assert arm.unacknowledged_slots(cr2) == []
    assert len(session.rail.executions) == 2
    cr3 = confirmed(session, arm)
    dead = arm.on_execute(session, request(session, ref=cr3, key="agent-6", req_id="dead"), 6)
    assert dead.reason == "rail_fault" and dead.details["debit_recorded"] is False
    assert session.store.get(cr3).status == "active" and session.store.get(cr3).uses == 0
    again = arm.on_execute(session, request(session, ref=cr3, key="agent-7", req_id="again"), 7)
    assert again.reason == "ok" and again.receipt["idempotency_key"] == arms.Tasdiq.stable_key(cr3, 0)
    assert len(session.rail.executions) == 3


def test_tasdiq_budget_slots_take_distinct_keys():
    session, _ = make_session()
    arm = arms.Tasdiq(revision_rule="invalidate", budget=2)
    cr = confirmed(session, arm, allowed_uses=2)
    first = arm.on_execute(session, request(session, ref=cr, key="a"), 1)
    second = arm.on_execute(session, request(session, ref=cr, key="b"), 2)
    assert first.details["slot"] == 0 and second.details["slot"] == 1
    assert {r.idempotency_key for r in session.rail.executions} == {arms.Tasdiq.stable_key(cr, 0), arms.Tasdiq.stable_key(cr, 1)}
    third = arm.on_execute(session, request(session, ref=cr, key="c"), 3)
    assert third.reason == "duplicate_replayed" and third.details["executed"] is False
    assert len(session.rail.executions) == 2


def _run_pair(arm_cls, rule="invalidate"):
    """Run executors A and B against one consent; B runs inside A's yield point."""
    session, _ = make_session()
    arm = arm_cls(revision_rule=rule)
    cr = confirmed(session, arm)
    req_a = request(session, ref=cr, key="a", req_id="A")
    req_b = request(session, ref=cr, key="b", req_id="B")
    outcomes: dict = {}

    def run_b() -> None:
        outcomes["b"] = arm.on_execute(session, req_b, 1)

    outcomes["a"] = arm.on_execute(session, req_a, 1, yield_point=run_b)
    return session, outcomes


def test_atomic_gate_executes_once_across_the_interleaving():
    session, outcomes = _run_pair(arms.Tasdiq)
    assert outcomes["a"].verdict == "ALLOW" and outcomes["a"].reason == "ok" and outcomes["a"].details["executed"] is True
    assert outcomes["b"].verdict == "DENY" and outcomes["b"].reason == "duplicate_in_flight"
    assert len(session.rail.executions) == 1
    session, outcomes = _run_pair(arms.Tasdiq, rule="cap_settle")
    assert len(session.rail.executions) == 1 and outcomes["b"].verdict == "DENY"


def test_nonatomic_ablation_executes_twice_across_the_interleaving():
    session, outcomes = _run_pair(arms.TasdiqNonAtomic)
    assert outcomes["b"].verdict == "ALLOW" and outcomes["b"].details["consumed_after_execution"] is True
    assert outcomes["a"].verdict == "ALLOW" and outcomes["a"].details["executed"] is True
    assert outcomes["a"].details["consumed_after_execution"] is False and outcomes["a"].details["consume_reason"] == "consumed"
    assert len(session.rail.executions) == 2
    assert session.store.get(outcomes["a"].consent_id).status == "consumed"
    quiet, _ = make_session()
    arm = arms.TasdiqNonAtomic()
    cr = confirmed(quiet, arm)
    assert arm.on_execute(quiet, request(quiet, ref=cr, key="x"), 1).reason == "ok"
    assert arm.on_execute(quiet, request(quiet, ref=cr, key="y"), 2).reason == "duplicate_replayed"
    assert arm.on_execute(quiet, request(quiet, amount=1, ref=cr, key="z"), 3).reason == "consumed"
    assert len(quiet.rail.executions) == 1


def _arm_with_consent(name: str):
    """Build an arm by name on a fresh session with one confirmed consent where the arm has one."""
    kwargs: dict = {}
    if name == "B2":
        kwargs = {"allowlist": [OMAR], "per_tx_cap_minor": 100000}
    if name == "B7":
        kwargs = {"initial_policy": {"payees": [OMAR], "cap_minor": 30000, "currency": "AED"}}
    arm = arms.make_arm(name, **kwargs)
    session, _ = make_session()
    cr = None
    if name not in ("B7", "RefuseAll"):
        cr = confirmed(session, arm)
    return session, arm, cr


@pytest.mark.parametrize("name", ALL_ARMS)
def test_every_arm_calls_yield_point_exactly_once_per_attempt(name):
    session, arm, cr = _arm_with_consent(name)
    calls: list[str] = []
    allowed = arm.on_execute(session, request(session, ref=cr, key="k1", claim=True), 1, yield_point=lambda: calls.append("allow"))
    assert calls == ["allow"], name
    denied = arm.on_execute(session, request(session, payee="P-404", amount=1, currency="XXX", ref=cr, key="k2"), 2, yield_point=lambda: calls.append("deny"))
    assert calls == ["allow", "deny"], name
    if name == "RefuseAll":
        assert allowed.verdict == "DENY" and denied.verdict == "DENY"
    else:
        assert allowed.verdict == "ALLOW", name
    if name not in ("B0", "PromptOnly", "TrustAgent", "B3"):
        assert denied.verdict == "DENY", name
    if name == "B3":
        assert denied.verdict == "ALLOW" and denied.details["elevated"] is True
    plain = arm.on_execute(session, request(session, ref=cr, key="k3"), 3)
    assert plain.verdict in ("ALLOW", "DENY") and len(calls) == 2


def test_yield_point_runs_after_consume_for_b4_and_before_execution_for_b0():
    session, _ = make_session()
    arm = arms.TransactionBound()
    cr = confirmed(session, arm)
    seen: dict = {}

    def observe() -> None:
        seen["status"] = session.store.get(cr).status
        seen["executions"] = len(session.rail.executions)

    arm.on_execute(session, request(session, ref=cr, key="k1"), 1, yield_point=observe)
    assert seen == {"status": "consumed", "executions": 0}
    loose, _ = make_session()
    plain = arms.NoMonitor()
    count: list[int] = []
    plain.on_execute(loose, request(loose, key="k1"), 1, yield_point=lambda: count.append(len(loose.rail.executions)))
    assert count == [0] and len(loose.rail.executions) == 1


def test_b4_concurrent_executors_consume_once():
    session, _ = make_session()
    arm = arms.TransactionBound()
    cr = confirmed(session, arm)
    decisions: list = []
    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        req = request(session, ref=cr, key=f"k{i}")
        barrier.wait()
        decisions.append(arm.on_execute(session, req, 1))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    allowed = [d for d in decisions if d.verdict == "ALLOW"]
    assert len(allowed) == 1 and len(session.rail.executions) == 1
    assert all(d.reason == "consumed" for d in decisions if d.verdict == "DENY")


# ----------------------------------------------------------------------------------------
# Cap increases, crash recovery and deferred revocation
# ----------------------------------------------------------------------------------------


def test_cap_only_increase_returns_to_the_surface_and_is_declined():
    session, _ = make_session(Intended(OMAR, 25000, "AED", 25250))
    arm = arms.TransactionBound()
    cr = confirmed(session, arm, amount=25000, cap=25250)
    raised = draft(session, amount=25000, cap=30000, revision=2, supersedes=cr)
    decision = arm.on_consent_request(session, raised, 1)
    assert decision.details["presented"] is True and decision.verdict == "DENY"
    assert decision.reason in ("cap_exceeded", "field_mismatch:total_max_minor")
    assert session.store.get(raised.cr_id).status == "revoked"
    assert session.store.get(cr).status == "active"
    fee = arm.on_execute(session, request(session, fee=4000, ref=cr, key="k1"), 2)
    assert fee.verdict == "DENY" and fee.reason == "cap_exceeded" and session.rail.executions == []
    lowered = draft(session, amount=25000, cap=25100, revision=2, supersedes=cr)
    inherited = arm.on_consent_request(session, lowered, 3)
    assert inherited.verdict == "ALLOW" and inherited.reason == "inherited" and inherited.details["presented"] is False
    assert session.store.get(cr).status == "superseded"

    session2, _ = make_session(Intended(OMAR, 25000, "AED", 25250))
    arm2 = arms.Tasdiq(revision_rule="invalidate")
    cr2 = confirmed(session2, arm2, amount=25000, cap=25250)
    raised2 = arm2.on_user_event(session2, UserEvent(kind="revise", cr_id=cr2, fields={"total_max_minor": 30000}), 1)
    assert raised2.details["presented"] is True and raised2.verdict == "DENY"
    assert raised2.reason in ("cap_exceeded", "field_mismatch:total_max_minor")
    assert session2.store.get(cr2).status == "superseded"
    assert arm2.on_execute(session2, request(session2, fee=4000, ref=cr2, key="k1"), 2).verdict == "DENY"
    assert session2.rail.executions == []


class _Crash(RuntimeError):
    pass


def _crash():
    raise _Crash("executor died at its yield point")


@pytest.mark.parametrize("make", [lambda: arms.TransactionBound(), lambda: arms.LedgerCapLease(), lambda: arms.Tasdiq(revision_rule="invalidate")])
def test_restart_after_crash_at_the_yield_point_completes_the_admitted_payment(make):
    session, _ = make_session()
    arm = make()
    cr = confirmed(session, arm)
    req = request(session, ref=cr, key="k1", req_id="R-A-1")
    with pytest.raises(_Crash):
        arm.on_execute(session, req, 1, yield_point=_crash)
    assert session.rail.executions == []
    if arm.name == "B5":
        assert [s.state for s in arm.lease_for(cr).slots] == ["Prepared"]
    else:
        assert session.store.get(cr).status == "consumed" and session.store.get(cr).uses == 1
    foreign = arm.on_execute(session, request(session, ref=cr, key="k1", req_id="R-B-1"), 2)
    assert foreign.verdict == "DENY" and foreign.reason == "duplicate_in_flight"
    assert session.rail.executions == []
    restarted = arm.on_execute(session, req, 3)
    assert restarted.verdict == "ALLOW" and restarted.reason == "ok" and restarted.details["executed"] is True
    assert len(session.rail.executions) == 1
    if arm.name == "B5":
        assert [s.state for s in arm.lease_for(cr).slots] == ["Committed"]
    else:
        assert session.store.get(cr).status == "consumed" and session.store.get(cr).uses == 1
    replayed = arm.on_execute(session, req, 4)
    if arm.name == "B5":
        assert replayed.verdict == "DENY" and replayed.reason == "consumed"
    else:
        assert replayed.verdict == "ALLOW" and replayed.reason == "duplicate_replayed"
        assert replayed.receipt["receipt_id"] == session.rail.executions[0].receipt_id
    fresh = arm.on_execute(session, request(session, ref=cr, key="k2", req_id="R-A-2"), 5)
    assert fresh.reason in ("consumed", "duplicate_replayed") and fresh.details["executed"] is False
    assert len(session.rail.executions) == 1


def test_restart_after_crash_replays_a_debit_the_rail_recorded():
    session, _ = make_session()
    arm = arms.Tasdiq(revision_rule="invalidate")
    cr = confirmed(session, arm)
    req = request(session, ref=cr, key="k1", req_id="R-A-1")
    key = arm.stable_key(cr, 0)

    def debit_then_die():
        session.rail.execute(request(session, ref=cr, key=key, req_id="R-A-1"))
        raise _Crash("died after the rail debited")

    with pytest.raises(_Crash):
        arm.on_execute(session, req, 1, yield_point=debit_then_die)
    assert len(session.rail.executions) == 1
    restarted = arm.on_execute(session, req, 2)
    assert restarted.verdict == "ALLOW" and restarted.reason == "duplicate_replayed"
    assert restarted.receipt["receipt_id"] == session.rail.executions[0].receipt_id
    assert len(session.rail.executions) == 1


def test_cancellation_during_an_execution_is_deferred_and_applied_on_release():
    session, _ = make_session(faults={"R-A-1": "timeout_before_debit"})
    arm = arms.Tasdiq(revision_rule="invalidate")
    cr = confirmed(session, arm)
    seen: list = []

    def cancel_while_in_flight():
        seen.append(arm.on_user_event(session, UserEvent(kind="cancel", cr_id=cr), 1))

    decision = arm.on_execute(session, request(session, ref=cr, key="k1", req_id="R-A-1"), 1, yield_point=cancel_while_in_flight)
    assert decision.reason == "rail_fault" and decision.details["executed"] is False
    assert seen[0].verdict == "DENY" and seen[0].reason == "consumed" and seen[0].details["revocation_deferred"] is True
    assert session.store.get(cr).status == "revoked"
    later = arm.on_execute(session, request(session, ref=cr, key="k2", req_id="R-A-2"), 2)
    assert later.verdict == "DENY" and later.reason == "revoked"
    assert session.rail.executions == []
