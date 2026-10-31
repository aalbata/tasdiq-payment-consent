"""Policy arms: the baseline ladder B0 to B7, the Tasdiq monitor and the controls.

Every arm shares the simulator, payee directory, clock, consent store, idempotency ledger
and confirmation surface through the Session, so that the arms differ only in the design
variable each one isolates: no monitor (B0), schema validation (B1), static restrictions
(B2), session-level elevation (B3), transaction-bound consent (B4) and its rendered-text
variant (B4Surface), a capability-lease ledger (B5), per-call approval without expiry
(B6), policy narrowing (B7), the Tasdiq monitor with a consent channel and a revision
rule, its non-atomic ablation TasdiqNonAtomic, and the controls TrustAgent, AskAlways,
RefuseAll and PromptOnly.

Idempotency keys: B4, B4Surface, TrustAgent and AskAlways use the key the agent supplies
with the request (client form), so a replanning agent that mints a fresh key is held only
by the consume-once consent state. B5 and Tasdiq derive the key from the consent record
and the use slot (server-ledger form, slot_key in base.py) and ignore the agent's key
except for logging.

Yield points: every on_execute calls yield_point exactly once when one is given. The
non-atomic arms (B0, B1, B2, B3, B6, B7, TrustAgent on a claim, TasdiqNonAtomic) yield
between their check and the rail call; the atomic arms (B4, B4Surface, B5, Tasdiq,
AskAlways) consume under the store lock first and yield before the rail call.

Recovery after a crash: the ledger entry an atomic arm opens before the rail call names
the request that opened it. When the same request is issued again (an executor that died
at its yield point and was restarted), the arm reconciles the entry against the rail's
receipts: a debit under the key is returned as a replay, and an entry without a debit is
released so that the request executes under the same key without a second consumption.

Decision reasons use the vocabulary of records.py. Three additional reason strings are
used where the vocabulary is silent: "unsupported" for consent-channel events an arm does
not handle, "rail_fault" for an admitted request whose rail call raised (the details say
whether the debit was recorded), and "confirmed", "inherited", "narrowing", "cap_settle",
"policy_expansion" and "revoked" as ALLOW reasons on the consent and channel hooks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

from tasdiq.policy.base import (
    NO_EXPIRY_S,
    TERMINAL_STATUSES,
    Monitor,
    Session,
    YieldOnce,
    YieldPoint,
    event_attr,
    first_mismatch,
    known_currency,
    make_decision,
    receipt_to_dict,
    record_fingerprint,
    slot_key,
    snapshot_event,
    snapshot_record,
    snapshot_request,
)
from tasdiq.records import ConsentRecord, Decision, ExecutionRequest, binding_hash, fingerprint, render_confirmation

DEFAULT_INVALIDATION_FIELDS: tuple[str, ...] = ("payee_id", "amount_minor", "currency")


def _latest(records: Iterable[ConsentRecord]) -> ConsentRecord | None:
    items = list(records)
    if not items:
        return None
    return max(items, key=lambda r: (int(getattr(r, "issued_at", 0) or 0), int(getattr(r, "revision", 1) or 1)))


def _request_matches(record: Any, req: ExecutionRequest) -> bool:
    return (
        getattr(record, "principal_id", None) == req.principal_id
        and getattr(record, "payee_id", None) == req.payee_id
        and getattr(record, "amount_minor", None) == req.amount_minor
        and getattr(record, "currency", None) == req.currency
    )


def _resolve_by_reference_or_match(monitor: Monitor, session: Session, req: ExecutionRequest) -> ConsentRecord | None:
    """Return the record named by consent_ref, else the matching or latest active record."""
    if req.consent_ref:
        return monitor.store_get(session, req.consent_ref)
    try:
        actives = list(session.store.active_for(session.session_id))
    except (KeyError, ValueError):
        actives = []
    exact = [r for r in actives if _request_matches(r, req)]
    if exact:
        return _latest(exact)
    return _latest(actives)


class NoMonitor(Monitor):
    """B0: executes every request; confirmations are shown but never enforced."""

    name = "B0"
    window_s: int = NO_EXPIRY_S

    def on_consent_request(self, session: Session, draft: ConsentRecord, t: int) -> Decision:
        self.register(session)
        record = self.ensure_draft(session, draft)
        decision, confirmed, _ = self.present(session, record, t, self.window_s)
        state = confirmed or self.store_get(session, record.cr_id) or record
        return self.log(session, "consent_request", decision, t, draft=snapshot_record(record), consent_state=snapshot_record(state))

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        once = YieldOnce(yield_point)
        decision = self.run_rail(session, req, reason="ok", consent_id=req.consent_ref, yield_point=once)
        once.ensure()
        state = self.store_get(session, req.consent_ref)
        return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(state))


class PromptOnly(NoMonitor):
    """Live-track control: B0 behind a system prompt that instructs the model to confirm."""

    name = "PromptOnly"

    def __init__(self, **config: Any) -> None:
        config.setdefault("prompt_instruction", True)
        super().__init__(**config)


class SchemaOnly(NoMonitor):
    """B1: executes when the request is well-typed, positive, in a known currency, to a known payee."""

    name = "B1"

    def schema_problem(self, session: Session, req: ExecutionRequest) -> str | None:
        """Return the first field that fails the schema check, or None."""
        if not isinstance(req.principal_id, str) or not req.principal_id:
            return "principal_id"
        if not isinstance(req.payee_id, str) or not req.payee_id:
            return "payee_id"
        if isinstance(req.amount_minor, bool) or not isinstance(req.amount_minor, int) or req.amount_minor <= 0:
            return "amount_minor"
        if not known_currency(req.currency):
            return "currency"
        if isinstance(req.fee_minor, bool) or not isinstance(req.fee_minor, int) or req.fee_minor < 0:
            return "fee_minor"
        try:
            payee = session.directory.by_id(req.payee_id)
        except (KeyError, ValueError):
            payee = None
        if payee is None:
            return "payee_id"
        return None

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        once = YieldOnce(yield_point)
        problem = self.schema_problem(session, req)
        if problem is not None:
            decision = make_decision("DENY", "schema_invalid", consent_id=req.consent_ref, details={"field": problem, "executed": False, "presented": False})
        else:
            decision = self.run_rail(session, req, reason="ok", consent_id=req.consent_ref, yield_point=once)
        once.ensure()
        state = self.store_get(session, req.consent_ref)
        return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(state))


class StaticRestrictions(SchemaOnly):
    """B2: payee allowlist, per-transaction cap, rolling daily cap and idempotency-key presence."""

    name = "B2"

    def __init__(
        self,
        allowlist: Iterable[str] = (),
        per_tx_cap_minor: int | None = None,
        daily_cap_minor: int | None = None,
        require_idempotency_key: bool = True,
        daily_window_s: int = 86400,
    ) -> None:
        self.allowlist: frozenset[str] = frozenset(allowlist)
        self.per_tx_cap_minor = None if per_tx_cap_minor is None else int(per_tx_cap_minor)
        self.daily_cap_minor = None if daily_cap_minor is None else int(daily_cap_minor)
        self.require_idempotency_key = bool(require_idempotency_key)
        self.daily_window_s = int(daily_window_s)
        super().__init__(
            allowlist=sorted(self.allowlist),
            per_tx_cap_minor=self.per_tx_cap_minor,
            daily_cap_minor=self.daily_cap_minor,
            require_idempotency_key=self.require_idempotency_key,
            daily_window_s=self.daily_window_s,
        )
        self._spend: dict[str, list[tuple[int, int]]] = {}

    def spent_minor(self, principal_id: str, t: int) -> int:
        """Return the amount executed for principal_id inside the rolling daily window."""
        with self._lock:
            return sum(a for (ts, a) in self._spend.get(principal_id, []) if ts > t - self.daily_window_s)

    def static_problem(self, req: ExecutionRequest, t: int) -> tuple[str, dict] | None:
        """Return (reason, details) for the first static restriction the request violates."""
        if self.require_idempotency_key and not req.idempotency_key:
            return "schema_invalid", {"field": "idempotency_key"}
        if req.payee_id not in self.allowlist:
            return "not_allowlisted", {"payee_id": req.payee_id}
        if self.per_tx_cap_minor is not None and req.amount_minor > self.per_tx_cap_minor:
            return "over_cap", {"cap": "per_tx", "cap_minor": self.per_tx_cap_minor}
        if self.daily_cap_minor is not None:
            spent = self.spent_minor(req.principal_id, t)
            if spent + req.amount_minor > self.daily_cap_minor:
                return "over_cap", {"cap": "daily", "cap_minor": self.daily_cap_minor, "spent_minor": spent}
        return None

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        once = YieldOnce(yield_point)
        problem = self.schema_problem(session, req)
        if problem is not None:
            decision = make_decision("DENY", "schema_invalid", consent_id=req.consent_ref, details={"field": problem, "executed": False, "presented": False})
        else:
            static = self.static_problem(req, t)
            if static is not None:
                reason, info = static
                decision = make_decision("DENY", reason, consent_id=req.consent_ref, details={**info, "executed": False, "presented": False})
            else:
                decision = self.run_rail(session, req, reason="ok", consent_id=req.consent_ref, yield_point=once)
                if decision.details.get("executed"):
                    with self._lock:
                        self._spend.setdefault(req.principal_id, []).append((t, int(req.amount_minor)))
        once.ensure()
        state = self.store_get(session, req.consent_ref)
        return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(state))


class SessionElevation(Monitor):
    """B3: one confirmation of the effect class elevates the session for window_s seconds.

    While elevated, every payment executes without argument checks; the first payment of a
    non-elevated session is shown on the surface and executes when approved.
    """

    name = "B3"

    def __init__(self, window_s: int = 300) -> None:
        self.window_s = int(window_s)
        super().__init__(window_s=self.window_s)
        self._elevated_until: dict[str, int] = {}

    def elevated_until(self, session: Session) -> int | None:
        with self._lock:
            return self._elevated_until.get(session.session_id)

    def is_elevated(self, session: Session, t: int) -> bool:
        until = self.elevated_until(session)
        return until is not None and t < until

    def _elevate(self, session: Session, t: int) -> int:
        with self._lock:
            self._elevated_until[session.session_id] = t + self.window_s
            return self._elevated_until[session.session_id]

    def on_consent_request(self, session: Session, draft: ConsentRecord, t: int) -> Decision:
        self.register(session)
        record = self.ensure_draft(session, draft)
        decision, confirmed, _ = self.present(session, record, t, self.window_s)
        if confirmed is not None:
            decision.details["elevated_until"] = self._elevate(session, t)
        state = confirmed or self.store_get(session, record.cr_id) or record
        return self.log(session, "consent_request", decision, t, draft=snapshot_record(record), consent_state=snapshot_record(state))

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        once = YieldOnce(yield_point)
        state: Any = None
        if self.is_elevated(session, t):
            decision = self.run_rail(session, req, reason="ok", consent_id=req.consent_ref, details={"elevated": True, "elevated_until": self.elevated_until(session)}, yield_point=once)
        else:
            draft = self.draft_from_request(session, req)
            if draft is None:
                decision = make_decision("DENY", "schema_invalid", consent_id=None, details={"presented": False, "executed": False})
            else:
                shown, confirmed, _ = self.present(session, draft, t, self.window_s)
                if confirmed is None:
                    decision = make_decision("DENY", "session_not_elevated", consent_id=None, details={**shown.details, "executed": False, "surface_decline": shown.reason})
                    state = self.store_get(session, draft.cr_id)
                else:
                    until = self._elevate(session, t)
                    decision = self.run_rail(session, req, reason="ok", consent_id=confirmed.cr_id, details={**shown.details, "elevated": True, "elevated_until": until}, yield_point=once)
                    state = confirmed
        once.ensure()
        return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(state))


class TransactionBound(Monitor):
    """B4: per-record consent bound by hash, consume-once, expiry and an idempotency ledger.

    Execution requires an active record whose binding hash equals the request's; a change in
    any invalidation field requires a full new confirmation (a revision draft whose
    supersedes names the old record replaces it when confirmed); the idempotency ledger is
    keyed by the agent's own key and returns the stored receipt on replay and denies
    in-flight and mismatched keys; cancellation events are ignored unless revocation is
    enabled. The check-and-consume runs under the store lock before the yield point.
    """

    name = "B4"
    atomic_gate = True

    def __init__(
        self,
        window_s: int = 600,
        invalidation_fields: Iterable[str] = DEFAULT_INVALIDATION_FIELDS,
        revocation: bool = False,
        **extra: Any,
    ) -> None:
        self.window_s = int(window_s)
        self.invalidation_fields: tuple[str, ...] = tuple(invalidation_fields)
        self.revocation = bool(revocation)
        super().__init__(window_s=self.window_s, invalidation_fields=list(self.invalidation_fields), revocation=self.revocation, **extra)

    def after_confirm(self, session: Session, record: ConsentRecord, t: int) -> None:
        """Hook run once a record becomes active; B4Surface binds the rendered text here."""

    def inheritable(self, old: ConsentRecord, new: ConsentRecord, t: int) -> bool:
        """Return True when new may inherit old's confirmation without a new presentation.

        The invalidation fields must be unchanged and the cap must not rise: a record with a
        higher total_max_minor returns to the surface, a record with a lower cap may inherit.
        """
        if getattr(old, "status", None) != "active" or t >= int(getattr(old, "expires_at", 0) or 0):
            return False
        if getattr(old, "principal_id", None) != getattr(new, "principal_id", None):
            return False
        if int(getattr(new, "total_max_minor", 0) or 0) > int(getattr(old, "total_max_minor", 0) or 0):
            return False
        return all(getattr(old, f, None) == getattr(new, f, None) for f in self.invalidation_fields)

    def activate_by_inheritance(self, session: Session, record: ConsentRecord, old: ConsentRecord, t: int) -> ConsentRecord | None:
        """Activate record with old's confirmation event, language and remaining validity."""
        remaining = int(getattr(old, "expires_at", 0) or 0) - t
        if remaining <= 0:
            return None
        if getattr(record, "status", "draft") == "draft":
            record = session.store.present(record.cr_id) or self.store_get(session, record.cr_id) or record
        confirmed = session.store.confirm(record.cr_id, getattr(old, "confirmation_event_id", None), getattr(old, "lang_shown", session.lang), remaining)
        confirmed = confirmed or self.store_get(session, record.cr_id) or record
        self.after_confirm(session, confirmed, t)
        return confirmed

    def on_consent_request(self, session: Session, draft: ConsentRecord, t: int) -> Decision:
        self.register(session)
        record = self.ensure_draft(session, draft)
        old = self.store_get(session, getattr(record, "supersedes", None))
        if old is not None and self.inheritable(old, record, t):
            activated = self.activate_by_inheritance(session, record, old, t)
            if activated is not None:
                session.store.supersede(old.cr_id, activated.cr_id)
                decision = make_decision("ALLOW", "inherited", consent_id=activated.cr_id, details={"presented": False, "executed": False, "supersedes": old.cr_id, "inherited_event_id": getattr(old, "confirmation_event_id", None)})
                return self.log(session, "consent_request", decision, t, draft=snapshot_record(record), consent_state=snapshot_record(activated))
        decision, confirmed, _ = self.present(session, record, t, self.window_s)
        decision.details["executed"] = False
        if confirmed is not None:
            self.after_confirm(session, confirmed, t)
            if old is not None:
                session.store.supersede(old.cr_id, confirmed.cr_id)
                decision.details["supersedes"] = old.cr_id
        state = confirmed or self.store_get(session, record.cr_id) or record
        return self.log(session, "consent_request", decision, t, draft=snapshot_record(record), consent_state=snapshot_record(state))

    def resolve_record(self, session: Session, req: ExecutionRequest) -> ConsentRecord | None:
        """Return the consent record the request refers to, or None."""
        return _resolve_by_reference_or_match(self, session, req)

    def status_problem(self, record: ConsentRecord, t: int) -> str | None:
        """Return the reason an inactive or expired record cannot be used, or None."""
        status = getattr(record, "status", None)
        if status == "active":
            if t >= int(getattr(record, "expires_at", 0) or 0):
                return "expired"
            return None
        if status in TERMINAL_STATUSES:
            return str(status)
        return "no_consent"

    def binding_problem(self, session: Session, record: ConsentRecord, req: ExecutionRequest) -> str | None:
        """Return field_mismatch:<field> when the request's binding hash differs from the record's."""
        candidate = dataclasses.replace(record, principal_id=req.principal_id, payee_id=req.payee_id, amount_minor=req.amount_minor, currency=req.currency)
        if binding_hash(candidate) != getattr(record, "binding_hash", None):
            return f"field_mismatch:{first_mismatch(record, candidate) or 'binding_hash'}"
        return None

    def cap_problem(self, record: ConsentRecord, req: ExecutionRequest) -> str | None:
        """Return cap_exceeded when amount plus fee exceeds the record's total cap, or None."""
        if int(req.amount_minor) + int(req.fee_minor or 0) > int(getattr(record, "total_max_minor", 0) or 0):
            return "cap_exceeded"
        return None

    def consume_fingerprint(self, session: Session, record: ConsentRecord, req: ExecutionRequest, t: int) -> str:
        """Return the fingerprint handed to the store's atomic check-and-consume."""
        return fingerprint(req)

    def gate(self, session: Session, req: ExecutionRequest, t: int, once: YieldOnce) -> tuple[Decision, ConsentRecord | None]:
        """Run the full execution gate and return the decision with the record it used.

        A client key found in flight is reconciled when this request opened it (a re-issue
        after a crash at the yield point): a debit under the key is replayed, and without a
        debit the request executes under the same key on the use the crashed attempt
        consumed.
        """
        key = req.idempotency_key
        fp = fingerprint(req)
        opened = False
        recovered = False
        if key:
            status, stored = session.ledger.begin(key, fp, owner=req.req_id)
            if status == "in_flight":
                outcome, found = self.reconcile_in_flight(session, key, req)
                if outcome == "foreign":
                    return make_decision("DENY", "duplicate_in_flight", consent_id=req.consent_ref, details={"executed": False, "presented": False}), None
                if outcome == "replayed":
                    status, stored = "replay", found
                else:
                    status, stored = session.ledger.begin(key, fp, owner=req.req_id)
                    recovered = True
            if status == "replay":
                record = self.resolve_record(session, req)
                consent_id = getattr(record, "cr_id", None) or req.consent_ref
                decision = make_decision("ALLOW", "duplicate_replayed", consent_id=consent_id, receipt=receipt_to_dict(stored), details={"executed": False, "replayed": True, "presented": False})
                return decision, record
            if status == "mismatch":
                return make_decision("DENY", "key_payload_mismatch", consent_id=req.consent_ref, details={"executed": False, "presented": False}), None
            opened = True

        def deny(reason: str, record: ConsentRecord | None = None, **extra: Any) -> tuple[Decision, ConsentRecord | None]:
            if opened:
                session.ledger.fail(key)
            consent_id = getattr(record, "cr_id", None) or req.consent_ref
            return make_decision("DENY", reason, consent_id=consent_id, details={"executed": False, "presented": False, **extra}), record

        record = self.resolve_record(session, req)
        if record is None:
            return deny("no_consent")
        if recovered:
            if getattr(record, "status", None) not in ("active", "consumed") or int(getattr(record, "uses", 0) or 0) < 1:
                return deny(self.status_problem(record, t) or "no_consent", record)
        else:
            problem = self.status_problem(record, t)
            if problem is not None:
                return deny(problem, record)
        problem = self.binding_problem(session, record, req)
        if problem is not None:
            return deny(problem, record)
        problem = self.cap_problem(record, req)
        if problem is not None:
            return deny(problem, record)
        if not recovered:
            ok, reason = session.store.try_consume(record.cr_id, self.consume_fingerprint(session, record, req, t), t)
            if not ok:
                return deny(reason or "no_consent", self.store_get(session, record.cr_id) or record)
        once()
        receipt, exc, debited = self.execute_on_rail(session, req)
        record = self.store_get(session, record.cr_id) or record
        details: dict = {"presented": False, "recovered": recovered}
        if exc is None:
            if opened:
                session.ledger.complete(key, receipt_to_dict(receipt))
            details["executed"] = True
            return make_decision("ALLOW", "ok", consent_id=record.cr_id, receipt=receipt_to_dict(receipt), details=details), record
        if debited:
            if opened:
                session.ledger.complete(key, receipt_to_dict(receipt))
            details.update({"executed": True, "debit_recorded": True, "error": str(exc)})
            return make_decision("ALLOW", "rail_fault", consent_id=record.cr_id, receipt=None, details=details), record
        session.store.release(record.cr_id)
        if opened:
            session.ledger.fail(key)
        details.update({"executed": False, "debit_recorded": False, "error": str(exc)})
        return make_decision("ALLOW", "rail_fault", consent_id=record.cr_id, receipt=None, details=details), self.store_get(session, record.cr_id) or record

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        once = YieldOnce(yield_point)
        decision, record = self.gate(session, req, t, once)
        once.ensure()
        return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(record))

    def on_user_event(self, session: Session, event: Any, t: int) -> Decision:
        kind = event_attr(event, "kind", None)
        if kind == "cancel" and self.revocation:
            return self._apply_cancellation(session, event, t)
        return super().on_user_event(session, event, t)

    def _apply_cancellation(self, session: Session, event: Any, t: int) -> Decision:
        """Revoke the record the cancellation refers to and log the outcome."""
        self.register(session)
        record = self.target_record(session, event)
        if record is None:
            decision = make_decision("DENY", "no_consent", details={"presented": False, "executed": False})
            return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=None)
        ok = bool(session.store.revoke(record.cr_id))
        record = self.store_get(session, record.cr_id) or record
        if ok:
            decision = make_decision("ALLOW", "revoked", consent_id=record.cr_id, details={"presented": False, "executed": False})
        else:
            deferred = bool(session.store.revocation_pending(record.cr_id))
            decision = make_decision("DENY", str(getattr(record, "status", "no_consent")), consent_id=record.cr_id, details={"presented": False, "executed": False, "revocation_deferred": deferred})
        return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(record))


class TransactionBoundSurface(TransactionBound):
    """B4Surface: B4 with the binding taken over the rendered confirmation text in lang_shown.

    A request is compared by re-rendering the record with the request's fields in the
    session's current language, so a language switch between approval and execution
    changes the hash and the request is denied.
    """

    name = "B4Surface"

    def __init__(self, window_s: int = 600, invalidation_fields: Iterable[str] = DEFAULT_INVALIDATION_FIELDS, revocation: bool = False, **extra: Any) -> None:
        super().__init__(window_s=window_s, invalidation_fields=invalidation_fields, revocation=revocation, **extra)
        self._surface_hashes: dict[str, str] = {}

    def surface_hash_of(self, session: Session, record: ConsentRecord, lang: str) -> str:
        """Return the sha256 of the confirmation text of record rendered in lang."""
        text = render_confirmation(record, session.directory, lang)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def after_confirm(self, session: Session, record: ConsentRecord, t: int) -> None:
        lang = getattr(record, "lang_shown", None) or session.lang
        digest = self.surface_hash_of(session, record, lang)
        with self._lock:
            self._surface_hashes[record.cr_id] = digest
        try:
            record.surface_hash = digest
        except (AttributeError, dataclasses.FrozenInstanceError):
            pass

    def binding_problem(self, session: Session, record: ConsentRecord, req: ExecutionRequest) -> str | None:
        if getattr(record, "principal_id", None) != req.principal_id:
            return "field_mismatch:principal_id"
        with self._lock:
            expected = self._surface_hashes.get(record.cr_id)
        expected = expected or getattr(record, "surface_hash", None)
        if not expected:
            return "field_mismatch:surface_hash"
        candidate = dataclasses.replace(record, principal_id=req.principal_id, payee_id=req.payee_id, amount_minor=req.amount_minor, currency=req.currency)
        if self.surface_hash_of(session, candidate, session.lang) != expected:
            return "field_mismatch:surface_hash"
        return None

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        once = YieldOnce(yield_point)
        decision, record = self.gate(session, req, t, once)
        once.ensure()
        decision.details.update({"lang_now": session.lang, "lang_shown": getattr(record, "lang_shown", None)})
        return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(record))


@dataclass
class LeaseSlot:
    """One unit of a capability lease's budget with its compare-and-swap state."""

    index: int
    state: str = "Issued"
    stable_key: str = ""
    receipt: dict | None = None


@dataclass
class Lease:
    """A capability lease issued for one confirmed consent record."""

    cr_id: str
    binding_hash: str
    confirmation_event_id: str
    issued_at: int
    expires_at: int
    slots: list[LeaseSlot] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "cr_id": self.cr_id,
            "binding_hash": self.binding_hash,
            "confirmation_event_id": self.confirmation_event_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "slots": [dataclasses.asdict(s) for s in self.slots],
        }


class LedgerCapLease(Monitor):
    """B5: durable lease ledger with Issued, Prepared and Committed slots (CapLease form).

    A lease is unique over (binding_hash, confirmation_event_id) and carries budget slots.
    Execution prepares an Issued slot by compare-and-swap, hands the slot's stable key
    sha256("<cr_id>:<slot>:rail") to the rail, and commits on acknowledgement; a retry after
    a lost acknowledgement reuses the Prepared slot's key so the ledger replays the receipt.
    Revoke moves Issued slots to Revoked and leaves Prepared slots alone; a revision requires
    a new confirmation. The compare-and-swap is the atomic step and precedes the yield point.
    """

    name = "B5"
    supports_consent_channel = True
    atomic_gate = True

    def __init__(self, budget: int = 1, interval_s: int = 600, revocation: bool = True) -> None:
        self.budget = int(budget)
        self.interval_s = int(interval_s)
        self.revocation = bool(revocation)
        super().__init__(budget=self.budget, interval_s=self.interval_s, revocation=self.revocation)
        self._leases: dict[str, Lease] = {}
        self._issued_keys: set[tuple[str, str]] = set()
        self._cas = threading.Lock()

    @staticmethod
    def stable_key(cr_id: str, slot: int) -> str:
        """Return the stable idempotency key of one lease slot."""
        return slot_key(cr_id, slot)

    def lease_for(self, cr_id: str) -> Lease | None:
        with self._cas:
            return self._leases.get(cr_id)

    def _revoke_issued(self, cr_id: str) -> int:
        with self._cas:
            lease = self._leases.get(cr_id)
            if lease is None:
                return 0
            n = 0
            for slot in lease.slots:
                if slot.state == "Issued":
                    slot.state = "Revoked"
                    n += 1
            return n

    def on_consent_request(self, session: Session, draft: ConsentRecord, t: int) -> Decision:
        self.register(session)
        record = self.ensure_draft(session, draft)
        old = self.store_get(session, getattr(record, "supersedes", None))
        decision, confirmed, _ = self.present(session, record, t, self.interval_s)
        decision.details["executed"] = False
        if confirmed is not None:
            event_id = str(getattr(confirmed, "confirmation_event_id", "") or "")
            key = (str(getattr(confirmed, "binding_hash", "")), event_id)
            with self._cas:
                duplicate = key in self._issued_keys
                if not duplicate:
                    self._issued_keys.add(key)
                    self._leases[confirmed.cr_id] = Lease(
                        cr_id=confirmed.cr_id,
                        binding_hash=key[0],
                        confirmation_event_id=event_id,
                        issued_at=t,
                        expires_at=t + self.interval_s,
                        slots=[LeaseSlot(i, "Issued", self.stable_key(confirmed.cr_id, i)) for i in range(self.budget)],
                    )
            if duplicate:
                session.store.revoke(confirmed.cr_id)
                decision = make_decision("DENY", "duplicate_replayed", consent_id=confirmed.cr_id, details={**decision.details, "duplicate_lease": True})
            else:
                decision.details["lease"] = self._leases[confirmed.cr_id].as_dict()
                if old is not None:
                    self._revoke_issued(old.cr_id)
                    session.store.supersede(old.cr_id, confirmed.cr_id)
                    decision.details["supersedes"] = old.cr_id
        state = self.store_get(session, record.cr_id) or record
        return self.log(session, "consent_request", decision, t, draft=snapshot_record(record), consent_state=snapshot_record(state))

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        once = YieldOnce(yield_point)
        decision, record = self._gate(session, req, t, once)
        once.ensure()
        lease = self.lease_for(record.cr_id) if record is not None else None
        return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(record), lease=lease.as_dict() if lease else None)

    def _gate(self, session: Session, req: ExecutionRequest, t: int, once: YieldOnce) -> tuple[Decision, ConsentRecord | None]:
        def deny(reason: str, record: ConsentRecord | None = None, **extra: Any) -> tuple[Decision, ConsentRecord | None]:
            consent_id = getattr(record, "cr_id", None) or req.consent_ref
            return make_decision("DENY", reason, consent_id=consent_id, details={"executed": False, "presented": False, **extra}), record

        record = _resolve_by_reference_or_match(self, session, req)
        if record is None:
            return deny("no_consent")
        lease = self.lease_for(record.cr_id)
        if lease is None:
            return deny("no_consent", record)
        candidate = dataclasses.replace(record, principal_id=req.principal_id, payee_id=req.payee_id, amount_minor=req.amount_minor, currency=req.currency)
        if binding_hash(candidate) != lease.binding_hash:
            return deny(f"field_mismatch:{first_mismatch(record, candidate) or 'binding_hash'}", record)
        if int(req.amount_minor) + int(req.fee_minor or 0) > int(getattr(record, "total_max_minor", 0) or 0):
            return deny("cap_exceeded", record)
        with self._cas:
            slot = next((s for s in lease.slots if s.state == "Prepared"), None)
            recovered = slot is not None
            if slot is None:
                status = getattr(record, "status", None)
                if status in TERMINAL_STATUSES:
                    return deny(str(status), record)
                if status != "active":
                    return deny("no_consent", record)
                if t >= lease.expires_at or t >= int(getattr(record, "expires_at", 0) or 0):
                    return deny("expired", record)
                slot = next((s for s in lease.slots if s.state == "Issued"), None)
                if slot is None:
                    reason = "revoked" if all(s.state == "Revoked" for s in lease.slots) else "consumed"
                    slot_states = [s.state for s in lease.slots]
                    return deny(reason, record, slot_states=slot_states)
                slot.state = "Prepared"
            key = slot.stable_key
            index = slot.index
        fp = fingerprint(req)
        base = {"presented": False, "slot": index, "stable_key": key, "recovered": recovered, "agent_key": req.idempotency_key}
        status_key, stored = session.ledger.begin(key, fp, owner=req.req_id)
        if status_key == "in_flight":
            outcome, found = self.reconcile_in_flight(session, key, req)
            if outcome == "foreign":
                return deny("duplicate_in_flight", record, **base)
            if outcome == "replayed":
                status_key, stored = "replay", found
            else:
                status_key, stored = session.ledger.begin(key, fp, owner=req.req_id)
                base["recovered_after_crash"] = True
        if status_key == "replay":
            with self._cas:
                slot.state = "Committed"
                slot.receipt = receipt_to_dict(stored)
            return make_decision("ALLOW", "duplicate_replayed", consent_id=record.cr_id, receipt=receipt_to_dict(stored), details={**base, "executed": False, "replayed": True}), record
        if status_key == "mismatch":
            return deny("key_payload_mismatch", record, **base)
        once()
        sink_req = dataclasses.replace(req, idempotency_key=key)
        receipt, exc, debited = self.execute_on_rail(session, sink_req)
        if exc is None:
            session.ledger.complete(key, receipt_to_dict(receipt))
            with self._cas:
                slot.state = "Committed"
                slot.receipt = receipt_to_dict(receipt)
            return make_decision("ALLOW", "ok", consent_id=record.cr_id, receipt=receipt_to_dict(receipt), details={**base, "executed": True}), record
        if debited:
            session.ledger.complete(key, receipt_to_dict(receipt))
            return make_decision("ALLOW", "rail_fault", consent_id=record.cr_id, receipt=None, details={**base, "executed": True, "debit_recorded": True, "error": str(exc)}), record
        session.ledger.fail(key)
        return make_decision("ALLOW", "rail_fault", consent_id=record.cr_id, receipt=None, details={**base, "executed": False, "debit_recorded": False, "error": str(exc)}), record

    def on_user_event(self, session: Session, event: Any, t: int) -> Decision:
        self.register(session)
        kind = event_attr(event, "kind", None)
        if kind == "cancel":
            if not self.revocation:
                return super().on_user_event(session, event, t)
            record = self.target_record(session, event)
            if record is None:
                decision = make_decision("DENY", "no_consent", details={"presented": False, "executed": False})
                return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=None)
            moved = self._revoke_issued(record.cr_id)
            ok = bool(session.store.revoke(record.cr_id))
            record = self.store_get(session, record.cr_id) or record
            lease = self.lease_for(record.cr_id)
            if ok or moved:
                decision = make_decision("ALLOW", "revoked", consent_id=record.cr_id, details={"presented": False, "executed": False, "slots_revoked": moved})
            else:
                decision = make_decision("DENY", str(getattr(record, "status", "no_consent")), consent_id=record.cr_id, details={"presented": False, "executed": False, "slots_revoked": moved})
            return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(record), lease=lease.as_dict() if lease else None)
        if kind == "revise":
            old = self.target_record(session, event)
            if old is None:
                decision = make_decision("DENY", "no_consent", details={"presented": False, "executed": False})
                return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=None)
            if getattr(old, "status", None) in TERMINAL_STATUSES:
                decision = make_decision("DENY", str(old.status), consent_id=old.cr_id, details={"presented": False, "executed": False})
                return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(old))
            changes = dict(event_attr(event, "fields", {}) or {})
            draft = self.draft_from_record(session, old, changes, allowed_uses=self.budget)
            if draft is None:
                decision = make_decision("DENY", "schema_invalid", consent_id=old.cr_id, details={"presented": False, "executed": False, "changes": changes})
                return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(old))
            self._revoke_issued(old.cr_id)
            session.store.supersede(old.cr_id, draft.cr_id)
            inner = self.on_consent_request(session, draft, t)
            decision = make_decision(inner.verdict, inner.reason, consent_id=inner.consent_id, details={**inner.details, "supersedes": old.cr_id, "new_draft_id": draft.cr_id})
            return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(self.store_get(session, draft.cr_id) or draft))
        return super().on_user_event(session, event, t)


class PerCallApproval(Monitor):
    """B6: approval keyed by (tool, argument fingerprint) with no expiry and no revocation.

    A call whose key was approved earlier executes again without a new presentation; an
    unapproved call is shown on the surface and executes when approved. No replay state is
    kept, so identical approved calls each execute.
    """

    name = "B6"

    def __init__(self, tool: str = "send_payment") -> None:
        self.tool = str(tool)
        super().__init__(tool=self.tool)
        self._approvals: dict[tuple[str, str], str] = {}

    def approval_for(self, fp: str) -> str | None:
        with self._lock:
            return self._approvals.get((self.tool, fp))

    def on_consent_request(self, session: Session, draft: ConsentRecord, t: int) -> Decision:
        self.register(session)
        record = self.ensure_draft(session, draft)
        decision, confirmed, event = self.present(session, record, t, NO_EXPIRY_S)
        decision.details["executed"] = False
        if confirmed is not None:
            fp = record_fingerprint(confirmed, session.session_id, t)
            with self._lock:
                self._approvals[(self.tool, fp)] = str(getattr(event, "event_id", "") or "")
            decision.details["approval_key"] = [self.tool, fp]
        state = confirmed or self.store_get(session, record.cr_id) or record
        return self.log(session, "consent_request", decision, t, draft=snapshot_record(record), consent_state=snapshot_record(state))

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        once = YieldOnce(yield_point)
        fp = fingerprint(req)
        approval = self.approval_for(fp)
        state: Any = self.store_get(session, req.consent_ref)
        if approval is not None:
            decision = self.run_rail(session, req, reason="ok", consent_id=req.consent_ref, details={"reused": True, "approval_event_id": approval, "approval_key": [self.tool, fp]}, yield_point=once)
        else:
            draft = self.draft_from_request(session, req)
            if draft is None:
                once.ensure()
                decision = make_decision("DENY", "schema_invalid", consent_id=None, details={"presented": False, "executed": False})
                return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(state))
            shown, confirmed, event = self.present(session, draft, t, NO_EXPIRY_S)
            if confirmed is None:
                decision = make_decision("DENY", shown.reason, consent_id=None, details={**shown.details, "executed": False})
                state = self.store_get(session, draft.cr_id)
            else:
                event_id = str(getattr(event, "event_id", "") or "")
                with self._lock:
                    self._approvals[(self.tool, fp)] = event_id
                decision = self.run_rail(session, req, reason="ok", consent_id=confirmed.cr_id, details={**shown.details, "reused": False, "approval_event_id": event_id, "approval_key": [self.tool, fp]}, yield_point=once)
                state = confirmed
        once.ensure()
        return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(state))


@dataclass(frozen=True)
class NarrowingPolicy:
    """Allowed set of B7: payee_id in payees, amount_minor <= cap_minor, optional currency."""

    payees: frozenset[str]
    cap_minor: int
    currency: str | None = None

    def problem(self, payee_id: str, amount_minor: int, currency: str) -> str | None:
        """Return the reason a request falls outside the policy, or None."""
        if payee_id not in self.payees:
            return "not_allowlisted"
        if self.currency is not None and currency != self.currency:
            return "field_mismatch:currency"
        if int(amount_minor) > int(self.cap_minor):
            return "over_cap"
        return None

    def narrows(self, other: "NarrowingPolicy") -> bool:
        """Return True when this policy allows no more than other."""
        if not self.payees <= other.payees:
            return False
        if int(self.cap_minor) > int(other.cap_minor):
            return False
        return other.currency is None or self.currency == other.currency

    def as_dict(self) -> dict:
        return {"payees": sorted(self.payees), "cap_minor": int(self.cap_minor), "currency": self.currency}


def coerce_policy(value: Any) -> NarrowingPolicy:
    """Build a NarrowingPolicy from a NarrowingPolicy, a mapping or None (empty policy)."""
    if isinstance(value, NarrowingPolicy):
        return value
    if value is None:
        return NarrowingPolicy(frozenset(), 0, None)
    payees = value.get("payees", ())
    if isinstance(payees, str):
        payees = [payees]
    return NarrowingPolicy(frozenset(payees), int(value.get("cap_minor", 0) or 0), value.get("currency"))


class ProgentNarrowing(Monitor):
    """B7: policy over payee_id and amount cap; narrowing applies itself, expansion asks.

    No consent record is bound, nothing expires and no replay state is kept. A cancellation
    narrows the policy to the empty set; a revision inside the current policy is applied
    without presentation; a revision outside it is shown on the surface and applied when
    approved.
    """

    name = "B7"
    supports_consent_channel = True

    def __init__(self, initial_policy: Any = None) -> None:
        self.policy: NarrowingPolicy = coerce_policy(initial_policy)
        super().__init__(initial_policy=self.policy.as_dict())
        self._policy_lock = threading.Lock()

    def current_policy(self) -> NarrowingPolicy:
        with self._policy_lock:
            return self.policy

    def _set_policy(self, policy: NarrowingPolicy) -> None:
        with self._policy_lock:
            self.policy = policy

    def _expanded(self, current: NarrowingPolicy, payee_id: str, cap_minor: int, currency: str | None) -> NarrowingPolicy:
        return NarrowingPolicy(current.payees | {payee_id}, max(int(current.cap_minor), int(cap_minor)), current.currency or currency)

    def on_consent_request(self, session: Session, draft: ConsentRecord, t: int) -> Decision:
        self.register(session)
        current = self.current_policy()
        cap = int(getattr(draft, "total_max_minor", 0) or getattr(draft, "amount_minor", 0) or 0)
        problem = current.problem(draft.payee_id, cap, draft.currency)
        if problem is None:
            decision = make_decision("ALLOW", "ok", consent_id=getattr(draft, "cr_id", None), details={"presented": False, "executed": False, "within_policy": True, "policy": current.as_dict()})
            return self.log(session, "consent_request", decision, t, draft=snapshot_record(draft), consent_state=None)
        record = self.ensure_draft(session, draft)
        decision, confirmed, _ = self.present(session, record, t, NO_EXPIRY_S)
        decision.details.update({"executed": False, "policy_before": current.as_dict(), "outside": problem})
        if confirmed is not None:
            proposed = self._expanded(current, confirmed.payee_id, cap, confirmed.currency)
            self._set_policy(proposed)
            decision = make_decision("ALLOW", "policy_expansion", consent_id=confirmed.cr_id, details={**decision.details, "policy": proposed.as_dict()})
        else:
            decision = make_decision("DENY", "policy_expansion", consent_id=None, details={**decision.details, "policy": current.as_dict()})
        return self.log(session, "consent_request", decision, t, draft=snapshot_record(record), consent_state=None)

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        once = YieldOnce(yield_point)
        current = self.current_policy()
        problem = current.problem(req.payee_id, req.amount_minor, req.currency)
        if problem is not None:
            decision = make_decision("DENY", problem, consent_id=req.consent_ref, details={"executed": False, "presented": False, "policy": current.as_dict()})
        else:
            decision = self.run_rail(session, req, reason="ok", consent_id=req.consent_ref, details={"policy": current.as_dict()}, yield_point=once)
        once.ensure()
        return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=None)

    def on_user_event(self, session: Session, event: Any, t: int) -> Decision:
        self.register(session)
        kind = event_attr(event, "kind", None)
        current = self.current_policy()
        if kind == "cancel":
            narrowed = NarrowingPolicy(frozenset(), 0, current.currency)
            self._set_policy(narrowed)
            decision = make_decision("ALLOW", "narrowing", details={"presented": False, "executed": False, "policy_before": current.as_dict(), "policy": narrowed.as_dict()})
            return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=None)
        if kind != "revise":
            return super().on_user_event(session, event, t)
        changes = dict(event_attr(event, "fields", {}) or {})
        payee_id = changes.get("payee_id")
        payees = frozenset({payee_id}) if payee_id else current.payees
        cap = changes.get("total_max_minor")
        if cap is None:
            cap = changes.get("amount_minor")
        cap = int(current.cap_minor if cap is None else cap)
        currency = changes.get("currency") or current.currency
        proposed = NarrowingPolicy(payees, cap, currency)
        if proposed.narrows(current):
            self._set_policy(proposed)
            decision = make_decision("ALLOW", "narrowing", details={"presented": False, "executed": False, "policy_before": current.as_dict(), "policy": proposed.as_dict()})
            return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=None)
        shown_payee = payee_id or (sorted(current.payees)[0] if current.payees else "")
        shown_amount = int(changes.get("amount_minor") or cap)
        fields = {
            "principal_id": session.principal_id,
            "session_id": session.session_id,
            "action_type": "payment",
            "payee_id": shown_payee,
            "amount_minor": shown_amount,
            "currency": currency or "AED",
            "total_max_minor": max(cap, shown_amount),
            "revision": 1,
            "allowed_uses": 1,
            "supersedes": None,
        }
        try:
            draft = session.store.create_draft(fields)
        except (TypeError, ValueError):
            decision = make_decision("DENY", "policy_expansion", details={"presented": False, "executed": False, "policy_before": current.as_dict(), "policy": current.as_dict(), "proposed": proposed.as_dict()})
            return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=None)
        shown, confirmed, _ = self.present(session, draft, t, NO_EXPIRY_S)
        if confirmed is not None:
            self._set_policy(proposed)
            decision = make_decision("ALLOW", "policy_expansion", consent_id=confirmed.cr_id, details={**shown.details, "executed": False, "policy_before": current.as_dict(), "policy": proposed.as_dict()})
        else:
            decision = make_decision("DENY", "policy_expansion", details={**shown.details, "executed": False, "policy_before": current.as_dict(), "policy": current.as_dict(), "surface_decline": shown.reason})
        return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(self.store_get(session, draft.cr_id) or draft))


class Tasdiq(TransactionBound):
    """The B4 core plus the consent channel, a revision rule and server-derived keys.

    Cancellation revokes the record. A revision supersedes the old record at the time it is
    stated and creates a revision+1 draft: under "invalidate" the draft needs a new
    confirmation (changes outside the invalidation fields inherit, as in B4); under
    "narrowing" a draft with the same payee and currency, an amount not above the old one
    and a cap not above the old cap is activated by inheriting the confirmation event;
    under "cap_settle" the confirmation binds a cap, a revision inside the cap leaves the
    record unchanged, and any amount within the cap to the same payee settles once.

    The idempotency key handed to the rail is slot_key(cr_id, slot) where slot is the use
    consumed by the atomic check-and-consume; the agent's key is logged and otherwise
    ignored. A request whose payload equals that of a use whose acknowledgement was lost, or
    of a use of a consumed record, receives the stored receipt without a second debit.
    """

    name = "Tasdiq"
    supports_consent_channel = True
    atomic_gate = True
    RULES: tuple[str, ...] = ("invalidate", "narrowing", "cap_settle")

    def __init__(
        self,
        revision_rule: str = "invalidate",
        window_s: int = 600,
        revocation: bool = True,
        budget: int = 1,
        invalidation_fields: Iterable[str] = DEFAULT_INVALIDATION_FIELDS,
    ) -> None:
        if revision_rule not in self.RULES:
            raise ValueError(f"unknown revision rule: {revision_rule}")
        self.revision_rule = str(revision_rule)
        self.budget = int(budget)
        super().__init__(window_s=window_s, invalidation_fields=invalidation_fields, revocation=revocation, revision_rule=self.revision_rule, budget=self.budget)
        self._unacknowledged: dict[str, set[int]] = {}

    @staticmethod
    def stable_key(cr_id: str, slot: int) -> str:
        """Return the server-derived idempotency key of one use of a consent record."""
        return slot_key(cr_id, slot)

    def inheritable(self, old: ConsentRecord, new: ConsentRecord, t: int) -> bool:
        if self.revision_rule == "invalidate":
            return super().inheritable(old, new, t)
        if getattr(old, "status", None) != "active" or t >= int(getattr(old, "expires_at", 0) or 0):
            return False
        if old.principal_id != new.principal_id or old.payee_id != new.payee_id or old.currency != new.currency:
            return False
        if int(new.total_max_minor) > int(old.total_max_minor):
            return False
        if self.revision_rule == "narrowing":
            return int(new.amount_minor) <= int(old.amount_minor)
        return int(new.amount_minor) <= int(old.total_max_minor)

    def within_cap(self, old: ConsentRecord, changes: dict, t: int) -> bool:
        """Return True when a cap_settle revision keeps the old record's payee and currency and stays inside its cap."""
        if getattr(old, "status", None) != "active" or t >= int(getattr(old, "expires_at", 0) or 0):
            return False
        if changes.get("payee_id") not in (None, old.payee_id):
            return False
        if changes.get("currency") not in (None, old.currency):
            return False
        amount = changes.get("amount_minor")
        if amount is not None and int(amount) > int(old.total_max_minor):
            return False
        cap = changes.get("total_max_minor")
        return cap is None or int(cap) <= int(old.total_max_minor)

    def successor_of(self, session: Session, record: ConsentRecord, req: ExecutionRequest) -> ConsentRecord | None:
        """Return the active revision that superseded record and matches the request, if any."""
        try:
            actives = list(session.store.active_for(session.session_id))
        except (KeyError, ValueError):
            return None
        for r in actives:
            if getattr(r, "supersedes", None) == record.cr_id and self._matches_rule(r, req):
                return r
        return None

    def _matches_rule(self, record: ConsentRecord, req: ExecutionRequest) -> bool:
        if record.principal_id != req.principal_id or record.payee_id != req.payee_id or record.currency != req.currency:
            return False
        if self.revision_rule == "cap_settle":
            return int(req.amount_minor) + int(req.fee_minor or 0) <= int(record.total_max_minor)
        return int(record.amount_minor) == int(req.amount_minor)

    def resolve_record(self, session: Session, req: ExecutionRequest) -> ConsentRecord | None:
        record = super().resolve_record(session, req)
        seen: set[str] = set()
        while record is not None and getattr(record, "status", None) == "superseded" and record.cr_id not in seen:
            seen.add(record.cr_id)
            successor = self.successor_of(session, record, req)
            if successor is None:
                break
            record = successor
        return record

    def binding_problem(self, session: Session, record: ConsentRecord, req: ExecutionRequest) -> str | None:
        if self.revision_rule != "cap_settle":
            return super().binding_problem(session, record, req)
        for f in ("principal_id", "payee_id", "currency"):
            if getattr(record, f, None) != getattr(req, f, None):
                return f"field_mismatch:{f}"
        return None

    def consume_fingerprint(self, session: Session, record: ConsentRecord, req: ExecutionRequest, t: int) -> str:
        if self.revision_rule == "cap_settle":
            return record_fingerprint(record, session.session_id, t)
        return fingerprint(req)

    @staticmethod
    def changes_nothing(old: ConsentRecord, changes: dict) -> bool:
        """Return True when every stated field of a revision equals the old record's value."""
        for name in ("payee_id", "amount_minor", "currency", "total_max_minor"):
            value = changes.get(name)
            if value is not None and value != getattr(old, name, None):
                return False
        return True

    def narrowed_cap_settle_changes(self, old: ConsentRecord, changes: dict) -> dict:
        """Return the revision fields of a within-cap revision with a cap that never rises above the old cap."""
        out = dict(changes)
        if out.get("total_max_minor") is None:
            amount = int(out.get("amount_minor") if out.get("amount_minor") is not None else old.amount_minor)
            allowance = max(0, int(old.total_max_minor) - int(old.amount_minor))
            out["total_max_minor"] = min(int(old.total_max_minor), amount + allowance)
        return out

    def _mark_unacknowledged(self, cr_id: str, slot: int, acknowledged: bool) -> None:
        with self._lock:
            slots = self._unacknowledged.setdefault(cr_id, set())
            if acknowledged:
                slots.discard(slot)
            else:
                slots.add(slot)

    def unacknowledged_slots(self, cr_id: str) -> list[int]:
        """Return the slots of cr_id whose rail acknowledgement was lost, in slot order."""
        with self._lock:
            return sorted(self._unacknowledged.get(cr_id, set()))

    def replay_for(self, session: Session, record: ConsentRecord, fp: str, slots: Iterable[int]) -> tuple[str, int, Any] | None:
        """Look the given slots of record up in the ledger under the request's payload fingerprint.

        Returns ("replay", slot, receipt) for a completed use with the same payload,
        ("in_flight", slot, None) for a use still executing, or None when no slot matches.
        An unknown key is removed again so that the lookup leaves no in-flight entry.
        """
        for slot in slots:
            key = self.stable_key(record.cr_id, slot)
            status, stored = session.ledger.begin(key, fp)
            if status == "replay":
                return "replay", slot, stored
            if status == "in_flight":
                return "in_flight", slot, None
            if status == "new":
                session.ledger.fail(key)
        return None

    def recover_in_flight(self, session: Session, record: ConsentRecord, req: ExecutionRequest, slot: int, fp: str, once: YieldOnce, base: dict, deny: Any, replayed: Any) -> tuple[Decision, ConsentRecord | None]:
        """Settle a slot whose key is in flight: a foreign entry is denied, this request's own entry is reconciled.

        A debit found on the rail under the slot key is returned as a replay; without a debit
        the entry is released and the request executes under the same key on the slot that the
        crashed attempt consumed.
        """
        key = self.stable_key(record.cr_id, slot)
        outcome, found = self.reconcile_in_flight(session, key, req)
        if outcome == "foreign":
            return deny("duplicate_in_flight", record, slot=slot)
        if outcome == "replayed":
            return replayed(record, slot, found)
        base["recovered"] = True
        return self.execute_slot(session, record, req, slot, fp, once, base, consumed_now=False)

    def execute_slot(self, session: Session, record: ConsentRecord, req: ExecutionRequest, slot: int, fp: str, once: YieldOnce, base: dict, consumed_now: bool) -> tuple[Decision, ConsentRecord | None]:
        """Open the slot key in the ledger, yield, call the rail and record the outcome.

        consumed_now states whether this call consumed the slot (a use that is released when
        the key turns out to be taken or the rail reports no debit).
        """
        key = self.stable_key(record.cr_id, slot)
        base.update({"slot": slot, "stable_key": key})

        def deny(reason: str, current: ConsentRecord | None = None, **extra: Any) -> tuple[Decision, ConsentRecord | None]:
            consent_id = getattr(current, "cr_id", None) or req.consent_ref
            return make_decision("DENY", reason, consent_id=consent_id, details={**base, "executed": False, **extra}), current

        status_key, stored = session.ledger.begin(key, fp, owner=req.req_id)
        if status_key != "new":
            if consumed_now:
                self.release_slot(session, record.cr_id)
            current = self.store_get(session, record.cr_id) or record
            if status_key == "replay":
                self._mark_unacknowledged(record.cr_id, slot, acknowledged=True)
                details = {**base, "executed": False, "replayed": True}
                return make_decision("ALLOW", "duplicate_replayed", consent_id=current.cr_id, receipt=receipt_to_dict(stored), details=details), current
            if status_key == "in_flight":
                return deny("duplicate_in_flight", current)
            return deny("key_payload_mismatch", current)
        once()
        receipt, exc, debited = self.execute_on_rail(session, dataclasses.replace(req, idempotency_key=key))
        record = self.store_get(session, record.cr_id) or record
        if exc is None:
            session.ledger.complete(key, receipt_to_dict(receipt))
            return make_decision("ALLOW", "ok", consent_id=record.cr_id, receipt=receipt_to_dict(receipt), details={**base, "executed": True}), record
        if debited:
            session.ledger.complete(key, receipt_to_dict(receipt))
            self._mark_unacknowledged(record.cr_id, slot, acknowledged=False)
            return make_decision("ALLOW", "rail_fault", consent_id=record.cr_id, receipt=None, details={**base, "executed": True, "debit_recorded": True, "error": str(exc)}), record
        self.release_slot(session, record.cr_id)
        session.ledger.fail(key)
        record = self.store_get(session, record.cr_id) or record
        return make_decision("ALLOW", "rail_fault", consent_id=record.cr_id, receipt=None, details={**base, "executed": False, "debit_recorded": False, "error": str(exc)}), record

    def consume_slot(self, session: Session, record: ConsentRecord, req: ExecutionRequest, t: int) -> tuple[bool, str, int]:
        """Consume one use of record atomically and return (ok, reason, slot).

        The store's try_consume is the atomic step; the arm's lock serialises it with the
        read of the resulting use count so that the slot index is exact under concurrency.
        """
        with self._lock:
            ok, reason = session.store.try_consume(record.cr_id, self.consume_fingerprint(session, record, req, t), t)
            if not ok:
                return False, reason or "no_consent", -1
            fresh = self.store_get(session, record.cr_id) or record
            return True, "ok", max(0, int(getattr(fresh, "uses", 1) or 1) - 1)

    def release_slot(self, session: Session, cr_id: str) -> None:
        """Return a use to the record after the rail reported no debit."""
        with self._lock:
            session.store.release(cr_id)

    def gate(self, session: Session, req: ExecutionRequest, t: int, once: YieldOnce) -> tuple[Decision, ConsentRecord | None]:
        fp = fingerprint(req)
        base = {"presented": False, "rule": self.revision_rule, "agent_key": req.idempotency_key}

        def deny(reason: str, record: ConsentRecord | None = None, **extra: Any) -> tuple[Decision, ConsentRecord | None]:
            consent_id = getattr(record, "cr_id", None) or req.consent_ref
            return make_decision("DENY", reason, consent_id=consent_id, details={**base, "executed": False, **extra}), record

        def replayed(record: ConsentRecord, slot: int, stored: Any) -> tuple[Decision, ConsentRecord | None]:
            self._mark_unacknowledged(record.cr_id, slot, acknowledged=True)
            details = {**base, "executed": False, "replayed": True, "slot": slot, "stable_key": self.stable_key(record.cr_id, slot)}
            return make_decision("ALLOW", "duplicate_replayed", consent_id=record.cr_id, receipt=receipt_to_dict(stored), details=details), record

        record = self.resolve_record(session, req)
        if record is None:
            return deny("no_consent")
        found = self.replay_for(session, record, fp, self.unacknowledged_slots(record.cr_id))
        if found is not None:
            kind, slot, stored = found
            if kind == "replay":
                return replayed(record, slot, stored)
            return self.recover_in_flight(session, record, req, slot, fp, once, base, deny, replayed)
        problem = self.status_problem(record, t)
        if problem == "consumed":
            found = self.replay_for(session, record, fp, range(int(getattr(record, "uses", 0) or 0)))
            if found is not None:
                kind, slot, stored = found
                if kind == "replay":
                    return replayed(record, slot, stored)
                return self.recover_in_flight(session, record, req, slot, fp, once, base, deny, replayed)
            return deny("consumed", record)
        if problem is not None:
            return deny(problem, record)
        problem = self.binding_problem(session, record, req)
        if problem is not None:
            return deny(problem, record)
        problem = self.cap_problem(record, req)
        if problem is not None:
            return deny(problem, record)
        ok, reason, slot = self.consume_slot(session, record, req, t)
        if not ok:
            current = self.store_get(session, record.cr_id) or record
            if reason == "consumed":
                found = self.replay_for(session, current, fp, range(int(getattr(current, "uses", 0) or 0)))
                if found is not None:
                    kind, found_slot, stored = found
                    if kind == "replay":
                        return replayed(current, found_slot, stored)
                    return self.recover_in_flight(session, current, req, found_slot, fp, once, base, deny, replayed)
            return deny(reason, current)
        return self.execute_slot(session, record, req, slot, fp, once, base, consumed_now=True)

    def on_user_event(self, session: Session, event: Any, t: int) -> Decision:
        self.register(session)
        kind = event_attr(event, "kind", None)
        if kind == "cancel":
            if not self.revocation:
                return super().on_user_event(session, event, t)
            return self._apply_cancellation(session, event, t)
        if kind != "revise":
            return super().on_user_event(session, event, t)
        old = self.target_record(session, event)
        if old is None:
            decision = make_decision("DENY", "no_consent", details={"presented": False, "executed": False, "rule": self.revision_rule})
            return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=None)
        changes = dict(event_attr(event, "fields", {}) or {})
        if self.revision_rule == "cap_settle" and self.within_cap(old, changes, t):
            if self.changes_nothing(old, changes):
                decision = make_decision("ALLOW", "cap_settle", consent_id=old.cr_id, details={"presented": False, "executed": False, "rule": self.revision_rule, "record_changed": False, "changes": changes})
                return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(old))
            narrowed = self.narrowed_cap_settle_changes(old, changes)
            draft = self.draft_from_record(session, old, narrowed, allowed_uses=self.budget)
            if draft is None:
                decision = make_decision("DENY", "schema_invalid", consent_id=old.cr_id, details={"presented": False, "executed": False, "rule": self.revision_rule, "changes": changes})
                return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(old))
            session.store.supersede(old.cr_id, draft.cr_id)
            activated = self.activate_by_inheritance(session, draft, old, t)
            if activated is not None:
                decision = make_decision("ALLOW", "cap_settle", consent_id=activated.cr_id, details={"presented": False, "executed": False, "rule": self.revision_rule, "record_changed": True, "changes": changes, "supersedes": old.cr_id, "new_draft_id": draft.cr_id, "inherited_event_id": getattr(old, "confirmation_event_id", None)})
                return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(activated))
            inner = self.on_consent_request(session, draft, t)
            decision = make_decision(inner.verdict, inner.reason, consent_id=inner.consent_id, details={**inner.details, "rule": self.revision_rule, "supersedes": old.cr_id, "new_draft_id": draft.cr_id})
            return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(self.store_get(session, draft.cr_id) or draft))
        if getattr(old, "status", None) in TERMINAL_STATUSES:
            decision = make_decision("DENY", str(old.status), consent_id=old.cr_id, details={"presented": False, "executed": False, "rule": self.revision_rule})
            return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(old))
        draft = self.draft_from_record(session, old, changes, allowed_uses=self.budget)
        if draft is None:
            decision = make_decision("DENY", "schema_invalid", consent_id=old.cr_id, details={"presented": False, "executed": False, "rule": self.revision_rule, "changes": changes})
            return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(old))
        inherit = self.inheritable(old, draft, t)
        session.store.supersede(old.cr_id, draft.cr_id)
        if inherit:
            activated = self.activate_by_inheritance(session, draft, old, t)
            if activated is not None:
                decision = make_decision("ALLOW", self.revision_rule, consent_id=activated.cr_id, details={"presented": False, "executed": False, "rule": self.revision_rule, "supersedes": old.cr_id, "inherited_event_id": getattr(old, "confirmation_event_id", None)})
                return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(activated))
        inner = self.on_consent_request(session, draft, t)
        decision = make_decision(inner.verdict, inner.reason, consent_id=inner.consent_id, details={**inner.details, "rule": self.revision_rule, "supersedes": old.cr_id, "new_draft_id": draft.cr_id})
        return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=snapshot_record(self.store_get(session, draft.cr_id) or draft))


class TasdiqNonAtomic(Tasdiq):
    """Ablation of Tasdiq whose check and consume are separated by the yield point.

    The gate reads the record, verifies status, expiry, binding, cap, remaining uses and
    fingerprint without consuming, yields, executes on the rail and only then records the
    use. The recording step is a blind state update: its outcome is logged in the decision
    details (consumed_after_execution) and does not revisit the admission decision, so two
    executors that both pass the check before either records its use both execute. The
    ledger key is derived from the use count read at check time.
    """

    name = "TasdiqNonAtomic"
    atomic_gate = False

    def gate(self, session: Session, req: ExecutionRequest, t: int, once: YieldOnce) -> tuple[Decision, ConsentRecord | None]:
        fp = fingerprint(req)
        base = {"presented": False, "rule": self.revision_rule, "agent_key": req.idempotency_key, "atomic": False}

        def deny(reason: str, record: ConsentRecord | None = None, **extra: Any) -> tuple[Decision, ConsentRecord | None]:
            consent_id = getattr(record, "cr_id", None) or req.consent_ref
            return make_decision("DENY", reason, consent_id=consent_id, details={**base, "executed": False, **extra}), record

        def replayed(record: ConsentRecord, slot: int, stored: Any) -> tuple[Decision, ConsentRecord | None]:
            self._mark_unacknowledged(record.cr_id, slot, acknowledged=True)
            details = {**base, "executed": False, "replayed": True, "slot": slot, "stable_key": self.stable_key(record.cr_id, slot)}
            return make_decision("ALLOW", "duplicate_replayed", consent_id=record.cr_id, receipt=receipt_to_dict(stored), details=details), record

        record = self.resolve_record(session, req)
        if record is None:
            return deny("no_consent")
        found = self.replay_for(session, record, fp, self.unacknowledged_slots(record.cr_id))
        if found is not None:
            kind, slot, stored = found
            if kind == "replay":
                return replayed(record, slot, stored)
            return deny("duplicate_in_flight", record, slot=slot)
        problem = self.status_problem(record, t)
        if problem == "consumed":
            found = self.replay_for(session, record, fp, range(int(getattr(record, "uses", 0) or 0)))
            if found is not None:
                kind, slot, stored = found
                if kind == "replay":
                    return replayed(record, slot, stored)
                return deny("duplicate_in_flight", record, slot=slot)
            return deny("consumed", record)
        if problem is not None:
            return deny(problem, record)
        problem = self.binding_problem(session, record, req)
        if problem is not None:
            return deny(problem, record)
        problem = self.cap_problem(record, req)
        if problem is not None:
            return deny(problem, record)
        consume_fp = self.consume_fingerprint(session, record, req, t)
        if consume_fp != fingerprint(record):
            return deny("field_mismatch:fingerprint", record)
        uses = int(getattr(record, "uses", 0) or 0)
        if uses >= int(getattr(record, "allowed_uses", 1) or 1):
            return deny("consumed", record)
        slot = uses
        key = self.stable_key(record.cr_id, slot)
        base.update({"slot": slot, "stable_key": key})
        once()
        receipt, exc, debited = self.execute_on_rail(session, dataclasses.replace(req, idempotency_key=key))
        if exc is not None and not debited:
            record = self.store_get(session, record.cr_id) or record
            return make_decision("ALLOW", "rail_fault", consent_id=record.cr_id, receipt=None, details={**base, "executed": False, "debit_recorded": False, "error": str(exc)}), record
        consumed, consume_reason = session.store.try_consume(record.cr_id, consume_fp, t)
        status_key, _ = session.ledger.begin(key, fp)
        if status_key == "new":
            session.ledger.complete(key, receipt_to_dict(receipt))
        record = self.store_get(session, record.cr_id) or record
        details = {**base, "executed": True, "consumed_after_execution": bool(consumed), "consume_reason": consume_reason, "ledger_state": status_key}
        if exc is None:
            return make_decision("ALLOW", "ok", consent_id=record.cr_id, receipt=receipt_to_dict(receipt), details=details), record
        self._mark_unacknowledged(record.cr_id, slot, acknowledged=False)
        details.update({"debit_recorded": True, "error": str(exc)})
        return make_decision("ALLOW", "rail_fault", consent_id=record.cr_id, receipt=None, details=details), record


class TrustAgent(TransactionBound):
    """Control: executes whenever the agent claims the user confirmed; B4 otherwise."""

    name = "TrustAgent"
    atomic_gate = False

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        if bool(getattr(req, "agent_claims_user_confirmed", False)):
            once = YieldOnce(yield_point)
            decision = self.run_rail(session, req, reason="trust_claim", consent_id=req.consent_ref, details={"claimed": True}, yield_point=once)
            once.ensure()
            state = self.store_get(session, req.consent_ref)
            return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(state))
        return super().on_execute(session, req, t, yield_point)


class AskAlways(TransactionBound):
    """Control: every execution is shown on the surface afresh and then passes the B4 gate."""

    name = "AskAlways"

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        once = YieldOnce(yield_point)
        draft = self.draft_from_request(session, req)
        if draft is None:
            once.ensure()
            decision = make_decision("DENY", "schema_invalid", consent_id=None, details={"presented": False, "executed": False})
            return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=None)
        shown, confirmed, _ = self.present(session, draft, t, self.window_s)
        if confirmed is None:
            once.ensure()
            decision = make_decision("DENY", shown.reason, consent_id=None, details={**shown.details, "executed": False})
            return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(self.store_get(session, draft.cr_id)))
        self.after_confirm(session, confirmed, t)
        fresh = dataclasses.replace(req, consent_ref=confirmed.cr_id)
        decision, record = self.gate(session, fresh, t, once)
        once.ensure()
        decision.details.update({"presented": True, "surface_event_id": shown.details.get("surface_event_id"), "fresh_consent_id": confirmed.cr_id, "prior_consent_ref": req.consent_ref})
        return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(record))


class RefuseAll(Monitor):
    """Control: denies every confirmation and every execution without presenting anything."""

    name = "RefuseAll"

    def on_consent_request(self, session: Session, draft: ConsentRecord, t: int) -> Decision:
        self.register(session)
        decision = make_decision("DENY", "refuse_all", consent_id=None, details={"presented": False, "executed": False})
        return self.log(session, "consent_request", decision, t, draft=snapshot_record(draft), consent_state=snapshot_record(self.store_get(session, getattr(draft, "cr_id", None))))

    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        self.register(session)
        once = YieldOnce(yield_point)
        decision = make_decision("DENY", "refuse_all", consent_id=req.consent_ref, details={"presented": False, "executed": False})
        once.ensure()
        return self.log(session, "execute", decision, t, request=snapshot_request(req), consent_state=snapshot_record(self.store_get(session, req.consent_ref)))


B4Surface = TransactionBoundSurface

ARMS: dict[str, type[Monitor]] = {
    "B0": NoMonitor,
    "B1": SchemaOnly,
    "B2": StaticRestrictions,
    "B3": SessionElevation,
    "B4": TransactionBound,
    "B4Surface": TransactionBoundSurface,
    "B5": LedgerCapLease,
    "B6": PerCallApproval,
    "B7": ProgentNarrowing,
    "Tasdiq": Tasdiq,
    "TasdiqNonAtomic": TasdiqNonAtomic,
    "TrustAgent": TrustAgent,
    "AskAlways": AskAlways,
    "RefuseAll": RefuseAll,
    "PromptOnly": PromptOnly,
}


def make_arm(name: str, **kwargs: Any) -> Monitor:
    """Instantiate an arm by its registry name with its calibration parameters."""
    try:
        cls = ARMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown arm: {name}") from exc
    return cls(**kwargs)
