"""Monitor base class, session container and helpers shared by the policy arms.

A monitor sits at the execution boundary of the simulated payment agent. The runner calls
on_consent_request when a draft consent record is to be confirmed, on_execute when the
agent submits a payment, on_user_event when the principal revises or cancels on the
consent channel, and on_tick when simulated time advances. Every hook returns a Decision
and appends exactly one log line to the session log, with the reason and a snapshot of
the consent record the decision was based on.

Race interleavings are driven through the yield_point argument of on_execute. Every arm
calls the callback exactly once per execution attempt when one is given. A non-atomic arm
calls it between its check and its state update or execution; an atomic arm performs its
check-and-consume under the store lock and calls it afterwards, before the rail call. On a
denial the callback runs after the check that produced the denial, so a runner that pauses
executors at their yield point never waits on an arm that denied.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable

from tasdiq.records import BINDING_FIELDS, ConsentRecord, Decision, ExecutionRequest, fingerprint

if TYPE_CHECKING:
    from tasdiq.clock import SimClock
    from tasdiq.payees import PayeeDirectory
    from tasdiq.rail import BankRail
    from tasdiq.store import ConsentStore, IdempotencyLedger
    from tasdiq.surface import ConfirmationSurface


YieldPoint = Callable[[], Any]

NO_EXPIRY_S: int = 10**12
"""Validity window used by arms whose approvals never expire."""

SNAPSHOT_FIELDS: tuple[str, ...] = (
    "cr_id",
    "principal_id",
    "session_id",
    "action_type",
    "payee_id",
    "amount_minor",
    "currency",
    "total_max_minor",
    "revision",
    "issued_at",
    "expires_at",
    "allowed_uses",
    "uses",
    "status",
    "supersedes",
    "confirmation_event_id",
    "binding_hash",
    "surface_hash",
    "lang_shown",
)

DRAFT_FIELDS: tuple[str, ...] = (
    "principal_id",
    "session_id",
    "action_type",
    "payee_id",
    "amount_minor",
    "currency",
    "total_max_minor",
    "revision",
    "allowed_uses",
    "supersedes",
)

TERMINAL_STATUSES: frozenset[str] = frozenset({"expired", "revoked", "superseded", "consumed"})

REASON_VOCABULARY: frozenset[str] = frozenset(
    {
        "no_consent",
        "expired",
        "revoked",
        "superseded",
        "consumed",
        "duplicate_replayed",
        "duplicate_in_flight",
        "key_payload_mismatch",
        "cap_exceeded",
        "batch_scope",
        "not_allowlisted",
        "over_cap",
        "session_not_elevated",
        "schema_invalid",
        "refuse_all",
        "trust_claim",
        "policy_expansion",
        "already_executed",
    }
)
REASON_STEMS: tuple[str, ...] = ("field_mismatch:", "ambiguous:")


class YieldOnce:
    """Wraps a yield-point callback so that it runs at most once.

    Calling the object runs the callback the first time only; ensure() runs it when it has
    not run yet, so an on_execute path that denied before reaching its designated yield
    point still yields exactly once. A None callback makes every call a no-op.
    """

    def __init__(self, yield_point: YieldPoint | None) -> None:
        self._fn = yield_point
        self.called: bool = False

    def __call__(self) -> None:
        if self.called or self._fn is None:
            return
        self.called = True
        self._fn()

    def ensure(self) -> None:
        """Run the callback when it has not run yet."""
        self()


@dataclass
class Session:
    """Everything one simulated agent session shares with the monitor.

    lang is the language of the principal's most recent utterance and is maintained by the
    runner; intent is the scenario's IntentState handed to the confirmation surface (None
    means the principal intends no payment); meta holds run-level fields (code_hash, model
    identity) merged into every log line.
    """

    session_id: str
    principal_id: str
    store: "ConsentStore"
    ledger: "IdempotencyLedger"
    rail: "BankRail"
    surface: "ConfirmationSurface"
    directory: "PayeeDirectory"
    clock: "SimClock"
    log: list[dict] = field(default_factory=list)
    lang: str = "en"
    intent: Any = None
    scenario_id: str = ""
    variant_id: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class UserEvent:
    """A revision or cancellation stated by the principal on the consent channel.

    kind is "revise" or "cancel"; cr_id names the record the event refers to, or None for
    the session's most recent active record; fields carries the revision values (payee_id,
    amount_minor, currency, total_max_minor) for a revision; lang is the language of the
    utterance; event_id identifies the event in the logs.
    """

    kind: str
    cr_id: str | None = None
    fields: dict = field(default_factory=dict)
    lang: str | None = None
    event_id: str = ""


class NoIntent:
    """Intent state of a principal who intends no payment."""

    def intended_at(self, t: int) -> None:
        return None


def config_hash(name: str, config: dict) -> str:
    """Return the sha256 of the canonical JSON of an arm name and its configuration."""
    payload = json.dumps({"name": name, "config": config}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_decision(
    verdict: str,
    reason: str,
    consent_id: str | None = None,
    receipt: dict | None = None,
    draft: ConsentRecord | None = None,
    details: dict | None = None,
) -> Decision:
    """Construct a Decision with every field set explicitly."""
    return Decision(
        verdict=verdict,
        reason=reason,
        consent_id=consent_id,
        receipt=receipt,
        draft=draft,
        details=dict(details or {}),
    )


def snapshot_record(record: Any) -> dict | None:
    """Return a JSON-ready snapshot of a consent record, or None."""
    if record is None:
        return None
    return {f: getattr(record, f, None) for f in SNAPSHOT_FIELDS}


def snapshot_request(req: Any) -> dict:
    """Return a JSON-ready snapshot of an execution request."""
    if dataclasses.is_dataclass(req) and not isinstance(req, type):
        return dataclasses.asdict(req)
    if isinstance(req, dict):
        return dict(req)
    return dict(vars(req))


def snapshot_event(event: Any) -> dict:
    """Return a JSON-ready snapshot of a user event given as a dataclass, object or mapping."""
    if dataclasses.is_dataclass(event) and not isinstance(event, type):
        return dataclasses.asdict(event)
    if isinstance(event, dict):
        return dict(event)
    return {k: v for k, v in vars(event).items() if not k.startswith("_")}


def event_attr(event: Any, name: str, default: Any = None) -> Any:
    """Read a field of a user event given as a dataclass, object or mapping."""
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def receipt_to_dict(receipt: Any) -> dict | None:
    """Return a receipt as a plain dictionary, or None."""
    if receipt is None:
        return None
    if isinstance(receipt, dict):
        return dict(receipt)
    if dataclasses.is_dataclass(receipt) and not isinstance(receipt, type):
        return dataclasses.asdict(receipt)
    return dict(vars(receipt))


def first_mismatch(record: Any, candidate: Any, fields: Iterable[str] = BINDING_FIELDS) -> str | None:
    """Return the first field in fields on which record and candidate differ, or None."""
    for f in fields:
        if getattr(record, f, None) != getattr(candidate, f, None):
            return f
    return None


def decline_reason(event: Any) -> tuple[str, str]:
    """Map a declined surface event to a Decision reason; returns (reason, raw_surface_reason)."""
    raw = str(getattr(event, "reason", "") or "")
    if raw in REASON_VOCABULARY or any(raw.startswith(p) and len(raw) > len(p) for p in REASON_STEMS):
        return raw, raw
    if raw in BINDING_FIELDS:
        return f"field_mismatch:{raw}", raw
    return "no_consent", raw


def record_fingerprint(record: Any, session_id: str, t: int) -> str:
    """Return the request fingerprint of a consent record's own binding tuple."""
    probe = ExecutionRequest(
        req_id="",
        session_id=session_id,
        principal_id=record.principal_id,
        payee_id=record.payee_id,
        amount_minor=record.amount_minor,
        currency=record.currency,
        fee_minor=0,
        consent_ref=getattr(record, "cr_id", None),
        idempotency_key=None,
        agent_claims_user_confirmed=False,
        t=t,
    )
    return fingerprint(probe)


def known_currency(code: Any) -> bool:
    """Return True when code is an ISO 4217 code of the currency table in tasdiq.money."""
    from tasdiq.money import CURRENCIES

    return isinstance(code, str) and code in CURRENCIES


def slot_key(cr_id: str, slot: int) -> str:
    """Return the server-derived idempotency key of one use of a consent record.

    The key is sha256 over "<cr_id>:<slot>:rail", shared by the lease ledger of B5 and the
    Tasdiq gate, so that a retry of the same slot reaches the ledger under the same key
    whatever key the agent supplied.
    """
    return hashlib.sha256(f"{cr_id}:{slot}:rail".encode("utf-8")).hexdigest()


class Monitor(ABC):
    """Abstract policy arm.

    Subclasses set name, implement on_consent_request and on_execute, and override
    on_user_event when they support the consent channel. Constructor keyword arguments are
    the calibration parameters and are stored in config; config_hash is the sha256 of the
    canonical JSON of name and config.
    """

    name: str = "Monitor"
    supports_consent_channel: bool = False
    atomic_gate: bool = False
    """True when the arm's check and consume are one step under the store lock."""

    def __init__(self, **config: Any) -> None:
        self.config: dict = dict(config)
        self.config_hash: str = config_hash(self.name, self.config)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    @abstractmethod
    def on_consent_request(self, session: Session, draft: ConsentRecord, t: int) -> Decision:
        """Handle a request to confirm draft; may present it on the surface."""

    @abstractmethod
    def on_execute(self, session: Session, req: ExecutionRequest, t: int, yield_point: YieldPoint | None = None) -> Decision:
        """Gate an execution request; ALLOW means the arm called rail.execute itself.

        yield_point, when given, is called exactly once: between the check and the state
        update or execution for a non-atomic arm, after the atomic check-and-consume for an
        atomic arm, and after the failing check on a denial.
        """

    def on_user_event(self, session: Session, event: Any, t: int) -> Decision:
        """Handle a revision or cancellation from the consent channel; unsupported by default."""
        self.register(session)
        decision = make_decision("DENY", "unsupported", details={"presented": False, "executed": False})
        return self.log(session, "user_event", decision, t, event=snapshot_event(event), consent_state=None)

    def on_tick(self, t: int) -> None:
        """Advance simulated time: expire consents and purge the ledger of every known session."""
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.store.sweep(t)
            purge = getattr(session.ledger, "purge", None)
            if purge is not None:
                purge(t)

    def register(self, session: Session) -> None:
        """Remember a session so that on_tick can sweep its store."""
        with self._lock:
            self._sessions[session.session_id] = session

    def log(self, session: Session, hook: str, decision: Decision, t: int, **extra: Any) -> Decision:
        """Append one log line for a decision and return the decision unchanged."""
        line = {
            "t": t,
            "hook": hook,
            "arm": self.name,
            "config_hash": self.config_hash,
            "session_id": session.session_id,
            "principal_id": session.principal_id,
            "scenario_id": session.scenario_id,
            "variant_id": session.variant_id,
            "verdict": decision.verdict,
            "reason": decision.reason,
            "consent_id": decision.consent_id,
            "receipt": decision.receipt,
            "details": dict(decision.details),
        }
        line.update(session.meta)
        line.update(extra)
        session.log.append(line)
        return decision

    def store_get(self, session: Session, cr_id: str | None) -> ConsentRecord | None:
        """Look a record up by id, treating unknown ids as None."""
        if not cr_id:
            return None
        try:
            return session.store.get(cr_id)
        except (KeyError, ValueError):
            return None

    def ensure_draft(self, session: Session, draft: ConsentRecord) -> ConsentRecord:
        """Return the store's copy of draft, creating it when the store does not know it."""
        record = self.store_get(session, getattr(draft, "cr_id", None))
        if record is not None:
            return record
        fields = {f: getattr(draft, f, None) for f in DRAFT_FIELDS}
        fields["session_id"] = fields.get("session_id") or session.session_id
        fields["action_type"] = fields.get("action_type") or "payment"
        fields["revision"] = int(fields.get("revision") or 1)
        fields["allowed_uses"] = int(fields.get("allowed_uses") or 1)
        if fields.get("total_max_minor") is None:
            fields["total_max_minor"] = fields["amount_minor"]
        cr_id = getattr(draft, "cr_id", None)
        if cr_id:
            fields["cr_id"] = cr_id
        return session.store.create_draft(fields)

    def draft_from_request(
        self,
        session: Session,
        req: ExecutionRequest,
        revision: int = 1,
        supersedes: str | None = None,
        allowed_uses: int = 1,
        total_max_minor: int | None = None,
    ) -> ConsentRecord | None:
        """Create a draft consent record carrying the request's binding tuple.

        Returns None when the request's fields cannot form a record: the store rejects them,
        or the currency is unknown and the record could not be rendered on the surface.
        """
        if not known_currency(req.currency):
            return None
        fields = {
            "principal_id": req.principal_id,
            "session_id": session.session_id,
            "action_type": "payment",
            "payee_id": req.payee_id,
            "amount_minor": req.amount_minor,
            "currency": req.currency,
            "total_max_minor": req.amount_minor + req.fee_minor if total_max_minor is None else total_max_minor,
            "revision": revision,
            "allowed_uses": allowed_uses,
            "supersedes": supersedes,
        }
        try:
            return session.store.create_draft(fields)
        except (TypeError, ValueError):
            return None

    def draft_from_record(
        self,
        session: Session,
        old: ConsentRecord,
        changes: dict,
        allowed_uses: int | None = None,
    ) -> ConsentRecord | None:
        """Create a revision+1 draft of old with the given field changes applied.

        A changed amount without a changed cap keeps the old record's fee allowance. Returns
        None when the changes cannot form a record (the store rejects them).
        """
        fields = {f: getattr(old, f, None) for f in DRAFT_FIELDS}
        for key in ("payee_id", "amount_minor", "currency", "total_max_minor"):
            if changes.get(key) is not None:
                fields[key] = changes[key]
        if changes.get("amount_minor") is not None and changes.get("total_max_minor") is None:
            fee_allowance = max(0, int(old.total_max_minor) - int(old.amount_minor))
            fields["total_max_minor"] = int(changes["amount_minor"]) + fee_allowance
        fields["revision"] = int(old.revision) + 1
        fields["supersedes"] = old.cr_id
        fields["allowed_uses"] = int(allowed_uses if allowed_uses is not None else getattr(old, "allowed_uses", 1) or 1)
        fields["session_id"] = session.session_id
        try:
            return session.store.create_draft(fields)
        except (TypeError, ValueError):
            return None

    def present(self, session: Session, record: ConsentRecord, t: int, window_s: int) -> tuple[Decision, ConsentRecord | None, Any]:
        """Show record on the confirmation surface and confirm or revoke it accordingly.

        Returns the decision, the confirmed record (None when declined) and the surface
        event. A record that is already active is shown again and stays active when
        approved; a record in a terminal status is denied without a presentation.
        """
        status = getattr(record, "status", "draft")
        if status in TERMINAL_STATUSES:
            return make_decision("DENY", str(status), consent_id=record.cr_id, details={"presented": False}), None, None
        if status == "draft":
            record = session.store.present(record.cr_id) or self.store_get(session, record.cr_id) or record
        intent = session.intent if session.intent is not None else NoIntent()
        event = session.surface.present(record, session.lang, intent)
        event_id = str(getattr(event, "event_id", "") or "")
        base = {"presented": True, "surface_event_id": event_id, "lang_shown": session.lang, "window_s": window_s}
        if getattr(event, "decision", "") == "approved":
            if status == "active":
                base["already_active"] = True
                return make_decision("ALLOW", "confirmed", consent_id=record.cr_id, details=base), record, event
            confirmed = session.store.confirm(record.cr_id, event_id, session.lang, window_s)
            confirmed = confirmed or self.store_get(session, record.cr_id) or record
            return make_decision("ALLOW", "confirmed", consent_id=confirmed.cr_id, details=base), confirmed, event
        session.store.revoke(record.cr_id)
        reason, raw = decline_reason(event)
        base["surface_reason"] = raw
        return make_decision("DENY", reason, details=base), None, event

    def execute_on_rail(self, session: Session, req: ExecutionRequest) -> tuple[Any, Exception | None, bool]:
        """Call rail.execute and report (receipt, exception, debit_recorded).

        A debit is recorded when the rail appended a receipt for this request even though it
        raised afterwards (a lost acknowledgement).
        """
        before = len(session.rail.executions)
        try:
            receipt = session.rail.execute(req)
        except Exception as exc:
            added = list(session.rail.executions[before:])
            match = None
            for r in added:
                if (
                    getattr(r, "payee_id", None) == req.payee_id
                    and getattr(r, "amount_minor", None) == req.amount_minor
                    and getattr(r, "currency", None) == req.currency
                    and getattr(r, "idempotency_key", None) == req.idempotency_key
                ):
                    match = r
                    break
            return match, exc, match is not None
        return receipt, None, True

    def reconcile_in_flight(self, session: Session, key: str, req: ExecutionRequest) -> tuple[str, Any]:
        """Settle a ledger entry in flight under key when this request opened it.

        Returns ("foreign", None) when another request holds the entry, ("replayed", receipt)
        when the rail holds a debit under the key (the entry is completed with that receipt),
        and ("orphaned", None) when no debit exists (the entry is removed, so the request can
        execute under the same key). A request is recognised by its req_id.
        """
        owner = session.ledger.owner_of(key)
        if not req.req_id or owner != req.req_id:
            return "foreign", None
        for receipt in list(session.rail.executions):
            if getattr(receipt, "idempotency_key", None) == key:
                session.ledger.complete(key, receipt_to_dict(receipt))
                return "replayed", receipt
        session.ledger.fail(key)
        return "orphaned", None

    def run_rail(
        self,
        session: Session,
        req: ExecutionRequest,
        reason: str = "ok",
        consent_id: str | None = None,
        details: dict | None = None,
        yield_point: YieldPoint | None = None,
    ) -> Decision:
        """Execute req on the rail without further checks and build the resulting decision.

        The yield point runs immediately before the rail call: for the arms that use this
        method the admission check is complete and no state has been updated at that moment.
        """
        info = dict(details or {})
        info.setdefault("presented", False)
        if yield_point is not None:
            yield_point()
        receipt, exc, debited = self.execute_on_rail(session, req)
        if exc is None:
            info["executed"] = True
            return make_decision("ALLOW", reason, consent_id=consent_id, receipt=receipt_to_dict(receipt), details=info)
        info.update({"executed": debited, "debit_recorded": debited, "error": str(exc)})
        return make_decision("ALLOW", "rail_fault", consent_id=consent_id, receipt=None, details=info)

    def latest_active(self, session: Session) -> ConsentRecord | None:
        """Return the most recently issued active record of the session, or None."""
        try:
            actives = list(session.store.active_for(session.session_id))
        except (KeyError, ValueError):
            return None
        if not actives:
            return None
        return max(actives, key=lambda r: (int(getattr(r, "issued_at", 0) or 0), int(getattr(r, "revision", 1) or 1)))

    def target_record(self, session: Session, event: Any) -> ConsentRecord | None:
        """Return the record a user event refers to: its cr_id or the latest active record."""
        cr_id = event_attr(event, "cr_id", None)
        if cr_id:
            return self.store_get(session, cr_id)
        return self.latest_active(session)
