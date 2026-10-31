"""Bounded explicit-state exploration of one consent record under the Tasdiq arm.

The explored system is the real monitor arm with the real consent store, idempotency
ledger, bank rail and confirmation surface, driven by a scripted principal and up to two
executors. The events are confirm, revise_lower, revise_higher, revoke, tick_expire and,
per executor e, prepare(e) (the executor runs the arm's execution gate up to its yield
point: the check step), commit(e) (the executor is released and completes: the commit
step), lost_ack_retry(e) (the executor commits under a rail timeout after the debit and
then retries the same request with the same idempotency key) and crash_restart(e) (the
executor dies at its yield point and a restarted executor re-issues the same request).

States are hashed over the consent records, the ledger entries, the rail receipts, the
executors' positions and requests, the arm's unacknowledged slots, the clock and the
principal's intent. Exploration proceeds breadth-first to a depth bound; a state reached
by two paths is expanded once. Because the arms are deterministic given the state, every
path is replayed from the initial state when it is examined. The invariants I1 to I7 of
the design are checked on every reachable state and a violation carries the event path
that reaches it. I7 is exercised on every record: the binding hash of a copy with another
payee_id must differ, and the arm must admit the record's own tuple after the directory's
display names change. A liveness check is reported beside the invariants: after
crash_restart(e) of an executor whose attempt the arm had admitted (its ledger entry was
in flight), the restarted request must end ALLOW, executed or replayed. The result is a
bounded check, never a proof.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from tasdiq.clock import SimClock
from tasdiq.payees import Payee, PayeeDirectory
from tasdiq.policy.arms import ARMS, make_arm
from tasdiq.policy.base import Monitor, Session, UserEvent, slot_key
from tasdiq.rail import BankRail, FaultPlan, FeePolicy
from tasdiq.records import BINDING_FIELDS, ConsentRecord, Decision, ExecutionRequest, IntendedRecord, binding_hash, render_confirmation
from tasdiq.store import ConsentStore, IdempotencyLedger
from tasdiq.surface import ConfirmationSurface

EXECUTORS: tuple[str, ...] = ("A", "B")
EVENTS: tuple[str, ...] = (
    "confirm",
    "revise_lower",
    "revise_higher",
    "revoke",
    "tick_expire",
    "prepare",
    "commit",
    "lost_ack_retry",
    "crash_restart",
)
EXECUTOR_EVENTS: frozenset[str] = frozenset({"prepare", "commit", "lost_ack_retry", "crash_restart"})
INVARIANTS: tuple[str, ...] = ("I1", "I2", "I3", "I4", "I5", "I6", "I7")
RULES: tuple[str, ...] = ("invalidate", "narrowing", "cap_settle")

PAYEE_ID = "P-1"
PRINCIPAL = "U-1"
SESSION_ID = "S-verify"
CURRENCY = "AED"


class CrashSignal(Exception):
    """Raised inside an executor's yield point to model the executor dying there."""


@dataclass(frozen=True)
class Event:
    """One exploration event, with the executor it concerns when it has one."""

    name: str
    executor: str | None = None

    @property
    def label(self) -> str:
        return self.name if self.executor is None else f"{self.name}({self.executor})"


@dataclass
class Transition:
    """One explored transition between two hashed states."""

    source: str
    event: str
    target: str
    depth: int


@dataclass
class Violation:
    """An invariant that failed in a reachable state, with the path that reaches it."""

    invariant: str
    detail: str
    path: list[str]
    state: str
    depth: int

    def to_dict(self) -> dict[str, Any]:
        return {"invariant": self.invariant, "detail": self.detail, "path": list(self.path), "state": self.state, "depth": self.depth}


@dataclass
class ExploreResult:
    """Outcome of one bounded exploration."""

    arm: str
    rule: str
    depth: int
    executors: int
    states: int
    transitions: int
    depth_reached: int
    violations: list[Violation]
    events_taken: dict[str, int]
    states_per_depth: dict[int, int]
    executions_max: int
    elapsed_seconds: float
    complete: bool
    intent_departures: list[Violation] = field(default_factory=list)
    """Executions that depart from the intent in force; reported beside the invariants, not as violations."""
    liveness_failures: list[Violation] = field(default_factory=list)
    """Restarts after a crash of an admitted attempt that did not end ALLOW; reported beside the invariants."""
    liveness_checks: int = 0
    """Number of crash_restart transitions whose crashed attempt the arm had admitted."""

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def live(self) -> bool:
        return not self.liveness_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "rule": self.rule,
            "depth": self.depth,
            "executors": self.executors,
            "states": self.states,
            "transitions": self.transitions,
            "depth_reached": self.depth_reached,
            "complete": self.complete,
            "violations": [v.to_dict() for v in self.violations],
            "n_violations": len(self.violations),
            "events_taken": dict(self.events_taken),
            "states_per_depth": {str(k): v for k, v in sorted(self.states_per_depth.items())},
            "executions_max": self.executions_max,
            "intent_departures": [v.to_dict() for v in self.intent_departures],
            "n_intent_departures": len(self.intent_departures),
            "liveness_failures": [v.to_dict() for v in self.liveness_failures],
            "n_liveness_failures": len(self.liveness_failures),
            "liveness_checks": self.liveness_checks,
            "elapsed_seconds": self.elapsed_seconds,
            "invariants": list(INVARIANTS),
            "events": list(EVENTS),
        }


class ParkedExecutor:
    """Runs an arm's on_execute in a thread parked at the arm's yield point.

    start() runs the check step and returns once the executor is parked or finished;
    commit() releases it and waits for the decision; crash() releases it with a
    CrashSignal so that its remaining steps never run.
    """

    def __init__(self, name: str, arm: Monitor, session: Session, req: ExecutionRequest, t: int) -> None:
        self.name = name
        self.arm = arm
        self.session = session
        self.req = req
        self.t = t
        self.decision: Decision | None = None
        self.error: BaseException | None = None
        self.crashed = False
        self._crash = False
        self._reached = threading.Event()
        self._release = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def _yield(self) -> None:
        self._reached.set()
        self._release.wait()
        if self._crash:
            raise CrashSignal(self.name)

    def _run(self) -> None:
        try:
            self.decision = self.arm.on_execute(self.session, self.req, self.t, yield_point=self._yield)
        except CrashSignal:
            self.crashed = True
        except BaseException as exc:
            self.error = exc
        finally:
            self._reached.set()
            self._done.set()

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"verify-{self.name}", daemon=True)
        self._thread.start()
        self._reached.wait()

    def commit(self) -> Decision | None:
        self._release.set()
        self._done.wait()
        if self._thread is not None:
            self._thread.join()
        if self.error is not None:
            raise self.error
        return self.decision

    def crash(self) -> None:
        self._crash = True
        self._release.set()
        self._done.wait()
        if self._thread is not None:
            self._thread.join()


class ScriptedIntent:
    """The principal's current intent, changed by the exploration events."""

    def __init__(self, record: IntendedRecord | None) -> None:
        self.record = record

    def intended_at(self, t: int) -> IntendedRecord | None:
        return self.record


class ObservedRail(BankRail):
    """Bank rail that reports every debit to an observer before performing it."""

    def __init__(self, clock: SimClock, observer: Callable[[ExecutionRequest], None]) -> None:
        super().__init__(clock, FeePolicy(), FaultPlan())
        self._observer = observer

    def execute(self, req: ExecutionRequest) -> Any:
        self._observer(req)
        return super().execute(req)


class World:
    """The explored system: one consent record, the arm, the store and up to two executors."""

    def __init__(
        self,
        rule: str = "invalidate",
        arm_name: str = "Tasdiq",
        window_s: int = 600,
        amount_minor: int = 25000,
        delta_minor: int = 5000,
        executors: Sequence[str] = EXECUTORS,
    ) -> None:
        self.rule = rule
        self.arm_name = arm_name
        self.window_s = int(window_s)
        self.delta = int(delta_minor)
        self.clock = SimClock(0)
        self.directory = PayeeDirectory([Payee(payee_id=PAYEE_ID, name_en="Omar Khalid", name_ar="عمر خالد", aliases=("Omar",), kind="domestic", iban_masked="AE** **** 4471")])
        self.store = ConsentStore(self.clock)
        self.ledger = IdempotencyLedger(self.clock)
        self.rail = ObservedRail(self.clock, self._observe_execution)
        self.intent = ScriptedIntent(IntendedRecord(PAYEE_ID, int(amount_minor), CURRENCY, None))
        self.surface = ConfirmationSurface(self.clock, self.directory, "exact")
        self.arm = make_arm(arm_name, **arm_arguments(arm_name, self.window_s, rule))
        self.session = Session(
            session_id=SESSION_ID,
            principal_id=PRINCIPAL,
            store=self.store,
            ledger=self.ledger,
            rail=self.rail,
            surface=self.surface,
            directory=self.directory,
            clock=self.clock,
            lang="en",
            intent=self.intent,
            scenario_id="verify",
            variant_id=f"verify-{rule}",
        )
        draft = self.store.create_draft(
            {
                "principal_id": PRINCIPAL,
                "session_id": SESSION_ID,
                "action_type": "payment",
                "payee_id": PAYEE_ID,
                "amount_minor": int(amount_minor),
                "currency": CURRENCY,
                "total_max_minor": int(amount_minor),
                "revision": 1,
                "allowed_uses": 1,
                "supersedes": None,
            }
        )
        self.current_id: str = draft.cr_id
        self.executor_names = tuple(executors)
        self.parked: dict[str, ParkedExecutor | None] = {name: None for name in self.executor_names}
        self.issued_keys: list[str] = []
        self.attempts = 0
        self.decisions: list[dict[str, Any]] = []
        self.execution_meta: list[dict[str, Any]] = []
        self.i5_failures: list[str] = []
        self.liveness_failures: list[str] = []
        self.path: list[str] = []

    # -- helpers --------------------------------------------------------------------------

    def current(self) -> ConsentRecord:
        return self.store.get(self.current_id)

    def known_keys(self) -> list[str]:
        """Every idempotency key that may name a ledger entry: the issued client keys and the slot keys."""
        keys: list[str] = list(self.issued_keys)
        for record in self.records():
            for slot in range(int(record.allowed_uses) + 1):
                keys.append(slot_key(record.cr_id, slot))
        return sorted(set(keys))

    def admitted(self, req: ExecutionRequest) -> bool:
        """True when the arm admitted the request's attempt: a ledger entry in flight is owned by it."""
        return any(self.ledger.state_of(k) == "in_flight" and self.ledger.owner_of(k) == req.req_id for k in self.known_keys())

    def records(self) -> list[ConsentRecord]:
        return self.store.all_records()

    def record_index(self, cr_id: str | None) -> int | None:
        if cr_id is None:
            return None
        for index, record in enumerate(self.records()):
            if record.cr_id == cr_id:
                return index
        return None

    def record_of_request(self, req: ExecutionRequest) -> ConsentRecord | None:
        """The record a rail request draws on: by its server-derived key, else by its consent reference."""
        key = req.idempotency_key
        for record in self.records():
            for slot in range(int(record.allowed_uses) + 1):
                if key is not None and slot_key(record.cr_id, slot) == key:
                    return record
        if req.consent_ref:
            try:
                return self.store.get(req.consent_ref)
            except KeyError:
                return None
        return None

    def _observe_execution(self, req: ExecutionRequest) -> None:
        record = self.record_of_request(req)
        intended = self.intent.record
        meta: dict[str, Any] = {
            "t": self.clock.now(),
            "payee_id": req.payee_id,
            "amount_minor": req.amount_minor,
            "currency": req.currency,
            "idempotency_key": req.idempotency_key,
            "cr_id": None if record is None else record.cr_id,
            "status": None if record is None else record.status,
            "uses": None if record is None else record.uses,
            "allowed_uses": None if record is None else record.allowed_uses,
            "expires_at": None if record is None else record.expires_at,
            "tuple_match": record is not None and (record.payee_id, record.amount_minor, record.currency, record.principal_id) == (req.payee_id, req.amount_minor, req.currency, req.principal_id),
            "cap_match": record is not None and (record.payee_id, record.currency, record.principal_id) == (req.payee_id, req.currency, req.principal_id),
            "within_cap": record is not None and int(req.amount_minor) + int(req.fee_minor or 0) <= int(record.total_max_minor),
            "intent_match": intended is not None and (intended.payee_id, intended.amount_minor, intended.currency) == (req.payee_id, req.amount_minor, req.currency),
            "path": list(self.path),
        }
        self.execution_meta.append(meta)

    def _request_for(self, executor: str) -> ExecutionRequest:
        self.attempts += 1
        current = self.current()
        intended = self.intent.record
        payee, amount, currency = (intended.payee_id, intended.amount_minor, intended.currency) if intended is not None else (current.payee_id, current.amount_minor, current.currency)
        key = f"k-{executor}-{self.attempts}"
        self.issued_keys.append(key)
        return ExecutionRequest(
            req_id=f"R-{executor}-{self.attempts}",
            session_id=SESSION_ID,
            principal_id=PRINCIPAL,
            payee_id=payee,
            amount_minor=int(amount),
            currency=currency,
            fee_minor=0,
            consent_ref=current.cr_id,
            idempotency_key=key,
            agent_claims_user_confirmed=False,
            t=self.clock.now(),
        )

    def _record_decision(self, event: Event, decision: Decision | None, **extra: Any) -> None:
        entry = {"event": event.label, "verdict": None if decision is None else decision.verdict, "reason": None if decision is None else decision.reason, "executed": bool(decision and decision.details.get("executed")), **extra}
        self.decisions.append(entry)

    def _successor(self, old_id: str) -> str:
        """The newest record that supersedes old_id, or old_id itself when none exists."""
        for record in reversed(self.records()):
            if record.supersedes == old_id:
                return record.cr_id
        return old_id

    # -- events ----------------------------------------------------------------------------

    def enabled(self) -> list[Event]:
        """The events applicable in the current state."""
        current = self.current()
        active = [r for r in self.records() if r.status == "active"]
        events: list[Event] = []
        if current.status in ("draft", "pending"):
            events.append(Event("confirm"))
        if current.status == "active":
            if current.amount_minor > self.delta:
                events.append(Event("revise_lower"))
            events.append(Event("revise_higher"))
        if current.status in ("active", "pending"):
            events.append(Event("revoke"))
        if active:
            events.append(Event("tick_expire"))
        for name in self.executor_names:
            if self.parked[name] is None:
                events.append(Event("prepare", name))
            else:
                events.append(Event("commit", name))
                events.append(Event("lost_ack_retry", name))
                events.append(Event("crash_restart", name))
        return events

    def apply(self, event: Event) -> None:
        """Apply one event to the world."""
        self.path.append(event.label)
        t = self.clock.now()
        if event.name == "confirm":
            decision = self.arm.on_consent_request(self.session, self.current(), t)
            self._record_decision(event, decision)
        elif event.name in ("revise_lower", "revise_higher"):
            current = self.current()
            amount = current.amount_minor - self.delta if event.name == "revise_lower" else current.amount_minor + self.delta
            fields = {"payee_id": current.payee_id, "amount_minor": int(amount), "currency": current.currency}
            self.intent.record = IntendedRecord(current.payee_id, int(amount), current.currency, None)
            user_event = UserEvent(kind="revise", cr_id=current.cr_id, fields=fields, lang="en", event_id=f"rev-{len(self.path)}")
            decision = self.arm.on_user_event(self.session, user_event, t)
            self._record_decision(event, decision)
            self.current_id = self._successor(current.cr_id)
        elif event.name == "revoke":
            self.intent.record = None
            user_event = UserEvent(kind="cancel", cr_id=self.current_id, lang="en", event_id=f"can-{len(self.path)}")
            decision = self.arm.on_user_event(self.session, user_event, t)
            self._record_decision(event, decision)
        elif event.name == "tick_expire":
            horizon = max(int(r.expires_at) for r in self.records() if r.status == "active" and r.expires_at is not None)
            self.clock.set(max(horizon, t))
            self.arm.on_tick(self.clock.now())
            self._record_decision(event, None, t=self.clock.now())
        elif event.name == "prepare":
            name = str(event.executor)
            executor = ParkedExecutor(name, self.arm, self.session, self._request_for(name), t)
            executor.start()
            if executor.finished:
                self._record_decision(event, executor.decision, finished_at_check=True)
                self.parked[name] = None
            else:
                self.parked[name] = executor
        elif event.name == "commit":
            name = str(event.executor)
            executor = self.parked[name]
            assert executor is not None
            decision = executor.commit()
            self.parked[name] = None
            self._record_decision(event, decision)
        elif event.name == "lost_ack_retry":
            name = str(event.executor)
            executor = self.parked[name]
            assert executor is not None
            self.rail.faults.add(executor.req.req_id, "timeout_after_debit")
            first = executor.commit()
            self.parked[name] = None
            receipts_before = list(self.rail.executions)
            retry = self.arm.on_execute(self.session, executor.req, self.clock.now())
            receipts_after = list(self.rail.executions)
            self._record_decision(event, retry, first_verdict=None if first is None else first.verdict, first_reason=None if first is None else first.reason)
            if first is not None and first.details.get("executed") and receipts_before:
                debited = receipts_before[-1]
                receipt = retry.receipt or {}
                if len(receipts_after) != len(receipts_before):
                    self.i5_failures.append(f"retry of {executor.req.req_id} produced a second debit")
                elif retry.verdict != "ALLOW" or receipt.get("receipt_id") != debited.receipt_id:
                    self.i5_failures.append(f"retry of {executor.req.req_id} returned {retry.verdict} {retry.reason} instead of receipt {debited.receipt_id}")
        elif event.name == "crash_restart":
            name = str(event.executor)
            executor = self.parked[name]
            assert executor is not None
            admitted = self.admitted(executor.req)
            executor.crash()
            self.parked[name] = None
            restarted = self.arm.on_execute(self.session, executor.req, self.clock.now())
            liveness: str | None = None
            if admitted:
                liveness = "ok" if restarted is not None and restarted.verdict == "ALLOW" else "failed"
                if liveness == "failed":
                    verdict = None if restarted is None else f"{restarted.verdict} {restarted.reason}"
                    self.liveness_failures.append(f"restart of {executor.req.req_id} after a crash of an admitted attempt ended {verdict}")
            self._record_decision(event, restarted, crashed=executor.crashed, admitted=admitted, liveness=liveness)
        else:
            raise ValueError(f"unknown event {event.label}")

    def close(self) -> None:
        """Release every parked executor by crashing it, so that no thread outlives the world."""
        for name, executor in self.parked.items():
            if executor is not None and not executor.finished:
                executor.crash()
            self.parked[name] = None

    # -- state and invariants -----------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """A JSON-ready view of the state, the basis of the state hash."""
        records = self.records()
        index = {r.cr_id: i for i, r in enumerate(records)}
        ledger = {k: self.ledger.state_of(k) for k in self.known_keys() if self.ledger.state_of(k) is not None}
        unacked = {}
        slots_of = getattr(self.arm, "unacknowledged_slots", None)
        if callable(slots_of):
            for record in records:
                slots = slots_of(record.cr_id)
                if slots:
                    unacked[index[record.cr_id]] = list(slots)
        return {
            "t": self.clock.now(),
            "current": index.get(self.current_id),
            "intent": None if self.intent.record is None else [self.intent.record.payee_id, self.intent.record.amount_minor, self.intent.record.currency],
            "records": [
                {
                    "i": index[r.cr_id],
                    "status": r.status,
                    "uses": r.uses,
                    "allowed_uses": r.allowed_uses,
                    "payee_id": r.payee_id,
                    "amount_minor": r.amount_minor,
                    "currency": r.currency,
                    "total_max_minor": r.total_max_minor,
                    "revision": r.revision,
                    "supersedes": index.get(r.supersedes) if r.supersedes else None,
                    "issued_at": r.issued_at,
                    "expires_at": r.expires_at,
                }
                for r in records
            ],
            "ledger": ledger,
            "receipts": [{"i": index.get(m["cr_id"]), "payee_id": m["payee_id"], "amount_minor": m["amount_minor"], "currency": m["currency"]} for m in self.execution_meta],
            "executors": {
                name: None
                if executor is None
                else {"payee_id": executor.req.payee_id, "amount_minor": executor.req.amount_minor, "currency": executor.req.currency, "consent_ref": index.get(executor.req.consent_ref), "key": executor.req.idempotency_key}
                for name, executor in self.parked.items()
            },
            "unacknowledged": unacked,
        }

    def state_hash(self) -> str:
        payload = json.dumps(self.state(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def intent_departures(self) -> list[str]:
        """Executions whose tuple differs from the principal's intent in force at execution time (informational)."""
        return [f"execution {n} of {m['amount_minor']} {m['currency']} to {m['payee_id']} departs from the intent in force" for n, m in enumerate(self.execution_meta, start=1) if not m["intent_match"]]

    def renamed_directory(self) -> PayeeDirectory:
        """A copy of the directory whose display names differ from the originals."""
        return PayeeDirectory([dataclasses.replace(p, name_en=f"{p.name_en} (other spelling)", name_ar=f"{p.name_ar} المعدل") for p in self.directory.all()])

    def binding_admits(self, record: ConsentRecord, payee_id: str, directory: PayeeDirectory) -> bool:
        """Whether the arm's binding check admits the record's tuple with the given payee under a directory."""
        probe = ExecutionRequest(req_id="probe", session_id=SESSION_ID, principal_id=record.principal_id, payee_id=payee_id, amount_minor=record.amount_minor, currency=record.currency, fee_minor=0, consent_ref=record.cr_id, idempotency_key=None, agent_claims_user_confirmed=False, t=self.clock.now())
        check = getattr(self.arm, "binding_problem", None)
        original = self.session.directory
        self.session.directory = directory
        try:
            if callable(check):
                return check(self.session, record, probe) is None
            candidate = dataclasses.replace(record, payee_id=payee_id)
            return binding_hash(candidate) == record.binding_hash
        finally:
            self.session.directory = original

    def i7_failures(self, records: Sequence[ConsentRecord]) -> list[tuple[str, str]]:
        """I7 on every record: payee_id changes alter the binding, display name changes do not."""
        failures: list[tuple[str, str]] = []
        renamed = self.renamed_directory()
        for record in records:
            if binding_hash(record) != record.binding_hash:
                failures.append(("I7", f"stored binding hash of {record.cr_id} differs from its fields"))
            other = dataclasses.replace(record, payee_id=f"{record.payee_id}-other")
            if binding_hash(other) == record.binding_hash:
                failures.append(("I7", f"a payee_id change leaves the binding of {record.cr_id} unchanged"))
            if render_confirmation(record, renamed, "en") == render_confirmation(record, self.directory, "en"):
                failures.append(("I7", f"the confirmation text of {record.cr_id} does not show the display name"))
            if not self.binding_admits(record, record.payee_id, renamed):
                failures.append(("I7", f"the binding of {record.cr_id} changes with the display name"))
            if self.binding_admits(record, f"{record.payee_id}-other", self.directory):
                failures.append(("I7", f"the binding of {record.cr_id} admits another payee_id"))
        return failures

    def check_invariants(self) -> list[tuple[str, str]]:
        """Check I1 to I7 in the current state and return (invariant, detail) pairs that fail."""
        failures: list[tuple[str, str]] = []
        records = self.records()
        by_id = {r.cr_id: r for r in records}
        per_record: Counter[str] = Counter()
        for n, meta in enumerate(self.execution_meta, start=1):
            cr_id = meta["cr_id"]
            if cr_id is None:
                failures.append(("I1", f"execution {n} draws on no consent record"))
                continue
            per_record[cr_id] += 1
            bound = meta["cap_match"] if self.rule == "cap_settle" else meta["tuple_match"]
            if not bound:
                failures.append(("I1", f"execution {n} differs from the binding of {cr_id}"))
            if meta["status"] not in ("active", "consumed"):
                failures.append(("I1", f"execution {n} found {cr_id} in status {meta['status']}"))
            if meta["status"] == "consumed" and int(meta["uses"] or 0) > per_record[cr_id]:
                failures.append(("I1", f"execution {n} reused a consumption of {cr_id}"))
            if not meta["within_cap"]:
                failures.append(("I1", f"execution {n} exceeds the cap of {cr_id}"))
            if meta["status"] in ("revoked", "expired") or (meta["expires_at"] is not None and meta["t"] >= int(meta["expires_at"])):
                failures.append(("I3", f"execution {n} after revocation or expiry of {cr_id}"))
            if meta["status"] == "superseded":
                failures.append(("I4", f"execution {n} under the superseded revision {cr_id}"))
        history = self.store.history()
        for record in records:
            if record.uses > record.allowed_uses:
                failures.append(("I2", f"{record.cr_id} has {record.uses} uses for {record.allowed_uses} allowed"))
            consumes = sum(1 for h in history if h["cr_id"] == record.cr_id and h["reason"] == "consume")
            releases = sum(1 for h in history if h["cr_id"] == record.cr_id and h["reason"] == "release" and h["kind"] == "transition")
            if consumes - releases > record.allowed_uses:
                failures.append(("I2", f"{record.cr_id} consumed {consumes - releases} times"))
            if per_record[record.cr_id] > record.allowed_uses:
                failures.append(("I6", f"{record.cr_id} executed {per_record[record.cr_id]} times for {record.allowed_uses} allowed"))
        failures.extend(self.i7_failures(records))
        for detail in self.i5_failures:
            failures.append(("I5", detail))
        if "payee_id" not in BINDING_FIELDS or any(name in BINDING_FIELDS for name in ("name_en", "name_ar", "display_name")):
            failures.append(("I7", "binding fields include a display name or omit payee_id"))
        seen: set[tuple[str, str]] = set()
        unique: list[tuple[str, str]] = []
        for item in failures:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique


def arm_arguments(arm_name: str, window_s: int, rule: str) -> dict[str, Any]:
    """Constructor arguments of an arm for the exploration, chosen by the constructor's signature."""
    params = inspect.signature(ARMS[arm_name].__init__).parameters
    kwargs: dict[str, Any] = {}
    if "window_s" in params:
        kwargs["window_s"] = int(window_s)
    if "interval_s" in params:
        kwargs["interval_s"] = int(window_s)
    if "revision_rule" in params:
        kwargs["revision_rule"] = rule
    return kwargs


def replay(path: Iterable[Event], **world_kwargs: Any) -> World:
    """Build a world and apply the events of a path in order."""
    world = World(**world_kwargs)
    try:
        for event in path:
            world.apply(event)
    except BaseException:
        world.close()
        raise
    return world


def explore(
    rule: str = "invalidate",
    depth: int = 10,
    arm_name: str = "Tasdiq",
    window_s: int = 600,
    executors: Sequence[str] = EXECUTORS,
    log: Callable[[str], None] | None = None,
) -> ExploreResult:
    """Explore every event path up to the depth bound and check the invariants on each state."""
    started = time.perf_counter()
    kwargs = {"rule": rule, "arm_name": arm_name, "window_s": window_s, "executors": tuple(executors)}
    initial = World(**kwargs)
    root = initial.state_hash()
    root_failures = initial.check_invariants()
    initial.close()
    seen: dict[str, int] = {root: 0}
    frontier: list[tuple[list[Event], str]] = [([], root)]
    transitions = 0
    events_taken: Counter[str] = Counter()
    violations: list[Violation] = []
    departures: list[Violation] = []
    liveness: list[Violation] = []
    liveness_checks = 0
    flagged: set[tuple[str, str]] = set()
    states_per_depth: Counter[int] = Counter({0: 1})
    executions_max = 0
    depth_reached = 0
    for invariant, detail in root_failures:
        violations.append(Violation(invariant, detail, [], root, 0))
        flagged.add((invariant, root))
    for level in range(int(depth)):
        next_frontier: list[tuple[list[Event], str]] = []
        for path, _ in frontier:
            parent = replay(path, **kwargs)
            events = parent.enabled()
            parent.close()
            for event in events:
                child_path = path + [event]
                world = replay(child_path, **kwargs)
                try:
                    digest = world.state_hash()
                    failures = world.check_invariants()
                    departed = world.intent_departures()
                    executions_max = max(executions_max, len(world.rail.executions))
                finally:
                    world.close()
                for detail in departed:
                    if ("intent", digest) not in flagged:
                        flagged.add(("intent", digest))
                        departures.append(Violation("intent", detail, [e.label for e in child_path], digest, level + 1))
                if event.name == "crash_restart" and world.decisions and world.decisions[-1].get("admitted"):
                    liveness_checks += 1
                    if world.decisions[-1].get("liveness") == "failed" and ("liveness", digest) not in flagged:
                        flagged.add(("liveness", digest))
                        liveness.append(Violation("liveness", world.liveness_failures[-1], [e.label for e in child_path], digest, level + 1))
                transitions += 1
                events_taken[event.name] += 1
                for invariant, detail in failures:
                    if (invariant, digest) not in flagged:
                        flagged.add((invariant, digest))
                        violations.append(Violation(invariant, detail, [e.label for e in child_path], digest, level + 1))
                if digest not in seen:
                    seen[digest] = level + 1
                    states_per_depth[level + 1] += 1
                    next_frontier.append((child_path, digest))
        if next_frontier:
            depth_reached = level + 1
        if log:
            log(f"explore {arm_name} {rule}: depth {level + 1}, {len(seen)} states, {transitions} transitions, {len(violations)} violations")
        frontier = next_frontier
        if not frontier:
            break
    elapsed = time.perf_counter() - started
    return ExploreResult(
        arm=arm_name,
        rule=rule,
        depth=int(depth),
        executors=len(tuple(executors)),
        states=len(seen),
        transitions=transitions,
        depth_reached=depth_reached,
        violations=violations,
        events_taken=dict(events_taken),
        states_per_depth=dict(states_per_depth),
        executions_max=executions_max,
        elapsed_seconds=elapsed,
        complete=not frontier,
        intent_departures=departures,
        liveness_failures=liveness,
        liveness_checks=liveness_checks,
    )


def explore_rules(
    depth: int = 10,
    rules: Sequence[str] = RULES,
    arm_name: str = "Tasdiq",
    window_s: int = 600,
    log: Callable[[str], None] | None = None,
) -> list[ExploreResult]:
    """Run the exploration once per revision rule."""
    return [explore(rule, depth, arm_name, window_s, log=log) for rule in rules]


def verify_settings(cfg: Any) -> tuple[int, list[str], int, list[str]]:
    """The (depth, rules, window, arms) a configuration asks the exploration to cover."""
    depth = int(cfg.get("explore_depth", 10) or 10)
    rules = [str(r) for r in (cfg.get("revision_rules") or RULES)]
    window = int((cfg.get("windows") or [600])[0])
    arms = [str(a) for a in (cfg.get("verify_arms") or ["Tasdiq", "TasdiqNonAtomic"])]
    return depth, rules, window, arms


def verify_counts(cfg: Any) -> dict[str, Any]:
    """Count the explorations (one per arm and rule) a configuration asks for; nothing is run."""
    from tasdiq.runners.config import RunConfig, load_config

    if not isinstance(cfg, RunConfig):
        cfg = load_config(cfg)
    depth, rules, window, arms = verify_settings(cfg)
    explorations = [{"arm": arm, "rule": rule} for arm in arms for rule in rules]
    return {
        "track": "verify",
        "run_id": cfg.run_id,
        "depth": depth,
        "window_s": window,
        "items": len(explorations),
        "pending": len(explorations),
        "models": {},
        "explorations": explorations,
    }


def run_verify(cfg: Any, log: Callable[[str], None] | None = print) -> Path:
    """Run the bounded exploration described by a configuration and write results/verify/<run_id>/explore.json.

    Reads explore_depth (default 10), revision_rules, windows (the first value) and
    verify_arms (default Tasdiq and TasdiqNonAtomic, the ablation that the exploration
    is expected to fault) from the configuration. The manifest records the wall-clock
    seconds of the run (elapsed_seconds, with started and ended as epoch seconds) and the
    peak resident set size of the process in bytes (peak_rss_bytes, through psutil).
    """
    from tasdiq.runners.config import RunConfig, load_config, write_json
    from tasdiq.runners.workers import peak_rss_bytes

    if not isinstance(cfg, RunConfig):
        cfg = load_config(cfg)
    run_dir = cfg.results_dir / "verify" / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    depth, rules, window, arms = verify_settings(cfg)
    results: list[ExploreResult] = []
    started = time.time()
    for arm in arms:
        results.extend(explore_rules(depth, rules, arm, window, log=log))
    ended = time.time()
    peak = peak_rss_bytes()
    document = {
        "track": "verify",
        "run_id": cfg.run_id,
        "config_hash": cfg.config_hash,
        "code_hash": cfg.code_hash,
        "depth": depth,
        "window_s": window,
        "results": [r.to_dict() for r in results],
        "elapsed_seconds": ended - started,
        "peak_rss_bytes": peak,
    }
    write_json(run_dir / "explore.json", document)
    write_json(
        run_dir / "manifest.json",
        {
            "track": "verify",
            "run_id": cfg.run_id,
            "config": cfg.to_dict(),
            "environment": cfg.environment(),
            "workers": 1,
            "peak_rss_bytes": peak,
            "n_items": len(results),
            "elapsed_seconds": ended - started,
            "started": started,
            "ended": ended,
        },
    )
    if log:
        for r in results:
            log(f"verify: {r.arm} {r.rule} depth {r.depth}: {r.states} states, {r.transitions} transitions, {len(r.violations)} violations, {len(r.liveness_failures)} liveness failures in {r.liveness_checks} checks")
    return run_dir
