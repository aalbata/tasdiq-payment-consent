"""Deterministic trace runner.

For every benchmark variant, arm, validity window and revision rule, the runner builds a
simulated session from the seed scenario (payee directory, clock, consent store, idempotency
ledger, bank rail with the scenario's faults, confirmation surface with the configured
reader) and drives the scenario's events in time order through the arm: scripted
confirmations become consent requests, revisions and cancellations become consent-channel
events, executions and retries become execution requests. The capture stage is either the
oracle (the variant's intended record, mode "oracle") or a stored capture output (mode
"captured"); the record it yields replaces the scenario's scripted record wherever the agent
would present or execute it.

Race events (two executors issuing requests at the same time) are run once per interleaving
of the reference policy. Each executor runs the arm's on_execute in its own thread, parked
at the arm's yield point until the runner releases it, so the order of checks and commits is
fixed by the runner and the outcome is deterministic; the threads carry no concurrency of
their own.

Every decision, execution, presentation and user event is written to
results/trace/<run_id>/<arm>.jsonl with scenario_id, variant_id, arm, config_hash and
code_hash; the reference labels of every session are written to sessions.jsonl and
aggregated per arm, window, rule, family and language. Every call of a monitor hook
(on_consent_request, on_execute, on_user_event) is timed with time.perf_counter; the
wall-clock seconds are written as monitor_seconds into the decision lines the hook logged
and collected per session, while simulated time is untouched.
"""

from __future__ import annotations

import inspect
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from tasdiq.analysis.metrics import enforcement_summary_by
from tasdiq.bench.render import Variant, build_all_variants
from tasdiq.bench.schema import (
    Ambiguous as SeedAmbiguous,
    IntendedBatch,
    IntendedRecord as SeedRecord,
    ScenarioSpec,
    load_seeds,
)
from tasdiq.bench.split import assign_splits
from tasdiq.clock import SimClock
from tasdiq.payees import Payee, PayeeDirectory
from tasdiq.policy.arms import ARMS, make_arm
from tasdiq.policy.base import Monitor, Session, UserEvent
from tasdiq.rail import BankRail, FaultPlan, FeePolicy
from tasdiq.records import ConsentRecord, Decision, ExecutionRequest, IntendedRecord
from tasdiq.reference import RACE_INTERLEAVINGS, label_race, label_session
from tasdiq.runners.config import RunConfig, append_jsonl, load_config, make_config, prepare_run_dir, read_jsonl, write_json
from tasdiq.runners.workers import Progress, make_key, plan_workers, run_in_workers, worker_state
from tasdiq.store import ConsentStore, IdempotencyLedger
from tasdiq.surface import ConfirmationSurface, IntentTimeline

Tuple3 = tuple[str, int, str]

CAPTURE_MODES: tuple[str, ...] = ("oracle", "captured")
FIXED_WINDOW_ARMS: tuple[str, ...] = ("B3",)
"""Arms whose window is a calibration of their own and is not swept."""
DEFAULT_LABEL_WINDOW_S: int = 600
"""Reference validity window W used to label arms that keep no window of their own (B0, B1, B2, B6, B7)."""
SWEPT_PARAMETERS: tuple[str, ...] = ("window_s", "interval_s", "revision_rule")
"""Constructor parameters set by the sweep; arm_params may not pin them for an arm the sweep covers."""
CALIBRATION_RULES: dict[str, str] = {"scenario_payees": "allowlist", "first_intended_payee_with_amount_cap": "initial_policy"}
"""Named calibration rules an arm_params value may carry, each with the parameter it applies to; resolved per scenario in arm_kwargs."""
AGGREGATE_KEYS: tuple[tuple[str, ...], ...] = (
    ("arm", "window_s", "rule"),
    ("arm", "window_s", "rule", "family"),
    ("arm", "window_s", "rule", "lang"),
    ("arm", "window_s", "rule", "family", "lang"),
)

_CONFIRM_REF = re.compile(r"confirmation at t=(\d+)")
_BATCH_REF = re.compile(r"batch item (\d+) of (\d+)")
_CLAIM_WORDS = re.compile(r"claim|approv|agree|accept|agent's own|treats|renewed", re.IGNORECASE)


class CrashSignal(Exception):
    """Raised inside an executor's yield point to model the executor process dying there."""


class GatedExecutor:
    """One executor of a race: runs an arm's on_execute in a thread parked at the yield point.

    start() runs the arm until its yield point (the check step) and returns once the executor
    is parked or finished; commit() releases it and waits for the decision (the commit step);
    crash() releases it with a CrashSignal so that the arm never reaches its rail call. The
    thread exists only to hold the arm's call frame between the two steps; the runner fixes
    the order of every step, so a race run is deterministic.
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
        self.seconds: float = 0.0
        """Wall-clock seconds the arm's on_execute spent running, excluding the time parked at the yield point."""
        self._parked_seconds = 0.0
        self._crash = False
        self._reached = threading.Event()
        self._release = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def _yield(self) -> None:
        self._reached.set()
        parked_from = time.perf_counter()
        self._release.wait()
        self._parked_seconds += time.perf_counter() - parked_from
        if self._crash:
            raise CrashSignal(self.name)

    def _run(self) -> None:
        started = time.perf_counter()
        try:
            self.decision = self.arm.on_execute(self.session, self.req, self.t, yield_point=self._yield)
        except CrashSignal:
            self.crashed = True
        except BaseException as exc:
            self.error = exc
        finally:
            self.seconds = max(0.0, time.perf_counter() - started - self._parked_seconds)
            self._reached.set()
            self._done.set()

    @property
    def started(self) -> bool:
        return self._thread is not None

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    @property
    def parked(self) -> bool:
        return self.started and self._reached.is_set() and not self._done.is_set()

    def start(self) -> None:
        """Run the check step: the arm proceeds until its yield point and parks there."""
        if self._thread is not None:
            raise RuntimeError(f"executor {self.name} already started")
        self._thread = threading.Thread(target=self._run, name=f"executor-{self.name}", daemon=True)
        self._thread.start()
        self._reached.wait()

    def commit(self) -> Decision | None:
        """Run the commit step: release the parked arm and wait for its decision."""
        if self._thread is None:
            raise RuntimeError(f"executor {self.name} not started")
        self._release.set()
        self._done.wait()
        self._thread.join()
        if self.error is not None:
            raise self.error
        return self.decision

    def crash(self) -> None:
        """Release the parked arm with a CrashSignal, so that its remaining steps never run."""
        if self._thread is None:
            raise RuntimeError(f"executor {self.name} not started")
        self._crash = True
        self._release.set()
        self._done.wait()
        self._thread.join()


def run_interleaving(order: str, executors: Mapping[str, GatedExecutor]) -> list[tuple[str, str]]:
    """Run two gated executors through one interleaving named as in RACE_INTERLEAVINGS.

    The order lists steps "<name>-check" and "<name>-commit" separated by commas; a check
    step starts the executor and a commit step commits it. Returns the steps performed.
    """
    steps: list[tuple[str, str]] = []
    for step in order.split(","):
        name, kind = step.strip().rsplit("-", 1)
        executor = executors[name]
        if kind == "check":
            executor.start()
        elif kind == "commit":
            executor.commit()
        else:
            raise ValueError(f"unknown race step {step!r}")
        steps.append((name, kind))
    return steps


# ----------------------------------------------------------------------------------------
# Session construction
# ----------------------------------------------------------------------------------------


def directory_for(spec: ScenarioSpec) -> PayeeDirectory:
    """Build the payee directory of a scenario from its payee entries (or bare identifiers)."""
    payees: list[Payee] = []
    if spec.payee_entries:
        for entry in spec.payee_entries:
            payees.append(
                Payee(
                    payee_id=entry.payee_id,
                    name_en=entry.name_en or entry.payee_id,
                    name_ar=entry.name_ar or entry.name_en or entry.payee_id,
                    aliases=tuple(entry.aliases),
                    kind=entry.kind or "domestic",
                    iban_masked=entry.iban_masked or "",
                )
            )
    else:
        for payee_id in spec.payees:
            payees.append(Payee(payee_id=payee_id, name_en=payee_id, name_ar=payee_id))
    return PayeeDirectory(payees)


def public_record(value: Any) -> Any:
    """Convert a seed record to the records.IntendedRecord the surface compares; batches and None pass through."""
    if value is None or isinstance(value, SeedAmbiguous):
        return None
    if isinstance(value, IntendedBatch):
        return value
    return IntendedRecord(
        payee_id=str(value.payee_id),
        amount_minor=int(value.amount_minor),
        currency=str(value.currency),
        total_max_minor=None if getattr(value, "total_max_minor", None) is None else int(value.total_max_minor),
    )


def tuple_of(value: Any) -> Tuple3 | None:
    """Return (payee_id, amount_minor, currency) of a record-like value, or None."""
    if value is None:
        return None
    payee = value.get("payee_id") if isinstance(value, Mapping) else getattr(value, "payee_id", None)
    amount = value.get("amount_minor") if isinstance(value, Mapping) else getattr(value, "amount_minor", None)
    currency = value.get("currency") if isinstance(value, Mapping) else getattr(value, "currency", None)
    if payee is None or amount is None or currency is None:
        return None
    return (str(payee), int(amount), str(currency))


def cap_of(value: Any) -> int | None:
    """Return the stated cap of a record-like value, or None."""
    cap = value.get("total_max_minor") if isinstance(value, Mapping) else getattr(value, "total_max_minor", None)
    return None if cap is None else int(cap)


@dataclass(frozen=True)
class Override:
    """The variant's replacement of one turn's intended value (a permission-altering edit).

    The replacement is bound to the turn the edit belongs to (turn_idx, edit_id): scripted
    records of other turns keep their tuples even when they repeat the edited turn's tuple.
    """

    turn_idx: int
    base: Any
    edited: Any
    edit_id: str | None = None

    def record_map(self) -> dict[Tuple3, Tuple3 | None]:
        """Map every scripted tuple of the edited turn to the tuple the edit yields (None drops it)."""
        base_items = list(self.base.items) if isinstance(self.base, IntendedBatch) else [self.base]
        edited_record = self.edited if isinstance(self.edited, SeedRecord) else None
        edited_items = list(self.edited.items) if isinstance(self.edited, IntendedBatch) else None
        out: dict[Tuple3, Tuple3 | None] = {}
        for position, item in enumerate(base_items):
            key = tuple_of(item)
            if key is None:
                continue
            if edited_items is not None:
                out[key] = tuple_of(edited_items[position]) if position < len(edited_items) else None
            elif edited_record is not None:
                same_payee = edited_record.payee_id == item.payee_id
                only_item = len(base_items) == 1
                any_payee_match = any(edited_record.payee_id == b.payee_id for b in base_items)
                out[key] = tuple_of(edited_record) if (same_payee or only_item or (not any_payee_match and position == 0)) else None
            else:
                out[key] = None
        return out


def override_for(spec: ScenarioSpec, variant: Variant) -> Override | None:
    """Return the override an edit variant applies, or None for base and paraphrase members."""
    if variant.kind != "edit":
        return None
    turn = spec.turn(variant.turn_idx)
    return Override(turn_idx=variant.turn_idx, base=turn.intended, edited=variant.intended, edit_id=variant.edit_id)


def trace_applicable(spec: ScenarioSpec, variant: Variant) -> bool:
    """True when the scenario's script can realise the variant: every non-edit, and edits of a recorded turn."""
    override = override_for(spec, variant)
    if override is None:
        return True
    return isinstance(override.base, (SeedRecord, IntendedBatch))


def edit_is_inquiry(variant: Variant) -> bool:
    """True when an edit variant turns its turn into an inquiry.

    An inquiry edit is marked by edit_kind or action_type inquiry with no intended record,
    or by an intended value ambiguous in the action (inquiry versus approval undecidable).
    An edit ambiguous in a record field is not an inquiry edit even when tagged as one.
    """
    if variant.kind != "edit":
        return False
    intended = variant.intended
    if isinstance(intended, SeedAmbiguous):
        return intended.field == "action_type"
    if isinstance(intended, (SeedRecord, IntendedBatch)):
        return False
    return str(variant.edit_kind or "").lower() == "inquiry" or str(variant.action_type or "").lower() == "inquiry"


def required_counts(spec: ScenarioSpec, variant: Variant) -> dict[Tuple3, int]:
    """Count, per tuple, the payments the scenario's expected final records state for the variant.

    Every final record is mapped through the edit variant's record map (a record of the edited
    turn becomes the tuple the edit yields and is dropped when the edit yields none); base and
    paraphrase members keep the records as stated.
    """
    override = override_for(spec, variant)
    mapping = override.record_map() if override is not None else {}
    counts: dict[Tuple3, int] = {}
    for record in spec.expected.final_records:
        key = tuple_of(record)
        if key is None:
            continue
        if key in mapping:
            mapped = mapping[key]
            if mapped is None:
                continue
            key = mapped
        counts[key] = counts.get(key, 0) + 1
    return counts


def intent_timeline_for(spec: ScenarioSpec, variant: Variant) -> IntentTimeline:
    """Build the surface's intent timeline from the scenario, with the edit of the variant applied at its turn.

    Every change keeps the kind the scenario states (additional marks a further payment). A
    payment or batch edit replaces the turn's change by the edited record under the same
    kind; an edit that states no record or is ambiguous in a record field replaces it by no
    intent; an inquiry edit removes the turn's change, so the intent stated before the turn
    stays in force. The timeline carries the counts of required_counts as its lower bound on
    the number of payments intended per tuple.
    """
    changes: list[tuple[int, Any, str | None]] = [(c.t, public_record(c.record), c.kind) for c in spec.intent_timeline]
    override = override_for(spec, variant)
    if override is not None:
        t_turn = spec.turn(override.turn_idx).t
        removed = [c for c in changes if c[0] == t_turn]
        changes = [c for c in changes if c[0] != t_turn]
        if not edit_is_inquiry(variant):
            kind = removed[-1][2] if removed else None
            changes.append((t_turn, public_record(override.edited), kind))
    return IntentTimeline(changes, required=required_counts(spec, variant))


def first_intended_record(spec: ScenarioSpec) -> Any:
    """The intended value of the first user turn that states a record or a batch, or None."""
    for turn in spec.turns:
        if turn.speaker != "user":
            continue
        if isinstance(turn.intended, (SeedRecord, IntendedBatch)):
            return turn.intended
    return None


def first_intended_payee_policy(spec: ScenarioSpec) -> dict[str, Any]:
    """B7 calibration: the payee of the first intended record with a cap equal to its amount.

    A batch seeds the policy with every item's payee and the largest item amount as the cap;
    a scenario whose turns state no record yields an empty policy, so every payment needs an
    expansion approval. Only the scenario's own statement of the first request is read.
    """
    intended = first_intended_record(spec)
    if intended is None:
        return {"payees": [], "cap_minor": 0, "currency": None}
    if isinstance(intended, IntendedBatch):
        items = list(intended.items)
        currencies = {str(item.currency) for item in items}
        return {
            "payees": sorted({str(item.payee_id) for item in items}),
            "cap_minor": max(int(item.amount_minor) for item in items) if items else 0,
            "currency": currencies.pop() if len(currencies) == 1 else None,
        }
    return {"payees": [str(intended.payee_id)], "cap_minor": int(intended.amount_minor), "currency": str(intended.currency)}


def resolve_calibration(name: str, kwargs: dict[str, Any], spec: ScenarioSpec) -> dict[str, Any]:
    """Replace named calibration rules in an arm's parameters by their value for one scenario.

    scenario_payees (parameter allowlist) is the scenario's payee list; first_intended_payee_with_amount_cap
    (parameter initial_policy) is the policy of first_intended_payee_policy. A rule named for any other
    parameter raises ValueError.
    """
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, str) and value in CALIBRATION_RULES:
            if CALIBRATION_RULES[value] != key:
                raise ValueError(f"calibration rule {value!r} is not defined for parameter {key!r} of arm {name}")
            out[key] = list(spec.payees) if value == "scenario_payees" else first_intended_payee_policy(spec)
        else:
            out[key] = value
    return out


def arm_kwargs(name: str, window_s: int | None, rule: str | None, spec: ScenarioSpec, cfg: RunConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Return the constructor arguments of an arm for one session.

    The configured arm_params are applied first (named calibration rules are resolved per
    scenario); B2 receives the scenario's payees as its allowlist unless configured; arms
    whose constructor takes window_s or interval_s take the swept window (B3 keeps its own
    fixed window); arms that take revision_rule take the rule. An arm_params entry that pins
    a parameter the sweep sets for that arm raises ValueError, so a configured value can
    never silently replace a swept one.
    """
    cls = ARMS[name]
    params = inspect.signature(cls.__init__).parameters
    arm_params = cfg.get("arm_params", {}) if hasattr(cfg, "get") else {}
    kwargs: dict[str, Any] = resolve_calibration(name, dict((arm_params or {}).get(name, {}) or {}), spec)
    if name == "B2" and "allowlist" not in kwargs:
        kwargs["allowlist"] = list(spec.payees)
    if window_s is not None and name not in FIXED_WINDOW_ARMS:
        pinned = [k for k in ("window_s", "interval_s") if k in kwargs and k in params]
        if pinned:
            raise ValueError(f"arm_params for {name} pins {pinned} although the window is swept")
        if "window_s" in params:
            kwargs["window_s"] = int(window_s)
        if "interval_s" in params:
            kwargs["interval_s"] = int(window_s)
    if rule is not None and "revision_rule" in params:
        if "revision_rule" in kwargs:
            raise ValueError(f"arm_params for {name} pins revision_rule although the rule is swept")
        kwargs["revision_rule"] = rule
    return kwargs


def arm_takes_window(name: str) -> bool:
    params = inspect.signature(ARMS[name].__init__).parameters
    return name not in FIXED_WINDOW_ARMS and ("window_s" in params or "interval_s" in params)


def label_window_for(name: str, cfg: RunConfig | Mapping[str, Any]) -> int:
    """The reference validity window W used to label an arm that is not swept over windows.

    A fixed-window arm (B3) is labelled with its own window, read from arm_params or the
    constructor default; an arm without any window is labelled with label_window_s from the
    configuration, by default DEFAULT_LABEL_WINDOW_S.
    """
    params = inspect.signature(ARMS[name].__init__).parameters
    arm_params = cfg.get("arm_params", {}) if hasattr(cfg, "get") else {}
    own = dict((arm_params or {}).get(name, {}) or {})
    if name in FIXED_WINDOW_ARMS:
        for key in ("window_s", "interval_s"):
            if key in own:
                return int(own[key])
            if key in params and params[key].default is not inspect.Parameter.empty:
                return int(params[key].default)
    return int(cfg.get("label_window_s", DEFAULT_LABEL_WINDOW_S) or DEFAULT_LABEL_WINDOW_S)


def arm_window_of(config: Mapping[str, Any]) -> int | None:
    """The validity window an arm enforces, read from its effective constructor arguments, or None."""
    for key in ("window_s", "interval_s"):
        if config.get(key) is not None:
            return int(config[key])
    return None


def arm_takes_rule(name: str) -> bool:
    return "revision_rule" in inspect.signature(ARMS[name].__init__).parameters


@dataclass
class ScriptedConsent:
    """A consent record the runner created from a scripted or channel confirmation."""

    turn_idx: int
    t_scripted: int
    batch_item: int | None
    cr_id: str
    event_id: str | None
    tuple3: Tuple3
    approved: bool


@dataclass
class CaptureResult:
    """What the capture stage yields for one turn: a record, an ask or nothing."""

    kind: str
    record: Tuple3 | None = None
    total_max_minor: int | None = None
    ask_field: str | None = None
    source: str = "oracle"


@dataclass
class SessionOutcome:
    """Everything one simulated session produced; monitor_seconds lists the wall-clock seconds of every hook call."""

    arm_log: list[dict[str, Any]]
    reference_log: list[dict[str, Any]]
    scenario_view: dict[str, Any]
    executions: int
    presentations: int
    interleaving: str | None = None
    race_steps: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    monitor_seconds: list[float] = field(default_factory=list)


class ScenarioRun:
    """Drives one scenario through one arm for one variant."""

    def __init__(
        self,
        spec: ScenarioSpec,
        variant: Variant,
        arm_name: str,
        arm_kwargs_: Mapping[str, Any],
        window_s: int | None,
        rule: str | None,
        cfg: RunConfig | Mapping[str, Any],
        captures: Mapping[str, Mapping[str, Any]] | None = None,
        interleaving: str | None = None,
    ) -> None:
        self.spec = spec
        self.variant = variant
        self.arm_name = arm_name
        self.arm_kwargs = dict(arm_kwargs_)
        self.window_s = window_s
        self.rule = rule
        self.cfg = cfg
        self.captures = captures or {}
        self.mode = str(cfg.get("capture_mode", "oracle"))
        self.interleaving = interleaving
        self.override = override_for(spec, variant)
        self.record_map: dict[int, dict[Tuple3, Tuple3 | None]] = {self.override.turn_idx: self.override.record_map()} if self.override else {}
        """Replacement tuples keyed by the turn index the edit belongs to, then by the scripted tuple of that turn."""
        self.principal = str(cfg.get("principal_id", "U-1"))
        self.notes: list[str] = []
        self.clock = SimClock(0)
        self.directory = directory_for(spec)
        self.store = ConsentStore(self.clock)
        self.ledger = IdempotencyLedger(self.clock)
        fee_cfg = dict(cfg.get("fee_policy", {}) or {})
        self.rail = BankRail(self.clock, FeePolicy(**fee_cfg), self._fault_plan())
        self.surface = ConfirmationSurface(self.clock, self.directory, str(cfg.get("user_model", "exact")), executions=lambda: self.rail.executions)
        self.arm = make_arm(arm_name, **self.arm_kwargs)
        self.session = Session(
            session_id=f"S-{spec.seed_id}-{variant.variant_id}",
            principal_id=self.principal,
            store=self.store,
            ledger=self.ledger,
            rail=self.rail,
            surface=self.surface,
            directory=self.directory,
            clock=self.clock,
            lang=variant.lang,
            intent=intent_timeline_for(spec, variant),
            scenario_id=spec.seed_id,
            variant_id=variant.variant_id,
            meta=dict(getattr(cfg, "meta", lambda: {})()),
        )
        self.reference_log: list[dict[str, Any]] = []
        self.consents: list[ScriptedConsent] = []
        self.channel_turns: set[int] = set()
        self.covered_confirms: list[int] = []
        self.pending_revision: dict[int, tuple[str | None, dict[str, Any]]] = {}
        self.records_by_tuple: dict[Tuple3, list[str]] = {}
        self._surface_seen = 0
        self._event_counter = 0
        self.race_steps: list[tuple[str, str]] = []
        self.monitor_seconds: list[float] = []

    # -- helpers -------------------------------------------------------------------------

    def _timed(self, hook: Callable[..., Decision], *args: Any) -> Decision:
        """Call a monitor hook, time it with perf_counter and stamp monitor_seconds on the lines it logged."""
        before = len(self.session.log)
        started = time.perf_counter()
        decision = hook(*args)
        seconds = time.perf_counter() - started
        self._stamp(before, seconds)
        return decision

    def _stamp(self, before: int, seconds: float, req_id: str | None = None) -> None:
        """Record one hook timing and write it into the arm log lines appended since index before.

        With req_id, only the execute lines whose request carries that id are stamped (the
        two executors of a race log into the same session).
        """
        self.monitor_seconds.append(float(seconds))
        for line in self.session.log[before:]:
            if req_id is not None:
                request = line.get("request") or {}
                if line.get("hook") != "execute" or request.get("req_id") != req_id:
                    continue
            line["monitor_seconds"] = float(seconds)

    def _fault_plan(self) -> FaultPlan:
        """Plan a timeout_after_debit on every request whose acknowledgement the scenario loses."""
        plan = FaultPlan()
        events = list(self.spec.events)
        for index, event in enumerate(events):
            if event.kind != "lost_ack":
                continue
            payload = event.payload
            if payload.get("payee_id") is not None:
                plan.add(self._req_id(index, payload), "timeout_after_debit")
                continue
            key = payload.get("repeats")
            for earlier in range(index - 1, -1, -1):
                prior = events[earlier]
                if prior.kind in ("execute", "retry") and prior.payload.get("idempotency_key") == key:
                    plan.add(self._req_id(earlier, prior.payload), "timeout_after_debit")
                    break
        return plan

    def _req_id(self, index: int, payload: Mapping[str, Any]) -> str:
        given = payload.get("req_id")
        return str(given) if given else f"R-{self.spec.seed_id}-{index}"

    def _turn_lang(self, turn_idx: int | None) -> str:
        if turn_idx is not None:
            try:
                lang = self.spec.turn(turn_idx).lang_of_record
            except KeyError:
                lang = None
            if lang:
                return lang
        return self.variant.lang

    def _capture_for_turn(self, turn_idx: int, scripted: Mapping[str, Any]) -> CaptureResult:
        """Return what the capture stage yields for the turn that the scripted record belongs to."""
        scripted_tuple = tuple_of(scripted)
        scripted_cap = cap_of(scripted)
        if self.mode == "captured":
            key = self.variant.variant_id if turn_idx == self.variant.turn_idx else f"{self.spec.seed_id}.t{turn_idx}.{self.variant.lang}.base"
            stored = self.captures.get(key)
            if stored is not None:
                kind = str(stored.get("kind"))
                if kind == "record":
                    rec = stored.get("record") or {}
                    produced = tuple_of(rec)
                    if produced is not None:
                        cap = cap_of(rec)
                        return CaptureResult("record", produced, cap, None, "captured")
                    return CaptureResult("no_record", source="captured")
                if kind == "ask":
                    return CaptureResult("ask", ask_field=stored.get("ask_field"), source="captured")
                if kind == "parse_failure":
                    return CaptureResult("parse_failure", source="captured")
                return CaptureResult("no_record", source="captured")
            self.notes.append(f"no capture output for {key}; oracle used")
        if self.override is not None and self.override.turn_idx == turn_idx:
            edited = self.override.edited
            if isinstance(edited, SeedAmbiguous):
                return CaptureResult("ask", ask_field=edited.field)
            if edited is None:
                return CaptureResult("no_record")
            mapped = self.record_map.get(turn_idx, {}).get(scripted_tuple) if scripted_tuple is not None else None
            if mapped is None:
                return CaptureResult("no_record")
            cap = None
            if isinstance(edited, SeedRecord) and tuple_of(edited) == mapped:
                cap = edited.total_max_minor
            elif isinstance(edited, IntendedBatch):
                for item in edited.items:
                    if tuple_of(item) == mapped:
                        cap = item.total_max_minor
                cap = cap if cap is not None else edited.total_max_minor
            return CaptureResult("record", mapped, cap)
        if scripted_tuple is None:
            return CaptureResult("no_record")
        cap = scripted_cap
        if cap is None:
            cap = self._intended_cap(turn_idx, scripted_tuple)
        return CaptureResult("record", scripted_tuple, cap)

    def _intended_cap(self, turn_idx: int, key: Tuple3) -> int | None:
        """The cap the scenario's intended value states for a tuple at a turn, or None."""
        try:
            intended = self.spec.turn(turn_idx).intended
        except KeyError:
            return None
        if isinstance(intended, SeedRecord) and tuple_of(intended) == key:
            return intended.total_max_minor
        if isinstance(intended, IntendedBatch):
            for item in intended.items:
                if tuple_of(item) == key and item.total_max_minor is not None:
                    return item.total_max_minor
        return None

    def _quote_fee(self, payee_id: str, amount_minor: int, currency: str) -> int:
        try:
            payee = self.directory.by_id(payee_id)
        except (KeyError, ValueError):
            payee = None
        if payee is None:
            return 0
        try:
            return int(self.rail.quote_fee(payee, amount_minor, currency))
        except ValueError:
            return 0

    def _map_tuple(self, key: Tuple3 | None, turn_idx: int | None) -> tuple[Tuple3 | None, bool]:
        """Apply the variant's record map to a scripted tuple of the given turn; returns (tuple, dropped).

        Only the tuples scripted for the edited turn are replaced; a tuple scripted for
        another turn is returned unchanged even when it equals the edited turn's tuple.
        """
        if key is None:
            return None, False
        mapping = self.record_map.get(turn_idx) if turn_idx is not None else None
        if mapping is not None and key in mapping:
            mapped = mapping[key]
            return mapped, mapped is None
        return key, False

    def _turn_states(self, turn_idx: int, key: Tuple3) -> bool:
        """True when the intended value of a user turn states the tuple (as a record or a batch item)."""
        try:
            turn = self.spec.turn(turn_idx)
        except KeyError:
            return False
        intended = turn.intended
        if isinstance(intended, SeedRecord):
            return tuple_of(intended) == key
        if isinstance(intended, IntendedBatch):
            return any(tuple_of(item) == key for item in intended.items)
        return False

    def _consent_turn(self, prose: Any) -> int | None:
        """The turn index of the scripted confirmation a consent reference names, or None."""
        match = _CONFIRM_REF.search("" if prose is None else str(prose))
        if not match:
            return None
        t_c = int(match.group(1))
        named = [c for c in self.consents if c.t_scripted == t_c]
        return named[-1].turn_idx if named else None

    def _turn_of_tuple(self, key: Tuple3 | None, t: int, consent_ref: Any = None, before_turn: int | None = None) -> int | None:
        """The user turn a scripted tuple belongs to at time t.

        The candidates are the user turns stated at or before t (and before before_turn when
        given) whose intended value states the tuple. A single candidate is the answer; among
        several, the turn of the confirmation the consent reference names wins, else the
        latest; without any candidate the referenced confirmation's turn is used, or None.
        """
        if key is None:
            return None
        candidates: list[int] = []
        for turn in self.spec.turns:
            if turn.speaker != "user" or int(turn.t) > int(t):
                continue
            if before_turn is not None and turn.idx >= before_turn:
                continue
            if self._turn_states(turn.idx, key):
                candidates.append(turn.idx)
        referenced = self._consent_turn(consent_ref)
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            return referenced if referenced in candidates else candidates[-1]
        return referenced

    def _latest_record(self, key: Tuple3 | None) -> ConsentRecord | None:
        """The record a scripted tuple refers to: the latest record with that tuple that has not been superseded.

        A superseded record is returned only when every record with the tuple is superseded
        (its successor carries the user's current request); without any record for the
        tuple the latest active record of the session is returned.
        """
        if key is not None:
            found: list[ConsentRecord] = []
            for cr_id in reversed(self.records_by_tuple.get(key) or []):
                try:
                    found.append(self.store.get(cr_id))
                except KeyError:
                    continue
            for record in found:
                if record.status != "superseded":
                    return record
            if found:
                return found[0]
        actives = self.store.active_for(self.session.session_id)
        return actives[-1] if actives else None

    def _remember(self, record: ConsentRecord) -> None:
        self.records_by_tuple.setdefault((record.payee_id, record.amount_minor, record.currency), []).append(record.cr_id)

    def _drain_surface(self) -> None:
        """Copy new surface presentations into the reference log."""
        for entry in self.surface.log[self._surface_seen :]:
            self.reference_log.append(
                {
                    "kind": "confirm",
                    "t": entry["t"],
                    "event_id": entry["event_id"],
                    "decision": entry["decision"],
                    "reason": entry["reason"],
                    "lang": entry["lang"],
                    "source": "surface",
                    "reader": entry["reader"],
                    "simulated": True,
                    "cr_id": entry["cr_id"],
                    "record": self._record_dict(entry["cr_id"]),
                }
            )
        self._surface_seen = len(self.surface.log)

    def _record_dict(self, cr_id: str | None) -> dict[str, Any] | None:
        if not cr_id:
            return None
        try:
            record = self.store.get(cr_id)
        except KeyError:
            return None
        return {
            "action_type": "payment",
            "payee_id": record.payee_id,
            "amount_minor": record.amount_minor,
            "currency": record.currency,
            "total_max_minor": record.total_max_minor,
            "revision": record.revision,
            "allowed_uses": record.allowed_uses,
        }

    def _runner_event(self, kind: str, t: int, **fields: Any) -> None:
        """Append a runner-level event to the arm log (asks, captures, scripted events)."""
        line = {"t": t, "hook": "runner", "event": kind, "arm": self.arm.name, "config_hash": self.arm.config_hash,
                "session_id": self.session.session_id, "scenario_id": self.spec.seed_id, "variant_id": self.variant.variant_id}
        line.update(self.session.meta)
        line.update(fields)
        self.session.log.append(line)

    def _resolve_consent_ref(self, prose: Any, key: Tuple3 | None) -> tuple[str | None, str | None, bool]:
        """Resolve the scripted consent reference to (cr_id, surface event id, agent_claims_user_confirmed)."""
        text = "" if prose is None else str(prose)
        match = _CONFIRM_REF.search(text)
        if match:
            t_c = int(match.group(1))
            candidates = [c for c in self.consents if c.t_scripted == t_c and c.approved]
            batch = _BATCH_REF.search(text)
            if batch:
                item = int(batch.group(1))
                exact = [c for c in candidates if c.batch_item == item]
                candidates = exact or candidates
            if key is not None:
                same = [c for c in candidates if c.tuple3 == key]
                candidates = same or candidates
            if candidates:
                chosen = candidates[-1]
                return chosen.cr_id, chosen.event_id, False
            return None, None, False
        return None, None, bool(_CLAIM_WORDS.search(text))

    # -- event handlers -------------------------------------------------------------------

    def _handle_confirm(self, index: int, t: int, payload: Mapping[str, Any]) -> None:
        turn_idx = int(payload.get("turn_idx", 0) or 0)
        scripted = payload.get("record") or {}
        batch_item = payload.get("batch_item")
        lang = self._turn_lang(turn_idx)
        self.session.lang = lang
        if turn_idx in self.channel_turns:
            covered = [c for c in self.consents if c.turn_idx == turn_idx]
            for c in covered:
                c.t_scripted = t
            self.covered_confirms.append(index)
            self._runner_event("confirm_covered", t, turn_idx=turn_idx, covered_by="consent_channel")
            return
        capture = self._capture_for_turn(turn_idx, scripted)
        self.reference_log.append(
            {
                "kind": "capture",
                "t": t,
                "turn_idx": turn_idx,
                "record": None if capture.record is None else {"action_type": "payment", "payee_id": capture.record[0], "amount_minor": capture.record[1], "currency": capture.record[2], "total_max_minor": capture.total_max_minor},
                "ask_field": capture.ask_field,
                "source": capture.source,
            }
        )
        self._runner_event("capture", t, turn_idx=turn_idx, capture_kind=capture.kind, record=capture.record, ask_field=capture.ask_field, source=capture.source)
        if capture.kind == "ask":
            self.reference_log.append({"kind": "ask", "t": t, "field": capture.ask_field, "turn_idx": turn_idx})
            self._runner_event("ask", t, turn_idx=turn_idx, field=capture.ask_field)
            self.consents.append(ScriptedConsent(turn_idx, t, batch_item, "", None, tuple_of(scripted) or ("", 0, ""), False))
            return
        if capture.kind != "record" or capture.record is None:
            return
        payee_id, amount_minor, currency = capture.record
        cap = capture.total_max_minor
        if cap is None:
            cap = amount_minor + self._quote_fee(payee_id, amount_minor, currency)
        fields: dict[str, Any] = {
            "principal_id": self.principal,
            "session_id": self.session.session_id,
            "action_type": "payment",
            "payee_id": payee_id,
            "amount_minor": amount_minor,
            "currency": currency,
            "total_max_minor": max(cap, amount_minor),
            "revision": 1,
            "allowed_uses": 1,
            "supersedes": None,
        }
        pending = self.pending_revision.pop(turn_idx, None)
        if pending is not None:
            old_id, _ = pending
            old = None
            if old_id:
                try:
                    old = self.store.get(old_id)
                except KeyError:
                    old = None
            if old is not None:
                fields["revision"] = int(old.revision) + 1
                fields["supersedes"] = old.cr_id
        try:
            draft = self.store.create_draft(fields)
        except (TypeError, ValueError) as exc:
            self.notes.append(f"draft rejected at t={t}: {exc}")
            return
        self._remember(draft)
        decision = self._timed(self.arm.on_consent_request, self.session, draft, t)
        self._drain_surface()
        event_id = decision.details.get("surface_event_id") or decision.details.get("inherited_event_id")
        approved = decision.verdict == "ALLOW"
        self.consents.append(ScriptedConsent(turn_idx, t, batch_item, draft.cr_id, event_id, (payee_id, amount_minor, currency), approved))

    def _handle_revise(self, t: int, payload: Mapping[str, Any]) -> None:
        turn_idx = int(payload.get("turn_idx", 0) or 0)
        new_record = payload.get("record") or {}
        superseded = tuple_of(payload.get("supersedes"))
        old_tuple, _ = self._map_tuple(superseded, self._turn_of_tuple(superseded, t, before_turn=turn_idx))
        self.session.lang = self._turn_lang(turn_idx)
        capture = self._capture_for_turn(turn_idx, new_record)
        old = self._latest_record(old_tuple)
        if capture.kind == "ask":
            self.reference_log.append({"kind": "ask", "t": t, "field": capture.ask_field, "turn_idx": turn_idx})
            self._runner_event("ask", t, turn_idx=turn_idx, field=capture.ask_field)
            self.channel_turns.add(turn_idx)
            return
        if capture.kind != "record" or capture.record is None:
            self._runner_event("revision_without_record", t, turn_idx=turn_idx, capture_kind=capture.kind)
            self.channel_turns.add(turn_idx)
            return
        payee_id, amount_minor, currency = capture.record
        fields: dict[str, Any] = {"payee_id": payee_id, "amount_minor": amount_minor, "currency": currency}
        if capture.total_max_minor is not None:
            fields["total_max_minor"] = int(capture.total_max_minor)
        self.pending_revision[turn_idx] = (old.cr_id if old is not None else None, fields)
        event = UserEvent(kind="revise", cr_id=old.cr_id if old is not None else None, fields=fields, lang=self.session.lang, event_id=f"rev-{self.spec.seed_id}-{turn_idx}")
        before = len(self.surface.log)
        decision = self._timed(self.arm.on_user_event, self.session, event, t)
        self._drain_surface()
        new_id = decision.details.get("new_draft_id")
        if not new_id and decision.consent_id and (old is None or decision.consent_id != old.cr_id):
            new_id = decision.consent_id
        presented = len(self.surface.log) > before
        channel_handled = decision.reason != "unsupported" and (presented or decision.verdict == "ALLOW")
        if channel_handled:
            self.channel_turns.add(turn_idx)
            self.pending_revision.pop(turn_idx, None)
            if new_id:
                try:
                    record = self.store.get(new_id)
                    self._remember(record)
                    event_id = decision.details.get("surface_event_id") or decision.details.get("inherited_event_id") or (old.confirmation_event_id if old is not None else None)
                    self.consents.append(ScriptedConsent(turn_idx, t, None, record.cr_id, event_id, (record.payee_id, record.amount_minor, record.currency), decision.verdict == "ALLOW"))
                except KeyError:
                    pass

    def _handle_cancel(self, t: int, payload: Mapping[str, Any]) -> None:
        turn_idx = int(payload.get("turn_idx", 0) or 0)
        scripted = tuple_of(payload.get("record"))
        key, _ = self._map_tuple(scripted, self._turn_of_tuple(scripted, t, before_turn=turn_idx))
        self.session.lang = self._turn_lang(turn_idx)
        target = self._latest_record(key)
        event = UserEvent(kind="cancel", cr_id=target.cr_id if target is not None else None, lang=self.session.lang, event_id=f"can-{self.spec.seed_id}-{turn_idx}")
        self._timed(self.arm.on_user_event, self.session, event, t)
        self._drain_surface()

    def _request_from(self, index: int, t: int, payload: Mapping[str, Any]) -> ExecutionRequest | None:
        scripted = tuple_of(payload)
        turn_idx = self._turn_of_tuple(scripted, t, payload.get("consent_ref"))
        key, dropped = self._map_tuple(scripted, turn_idx)
        if key is None:
            return None
        if dropped:
            self._runner_event("execute_dropped", t, reason="the capture stage yielded no record for this tuple", scripted=scripted, turn_idx=turn_idx)
            return None
        payee_id, amount_minor, currency = key
        fee = payload.get("fee_minor")
        fee_minor = int(fee) if fee is not None else self._quote_fee(payee_id, amount_minor, currency)
        cr_id, event_id, claims = self._resolve_consent_ref(payload.get("consent_ref"), key)
        req = ExecutionRequest(
            req_id=self._req_id(index, payload),
            session_id=self.session.session_id,
            principal_id=self.principal,
            payee_id=payee_id,
            amount_minor=amount_minor,
            currency=currency,
            fee_minor=fee_minor,
            consent_ref=cr_id,
            idempotency_key=(str(payload["idempotency_key"]) if payload.get("idempotency_key") else None),
            agent_claims_user_confirmed=claims,
            t=t,
        )
        setattr(req, "consent_ref_text", payload.get("consent_ref"))
        setattr(req, "consent_event_id", event_id)
        setattr(req, "executor", payload.get("executor"))
        return req

    def _record_execution(self, req: ExecutionRequest, decision: Decision | None, t: int, before: int, crashed: bool = False) -> None:
        new_receipts = list(self.rail.executions[before:])
        event_id = getattr(req, "consent_event_id", None)
        prose = getattr(req, "consent_ref_text", None)
        consent_ref = event_id if event_id else (str(prose) if prose else None)
        base = {
            "t": t,
            "payee_id": req.payee_id,
            "amount_minor": req.amount_minor,
            "currency": req.currency,
            "fee_minor": req.fee_minor,
            "req_id": req.req_id,
            "consent_ref": consent_ref,
            "consent_cr_id": req.consent_ref,
            "executor": getattr(req, "executor", None),
            "agent_claims_user_confirmed": req.agent_claims_user_confirmed,
            "verdict": None if decision is None else decision.verdict,
            "reason": "crashed" if crashed else (None if decision is None else decision.reason),
        }
        if new_receipts:
            for receipt in new_receipts:
                self.reference_log.append({**base, "kind": "execute", "executed": True, "receipt_id": receipt.receipt_id, "idempotency_key": receipt.idempotency_key})
            return
        if decision is None:
            self.reference_log.append({**base, "kind": "crash", "executed": False, "idempotency_key": req.idempotency_key})
            return
        if decision.verdict == "DENY":
            self.reference_log.append({**base, "kind": "execute", "executed": False, "idempotency_key": req.idempotency_key})
        elif decision.reason == "duplicate_replayed":
            self.reference_log.append({**base, "kind": "replay", "executed": False, "idempotency_key": req.idempotency_key})
        else:
            self.reference_log.append({**base, "kind": "rail_fault", "executed": False, "idempotency_key": req.idempotency_key})

    def _handle_execute(self, index: int, t: int, payload: Mapping[str, Any]) -> None:
        req = self._request_from(index, t, payload)
        if req is None:
            return
        before = len(self.rail.executions)
        decision = self._timed(self.arm.on_execute, self.session, req, t)
        self._drain_surface()
        self._record_execution(req, decision, t, before)

    def _handle_race(self, group: Sequence[tuple[int, Mapping[str, Any]]], t: int) -> None:
        requests: dict[str, ExecutionRequest] = {}
        for index, payload in group:
            name = str(payload.get("executor"))
            req = self._request_from(index, t, payload)
            if req is not None:
                requests[name] = req
        if len(requests) < 2:
            for name, req in requests.items():
                before = len(self.rail.executions)
                decision = self._timed(self.arm.on_execute, self.session, req, t)
                self._drain_surface()
                self._record_execution(req, decision, t, before)
            return
        order = self.interleaving or RACE_INTERLEAVINGS[0]
        names = sorted(requests)
        mapping = {"A": names[0], "B": names[1]}
        executors = {label: GatedExecutor(label, self.arm, self.session, requests[name], t) for label, name in mapping.items()}
        recorded: set[str] = set()
        log_before = len(self.session.log)
        for step in order.split(","):
            label, kind = step.strip().rsplit("-", 1)
            executor = executors[label]
            if kind == "check":
                executor.start()
            else:
                before = len(self.rail.executions)
                decision = executor.commit()
                self._record_execution(executor.req, decision, t, before)
                recorded.add(label)
            self.race_steps.append((label, kind))
        for label in sorted(executors):
            if label not in recorded:
                executor = executors[label]
                before = len(self.rail.executions)
                self._record_execution(executor.req, executor.commit(), t, before)
        for label in sorted(executors):
            executor = executors[label]
            self._stamp(log_before, executor.seconds, req_id=executor.req.req_id)
        self._drain_surface()

    # -- main loop -----------------------------------------------------------------------

    def run(self) -> SessionOutcome:
        events = list(self.spec.events)
        index = 0
        while index < len(events):
            event = events[index]
            t = int(event.t)
            self.clock.set(t)
            self.arm.on_tick(t)
            kind = event.kind
            payload = event.payload
            if kind == "execute" and payload.get("executor") is not None:
                group: list[tuple[int, Mapping[str, Any]]] = [(index, payload)]
                look = index + 1
                while look < len(events) and events[look].kind == "execute" and int(events[look].t) == t and events[look].payload.get("executor") is not None:
                    group.append((look, events[look].payload))
                    look += 1
                if len(group) > 1:
                    self._handle_race(group, t)
                    index = look
                    continue
            if kind == "confirm":
                self._handle_confirm(index, t, payload)
            elif kind == "revise":
                self._handle_revise(t, payload)
            elif kind == "cancel":
                self._handle_cancel(t, payload)
            elif kind in ("execute", "retry"):
                self._handle_execute(index, t, payload)
            elif kind == "lost_ack":
                if payload.get("payee_id") is not None:
                    self._handle_execute(index, t, payload)
                else:
                    self._runner_event("lost_ack", t, repeats=payload.get("repeats"))
            elif kind == "untrusted_content":
                self._runner_event("untrusted_content", t, turn_idx=payload.get("turn_idx"), induces=payload.get("induces"))
            elif kind in ("delay", "lapse", "crash"):
                self._runner_event(kind, t, **{k: v for k, v in payload.items() if k in ("seconds", "step", "turn_idx")})
            index += 1
        self._drain_surface()
        return SessionOutcome(
            arm_log=list(self.session.log),
            reference_log=list(self.reference_log),
            scenario_view=self._scenario_view(),
            executions=len(self.rail.executions),
            presentations=self.surface.presentations(),
            interleaving=self.interleaving,
            race_steps=list(self.race_steps),
            notes=list(self.notes),
            monitor_seconds=list(self.monitor_seconds),
        )

    def _scenario_view(self) -> dict[str, Any]:
        """The scenario as handed to the reference, without scripted confirmations the consent channel covered."""
        view = self.spec.to_dict()
        view["events"] = [e for i, e in enumerate(view["events"]) if i not in set(self.covered_confirms)]
        return view


# ----------------------------------------------------------------------------------------
# Session enumeration and labelling
# ----------------------------------------------------------------------------------------


def has_race(spec: ScenarioSpec) -> bool:
    """True when the scenario has two execute events at one time with distinct executors."""
    by_t: dict[int, set[str]] = {}
    for event in spec.events:
        if event.kind == "execute" and event.payload.get("executor") is not None:
            by_t.setdefault(int(event.t), set()).add(str(event.payload["executor"]))
    return any(len(names) >= 2 for names in by_t.values())


def variant_dict(variant: Variant) -> dict[str, Any]:
    return variant.to_dict()


def label_outcome(spec: ScenarioSpec, variant: Variant, outcome: SessionOutcome, window_s: int) -> dict[str, Any]:
    """Compute the reference labels of one session outcome under both confirmation rules."""
    view = outcome.scenario_view
    vdict = variant_dict(variant)
    primary = label_session(view, vdict, outcome.reference_log, window_s, rule="invalidate")
    secondary = label_session(view, vdict, outcome.reference_log, window_s, rule="narrowing")
    record = primary.as_record()
    record["unnecessary_confirmations_narrowing"] = secondary.unnecessary_confirmations
    record["execution_labels"] = [lab.as_dict() for lab in primary.execution_labels]
    record["blocked_labels"] = [lab.as_dict() for lab in primary.blocked_labels]
    return record


def session_record(
    cfg: RunConfig,
    spec: ScenarioSpec,
    variant: Variant,
    arm_name: str,
    window_s: int | None,
    rule: str | None,
    split: str,
    labels: dict[str, Any],
    outcome: SessionOutcome,
    arm_config_hash: str,
    interleavings: list[str] | None = None,
    arm_config: Mapping[str, Any] | None = None,
    label_window_s: int | None = None,
) -> dict[str, Any]:
    """Assemble the session record written to sessions.jsonl.

    The identification fields are written after the labels so that the arm's revision rule
    and window take precedence over the label details (whose rule names the confirmation
    rule the labels were computed under, kept as label_rule). arm_config holds the effective
    constructor arguments of the arm, arm_window_s the window it enforces (None for an arm
    without one) and label_window_s the reference validity window W the labels used.
    monitor_seconds lists the wall-clock seconds of every monitor hook call of the session
    (timing only; simulated time is unaffected) and monitor_calls their number.
    """
    record: dict[str, Any] = dict(labels)
    record["label_rule"] = record.pop("rule", None)
    effective = dict(arm_config or {})
    record.update({
        "record": "session",
        "run_id": cfg.run_id,
        "config_hash": cfg.config_hash,
        "code_hash": cfg.code_hash,
        "scenario_id": spec.seed_id,
        "seed_id": spec.seed_id,
        "variant_id": variant.variant_id,
        "pair_id": variant.pair_id,
        "family": spec.family,
        "phenomena": list(spec.phenomena),
        "lang": variant.lang,
        "kind": variant.kind,
        "edit_kind": variant.edit_kind,
        "turn_idx": variant.turn_idx,
        "split": split,
        "arm": arm_name,
        "arm_config_hash": arm_config_hash,
        "arm_config": effective,
        "window_s": window_s,
        "arm_window_s": arm_window_of(effective),
        "label_window_s": window_s if label_window_s is None else int(label_window_s),
        "rule": rule if rule is not None else "none",
        "mode": str(cfg.get("capture_mode", "oracle")),
        "reader": str(cfg.get("user_model", "exact")),
        "rail_executions": outcome.executions,
        "presentations": outcome.presentations,
        "interleaving": outcome.interleaving,
        "interleavings": interleavings or [],
        "notes": outcome.notes,
        "monitor_seconds": [float(v) for v in outcome.monitor_seconds],
        "monitor_calls": len(outcome.monitor_seconds),
    })
    return record


@dataclass
class TraceItem:
    """One unit of trace work: a variant under one arm, window and rule."""

    seed_id: str
    variant_id: str
    arm: str
    window_s: int | None
    rule: str | None

    @property
    def key(self) -> str:
        return make_key(self.variant_id, self.arm, self.window_s, self.rule)


def enumerate_items(cfg: RunConfig, specs: Sequence[ScenarioSpec], variants: Sequence[Variant], splits: Mapping[str, str]) -> list[TraceItem]:
    """List the trace work units selected by the configuration (excluded near-duplicate variants are left out)."""
    wanted_splits = set(cfg.get("splits", ["test"]) or ["test"])
    langs = set(cfg.get("languages", ["en", "ar", "mix"]) or ["en", "ar", "mix"])
    kinds = set(cfg.get("trace_kinds", ["base", "paraphrase", "edit"]) or ["base", "paraphrase", "edit"])
    excluded = excluded_variants(cfg)
    by_seed = {s.seed_id: s for s in specs}
    items: list[TraceItem] = []
    for variant in variants:
        spec = by_seed.get(variant.seed_id)
        if spec is None or variant.dialect_extra or variant.variant_id in excluded:
            continue
        if splits.get(variant.seed_id) not in wanted_splits or variant.lang not in langs or variant.kind not in kinds:
            continue
        if not trace_applicable(spec, variant):
            continue
        for arm in cfg.get("arms", []) or []:
            windows = [int(w) for w in (cfg.get("windows", [600]) or [600])] if arm_takes_window(arm) else [label_window_for(arm, cfg)]
            rules = list(cfg.get("revision_rules", ["invalidate"]) or ["invalidate"]) if arm_takes_rule(arm) else [None]
            for window in windows:
                for rule in rules:
                    items.append(TraceItem(variant.seed_id, variant.variant_id, arm, window, rule))
    limit = cfg.get("limit")
    if limit:
        items = items[: int(limit)]
    return items


def select_specs(cfg: RunConfig, specs: Sequence[ScenarioSpec]) -> list[ScenarioSpec]:
    """Restrict the seeds to the configured seed_ids (a list, or "all")."""
    wanted = cfg.get("seed_ids", "all")
    if wanted in (None, "all", ["all"]):
        return [s for s in specs if not s.dialect_extra]
    wanted_set = {str(x) for x in wanted}
    return [s for s in specs if s.seed_id in wanted_set and not s.dialect_extra]


def excluded_variants(cfg: RunConfig) -> frozenset[str]:
    """The variant ids the near-duplicate rule excluded from the test split (data/splits.json), or none."""
    splits_path = cfg.data_dir / "splits.json"
    if not splits_path.is_file():
        return frozenset()
    import json

    with open(splits_path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    return frozenset(str(v) for v in (document.get("excluded_variants") or []))


def load_benchmark(cfg: RunConfig) -> tuple[list[ScenarioSpec], list[Variant], dict[str, str]]:
    """Load the seeds, build their variants and read or compute the seed-level splits."""
    specs = select_specs(cfg, load_seeds(cfg.seeds_dir))
    variants = build_all_variants(specs)
    splits_path = cfg.data_dir / "splits.json"
    splits: dict[str, str] = {}
    if splits_path.is_file():
        import json

        with open(splits_path, "r", encoding="utf-8") as handle:
            splits = dict(json.load(handle).get("seed_splits", {}))
    missing = [s for s in specs if s.seed_id not in splits]
    if missing:
        computed = assign_splits(specs, dev_fraction=float(cfg.get("dev_fraction", 0.25)), seed=int(cfg.get("split_seed", 20261006)))
        for seed_id, split in computed.items():
            splits.setdefault(seed_id, split)
    return specs, variants, splits


def load_captures(cfg: RunConfig) -> dict[str, dict[str, Any]]:
    """Read stored capture outputs (one JSON line per variant) keyed by variant_id for mode "captured"."""
    if str(cfg.get("capture_mode", "oracle")) != "captured":
        return {}
    path = cfg.get("capture_file")
    if not path:
        raise ValueError("capture_mode captured needs capture_file")
    file = Path(str(path))
    if not file.is_absolute():
        file = cfg.root / file
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(file):
        vid = row.get("variant_id")
        if vid:
            out[str(vid)] = row
    return out


def run_item(cfg: RunConfig, spec: ScenarioSpec, variant: Variant, item: TraceItem, split: str, captures: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one trace item (all interleavings for a race scenario) and return (session record, event lines)."""
    swept_window = item.window_s if arm_takes_window(item.arm) else None
    kwargs = arm_kwargs(item.arm, swept_window, item.rule, spec, cfg)
    window_for_labels = int(item.window_s if item.window_s is not None else label_window_for(item.arm, cfg))
    lines: list[dict[str, Any]] = []
    if has_race(spec):
        logs: dict[str, Any] = {}
        outcomes: dict[str, SessionOutcome] = {}
        for order in RACE_INTERLEAVINGS:
            run = ScenarioRun(spec, variant, item.arm, kwargs, item.window_s, item.rule, cfg, captures, interleaving=order)
            outcome = run.run()
            outcomes[order] = outcome
            logs[order] = outcome.reference_log
            for line in outcome.arm_log:
                lines.append({"record": "event", "window_s": item.window_s, "rule": item.rule, "interleaving": order, **line})
        view = outcomes[RACE_INTERLEAVINGS[0]].scenario_view
        race = label_race(view, variant_dict(variant), logs, window_for_labels, rule="invalidate")
        labels = race.as_record()
        secondary = label_race(view, variant_dict(variant), logs, window_for_labels, rule="narrowing")
        labels["unnecessary_confirmations_narrowing"] = secondary.worst_label.unnecessary_confirmations
        labels["execution_labels"] = [lab.as_dict() for lab in race.worst_label.execution_labels]
        labels["blocked_labels"] = [lab.as_dict() for lab in race.worst_label.blocked_labels]
        worst = outcomes[race.worst]
        arm_hash = run.arm.config_hash
        record = session_record(cfg, spec, variant, item.arm, item.window_s, item.rule, split, labels, worst, arm_hash, list(RACE_INTERLEAVINGS), run.arm.config, window_for_labels)
        record["interleaving"] = race.worst
        record["interleaving_labels"] = {name: {"intent_violations": [lab.intent_class for lab in lab_.intent_violations], "violations": [lab.reason for lab in lab_.violations], "executions": lab_.executions, "completed": lab_.completed} for name, lab_ in race.labels.items()}
        return record, lines
    run = ScenarioRun(spec, variant, item.arm, kwargs, item.window_s, item.rule, cfg, captures)
    outcome = run.run()
    for line in outcome.arm_log:
        lines.append({"record": "event", "window_s": item.window_s, "rule": item.rule, "interleaving": None, **line})
    labels = label_outcome(spec, variant, outcome, window_for_labels)
    record = session_record(cfg, spec, variant, item.arm, item.window_s, item.rule, split, labels, outcome, run.arm.config_hash, None, run.arm.config, window_for_labels)
    return record, lines


def _worker_benchmark(cfg: RunConfig) -> tuple[dict[str, ScenarioSpec], dict[str, Variant], dict[str, str], dict[str, dict[str, Any]]]:
    """The benchmark and stored captures of a configuration, loaded once per worker process and kept in its state."""
    state = worker_state()
    if state.get("benchmark_config_hash") != cfg.config_hash:
        specs, variants, splits = load_benchmark(cfg)
        state["benchmark"] = ({s.seed_id: s for s in specs}, {v.variant_id: v for v in variants}, splits, load_captures(cfg))
        state["benchmark_config_hash"] = cfg.config_hash
    return state["benchmark"]


def _run_batch(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entry: run the trace items of a batch and return their records and event lines."""
    cfg = make_config(payload["values"], source=Path(payload["source"]) if payload.get("source") else None)
    by_seed, by_variant, splits, captures = _worker_benchmark(cfg)
    sessions: list[dict[str, Any]] = []
    lines_by_arm: dict[str, list[dict[str, Any]]] = {}
    for raw in payload["items"]:
        item = TraceItem(**raw)
        record, lines = run_item(cfg, by_seed[item.seed_id], by_variant[item.variant_id], item, splits.get(item.seed_id, "test"), captures)
        sessions.append(record)
        lines_by_arm.setdefault(item.arm, []).extend(lines)
    return {"sessions": sessions, "lines_by_arm": lines_by_arm}


def aggregate_sessions(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate session records per arm, window, rule, family and language."""
    out: dict[str, Any] = {"n_sessions": len(sessions), "groups": []}
    for keys in AGGREGATE_KEYS:
        summaries = enforcement_summary_by(list(sessions), keys)
        for values, summary in summaries.items():
            out["groups"].append({"keys": dict(zip(keys, values)), "summary": summary.as_dict()})
    return out


def trace_counts(cfg: RunConfig | Path | str) -> dict[str, Any]:
    """Count the trace work units (sessions) of a configuration per arm, with the pending ones; nothing is run."""
    if not isinstance(cfg, RunConfig):
        cfg = load_config(cfg)
    specs, variants, splits = load_benchmark(cfg)
    items = enumerate_items(cfg, specs, variants, splits)
    progress = Progress(cfg.results_dir / "trace" / cfg.run_id / "done.jsonl")
    per_arm: dict[str, int] = {}
    for it in items:
        per_arm[it.arm] = per_arm.get(it.arm, 0) + 1
    pending = sum(1 for it in items if it.key not in progress)
    selected = {it.variant_id for it in items}
    return {
        "track": "trace",
        "run_id": cfg.run_id,
        "n_seeds": len(specs),
        "n_variants": len(variants),
        "n_variants_selected": len(selected),
        "items": len(items),
        "pending": pending,
        "models": {},
        "arms": per_arm,
        "windows": [int(w) for w in (cfg.get("windows", [600]) or [600])],
        "revision_rules": list(cfg.get("revision_rules", ["invalidate"]) or ["invalidate"]),
    }


def run_trace(cfg: RunConfig | Path | str, log: Callable[[str], None] | None = print) -> Path:
    """Run the trace track for a configuration and return the run directory.

    The run directory results/trace/<run_id> receives <arm>.jsonl (event lines),
    sessions.jsonl (one labelled record per session), aggregates.json, manifest.json and
    done.jsonl (completed keys for resumption). The pending sessions are run one seed at
    a time, so the records of a seed are written as soon as its sessions are labelled and
    a restarted run resumes at the first seed not yet written. A run without resumption
    removes the files of an earlier run of the same run_id before it writes.
    """
    if not isinstance(cfg, RunConfig):
        cfg = load_config(cfg)
    resume = bool(cfg.get("resume", True))
    run_dir = prepare_run_dir(cfg.results_dir / "trace" / cfg.run_id, resume)
    progress = Progress(run_dir / "done.jsonl") if resume else Progress(None)
    specs, variants, splits = load_benchmark(cfg)
    items = enumerate_items(cfg, specs, variants, splits)
    pending = [it for it in items if it.key not in progress]
    if log:
        log(f"trace: {len(specs)} seeds, {len(variants)} variants, {len(items)} sessions ({len(pending)} pending)")
    import time

    started = time.time()
    plan = plan_workers(cfg, log)
    workers = int(cfg.get("trace_workers", plan.workers) or 1)
    by_seed_items: dict[str, list[TraceItem]] = {}
    for it in pending:
        by_seed_items.setdefault(it.seed_id, []).append(it)
    payloads = [{"values": cfg.values, "source": cfg.source.as_posix() if cfg.source else None, "items": [it.__dict__ for it in seed_items]} for seed_items in by_seed_items.values()]
    n_done = 0
    peak = 0
    for result in run_in_workers(_run_batch, payloads, workers, plan=plan, log=log):
        peak = max(peak, int(result.get("peak_rss_bytes", 0)))
        for arm, lines in result["lines_by_arm"].items():
            append_jsonl(run_dir / f"{arm}.jsonl", lines)
        append_jsonl(run_dir / "sessions.jsonl", result["sessions"])
        for record in result["sessions"]:
            progress.mark(make_key(record["variant_id"], record["arm"], record["window_s"], None if record["rule"] == "none" else record["rule"]))
            n_done += 1
        if log:
            log(f"trace: {n_done}/{len(pending)} sessions written")
    sessions = read_jsonl(run_dir / "sessions.jsonl")
    write_json(run_dir / "aggregates.json", aggregate_sessions(sessions))
    elapsed = time.time() - started
    manifest = {
        "track": "trace",
        "run_id": cfg.run_id,
        "config": cfg.to_dict(),
        "environment": cfg.environment(),
        "workers": plan.to_dict(),
        "peak_rss_bytes": peak,
        "n_seeds": len(specs),
        "n_variants": len(variants),
        "n_items": len(items),
        "n_sessions_written": len(sessions),
        "n_skipped_inapplicable": sum(1 for v in variants if not trace_applicable(next(s for s in specs if s.seed_id == v.seed_id), v)),
        "n_excluded_variants": len(excluded_variants(cfg)),
        "splits": dict(sorted(splits.items())),
        "elapsed_seconds": elapsed,
        "started": started,
        "ended": started + elapsed,
    }
    write_json(run_dir / "manifest.json", manifest)
    if log:
        log(f"trace: run {cfg.run_id} complete in {elapsed:.1f} s; {len(sessions)} sessions in {run_dir}")
    return run_dir


PREMISE_RULES: frozenset[str] = frozenset({"none", "invalidate", "narrowing"})
"""Revision rules under which the H2 premise is checked; cap_settle relaxes the binding to a cap by design."""

MEASURED_CLASSES: dict[str, frozenset[str]] = {
    "B4": frozenset({"cancelled", "superseded", "lapsed"}),
    "B5": frozenset({"lapsed"}),
    "Tasdiq": frozenset({"lapsed"}),
    "Tasdiq/cap_settle": frozenset({"superseded", "amount_change"}),
}
"""Intent-violation classes that are measured outcomes rather than premise failures.

An entry keyed by an arm name applies under every revision rule; an entry keyed by
"<arm>/<rule>" adds classes for that rule only. The windowed arms B4, B5 and Tasdiq hold a
consent for the validity window, so an execution after a stated lapse of intent inside the
window is the stale-execution quantity the window curve measures: their lapsed class is a
measured class under every rule. B4 has no consent channel, so a cancellation or a revision
stated after its confirmation is not applied until a new confirmation arrives: its
cancelled and superseded classes are the measured consent-channel gap. Under cap_settle a
revision inside the cap settles the record as a narrowing, so a stated revision that the
open mandate still covers executes at the earlier amount: the superseded and amount_change
classes of Tasdiq under cap_settle are the measured cost of that form. The premise
therefore requires zero intent violations of the classes substitution, amount_change,
currency_change, duplicate, untrusted_consent, cancelled, superseded, batch_scope and
cap_exceeded for Tasdiq under invalidate and narrowing.
"""


def measured_classes(arm: str, rule: str | None = None) -> frozenset[str]:
    """Return the measured intent-violation classes of an arm under a revision rule.

    The result is the union of the arm's entry in MEASURED_CLASSES and the entry of
    "<arm>/<rule>" when the rule is given; an arm without entries has no measured class.
    """
    classes = set(MEASURED_CLASSES.get(arm, frozenset()))
    if rule is not None:
        classes.update(MEASURED_CLASSES.get(f"{arm}/{rule}", frozenset()))
    return frozenset(classes)


def premise_check(sessions: Iterable[Mapping[str, Any]], arms: Iterable[str] = ("B4", "Tasdiq")) -> dict[str, Any]:
    """Count intent-layer violations of the named arms on oracle sessions (the H2 premise).

    Returns, per arm, the number of intent violations over the sessions run under the rules
    in PREMISE_RULES, excluding the classes measured_classes returns for the arm and rule,
    and a further entry "<arm>/<rule>" for sessions under any other rule (cap_settle), again
    excluding its measured classes. The entry "measured" holds the excluded counts per class
    for every arm and rule that has measured classes. The premise holds when every count
    outside "measured" is zero: Tasdiq must show zero intent violations under invalidate and
    narrowing on the oracle trace track.
    """
    counts: dict[str, Any] = {arm: 0 for arm in arms}
    measured: dict[str, dict[str, int]] = {}
    for record in sessions:
        arm = record.get("arm")
        if arm not in counts or record.get("mode", "oracle") != "oracle":
            continue
        rule = str(record.get("rule") or "none")
        key = arm if rule in PREMISE_RULES else f"{arm}/{rule}"
        excluded = measured_classes(arm, rule)
        for cls in record.get("intent_violations") or []:
            name = str(cls).split(":", 1)[0]
            if name in excluded:
                bucket = measured.setdefault(key, {})
                bucket[name] = bucket.get(name, 0) + 1
            else:
                counts[key] = counts.get(key, 0) + 1
    counts["measured"] = measured
    return counts
