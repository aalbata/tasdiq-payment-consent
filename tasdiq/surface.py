"""Simulated confirmation surface.

The surface renders a draft consent record in the requested language and asks a simulated reader to approve or decline
it against the scenario's current intent. Two readers are defined by the reference policy: the exact reader, which
approves only the intended record, and the inattentive reader, a sensitivity model that compares the displayed payee
name by its first token and tolerates a small amount difference. Both readers decline a record whose intended payment
has already been executed: when the surface is given read access to the rail's executions, a presentation is declined
with reason already_executed once the reader counts at least as many executions of the intended record as the
scenario intends payments of it (a record intended in addition to the record in force counts once more). Every
presentation is logged with simulated=True and the reader's name, so that interruptions can be counted, results of
the two readers are never pooled, and no simulated acknowledgement can be mistaken for a human one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable

from tasdiq.clock import SimClock
from tasdiq.records import ConsentRecord, IntendedRecord, NameRenderer, render_confirmation, text_hash

USER_COMPARED_FIELDS: tuple[str, ...] = ("payee_id", "amount_minor", "currency")
READERS: tuple[str, ...] = ("exact", "inattentive")
INATTENTIVE_AMOUNT_TOLERANCE_PERCENT: int = 5
ALREADY_EXECUTED: str = "already_executed"
"""Decline reason of a presentation whose intended payment has already been executed."""
ADDITIONAL: str = "additional"
"""Kind of an intent change whose record is intended in addition to the records in force."""

Tuple3 = tuple[str, int, str]


def record_tuple(value: Any) -> Tuple3 | None:
    """Return (payee_id, amount_minor, currency) of a record-like value or mapping, or None when a field is missing."""
    if value is None:
        return None
    get = value.get if isinstance(value, Mapping) else lambda name, default=None: getattr(value, name, default)
    payee, amount, currency = get("payee_id"), get("amount_minor"), get("currency")
    if payee is None or amount is None or currency is None:
        return None
    return (str(payee), int(amount), str(currency))


def record_tuples(value: Any) -> list[Tuple3]:
    """Return the tuples a record states: the items of a batch, the record's own tuple, or none for None."""
    if value is None:
        return []
    items = getattr(value, "items", None)
    if items is not None and not isinstance(value, Mapping):
        out = [record_tuple(item) for item in items]
        return [key for key in out if key is not None]
    key = record_tuple(value)
    return [] if key is None else [key]


def intended_count(state: Any, key: Tuple3, t: int) -> int:
    """Return the number of payments of the tuple key the principal intends at simulated time t.

    An intent state with an intended_count method answers for itself; otherwise the record in force at t intends
    the tuple once when it states it (as a record or as a batch item) and not at all otherwise.
    """
    counter = getattr(state, "intended_count", None)
    if callable(counter):
        return int(counter(key, t))
    return 1 if key in record_tuples(state.intended_at(t)) else 0


@runtime_checkable
class IntentState(Protocol):
    """The scenario's current intent, queried at a simulated time.

    A state may also define intended_count(key, t), the number of payments of a tuple intended at t; without it
    the surface counts one payment of every tuple the record in force states (see intended_count).
    """

    def intended_at(self, t: int) -> IntendedRecord | None:
        """Return the record the principal intends at simulated time t, or None when no payment is intended."""
        ...


@dataclass(frozen=True)
class StaticIntent:
    """An intent that does not change over time; None means the principal intends no payment."""

    record: IntendedRecord | None

    def intended_at(self, t: int) -> IntendedRecord | None:
        """Return the same record for every t."""
        return self.record

    def intended_count(self, key: Tuple3, t: int) -> int:
        """Return 1 when the record states the tuple, else 0."""
        return 1 if key in record_tuples(self.record) else 0


@dataclass
class IntentTimeline:
    """An intent given as changes in time order; the record in force at t is the latest change at or before t.

    A change is (t, record) or (t, record, kind). Before the first change the principal intends no payment. A
    change to None models a cancellation or a lapse. A change of kind additional states a record intended in
    addition to the records in force (a further payment); a change of any other kind replaces them. required
    gives, per tuple, a lower bound on the number of payments intended, taken from the scenario's expected final
    records; intended_count returns the larger of the timeline's count and that bound.
    """

    changes: list[tuple[Any, ...]]
    required: Mapping[Tuple3, int] | None = None

    def __post_init__(self) -> None:
        normalised: list[tuple[int, Any, str | None]] = []
        for change in self.changes:
            if len(change) == 2:
                t_change, record = change
                kind: str | None = None
            elif len(change) == 3:
                t_change, record, kind = change
            else:
                raise ValueError("an intent change is (t, record) or (t, record, kind)")
            normalised.append((int(t_change), record, None if kind is None else str(kind)))
        self.changes = sorted(normalised, key=lambda c: c[0])

    def intended_at(self, t: int) -> IntendedRecord | None:
        """Return the record of the latest change at or before t, or None before the first change."""
        current: IntendedRecord | None = None
        for t_change, record, _ in self.changes:
            if t_change > t:
                break
            current = record
        return current

    def intended_count(self, key: Tuple3, t: int) -> int:
        """Return how many payments of the tuple the changes at or before t intend, at least the required bound."""
        in_force: list[Any] = []
        for t_change, record, kind in self.changes:
            if t_change > t:
                break
            if kind == ADDITIONAL and record is not None:
                in_force.append(record)
            else:
                in_force = [] if record is None else [record]
        count = sum(1 for record in in_force if key in record_tuples(record))
        return max(count, int((self.required or {}).get(key, 0)))


def select_intended(draft: ConsentRecord, intended: Any) -> Any:
    """Return the single intended record a draft is compared with.

    A plain record is returned as is. A batch intent (an object with an items sequence of records) yields the item
    that equals the draft on payee_id, amount_minor and currency, otherwise the first item with the draft's payee_id,
    otherwise the first item; an item without a stated cap inherits the batch cap so that the reader can apply it.
    """
    items = getattr(intended, "items", None)
    if intended is None or items is None:
        return intended
    items = list(items)
    if not items:
        return None
    chosen = None
    for item in items:
        if all(getattr(item, name, None) == getattr(draft, name) for name in USER_COMPARED_FIELDS):
            chosen = item
            break
    if chosen is None:
        chosen = next((item for item in items if getattr(item, "payee_id", None) == draft.payee_id), items[0])
    cap = getattr(chosen, "total_max_minor", None)
    batch_cap = getattr(intended, "total_max_minor", None)
    if cap is None and batch_cap is not None:
        return IntendedRecord(
            payee_id=str(getattr(chosen, "payee_id")),
            amount_minor=int(getattr(chosen, "amount_minor")),
            currency=str(getattr(chosen, "currency")),
            total_max_minor=int(batch_cap),
        )
    return chosen


def first_mismatch(draft: ConsentRecord, intended: Any) -> str | None:
    """Return the reason the exact reader declines the draft, or None when the reader approves.

    The reader approves when the draft equals the intended record on payee_id, amount_minor and currency and the
    draft's cap does not exceed the intended cap when one is stated. The reason names the first differing field in
    that order (field_mismatch:<field>), cap_exceeded for the cap, or no_intent when there is no current intent. A
    batch intent is compared through the item select_intended picks.
    """
    intended = select_intended(draft, intended)
    if intended is None:
        return "no_intent"
    for name in USER_COMPARED_FIELDS:
        if getattr(draft, name) != getattr(intended, name):
            return f"field_mismatch:{name}"
    cap = getattr(intended, "total_max_minor", None)
    if cap is not None and draft.total_max_minor > cap:
        return "cap_exceeded"
    return None


@runtime_checkable
class UserModel(Protocol):
    """Decides, for a draft shown in a language and the current intent, whether the simulated reader approves."""

    def decide(self, draft: ConsentRecord, intended: IntendedRecord | None, t: int, lang: str) -> tuple[str, str]:
        """Return ("approved", "match") or ("declined", reason) for a draft shown in lang at time t."""
        ...


class ExactIntentUser:
    """Primary reader: approves exactly the intended record and declines on the first mismatching field."""

    name: str = "exact"

    def decide(self, draft: ConsentRecord, intended: IntendedRecord | None, t: int, lang: str) -> tuple[str, str]:
        """Approve when first_mismatch finds no difference; otherwise decline with that reason. lang is not used."""
        reason = first_mismatch(draft, intended)
        if reason is None:
            return "approved", "match"
        return "declined", reason

    def same_payment(self, execution: Any, intended: Any, lang: str) -> bool:
        """True when an execution equals the intended record on payee_id, amount_minor and currency."""
        key = record_tuple(execution)
        return key is not None and key == record_tuple(intended)


class InattentiveUser:
    """Sensitivity reader of the reference policy.

    Approves when the displayed payee name shares its first whitespace-delimited token with the intended payee's
    display name in the shown language, the amount is within tolerance_percent of the intended amount (integer
    arithmetic on minor units) and the currency matches. The cap is not inspected. A decline names the first failing
    condition in the order payee_id, amount_minor, currency, or no_intent when the scenario states no intent. Without a
    directory the payee identifier serves as the display name.
    """

    name: str = "inattentive"

    def __init__(
        self, directory: NameRenderer | None, tolerance_percent: int = INATTENTIVE_AMOUNT_TOLERANCE_PERCENT
    ) -> None:
        if not isinstance(tolerance_percent, int) or isinstance(tolerance_percent, bool) or tolerance_percent < 0:
            raise ValueError("tolerance_percent must be a non-negative integer")
        self._directory = directory
        self._tolerance = tolerance_percent

    def _first_token(self, payee_id: str, lang: str) -> str:
        name = self._directory.render_name(payee_id, lang) if self._directory is not None else payee_id
        tokens = name.split()
        return tokens[0] if tokens else name

    def _mismatch(self, payee_id: str, amount_minor: int, currency: str, intended: Any, lang: str) -> str | None:
        """Return the first failing condition of the reader's tests against the intended record, or None."""
        if self._first_token(payee_id, lang) != self._first_token(intended.payee_id, lang):
            return "field_mismatch:payee_id"
        if abs(amount_minor - intended.amount_minor) * 100 > self._tolerance * intended.amount_minor:
            return "field_mismatch:amount_minor"
        if currency != intended.currency:
            return "field_mismatch:currency"
        return None

    def decide(self, draft: ConsentRecord, intended: Any, t: int, lang: str) -> tuple[str, str]:
        """Apply the first-token name test, the amount tolerance and the currency test in that order."""
        intended = select_intended(draft, intended)
        if intended is None:
            return "declined", "no_intent"
        reason = self._mismatch(draft.payee_id, draft.amount_minor, draft.currency, intended, lang)
        if reason is not None:
            return "declined", reason
        return "approved", "match"

    def same_payment(self, execution: Any, intended: Any, lang: str) -> bool:
        """True when the reader's own tests accept an execution as a payment of the intended record."""
        key = record_tuple(execution)
        if key is None or intended is None:
            return False
        return self._mismatch(key[0], key[1], key[2], intended, lang) is None


def make_user_model(user_model: str | UserModel | None, directory: NameRenderer | None) -> UserModel:
    """Return the reader named by a string ("exact" or "inattentive"), the exact reader for None, or the object given."""
    if user_model is None or user_model == "exact":
        return ExactIntentUser()
    if user_model == "inattentive":
        return InattentiveUser(directory)
    if isinstance(user_model, str):
        raise ValueError(f"unknown reader {user_model!r}; expected one of {READERS}")
    return user_model


def reader_name(user_model: Any) -> str:
    """Return the name a reader reports in log lines: its name attribute, or its class name."""
    name = getattr(user_model, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(user_model).__name__


@dataclass(frozen=True)
class SurfaceEvent:
    """Outcome of one presentation on the surface."""

    event_id: str
    t: int
    decision: str
    shown_text: str
    lang: str
    reason: str
    cr_id: str | None = None
    simulated: bool = True
    reader: str = "exact"


class ConfirmationSurface:
    """Trusted surface that shows a draft record to a simulated reader and records the outcome.

    user_model selects the reader: "exact" (default) or "inattentive" as defined by the reference policy, or any object
    with the UserModel interface. executions, when given, returns the rail's executions (records with payee_id,
    amount_minor and currency) and enables the already_executed rule: an approved presentation is turned into a
    decline once the reader counts at least as many executions of the intended record as the intent state says are
    intended. A reader with a same_payment method decides which executions count; any other reader counts the
    executions whose tuple equals the intended record. Without executions the rule is inactive.
    """

    def __init__(
        self,
        clock: SimClock,
        directory: NameRenderer | None,
        user_model: str | UserModel | None = None,
        executions: Callable[[], Iterable[Any]] | None = None,
    ) -> None:
        self._clock = clock
        self._directory = directory
        self._executions = executions
        self._user = make_user_model(user_model, directory)
        self.reader: str = reader_name(self._user)
        self.events: list[SurfaceEvent] = []
        self.log: list[dict[str, Any]] = []
        self._counter = 0

    def _next_event_id(self, cr_id: str, t: int) -> str:
        self._counter += 1
        digest = hashlib.sha256(f"{cr_id}|{t}|{self._counter}".encode("utf-8")).hexdigest()
        return f"sev-{digest[:12]}"

    def render(self, draft: ConsentRecord, lang: str) -> str:
        """Return the text the surface shows for a draft in a language."""
        return render_confirmation(draft, self._directory, lang)

    def executed_count(self, intended: Any, lang: str) -> int:
        """Return the number of rail executions the reader counts as payments of the intended record."""
        if self._executions is None or intended is None:
            return 0
        matcher = getattr(self._user, "same_payment", None)
        count = 0
        for execution in self._executions():
            if callable(matcher):
                matched = bool(matcher(execution, intended, lang))
            else:
                key = record_tuple(execution)
                matched = key is not None and key == record_tuple(intended)
            count += int(matched)
        return count

    def already_executed(self, draft: ConsentRecord, intended: IntentState, current: Any, t: int, lang: str) -> bool:
        """True when every payment the scenario intends of the record the draft matches has been executed.

        The record is the intended record select_intended picks for the draft; the number of intended payments
        comes from intended_count on the intent state and the executed number from executed_count. A record
        the state does not intend at all never satisfies the rule.
        """
        chosen = select_intended(draft, current)
        key = record_tuple(chosen)
        if key is None:
            return False
        wanted = intended_count(intended, key, t)
        return wanted > 0 and self.executed_count(chosen, lang) >= wanted

    def present(self, draft: ConsentRecord, lang: str, intended: IntentState) -> SurfaceEvent:
        """Show the draft in the given language and return the simulated reader's decision at the current time.

        A draft the reader approves is declined with reason already_executed when already_executed holds.
        """
        t = self._clock.now()
        shown_text = self.render(draft, lang)
        current = intended.intended_at(t)
        decision, reason = self._user.decide(draft, current, t, lang)
        if decision == "approved" and self.already_executed(draft, intended, current, t, lang):
            decision, reason = "declined", ALREADY_EXECUTED
        event = SurfaceEvent(
            event_id=self._next_event_id(draft.cr_id, t),
            t=t,
            decision=decision,
            shown_text=shown_text,
            lang=lang,
            reason=reason,
            cr_id=draft.cr_id,
            simulated=True,
            reader=self.reader,
        )
        self.events.append(event)
        self.log.append(
            {
                "event": "surface_presentation",
                "event_id": event.event_id,
                "t": t,
                "cr_id": draft.cr_id,
                "session_id": draft.session_id,
                "principal_id": draft.principal_id,
                "lang": lang,
                "reader": self.reader,
                "decision": decision,
                "reason": reason,
                "shown_text": shown_text,
                "binding_hash": draft.binding_hash,
                "surface_hash": text_hash(shown_text),
                "revision": draft.revision,
                "simulated": True,
            }
        )
        return event

    def presentations(self, session_id: str | None = None) -> int:
        """Return the number of presentations, restricted to one session_id when given."""
        if session_id is None:
            return len(self.events)
        return sum(1 for entry in self.log if entry["session_id"] == session_id)
