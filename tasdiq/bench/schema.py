"""Seed scenario schema for the Tasdiq benchmark.

A seed is a YAML document that describes one scenario: the payees it mentions, the
dialogue turns with their English, Modern Standard Arabic and code-switched members,
the meaning-preserving paraphrases and permission-altering edits of each turn, the
timed events that drive the simulator, the timeline of the principal's intent, and
the expected outcome. A seed file holds either one seed document or a mapping with
the single key seeds whose value is a list of seed documents. This module loads
seed files into frozen dataclasses and validates them. Every validation failure
raises SeedValidationError with a message that names the seed and the field.

Seed layout (YAML keys):

    seed_id: ov-001
    family: ordinary_valid
    phenomena: [digit_script, currency_words]
    description: one sentence describing the scenario
    payees:
      - payee_id: P-114
        name_en: Omar Khalid
        name_ar: عمر خالد
        aliases: [Omar, 3omar]
        kind: domestic
        iban_masked: "AE** **** 4471"
    default_currency: AED
    dialect_extra: false
    turns:
      - idx: 0
        t: 0
        speaker: user
        action_type: payment
        text: {en: ..., ar: ..., mix: ...}
        paraphrases: {en: [..., ...], ar: [..., ...], mix: [..., ...]}
        edits:
          - edit_id: e1
            kind: payee
            text: {en: ..., ar: ..., mix: ...}
            intended: {payee_id: P-115, amount_minor: 25000, currency: AED}
          - edit_id: e2
            kind: inquiry
            action_type: inquiry
            text: {en: ..., ar: ..., mix: ...}
            intended: null
        intended: {payee_id: P-114, amount_minor: 25000, currency: AED, total_max_minor: 25000}
        lang_of_record: en
    events:
      - {t: 10, kind: confirm, payload: {turn_idx: 0}}
    intent_timeline:
      - {t: 0, record: {payee_id: P-114, amount_minor: 25000, currency: AED}, note: initial request}
    expected:
      final_records: [{payee_id: P-114, amount_minor: 25000, currency: AED}]
      must_not_execute:
        - {payee_id: P-115, amount_minor: 25000, currency: AED, reason: similar-name substitution}
        - {payee_id: P-114, before_t: 10, reason: no consent exists before the confirmation}
        - Any second debit of the consent under a different idempotency key.
    provenance: {authored_by: author-seed, review_status: unreviewed, notes: ""}

The payees list holds either bare payee identifiers or payee entries with the
directory fields shown above. The intended value of a turn or an edit takes one of
four forms: a record mapping with payee_id, amount_minor and currency (optional
total_max_minor, the cap on amount plus fees, absent when the utterance states no
cap; optional allowed_uses, 1 unless the utterance states a batch count; optional
reason, free text that takes no part in comparisons); a batch mapping
{batch: [record, ...]} listing the records of one instruction in order, with an
optional total_max_minor, the cap on the sum of the amounts plus fees (every item
then shares one currency); the mapping {ambiguous: field} naming the field a
correct capture must ask about; or null (also spelled {no_record: true}) for an
utterance that must not yield a record. A turn whose action_type is payment or
revision must carry a record, a batch or an ambiguous field.

An edit may state its own action_type; when it does not, an inquiry edit is a
payment when its intended value is a record and an inquiry when it has none, and
every other edit keeps the action_type of its turn. Entries of must_not_execute
are either mappings of field constraints (payee_id, amount_minor, currency,
fee_minor, before_t, account, reason; a field left out constrains nothing) or plain strings
that describe the forbidden execution in prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

FAMILIES: tuple[str, ...] = (
    "view_vs_pay",
    "recipient_substitution",
    "amount_currency_change",
    "total_cost_change",
    "cancellation",
    "expiry_stale",
    "repeated_execution",
    "batch_scope",
    "untrusted_consent",
    "ordinary_valid",
    "ambiguous",
    "revision",
    "negation",
    "numerals_currency",
    "cross_turn_reference",
)

PHENOMENA: tuple[str, ...] = (
    "digit_script",
    "currency_words",
    "number_words",
    "negation",
    "inquiry_vs_approval",
    "cross_turn_reference",
    "cross_language_revision",
    "code_switching",
    "arabizi",
    "similar_names",
)

LANGS: tuple[str, ...] = ("en", "ar", "mix")
SPEAKERS: tuple[str, ...] = ("user", "agent", "content")
ACTION_TYPES: tuple[str, ...] = ("payment", "inquiry", "revision", "cancellation", "other")
RECORD_ACTION_TYPES: tuple[str, ...] = ("payment", "revision")
EDIT_KINDS: tuple[str, ...] = ("payee", "amount", "currency", "negation", "inquiry", "reference", "batch")
EVENT_KINDS: tuple[str, ...] = (
    "confirm",
    "revise",
    "cancel",
    "lapse",
    "delay",
    "execute",
    "retry",
    "lost_ack",
    "untrusted_content",
    "crash",
)
REVIEW_STATUSES: tuple[str, ...] = ("unreviewed", "author-reviewed", "second-reviewer")
INTENT_KINDS: tuple[str, ...] = ("request", "revise", "additional", "cancel", "lapse", "none")
NO_RECORD_ACTION_TYPES: tuple[str, ...] = ("inquiry", "cancellation", "other")
PAYEE_KINDS: tuple[str, ...] = ("domestic", "international", "merchant")
AMBIGUOUS_FIELDS: tuple[str, ...] = ("payee_id", "amount_minor", "currency", "action_type", "reference")
AMBIGUOUS_FIELD_ALIASES: dict[str, str] = {
    "payee": "payee_id",
    "recipient": "payee_id",
    "amount": "amount_minor",
    "action": "action_type",
    "ref": "reference",
}

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")
_ARABIC_LETTER = re.compile(r"[؀-ۿ]")


class SeedValidationError(ValueError):
    """Raised when a seed document violates the schema; names the seed and the field."""

    def __init__(self, seed: str, field_path: str, problem: str) -> None:
        self.seed = seed
        self.field_path = field_path
        self.problem = problem
        super().__init__(f"seed {seed}: field {field_path}: {problem}")


@dataclass(frozen=True)
class IntendedRecord:
    """Canonical transaction record the principal intends (amounts in minor units).

    total_max_minor is the cap on amount plus fees; None means the utterance states
    no cap. reason is free text for the seed reader and takes no part in equality.
    """

    payee_id: str
    amount_minor: int
    currency: str
    total_max_minor: int | None = None
    allowed_uses: int = 1
    reason: str = field(default="", compare=False)

    @property
    def cap(self) -> int:
        """The stated cap, or the amount itself when no cap is stated."""
        return self.amount_minor if self.total_max_minor is None else self.total_max_minor

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "payee_id": self.payee_id,
            "amount_minor": self.amount_minor,
            "currency": self.currency,
            "total_max_minor": self.total_max_minor,
            "allowed_uses": self.allowed_uses,
        }
        if self.reason:
            out["reason"] = self.reason
        return out


@dataclass(frozen=True)
class IntendedBatch:
    """A batch of payments the principal intends as one instruction.

    items are the records of the batch in the order stated. total_max_minor is the
    cap on the sum of the amounts plus fees; None means the utterance states no
    cap. reason is free text for the seed reader and takes no part in equality.
    """

    items: tuple[IntendedRecord, ...]
    total_max_minor: int | None = None
    reason: str = field(default="", compare=False)

    @property
    def amount_minor(self) -> int:
        """The sum of the item amounts."""
        return sum(item.amount_minor for item in self.items)

    @property
    def cap(self) -> int:
        """The stated cap, or the sum of the item amounts when no cap is stated."""
        return self.amount_minor if self.total_max_minor is None else self.total_max_minor

    @property
    def allowed_uses(self) -> int:
        """The number of items, one execution per item."""
        return len(self.items)

    @property
    def payee_ids(self) -> tuple[str, ...]:
        return tuple(item.payee_id for item in self.items)

    @property
    def currencies(self) -> tuple[str, ...]:
        """The distinct currencies of the items in order of first appearance."""
        seen: list[str] = []
        for item in self.items:
            if item.currency not in seen:
                seen.append(item.currency)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"batch": [item.to_dict() for item in self.items]}
        if self.total_max_minor is not None:
            out["total_max_minor"] = self.total_max_minor
        if self.reason:
            out["reason"] = self.reason
        return out


@dataclass(frozen=True)
class Ambiguous:
    """Marks an utterance whose correct capture asks about the named field."""

    field: str

    @property
    def ambiguous(self) -> str:
        return self.field

    def to_dict(self) -> dict[str, Any]:
        return {"ambiguous": self.field}


IntendedPayment = IntendedRecord | IntendedBatch
Intended = IntendedRecord | IntendedBatch | Ambiguous | None


@dataclass(frozen=True)
class PayeeEntry:
    """A payee of the scenario with the directory fields the seed states."""

    payee_id: str
    name_en: str | None = None
    name_ar: str | None = None
    aliases: tuple[str, ...] = ()
    kind: str | None = None
    iban_masked: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "payee_id": self.payee_id,
            "name_en": self.name_en,
            "name_ar": self.name_ar,
            "aliases": list(self.aliases),
            "kind": self.kind,
            "iban_masked": self.iban_masked,
        }


@dataclass(frozen=True)
class LangText:
    """One utterance in its three language members."""

    en: str
    ar: str
    mix: str

    def get(self, lang: str) -> str:
        if lang not in LANGS:
            raise KeyError(lang)
        return getattr(self, lang)

    def to_dict(self) -> dict[str, str]:
        return {"en": self.en, "ar": self.ar, "mix": self.mix}


@dataclass(frozen=True)
class Paraphrases:
    """Meaning-preserving rewordings of a turn per language."""

    en: tuple[str, ...] = ()
    ar: tuple[str, ...] = ()
    mix: tuple[str, ...] = ()

    def get(self, lang: str) -> tuple[str, ...]:
        if lang not in LANGS:
            raise KeyError(lang)
        return getattr(self, lang)

    def to_dict(self) -> dict[str, list[str]]:
        return {"en": list(self.en), "ar": list(self.ar), "mix": list(self.mix)}


@dataclass(frozen=True)
class Edit:
    """A permission-altering rewording of a turn with its own intended value.

    action_type states what the edited utterance does; it may differ from the
    action_type of the turn (an inquiry edit of a payment turn carries inquiry).
    """

    edit_id: str
    kind: str
    text: LangText
    intended: Intended
    action_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "kind": self.kind,
            "action_type": self.action_type,
            "text": self.text.to_dict(),
            "intended": intended_to_dict(self.intended),
        }


@dataclass(frozen=True)
class Turn:
    """One dialogue turn with its members, paraphrases, edits and intended value."""

    idx: int
    t: int
    speaker: str
    action_type: str
    text: LangText
    paraphrases: Paraphrases
    edits: tuple[Edit, ...]
    intended: Intended
    lang_of_record: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "t": self.t,
            "speaker": self.speaker,
            "action_type": self.action_type,
            "text": self.text.to_dict(),
            "paraphrases": self.paraphrases.to_dict(),
            "edits": [e.to_dict() for e in self.edits],
            "intended": intended_to_dict(self.intended),
            "lang_of_record": self.lang_of_record,
        }


@dataclass(frozen=True)
class Event:
    """A timed simulator event with a free-form payload."""

    t: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"t": self.t, "kind": self.kind, "payload": dict(self.payload)}


@dataclass(frozen=True)
class IntentChange:
    """A change of the principal's intent at time t.

    record is the record or batch intended from t on; None means no payment is
    intended (a cancellation or a lapse). kind, when stated, names the change
    (request, revise, additional, cancel, lapse or none); additional means the record
    is intended in addition to the record in force, as a second independent payment.
    """

    t: int
    record: IntendedPayment | None
    note: str = ""
    kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "record": None if self.record is None else self.record.to_dict(),
            "note": self.note,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Forbidden:
    """An execution the scenario forbids, stated as field constraints or in prose.

    A field left None constrains nothing. fee_minor constrains the fee quoted for
    the execution. before_t limits the entry to executions earlier than that time.
    account names a masked destination outside the payee directory. reason is the
    author's description and takes no part in matching; an entry written as a plain
    string in the seed file carries only a reason.
    """

    payee_id: str | None = None
    amount_minor: int | None = None
    currency: str | None = None
    fee_minor: int | None = None
    before_t: int | None = None
    account: str | None = None
    reason: str = ""

    @property
    def record(self) -> IntendedRecord | None:
        """The forbidden record when payee_id, amount_minor and currency are all stated."""
        if self.payee_id is None or self.amount_minor is None or self.currency is None:
            return None
        return IntendedRecord(payee_id=self.payee_id, amount_minor=self.amount_minor, currency=self.currency)

    @property
    def is_prose(self) -> bool:
        """True when the entry states no field and only describes the forbidden execution."""
        return (
            self.payee_id is None
            and self.amount_minor is None
            and self.currency is None
            and self.fee_minor is None
            and self.before_t is None
            and self.account is None
        )

    def matches(
        self,
        payee_id: str | None,
        amount_minor: int,
        currency: str,
        t: int | None = None,
        fee_minor: int | None = None,
    ) -> bool:
        """True when an execution of the given tuple at time t with the given fee falls under this entry.

        Every stated field must equal the executed value (a stated fee_minor requires
        the fee to be given); before_t requires t to be given and earlier than it;
        account requires the execution to name no directory payee (payee_id None).
        A prose entry matches nothing.
        """
        if self.is_prose:
            return False
        if self.payee_id is not None and payee_id != self.payee_id:
            return False
        if self.amount_minor is not None and amount_minor != self.amount_minor:
            return False
        if self.currency is not None and currency != self.currency:
            return False
        if self.fee_minor is not None and fee_minor != self.fee_minor:
            return False
        if self.before_t is not None and (t is None or t >= self.before_t):
            return False
        if self.account is not None and payee_id is not None:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in ("payee_id", "amount_minor", "currency", "fee_minor", "before_t", "account"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.reason or not out:
            out["reason"] = self.reason
        return out


@dataclass(frozen=True)
class ExpectedOutcome:
    """Records that must execute exactly once and executions that must never happen."""

    final_records: tuple[IntendedRecord, ...]
    must_not_execute: tuple[Forbidden, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_records": [r.to_dict() for r in self.final_records],
            "must_not_execute": [f.to_dict() for f in self.must_not_execute],
        }


@dataclass(frozen=True)
class Provenance:
    """Authorship and review status of a seed.

    review_status is the status of the seed's English content; reviewed_langs lists the
    further languages whose content has been reviewed to that status. The Arabic and
    code-switched members of a seed stay unreviewed until their language is listed.
    """

    authored_by: str
    review_status: str
    notes: str = ""
    reviewed_langs: tuple[str, ...] = ()

    def status_for(self, lang: str) -> str:
        """The review status of the members of one language."""
        if lang == "en" or lang in self.reviewed_langs:
            return self.review_status
        return "unreviewed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "authored_by": self.authored_by,
            "review_status": self.review_status,
            "notes": self.notes,
            "reviewed_langs": list(self.reviewed_langs),
        }


@dataclass(frozen=True)
class ScenarioSpec:
    """A validated seed scenario.

    payees holds the payee identifiers; payee_entries holds the directory fields when
    the seed states them (empty when the seed lists bare identifiers).
    """

    seed_id: str
    family: str
    phenomena: tuple[str, ...]
    description: str
    payees: tuple[str, ...]
    default_currency: str
    turns: tuple[Turn, ...]
    events: tuple[Event, ...]
    intent_timeline: tuple[IntentChange, ...]
    expected: ExpectedOutcome
    provenance: Provenance
    payee_entries: tuple[PayeeEntry, ...] = ()
    dialect_extra: bool = False
    source: str = field(default="", compare=False)

    def user_turns(self) -> tuple[Turn, ...]:
        return tuple(t for t in self.turns if t.speaker == "user")

    def turn(self, idx: int) -> Turn:
        for t in self.turns:
            if t.idx == idx:
                return t
        raise KeyError(idx)

    def intended_at(self, t: int) -> IntendedPayment | None:
        """Return the record or batch the principal intends at time t, or None when no payment is intended."""
        current: IntendedPayment | None = None
        for change in self.intent_timeline:
            if change.t <= t:
                current = change.record
            else:
                break
        return current

    def to_dict(self) -> dict[str, Any]:
        payees: list[Any] = [p.to_dict() for p in self.payee_entries] if self.payee_entries else list(self.payees)
        return {
            "seed_id": self.seed_id,
            "family": self.family,
            "phenomena": list(self.phenomena),
            "description": self.description,
            "payees": payees,
            "default_currency": self.default_currency,
            "dialect_extra": self.dialect_extra,
            "turns": [t.to_dict() for t in self.turns],
            "events": [e.to_dict() for e in self.events],
            "intent_timeline": [c.to_dict() for c in self.intent_timeline],
            "expected": self.expected.to_dict(),
            "provenance": self.provenance.to_dict(),
        }


def intended_to_dict(value: Intended) -> dict[str, Any] | None:
    """Encode an intended value in the same shape the seed files use; None stays None."""
    if value is None:
        return None
    if isinstance(value, (IntendedRecord, IntendedBatch, Ambiguous)):
        return value.to_dict()
    raise TypeError(f"not an intended value: {value!r}")


def intended_from_dict(value: Mapping[str, Any] | None) -> Intended:
    """Decode an intended value written by intended_to_dict."""
    return _parse_intended(value, "record", "intended", allow_null=True)


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""


def _construct_mapping(loader: _StrictLoader, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_seed_file(path: Path | str) -> list[ScenarioSpec]:
    """Load and validate every seed in one file (a single seed document or a seeds list)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = yaml.load(handle, Loader=_StrictLoader)
        except yaml.YAMLError as exc:
            raise SeedValidationError(path.name, "<document>", f"YAML parse error: {exc}") from exc
    if isinstance(data, Mapping) and set(data) == {"seeds"}:
        items = data["seeds"]
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or not items:
            raise SeedValidationError(path.name, "seeds", "must be a non-empty list of seed documents")
        return [parse_spec(item, source=f"{path.name}#{i}") for i, item in enumerate(items)]
    return [parse_spec(data, source=path.name)]


def load_seed(path: Path | str) -> ScenarioSpec:
    """Load and validate a file that holds exactly one seed."""
    specs = load_seed_file(path)
    if len(specs) != 1:
        raise SeedValidationError(Path(path).name, "seeds", f"expected one seed, found {len(specs)}")
    return specs[0]


def is_live_task_document(data: Any) -> bool:
    """Return True when a decoded YAML document is a live-task file (a top-level tasks list, no seeds)."""
    return isinstance(data, Mapping) and "tasks" in data and "seeds" not in data and "seed_id" not in data


def is_seed_file(path: Path | str) -> bool:
    """Return True when a YAML file holds seed scenarios rather than another document kind.

    A file whose top-level mapping carries a tasks key and no seeds (the live-task file,
    which may also carry provenance) is not a seed file; every other YAML file in the
    seed directory is.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = yaml.load(handle, Loader=_StrictLoader)
        except yaml.YAMLError:
            return True
    return not is_live_task_document(data)


def load_seeds(directory: Path | str, pattern: str = "*.yaml") -> list[ScenarioSpec]:
    """Load every seed in a directory, in file-name order; seed ids must be unique.

    Files that hold live tasks (a top-level tasks key) are skipped.
    """
    directory = Path(directory)
    specs: list[ScenarioSpec] = []
    seen: dict[str, str] = {}
    for path in sorted(directory.glob(pattern)):
        if not is_seed_file(path):
            continue
        for spec in load_seed_file(path):
            if spec.seed_id in seen:
                raise SeedValidationError(
                    spec.seed_id, "seed_id", f"duplicate seed id in {seen[spec.seed_id]} and {spec.source}"
                )
            seen[spec.seed_id] = spec.source
            specs.append(spec)
    return specs


def parse_spec(data: Any, source: str = "<memory>") -> ScenarioSpec:
    """Validate a decoded seed document and build a ScenarioSpec."""
    seed = source
    top = _expect_mapping(data, seed, "<document>")
    seed = _expect_str(top.get("seed_id"), seed, "seed_id")
    if not _ID_PATTERN.match(seed):
        raise SeedValidationError(seed, "seed_id", "must contain only letters, digits, '_', '.' and '-'")
    _check_keys(
        top,
        seed,
        "<document>",
        required=(
            "seed_id",
            "family",
            "phenomena",
            "description",
            "payees",
            "default_currency",
            "turns",
            "events",
            "intent_timeline",
            "expected",
            "provenance",
        ),
        optional=("dialect_extra",),
    )
    family = _expect_str(top["family"], seed, "family")
    if family not in FAMILIES:
        raise SeedValidationError(seed, "family", f"unknown family {family!r}; expected one of {list(FAMILIES)}")
    phenomena = _expect_str_list(top["phenomena"], seed, "phenomena", allow_empty=True)
    for i, tag in enumerate(phenomena):
        if tag not in PHENOMENA:
            raise SeedValidationError(seed, f"phenomena[{i}]", f"unknown phenomenon {tag!r}; expected one of {list(PHENOMENA)}")
    if len(set(phenomena)) != len(phenomena):
        raise SeedValidationError(seed, "phenomena", "duplicate tags")
    description = _expect_str(top["description"], seed, "description")
    payees, payee_entries = _parse_payees(top["payees"], seed)
    default_currency = _expect_currency(top["default_currency"], seed, "default_currency")
    dialect_extra = top.get("dialect_extra", False)
    if not isinstance(dialect_extra, bool):
        raise SeedValidationError(seed, "dialect_extra", "must be a boolean")

    turns = _parse_turns(top["turns"], seed)
    events = _parse_events(top["events"], seed)
    timeline = _parse_timeline(top["intent_timeline"], seed)
    expected = _parse_expected(top["expected"], seed)
    provenance = _parse_provenance(top["provenance"], seed)

    spec = ScenarioSpec(
        seed_id=seed,
        family=family,
        phenomena=tuple(phenomena),
        description=description,
        payees=payees,
        default_currency=default_currency,
        turns=turns,
        events=events,
        intent_timeline=timeline,
        expected=expected,
        provenance=provenance,
        payee_entries=payee_entries,
        dialect_extra=dialect_extra,
        source=source,
    )
    _check_payee_references(spec)
    return spec


def _parse_payees(value: Any, seed: str) -> tuple[tuple[str, ...], tuple[PayeeEntry, ...]]:
    items = _expect_list(value, seed, "payees")
    if not items:
        raise SeedValidationError(seed, "payees", "must not be empty")
    ids: list[str] = []
    entries: list[PayeeEntry] = []
    for i, item in enumerate(items):
        where = f"payees[{i}]"
        if isinstance(item, str):
            ids.append(_expect_str(item, seed, where))
            continue
        mapping = _expect_mapping(item, seed, where)
        _check_keys(
            mapping,
            seed,
            where,
            required=("payee_id",),
            optional=("name_en", "name_ar", "aliases", "kind", "iban_masked"),
        )
        payee_id = _expect_str(mapping["payee_id"], seed, f"{where}.payee_id")
        name_en = mapping.get("name_en")
        if name_en is not None:
            name_en = _expect_str(name_en, seed, f"{where}.name_en")
        name_ar = mapping.get("name_ar")
        if name_ar is not None:
            name_ar = _expect_str(name_ar, seed, f"{where}.name_ar")
        aliases = tuple(_expect_str_list(mapping.get("aliases", []), seed, f"{where}.aliases", allow_empty=True))
        kind = mapping.get("kind")
        if kind is not None:
            kind = _expect_choice(kind, seed, f"{where}.kind", PAYEE_KINDS)
        iban_masked = mapping.get("iban_masked")
        if iban_masked is not None:
            iban_masked = _expect_str(iban_masked, seed, f"{where}.iban_masked")
            if "*" not in iban_masked:
                raise SeedValidationError(seed, f"{where}.iban_masked", "must be masked (contain at least one '*')")
        ids.append(payee_id)
        entries.append(
            PayeeEntry(payee_id=payee_id, name_en=name_en, name_ar=name_ar, aliases=aliases, kind=kind, iban_masked=iban_masked)
        )
    if entries and len(entries) != len(ids):
        raise SeedValidationError(seed, "payees", "list either bare payee ids or payee entries, not both")
    if len(set(ids)) != len(ids):
        raise SeedValidationError(seed, "payees", "duplicate payee ids")
    return tuple(ids), tuple(entries)


def _parse_turns(value: Any, seed: str) -> tuple[Turn, ...]:
    items = _expect_list(value, seed, "turns")
    if not items:
        raise SeedValidationError(seed, "turns", "at least one turn is required")
    turns: list[Turn] = []
    edit_ids: set[str] = set()
    previous_t: int | None = None
    for position, item in enumerate(items):
        where = f"turns[{position}]"
        turn = _expect_mapping(item, seed, where)
        _check_keys(
            turn,
            seed,
            where,
            required=("idx", "t", "speaker", "action_type", "text", "intended"),
            optional=("paraphrases", "edits", "lang_of_record"),
        )
        idx = _expect_int(turn["idx"], seed, f"{where}.idx", minimum=0)
        if idx != position:
            raise SeedValidationError(seed, f"{where}.idx", f"expected {position} (turns are numbered consecutively from 0)")
        t = _expect_int(turn["t"], seed, f"{where}.t", minimum=0)
        if previous_t is not None and t < previous_t:
            raise SeedValidationError(seed, f"{where}.t", f"turn times must not decrease (previous turn at t={previous_t})")
        previous_t = t
        speaker = _expect_choice(turn["speaker"], seed, f"{where}.speaker", SPEAKERS)
        action_type = _expect_choice(turn["action_type"], seed, f"{where}.action_type", ACTION_TYPES)
        text = _parse_lang_text(turn["text"], seed, f"{where}.text")
        paraphrases = _parse_paraphrases(turn.get("paraphrases"), seed, f"{where}.paraphrases")
        intended = _parse_intended(turn["intended"], seed, f"{where}.intended", allow_null=True)
        if intended is None and action_type in RECORD_ACTION_TYPES:
            raise SeedValidationError(
                seed, f"{where}.intended", f"a turn with action_type {action_type!r} needs an intended record or an ambiguous field"
            )
        lang_of_record = turn.get("lang_of_record")
        if lang_of_record is not None:
            lang_of_record = _expect_choice(lang_of_record, seed, f"{where}.lang_of_record", LANGS)
        edits = _parse_edits(turn.get("edits"), seed, f"{where}.edits", intended, action_type, edit_ids)
        turns.append(
            Turn(
                idx=idx,
                t=t,
                speaker=speaker,
                action_type=action_type,
                text=text,
                paraphrases=paraphrases,
                edits=edits,
                intended=intended,
                lang_of_record=lang_of_record,
            )
        )
    return tuple(turns)


def _parse_edits(
    value: Any, seed: str, where: str, base_intended: Intended, turn_action_type: str, edit_ids: set[str]
) -> tuple[Edit, ...]:
    if value is None:
        return ()
    items = _expect_list(value, seed, where)
    edits: list[Edit] = []
    for i, item in enumerate(items):
        sub = f"{where}[{i}]"
        edit = _expect_mapping(item, seed, sub)
        _check_keys(edit, seed, sub, required=("edit_id", "kind", "text", "intended"), optional=("action_type",))
        edit_id = _expect_str(edit["edit_id"], seed, f"{sub}.edit_id")
        if not _ID_PATTERN.match(edit_id):
            raise SeedValidationError(seed, f"{sub}.edit_id", "must contain only letters, digits, '_', '.' and '-'")
        if edit_id in edit_ids:
            raise SeedValidationError(seed, f"{sub}.edit_id", f"duplicate edit id {edit_id!r} within the seed")
        edit_ids.add(edit_id)
        kind = _expect_choice(edit["kind"], seed, f"{sub}.kind", EDIT_KINDS)
        text = _parse_lang_text(edit["text"], seed, f"{sub}.text")
        intended = _parse_intended(edit["intended"], seed, f"{sub}.intended", allow_null=True)
        if intended == base_intended:
            raise SeedValidationError(seed, f"{sub}.intended", "an edit must alter the intended value of its turn")
        action_type = edit.get("action_type")
        if action_type is None:
            action_type = default_edit_action_type(kind, intended, turn_action_type)
        else:
            action_type = _expect_choice(action_type, seed, f"{sub}.action_type", ACTION_TYPES)
        if action_type in NO_RECORD_ACTION_TYPES and isinstance(intended, (IntendedRecord, IntendedBatch)):
            raise SeedValidationError(seed, f"{sub}.action_type", f"an edit with action_type {action_type!r} cannot intend a record")
        edits.append(Edit(edit_id=edit_id, kind=kind, text=text, intended=intended, action_type=action_type))
    return tuple(edits)


def default_edit_action_type(kind: str, intended: Intended, turn_action_type: str) -> str:
    """The action_type of an edit that states none.

    An inquiry edit, and any edit of a turn whose action yields no record (inquiry,
    cancellation, other), is a payment when its intended value is a record; an inquiry
    edit without a record is an inquiry; an edit marked ambiguous, and every other edit,
    keeps the action_type of its turn.
    """
    if isinstance(intended, (IntendedRecord, IntendedBatch)) and (kind == "inquiry" or turn_action_type in NO_RECORD_ACTION_TYPES):
        return "payment"
    if kind == "inquiry" and intended is None:
        return "inquiry"
    return turn_action_type


def _parse_lang_text(value: Any, seed: str, where: str) -> LangText:
    mapping = _expect_mapping(value, seed, where)
    _check_keys(mapping, seed, where, required=LANGS, optional=())
    members = {lang: _expect_str(mapping[lang], seed, f"{where}.{lang}") for lang in LANGS}
    if not _ARABIC_LETTER.search(members["ar"]):
        raise SeedValidationError(seed, f"{where}.ar", "the Arabic member contains no Arabic-script characters")
    if _ARABIC_LETTER.search(members["en"]):
        raise SeedValidationError(seed, f"{where}.en", "the English member contains Arabic-script characters")
    return LangText(en=members["en"], ar=members["ar"], mix=members["mix"])


def _parse_paraphrases(value: Any, seed: str, where: str) -> Paraphrases:
    if value is None:
        return Paraphrases()
    mapping = _expect_mapping(value, seed, where)
    _check_keys(mapping, seed, where, required=(), optional=LANGS)
    lists = {lang: tuple(_expect_str_list(mapping.get(lang, []), seed, f"{where}.{lang}", allow_empty=True)) for lang in LANGS}
    for i, text in enumerate(lists["ar"]):
        if not _ARABIC_LETTER.search(text):
            raise SeedValidationError(seed, f"{where}.ar[{i}]", "the Arabic paraphrase contains no Arabic-script characters")
    for i, text in enumerate(lists["en"]):
        if _ARABIC_LETTER.search(text):
            raise SeedValidationError(seed, f"{where}.en[{i}]", "the English paraphrase contains Arabic-script characters")
    return Paraphrases(en=lists["en"], ar=lists["ar"], mix=lists["mix"])


def _parse_intended(value: Any, seed: str, where: str, allow_null: bool) -> Intended:
    if value is None:
        if allow_null:
            return None
        raise SeedValidationError(seed, where, "intended record or ambiguous field required")
    mapping = _expect_mapping(value, seed, where)
    if "ambiguous" in mapping:
        _check_keys(mapping, seed, where, required=("ambiguous",), optional=())
        name = _expect_str(mapping["ambiguous"], seed, f"{where}.ambiguous")
        name = AMBIGUOUS_FIELD_ALIASES.get(name, name)
        if name not in AMBIGUOUS_FIELDS:
            raise SeedValidationError(seed, f"{where}.ambiguous", f"unknown field {name!r}; expected one of {list(AMBIGUOUS_FIELDS)}")
        return Ambiguous(field=name)
    if "no_record" in mapping:
        _check_keys(mapping, seed, where, required=("no_record",), optional=())
        if mapping["no_record"] is not True:
            raise SeedValidationError(seed, f"{where}.no_record", "must be true")
        return None
    return _parse_payment(mapping, seed, where)


def _parse_payment(value: Any, seed: str, where: str) -> IntendedPayment:
    """Parse a record mapping or a batch mapping."""
    mapping = _expect_mapping(value, seed, where)
    if "batch" in mapping:
        return _parse_batch(mapping, seed, where)
    return _parse_record(mapping, seed, where)


def _parse_batch(value: Any, seed: str, where: str) -> IntendedBatch:
    mapping = _expect_mapping(value, seed, where)
    _check_keys(mapping, seed, where, required=("batch",), optional=("total_max_minor", "reason"))
    raw_items = _expect_list(mapping["batch"], seed, f"{where}.batch")
    if not raw_items:
        raise SeedValidationError(seed, f"{where}.batch", "must list at least one record")
    items = tuple(_parse_record(item, seed, f"{where}.batch[{i}]") for i, item in enumerate(raw_items))
    total_max_minor = mapping.get("total_max_minor")
    if total_max_minor is not None:
        total_max_minor = _expect_int(total_max_minor, seed, f"{where}.total_max_minor", minimum=1)
        if len({item.currency for item in items}) != 1:
            raise SeedValidationError(seed, f"{where}.batch", "a batch with a cap must have one currency across its items")
        if total_max_minor < sum(item.amount_minor for item in items):
            raise SeedValidationError(seed, f"{where}.total_max_minor", "must not be below the sum of the item amounts")
    reason = mapping.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise SeedValidationError(seed, f"{where}.reason", "must be a string")
    return IntendedBatch(items=items, total_max_minor=total_max_minor, reason=reason)


def _parse_record(value: Any, seed: str, where: str) -> IntendedRecord:
    mapping = _expect_mapping(value, seed, where)
    _check_keys(
        mapping,
        seed,
        where,
        required=("payee_id", "amount_minor", "currency"),
        optional=("total_max_minor", "allowed_uses", "reason"),
    )
    payee_id = _expect_str(mapping["payee_id"], seed, f"{where}.payee_id")
    amount_minor = _expect_int(mapping["amount_minor"], seed, f"{where}.amount_minor", minimum=1)
    currency = _expect_currency(mapping["currency"], seed, f"{where}.currency")
    total_max_minor = mapping.get("total_max_minor")
    if total_max_minor is not None:
        total_max_minor = _expect_int(total_max_minor, seed, f"{where}.total_max_minor", minimum=1)
        if total_max_minor < amount_minor:
            raise SeedValidationError(seed, f"{where}.total_max_minor", "must not be below amount_minor")
    allowed_uses = _expect_int(mapping.get("allowed_uses", 1), seed, f"{where}.allowed_uses", minimum=1)
    reason = mapping.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise SeedValidationError(seed, f"{where}.reason", "must be a string")
    return IntendedRecord(
        payee_id=payee_id,
        amount_minor=amount_minor,
        currency=currency,
        total_max_minor=total_max_minor,
        allowed_uses=allowed_uses,
        reason=reason,
    )


def _parse_events(value: Any, seed: str) -> tuple[Event, ...]:
    items = _expect_list(value, seed, "events")
    events: list[Event] = []
    previous_t: int | None = None
    for i, item in enumerate(items):
        where = f"events[{i}]"
        event = _expect_mapping(item, seed, where)
        _check_keys(event, seed, where, required=("t", "kind"), optional=("payload",))
        t = _expect_int(event["t"], seed, f"{where}.t", minimum=0)
        if previous_t is not None and t < previous_t:
            raise SeedValidationError(seed, f"{where}.t", f"events must be in time order (previous event at t={previous_t})")
        previous_t = t
        kind = _expect_choice(event["kind"], seed, f"{where}.kind", EVENT_KINDS)
        payload = event.get("payload")
        if payload is None:
            payload = {}
        payload = _expect_mapping(payload, seed, f"{where}.payload")
        for key in payload:
            if not isinstance(key, str):
                raise SeedValidationError(seed, f"{where}.payload", "payload keys must be strings")
        events.append(Event(t=t, kind=kind, payload=dict(payload)))
    return tuple(events)


def _parse_timeline(value: Any, seed: str) -> tuple[IntentChange, ...]:
    items = _expect_list(value, seed, "intent_timeline")
    if not items:
        raise SeedValidationError(seed, "intent_timeline", "at least one intent change is required")
    changes: list[IntentChange] = []
    previous_t: int | None = None
    for i, item in enumerate(items):
        where = f"intent_timeline[{i}]"
        change = _expect_mapping(item, seed, where)
        _check_keys(change, seed, where, required=("t", "record"), optional=("note", "kind"))
        t = _expect_int(change["t"], seed, f"{where}.t", minimum=0)
        if previous_t is not None and t < previous_t:
            raise SeedValidationError(seed, f"{where}.t", f"intent changes must be in time order (previous change at t={previous_t})")
        previous_t = t
        record = None if change["record"] is None else _parse_payment(change["record"], seed, f"{where}.record")
        note = change.get("note", "")
        if note is None:
            note = ""
        if not isinstance(note, str):
            raise SeedValidationError(seed, f"{where}.note", "must be a string")
        kind = change.get("kind")
        if kind is not None:
            kind = _expect_choice(kind, seed, f"{where}.kind", INTENT_KINDS)
            if kind in ("request", "revise", "additional") and record is None:
                raise SeedValidationError(seed, f"{where}.kind", f"an intent change of kind {kind!r} needs a record")
            if kind in ("cancel", "lapse", "none") and record is not None:
                raise SeedValidationError(seed, f"{where}.kind", f"an intent change of kind {kind!r} must have record null")
        changes.append(IntentChange(t=t, record=record, note=note, kind=kind))
    return tuple(changes)


def _parse_expected(value: Any, seed: str) -> ExpectedOutcome:
    mapping = _expect_mapping(value, seed, "expected")
    _check_keys(mapping, seed, "expected", required=("final_records", "must_not_execute"), optional=())
    final_records = tuple(
        _parse_record(item, seed, f"expected.final_records[{i}]")
        for i, item in enumerate(_expect_list(mapping["final_records"], seed, "expected.final_records"))
    )
    must_not = tuple(
        _parse_forbidden(item, seed, f"expected.must_not_execute[{i}]")
        for i, item in enumerate(_expect_list(mapping["must_not_execute"], seed, "expected.must_not_execute"))
    )
    return ExpectedOutcome(final_records=final_records, must_not_execute=must_not)


def _parse_forbidden(value: Any, seed: str, where: str) -> Forbidden:
    if isinstance(value, str):
        return Forbidden(reason=_expect_str(value, seed, where))
    mapping = _expect_mapping(value, seed, where)
    _check_keys(
        mapping,
        seed,
        where,
        required=(),
        optional=("payee_id", "amount_minor", "currency", "fee_minor", "before_t", "account", "reason"),
    )
    payee_id = mapping.get("payee_id")
    if payee_id is not None:
        payee_id = _expect_str(payee_id, seed, f"{where}.payee_id")
    amount_minor = mapping.get("amount_minor")
    if amount_minor is not None:
        amount_minor = _expect_int(amount_minor, seed, f"{where}.amount_minor", minimum=1)
    currency = mapping.get("currency")
    if currency is not None:
        currency = _expect_currency(currency, seed, f"{where}.currency")
    fee_minor = mapping.get("fee_minor")
    if fee_minor is not None:
        fee_minor = _expect_int(fee_minor, seed, f"{where}.fee_minor", minimum=0)
    before_t = mapping.get("before_t")
    if before_t is not None:
        before_t = _expect_int(before_t, seed, f"{where}.before_t", minimum=0)
    account = mapping.get("account")
    if account is not None:
        account = _expect_str(account, seed, f"{where}.account")
        if "*" not in account:
            raise SeedValidationError(seed, f"{where}.account", "must be masked (contain at least one '*')")
    reason = mapping.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise SeedValidationError(seed, f"{where}.reason", "must be a string")
    entry = Forbidden(
        payee_id=payee_id,
        amount_minor=amount_minor,
        currency=currency,
        fee_minor=fee_minor,
        before_t=before_t,
        account=account,
        reason=reason,
    )
    if entry.is_prose and not reason.strip():
        raise SeedValidationError(seed, where, "must state at least one field or a reason")
    return entry


def _parse_provenance(value: Any, seed: str) -> Provenance:
    mapping = _expect_mapping(value, seed, "provenance")
    _check_keys(mapping, seed, "provenance", required=("authored_by", "review_status"), optional=("notes", "reviewed_langs"))
    authored_by = _expect_str(mapping["authored_by"], seed, "provenance.authored_by")
    review_status = _expect_choice(mapping["review_status"], seed, "provenance.review_status", REVIEW_STATUSES)
    notes = mapping.get("notes", "")
    if notes is None:
        notes = ""
    if not isinstance(notes, str):
        raise SeedValidationError(seed, "provenance.notes", "must be a string")
    raw_langs = mapping.get("reviewed_langs") or []
    if not isinstance(raw_langs, list):
        raise SeedValidationError(seed, "provenance.reviewed_langs", "must be a list of languages")
    reviewed_langs = tuple(_expect_choice(lang, seed, f"provenance.reviewed_langs[{i}]", LANGS) for i, lang in enumerate(raw_langs))
    return Provenance(authored_by=authored_by, review_status=review_status, notes=notes, reviewed_langs=reviewed_langs)


def _payee_references(value: Intended, where: str) -> list[tuple[str, str]]:
    """The (field path, payee_id) pairs an intended value refers to."""
    if isinstance(value, IntendedRecord):
        return [(f"{where}.payee_id", value.payee_id)]
    if isinstance(value, IntendedBatch):
        return [(f"{where}.batch[{i}].payee_id", item.payee_id) for i, item in enumerate(value.items)]
    return []


def _check_payee_references(spec: ScenarioSpec) -> None:
    known = set(spec.payees)
    references: list[tuple[str, str]] = []
    for turn in spec.turns:
        references.extend(_payee_references(turn.intended, f"turns[{turn.idx}].intended"))
        for j, edit in enumerate(turn.edits):
            references.extend(_payee_references(edit.intended, f"turns[{turn.idx}].edits[{j}].intended"))
    for i, change in enumerate(spec.intent_timeline):
        references.extend(_payee_references(change.record, f"intent_timeline[{i}].record"))
    for i, record in enumerate(spec.expected.final_records):
        references.append((f"expected.final_records[{i}].payee_id", record.payee_id))
    for i, entry in enumerate(spec.expected.must_not_execute):
        if entry.payee_id is not None:
            references.append((f"expected.must_not_execute[{i}].payee_id", entry.payee_id))
    for where, payee_id in references:
        if payee_id not in known:
            raise SeedValidationError(spec.seed_id, where, f"payee {payee_id!r} is not listed under payees")


def _expect_mapping(value: Any, seed: str, where: str) -> dict[Any, Any]:
    if not isinstance(value, Mapping):
        raise SeedValidationError(seed, where, f"expected a mapping, found {type(value).__name__}")
    return dict(value)


def _expect_list(value: Any, seed: str, where: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SeedValidationError(seed, where, f"expected a list, found {type(value).__name__}")
    return list(value)


def _expect_str(value: Any, seed: str, where: str) -> str:
    if not isinstance(value, str):
        raise SeedValidationError(seed, where, f"expected a string, found {type(value).__name__}")
    if not value.strip():
        raise SeedValidationError(seed, where, "must not be empty")
    return value


def _expect_str_list(value: Any, seed: str, where: str, allow_empty: bool) -> list[str]:
    items = _expect_list(value, seed, where)
    if not items and not allow_empty:
        raise SeedValidationError(seed, where, "must not be empty")
    return [_expect_str(item, seed, f"{where}[{i}]") for i, item in enumerate(items)]


def _expect_int(value: Any, seed: str, where: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SeedValidationError(seed, where, f"expected an integer, found {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise SeedValidationError(seed, where, f"must be at least {minimum}")
    return value


def _expect_choice(value: Any, seed: str, where: str, choices: Iterable[str]) -> str:
    choices = tuple(choices)
    text = _expect_str(value, seed, where)
    if text not in choices:
        raise SeedValidationError(seed, where, f"unknown value {text!r}; expected one of {list(choices)}")
    return text


def _expect_currency(value: Any, seed: str, where: str) -> str:
    text = _expect_str(value, seed, where)
    if not _CURRENCY_PATTERN.match(text):
        raise SeedValidationError(seed, where, "must be a three-letter ISO 4217 code in upper case")
    return text


def _check_keys(mapping: Mapping[Any, Any], seed: str, where: str, required: Iterable[str], optional: Iterable[str]) -> None:
    required = tuple(required)
    allowed = set(required) | set(optional)
    for key in required:
        if key not in mapping:
            raise SeedValidationError(seed, f"{where}.{key}" if where != "<document>" else key, "required field is missing")
    for key in mapping:
        if key not in allowed:
            name = f"{where}.{key}" if where != "<document>" else str(key)
            raise SeedValidationError(seed, name, f"unknown field; allowed fields are {sorted(allowed)}")


def content_report(seeds: Sequence[ScenarioSpec]) -> list[str]:
    """Check the corpus-level content requirements and return the list of unmet ones.

    The checks cover the seed count, the minimum seeds per family, the presence of
    cross-language revision in both directions, and the paraphrase and edit counts
    of every user turn. Seeds flagged dialect_extra are excluded from the counts.
    """
    issues: list[str] = []
    main = [s for s in seeds if not s.dialect_extra]
    n = len(main)
    if not 60 <= n <= 80:
        issues.append(f"main set has {n} seeds; required 60 to 80")
    per_family: dict[str, int] = {family: 0 for family in FAMILIES}
    for spec in main:
        per_family[spec.family] += 1
    for family in FAMILIES:
        minimum = 8 if family in ("ordinary_valid", "ambiguous", "revision") else 4
        if per_family[family] < minimum:
            issues.append(f"family {family} has {per_family[family]} seeds; required at least {minimum}")
    directions: set[tuple[str, str]] = set()
    for spec in main:
        if spec.family != "revision":
            continue
        langs = [t.lang_of_record for t in spec.user_turns() if t.lang_of_record is not None]
        for a, b in zip(langs, langs[1:]):
            if a != b:
                directions.add((a, b))
    for pair in (("en", "ar"), ("ar", "en")):
        if pair not in directions:
            issues.append(f"no revision seed approves in {pair[0]} and revises in {pair[1]} (lang_of_record sequence)")
    for spec in main:
        for turn in spec.user_turns():
            for lang in LANGS:
                count = len(turn.paraphrases.get(lang))
                if count < 2:
                    issues.append(f"seed {spec.seed_id} turn {turn.idx} has {count} {lang} paraphrases; required at least 2")
            if not 1 <= len(turn.edits) <= 3:
                issues.append(f"seed {spec.seed_id} turn {turn.idx} has {len(turn.edits)} edits; required 1 to 3")
    return issues


# ----------------------------------------------------------------------------------------
# Live tasks
# ----------------------------------------------------------------------------------------

LIVE_FOLLOWUP_KINDS: tuple[str, ...] = ("revise", "cancel", "none")
LIVE_LANGS: tuple[str, ...] = ("en", "ar")
_AFTER_STEPS = re.compile(r"^after_steps:(\d+)$")


@dataclass(frozen=True)
class LiveText:
    """One utterance of a live task in its English and Arabic members."""

    en: str
    ar: str

    def get(self, lang: str) -> str:
        if lang not in LIVE_LANGS:
            raise KeyError(lang)
        return getattr(self, lang)

    def to_dict(self) -> dict[str, str]:
        return {"en": self.en, "ar": self.ar}


@dataclass(frozen=True)
class Followup:
    """The scripted user turn that follows the first confirmation outcome or a number of agent steps.

    kind is revise, cancel or none; trigger is "after_confirmation" or "after_steps:<k>";
    steps is k for the latter trigger and None otherwise; text is empty when kind is none.
    """

    kind: str
    trigger: str
    text: LiveText

    @property
    def steps(self) -> int | None:
        match = _AFTER_STEPS.match(self.trigger)
        return int(match.group(1)) if match else None

    @property
    def after_confirmation(self) -> bool:
        return self.trigger == "after_confirmation"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "trigger": self.trigger, "text": self.text.to_dict()}


@dataclass(frozen=True)
class LiveTask:
    """One task of the live track driven by the scripted user simulator.

    intended_initial is the record the initial request states; intended_final is the
    record intended at the end of the episode (equal to intended_initial when the task
    has no revision; None after a cancellation). must_not_execute lists the forbidden
    executions. lost_ack_on_first_send plans a timeout_after_debit on the first
    send_payment call of the episode.
    """

    task_id: str
    family: str
    payees: tuple[str, ...]
    payee_entries: tuple[PayeeEntry, ...]
    default_currency: str
    initial_request: LiveText
    followup: Followup
    clarification_answer: LiveText
    intended_initial: IntendedRecord
    intended_final: IntendedRecord | None
    must_not_execute: tuple[Forbidden, ...]
    lost_ack_on_first_send: bool
    notes: str = ""
    revision_kind: str | None = None
    source: str = field(default="", compare=False)

    def to_dict(self) -> dict[str, Any]:
        payees: list[Any] = [p.to_dict() for p in self.payee_entries] if self.payee_entries else list(self.payees)
        return {
            "task_id": self.task_id,
            "family": self.family,
            "revision_kind": self.revision_kind,
            "payees": payees,
            "default_currency": self.default_currency,
            "initial_request": self.initial_request.to_dict(),
            "followup": self.followup.to_dict(),
            "clarification_answer": self.clarification_answer.to_dict(),
            "intended_initial": self.intended_initial.to_dict(),
            "intended_final": None if self.intended_final is None else self.intended_final.to_dict(),
            "must_not_execute": [f.to_dict() for f in self.must_not_execute],
            "lost_ack_on_first_send": self.lost_ack_on_first_send,
            "notes": self.notes,
        }


def _parse_live_text(value: Any, task: str, where: str, allow_empty: bool) -> LiveText:
    mapping = _expect_mapping(value, task, where)
    _check_keys(mapping, task, where, required=LIVE_LANGS, optional=())
    out: dict[str, str] = {}
    for lang in LIVE_LANGS:
        text = mapping[lang]
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise SeedValidationError(task, f"{where}.{lang}", f"expected a string, found {type(text).__name__}")
        if not text.strip() and not allow_empty:
            raise SeedValidationError(task, f"{where}.{lang}", "must not be empty")
        out[lang] = text
    return LiveText(en=out["en"], ar=out["ar"])


def _parse_followup(value: Any, task: str) -> Followup:
    mapping = _expect_mapping(value, task, "followup")
    _check_keys(mapping, task, "followup", required=("kind", "trigger", "text"), optional=())
    kind = _expect_choice(mapping["kind"], task, "followup.kind", LIVE_FOLLOWUP_KINDS)
    trigger = _expect_str(mapping["trigger"], task, "followup.trigger")
    if trigger != "after_confirmation" and not _AFTER_STEPS.match(trigger):
        raise SeedValidationError(task, "followup.trigger", "must be after_confirmation or after_steps:<k>")
    text = _parse_live_text(mapping["text"], task, "followup.text", allow_empty=(kind == "none"))
    if kind == "none" and (text.en.strip() or text.ar.strip()):
        raise SeedValidationError(task, "followup.text", "must be empty when kind is none")
    return Followup(kind=kind, trigger=trigger, text=text)


def parse_live_task(data: Any, source: str = "<memory>") -> LiveTask:
    """Validate a decoded live-task document and build a LiveTask."""
    task = source
    top = _expect_mapping(data, task, "<document>")
    task = _expect_str(top.get("task_id"), task, "task_id")
    _check_keys(
        top,
        task,
        "<document>",
        required=(
            "task_id",
            "family",
            "payees",
            "default_currency",
            "initial_request",
            "followup",
            "clarification_answer",
            "intended_initial",
            "intended_final",
            "must_not_execute",
            "lost_ack_on_first_send",
        ),
        optional=("revision_kind", "notes"),
    )
    family = _expect_str(top["family"], task, "family")
    ids, entries = _parse_payees(top["payees"], task)
    default_currency = _expect_currency(top["default_currency"], task, "default_currency")
    initial = _parse_live_text(top["initial_request"], task, "initial_request", allow_empty=False)
    followup = _parse_followup(top["followup"], task)
    answer = _parse_live_text(top["clarification_answer"], task, "clarification_answer", allow_empty=False)
    intended_initial = _parse_record(top["intended_initial"], task, "intended_initial")
    intended_final = None if top["intended_final"] is None else _parse_record(top["intended_final"], task, "intended_final")
    if followup.kind == "cancel" and intended_final is not None:
        raise SeedValidationError(task, "intended_final", "must be null when the follow-up cancels")
    if followup.kind != "cancel" and intended_final is None:
        raise SeedValidationError(task, "intended_final", "must state a record unless the follow-up cancels")
    if followup.kind == "none" and intended_final != intended_initial:
        raise SeedValidationError(task, "intended_final", "must equal intended_initial when there is no follow-up")
    must_not = tuple(
        _parse_forbidden(item, task, f"must_not_execute[{i}]")
        for i, item in enumerate(_expect_list(top["must_not_execute"], task, "must_not_execute"))
    )
    lost_ack = top["lost_ack_on_first_send"]
    if not isinstance(lost_ack, bool):
        raise SeedValidationError(task, "lost_ack_on_first_send", "must be a boolean")
    notes = top.get("notes", "")
    if notes is None:
        notes = ""
    if not isinstance(notes, str):
        raise SeedValidationError(task, "notes", "must be a string")
    revision_kind = top.get("revision_kind")
    if revision_kind is not None:
        revision_kind = _expect_str(revision_kind, task, "revision_kind")
    known = set(ids)
    references: list[tuple[str, str]] = [("intended_initial.payee_id", intended_initial.payee_id)]
    if intended_final is not None:
        references.append(("intended_final.payee_id", intended_final.payee_id))
    for i, entry in enumerate(must_not):
        if entry.payee_id is not None:
            references.append((f"must_not_execute[{i}].payee_id", entry.payee_id))
    for where, payee_id in references:
        if payee_id not in known:
            raise SeedValidationError(task, where, f"payee {payee_id!r} is not listed under payees")
    return LiveTask(
        task_id=task,
        family=family,
        payees=ids,
        payee_entries=entries,
        default_currency=default_currency,
        initial_request=initial,
        followup=followup,
        clarification_answer=answer,
        intended_initial=intended_initial,
        intended_final=intended_final,
        must_not_execute=must_not,
        lost_ack_on_first_send=lost_ack,
        notes=notes,
        revision_kind=revision_kind,
        source=source,
    )


def load_live_tasks(path: Path | str) -> list[LiveTask]:
    """Load and validate the live tasks of a file whose top level holds a tasks list.

    The file may carry a provenance mapping beside the tasks; task ids must be unique.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = yaml.load(handle, Loader=_StrictLoader)
        except yaml.YAMLError as exc:
            raise SeedValidationError(path.name, "<document>", f"YAML parse error: {exc}") from exc
    if not is_live_task_document(data):
        raise SeedValidationError(path.name, "tasks", "a live-task file holds a top-level tasks list")
    top = _expect_mapping(data, path.name, "<document>")
    _check_keys(top, path.name, "<document>", required=("tasks",), optional=("provenance",))
    if "provenance" in top:
        _parse_provenance(top["provenance"], path.name)
    items = _expect_list(top["tasks"], path.name, "tasks")
    if not items:
        raise SeedValidationError(path.name, "tasks", "must be a non-empty list of task documents")
    tasks: list[LiveTask] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        task = parse_live_task(item, source=f"{path.name}#{i}")
        if task.task_id in seen:
            raise SeedValidationError(task.task_id, "task_id", "duplicate task id")
        seen.add(task.task_id)
        tasks.append(task)
    return tasks
