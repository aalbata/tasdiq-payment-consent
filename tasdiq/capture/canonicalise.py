"""Canonicalisation of a RawExtraction into record fields.

The payee mention is resolved through the directory (exact match after text normalisation,
all matches kept), the amount text is parsed into integer minor units with
``tasdiq.money.parse_amount`` (Eastern and Western digits, number words, separators,
currency words), the currency follows the precedence explicit text, then the previous record
for a referring utterance, then the default, and a referring utterance inherits missing
fields from the previous record. The ``noncanon`` variant skips all of this and takes the
model's own integer amount, currency code and exactly spelled display name.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .extract import RawExtraction

BINDING_FIELDS = ("payee_id", "amount_minor", "currency")
REFERRING_ACTIONS = ("revision", "cancellation")
BLOCKING_AMOUNT_FLAGS = ("multiple_amounts", "unrecognised_fraction", "unrecognised_number_word", "negative")

AmountParser = Callable[..., Any]

_HARAKAT = re.compile("[ؐ-ًؚ-ْٰـ]")


@dataclass
class CanonResult:
    """Record fields produced by the canonicaliser, with their provenance.

    ``payee_matches`` lists every directory identifier the mention matched; ``payee_id`` is
    set only for exactly one match (or an inherited value). ``amount_candidates`` and
    ``amount_flags`` come from the amount parser. ``ask_field`` names the first binding
    field that could not be resolved, or None. ``had_previous`` records whether a previous
    record was available, so that the gate can distinguish an unresolved reference from a
    resolved one.
    """

    action_type: str
    payee_id: str | None = None
    payee_matches: list[str] = field(default_factory=list)
    amount_minor: int | None = None
    currency: str | None = None
    amount_candidates: list[tuple[int, str | None]] = field(default_factory=list)
    amount_flags: set[str] = field(default_factory=set)
    payee_source: str | None = None
    amount_source: str | None = None
    currency_source: str | None = None
    negated: bool = False
    refers_to_previous: bool = False
    had_previous: bool = False
    ask_field: str | None = None
    mode: str = "canon"
    notes: list[str] = field(default_factory=list)

    def record_fields(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "payee_id": self.payee_id,
            "amount_minor": self.amount_minor,
            "currency": self.currency,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.record_fields(),
            "payee_matches": list(self.payee_matches),
            "amount_candidates": [list(c) for c in self.amount_candidates],
            "amount_flags": sorted(self.amount_flags),
            "payee_source": self.payee_source,
            "amount_source": self.amount_source,
            "currency_source": self.currency_source,
            "negated": self.negated,
            "refers_to_previous": self.refers_to_previous,
            "had_previous": self.had_previous,
            "ask_field": self.ask_field,
            "mode": self.mode,
            "notes": list(self.notes),
        }


def record_field(record: Any, name: str) -> Any:
    """Read ``name`` from a mapping or an object with attributes; None when absent."""
    if record is None:
        return None
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def iter_payees(directory: Any) -> list[Any]:
    """Every payee of a directory, read through ``payees``, ``all()`` or iteration."""
    source = getattr(directory, "payees", None)
    if source is None and hasattr(directory, "all"):
        source = directory.all()
    if source is None:
        source = getattr(directory, "_payees", None)
    if source is None:
        source = directory
    if isinstance(source, Mapping):
        return list(source.values())
    return list(source)


def exact_display_match(directory: Any, name: str | None) -> list[str]:
    """Identifiers of payees whose English name, Arabic name or an alias equals ``name``
    exactly (whitespace trimmed, no other normalisation)."""
    if not name or not str(name).strip():
        return []
    wanted = str(name).strip()
    matches: list[str] = []
    for payee in iter_payees(directory):
        labels = [record_field(payee, "name_en"), record_field(payee, "name_ar")]
        labels.extend(record_field(payee, "aliases") or [])
        if any(label is not None and str(label).strip() == wanted for label in labels):
            pid = record_field(payee, "payee_id")
            if pid is not None and pid not in matches:
                matches.append(str(pid))
    return matches


def _default_amount_parser() -> AmountParser:
    from tasdiq.money import parse_amount

    return parse_amount


def minor_unit_codes(text: str | None) -> frozenset[str]:
    """Codes whose minor unit the currency text names, through tasdiq.money; empty when it names none."""
    if text is None or not str(text).strip():
        return frozenset()
    try:
        from tasdiq.money import lookup_minor_unit
    except ImportError:
        return frozenset()
    return lookup_minor_unit(str(text))


def _currency_table() -> Mapping[str, Any] | None:
    try:
        from tasdiq.money import CURRENCIES
    except ImportError:
        return None
    return CURRENCIES


def _loose(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = _HARAKAT.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = text.strip().strip(".,;:!?()[]\"'").strip()
    return " ".join(text.lower().split())


def currency_code_from_text(text: str | None, currencies: Mapping[str, Any] | None = None) -> str | None:
    """ISO 4217 code for a currency word, symbol or code, or None when unrecognised.

    ``currencies`` defaults to ``tasdiq.money.CURRENCIES``; without that table only bare
    three-letter codes are accepted.
    """
    if text is None or not str(text).strip():
        return None
    table = currencies if currencies is not None else _currency_table()
    raw = str(text).strip()
    code = raw.upper()
    if len(code) == 3 and code.isalpha():
        if table is None or code in table:
            return code
    if table is None:
        return None
    loose = _loose(raw)
    for key, info in table.items():
        labels: list[str] = [str(record_field(info, "code") or key)]
        for attr in ("names_en", "names_ar", "symbols"):
            labels.extend(str(x) for x in (record_field(info, attr) or []))
        for label in labels:
            if label and (_loose(label) == loose or label.strip() == raw):
                return str(record_field(info, "code") or key)
    return None


def _ids(payees: Iterable[Any]) -> list[str]:
    ids: list[str] = []
    for payee in payees:
        pid = record_field(payee, "payee_id")
        if pid is not None and str(pid) not in ids:
            ids.append(str(pid))
    return ids


def _inherits_previous(
    action_type: str,
    refers: bool,
    had_previous: bool,
    payee_id: str | None,
    previous_record: Any,
    notes: list[str],
) -> bool:
    """Whether amount and currency may be inherited from the previous record.

    A referring utterance inherits them, except a cancellation that names a payee other
    than the previous record's, which concerns a different payment.
    """
    if not (refers and had_previous):
        return False
    previous_payee = record_field(previous_record, "payee_id")
    if (
        action_type == "cancellation"
        and payee_id is not None
        and previous_payee is not None
        and payee_id != str(previous_payee)
    ):
        notes.append("cancellation_names_other_payee")
        return False
    return True


def _first_unresolved(action_type: str, payee_id: str | None, amount: int | None, currency: str | None) -> str | None:
    if payee_id is None:
        return "payee_id"
    if action_type == "cancellation":
        return None
    if amount is None:
        return "amount_minor"
    if currency is None:
        return "currency"
    return None


def canonicalise(
    raw: RawExtraction,
    directory: Any,
    default_currency: str | None,
    previous_record: Any = None,
    *,
    utterance: str | None = None,
    amount_parser: AmountParser | None = None,
    currencies: Mapping[str, Any] | None = None,
) -> CanonResult:
    """Resolve a RawExtraction into canonical record fields.

    ``utterance`` is parsed for an amount when the model returned no amount text (or text
    without any recognisable number). ``amount_parser`` defaults to
    ``tasdiq.money.parse_amount`` and must accept ``(text, default_currency)``;
    ``currencies`` defaults to ``tasdiq.money.CURRENCIES`` and is used to read
    ``currency_text``. A ``currency_text`` that names a minor unit ("fils", "فلس") is
    appended to the amount text so that the parser applies the minor scale. When the
    currency named next to the number differs from ``currency_text``, the number's own
    currency is kept (it set the scale of the amount) and ``currency_conflict`` is noted
    for the gate.
    """
    parser = amount_parser or _default_amount_parser()
    refers = bool(raw.refers_to_previous) or raw.action_type in REFERRING_ACTIONS
    had_previous = previous_record is not None
    notes: list[str] = []

    payee_id: str | None = None
    matches: list[str] = []
    payee_source: str | None = None
    mention = (raw.payee_mention or "").strip()
    if mention:
        matches = _ids(directory.resolve(mention))
        if len(matches) == 1:
            payee_id = matches[0]
            payee_source = "mention"
    elif refers and had_previous and record_field(previous_record, "payee_id") is not None:
        payee_id = str(record_field(previous_record, "payee_id"))
        matches = [payee_id]
        payee_source = "previous"

    inherit = _inherits_previous(raw.action_type, refers, had_previous, payee_id, previous_record, notes)

    explicit_code = currency_code_from_text(raw.currency_text, currencies)
    minor_codes = minor_unit_codes(raw.currency_text) if explicit_code is None else frozenset()
    if raw.currency_text and explicit_code is None and not minor_codes:
        notes.append("currency_text_unrecognised")
    if minor_codes:
        notes.append("currency_text_minor_unit")
    previous_currency = record_field(previous_record, "currency") if inherit else None
    fallback_currency = explicit_code or (str(previous_currency) if previous_currency else None) or default_currency
    fallback_source = (
        "explicit" if explicit_code else "previous" if previous_currency else "default" if default_currency else None
    )

    amount: int | None = None
    amount_source: str | None = None
    parsed_currency: str | None = None
    parsed_explicit = False
    parsed_source: str | None = None
    flags: set[str] = set()
    candidates: list[tuple[int, str | None]] = []
    amount_text = (raw.amount_text or "").strip()
    if amount_text and minor_codes and str(raw.currency_text).strip().lower() not in amount_text.lower():
        amount_text = f"{amount_text} {str(raw.currency_text).strip()}"
    texts: list[tuple[str, str]] = []
    if amount_text:
        texts.append((amount_text, "amount_text"))
    if utterance and str(utterance).strip():
        texts.append((str(utterance), "utterance"))
    for text, source in texts:
        if source == "utterance" and amount_text:
            if candidates or flags.intersection(BLOCKING_AMOUNT_FLAGS):
                break
            notes.append("amount_text_unparsed")
        parse = parser(text, fallback_currency)
        parse_flags = set(record_field(parse, "flags") or ())
        flags |= parse_flags
        candidates = [(int(a), c) for a, c in (record_field(parse, "candidates") or [])]
        amount = record_field(parse, "amount_minor")
        parsed_currency = record_field(parse, "currency")
        parsed_explicit = parsed_currency is not None and "no_currency" not in parse_flags
        parsed_source = source
        if amount is not None:
            amount_source = source
            break
    if amount is None and not candidates and not flags.intersection(BLOCKING_AMOUNT_FLAGS):
        if inherit and record_field(previous_record, "amount_minor") is not None:
            amount = int(record_field(previous_record, "amount_minor"))
            amount_source = "previous"

    currency: str | None = None
    currency_source: str | None = None
    if explicit_code and parsed_explicit and str(parsed_currency).upper() != explicit_code:
        currency, currency_source = str(parsed_currency), parsed_source
        notes.append("currency_conflict")
    elif explicit_code:
        currency, currency_source = explicit_code, "explicit"
    elif parsed_explicit:
        currency, currency_source = str(parsed_currency), parsed_source
    elif amount_source == "previous" and previous_currency:
        currency, currency_source = str(previous_currency), "previous"
    elif fallback_currency:
        currency, currency_source = fallback_currency, fallback_source
    if currency is not None:
        currency = str(currency).upper()

    return CanonResult(
        action_type=raw.action_type,
        payee_id=payee_id,
        payee_matches=matches,
        amount_minor=amount,
        currency=currency,
        amount_candidates=candidates,
        amount_flags=flags,
        payee_source=payee_source,
        amount_source=amount_source,
        currency_source=currency_source,
        negated=bool(raw.negated),
        refers_to_previous=refers,
        had_previous=had_previous,
        ask_field=_first_unresolved(raw.action_type, payee_id, amount, currency),
        mode="canon",
        notes=notes,
    )


def canonicalise_noncanon(
    raw: RawExtraction,
    directory: Any,
    default_currency: str | None,
    previous_record: Any = None,
) -> CanonResult:
    """Record fields without the canonicaliser: the model's integer ``amount_minor``, its
    currency code (else the default) and its display name matched exactly against the
    directory. A referring utterance still inherits missing fields from the previous record.
    """
    refers = bool(raw.refers_to_previous) or raw.action_type in REFERRING_ACTIONS
    had_previous = previous_record is not None
    notes: list[str] = []

    payee_id: str | None = None
    payee_source: str | None = None
    matches = exact_display_match(directory, raw.payee_mention)
    if len(matches) == 1:
        payee_id, payee_source = matches[0], "model"
    elif not (raw.payee_mention or "").strip() and refers and had_previous:
        prev = record_field(previous_record, "payee_id")
        if prev is not None:
            payee_id, matches, payee_source = str(prev), [str(prev)], "previous"

    inherit = _inherits_previous(raw.action_type, refers, had_previous, payee_id, previous_record, notes)

    amount = raw.amount_minor
    amount_source = "model" if amount is not None else None
    if amount is None and inherit and record_field(previous_record, "amount_minor") is not None:
        amount, amount_source = int(record_field(previous_record, "amount_minor")), "previous"

    currency: str | None = None
    currency_source: str | None = None
    code = (raw.currency or "").strip().upper()
    if len(code) == 3 and code.isalpha():
        currency, currency_source = code, "model"
    elif code:
        notes.append("currency_code_invalid")
    if currency is None and inherit and record_field(previous_record, "currency"):
        currency, currency_source = str(record_field(previous_record, "currency")).upper(), "previous"
    if currency is None and default_currency:
        currency, currency_source = str(default_currency).upper(), "default"

    candidates = [(int(amount), currency)] if amount is not None else []
    return CanonResult(
        action_type=raw.action_type,
        payee_id=payee_id,
        payee_matches=matches,
        amount_minor=amount,
        currency=currency,
        amount_candidates=candidates,
        amount_flags=set(),
        payee_source=payee_source,
        amount_source=amount_source,
        currency_source=currency_source,
        negated=bool(raw.negated),
        refers_to_previous=refers,
        had_previous=had_previous,
        ask_field=_first_unresolved(raw.action_type, payee_id, amount, currency),
        mode="noncanon",
        notes=notes,
    )
