"""Ambiguity gate between canonicalisation and the consent channel.

The gate turns a CanonResult into a capture record, a clarification request naming one
field, or no record (for an inquiry or an unrelated utterance). It asks when the payee
mention matches zero or several payees, the amount is missing or indeterminate (several
amounts, unrecognised fraction, a sign or an unrecognised number word), the currency is
unknown with no default or the currency named next to the number conflicts with the
model's currency text, the utterance refers to a previous payment that does not exist, a
negation carries no replacement (the only amount in the utterance is the rejected one),
the model is unsure whether the utterance is a request at all, or the model marked a
binding field as uncertain. The ``no_gate`` ablation binds the first payee match, the
first amount candidate and the default currency without asking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonicalise import BINDING_FIELDS, AmountParser, CanonResult, record_field
from .extract import RawExtraction

ASK_FIELDS = ("payee_id", "amount_minor", "currency", "reference", "action_type")

UNCERTAIN_TO_FIELD: dict[str, str] = {
    "payee_mention": "payee_id",
    "payee": "payee_id",
    "payee_id": "payee_id",
    "amount_text": "amount_minor",
    "amount": "amount_minor",
    "amount_minor": "amount_minor",
    "currency_text": "currency",
    "currency": "currency",
    "action_type": "action_type",
}

NO_RECORD_ACTIONS = ("inquiry", "other")


@dataclass
class CaptureRecord:
    """Record fields handed to the consent channel.

    ``amount_minor`` and ``currency`` may be None only for a cancellation that names the
    payee without restating the amount. ``total_max_minor`` is left to the monitor, which
    adds the quoted fee.
    """

    action_type: str
    payee_id: str | None
    amount_minor: int | None
    currency: str | None
    total_max_minor: int | None = None

    def binding(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in BINDING_FIELDS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "payee_id": self.payee_id,
            "amount_minor": self.amount_minor,
            "currency": self.currency,
            "total_max_minor": self.total_max_minor,
        }


@dataclass
class AskOrRecord:
    """Gate outcome: ``kind`` is "record", "ask" or "no_record"."""

    kind: str
    record: CaptureRecord | None = None
    ask_field: str | None = None
    reason: str = ""

    @property
    def is_record(self) -> bool:
        return self.kind == "record"

    @property
    def is_ask(self) -> bool:
        return self.kind == "ask"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "record": self.record.to_dict() if self.record else None,
            "ask_field": self.ask_field,
            "reason": self.reason,
        }


def ask(field_name: str, reason: str) -> AskOrRecord:
    return AskOrRecord(kind="ask", ask_field=field_name, reason=reason)


def no_record(reason: str) -> AskOrRecord:
    return AskOrRecord(kind="no_record", reason=reason)


def _record(canon: CanonResult) -> AskOrRecord:
    return AskOrRecord(
        kind="record",
        record=CaptureRecord(
            action_type=canon.action_type,
            payee_id=canon.payee_id,
            amount_minor=canon.amount_minor,
            currency=canon.currency,
        ),
        reason="ok",
    )


def uncertain_binding_fields(raw: RawExtraction) -> list[str]:
    """Binding fields (and action_type) the model marked as uncertain, in record names."""
    names: list[str] = []
    for item in raw.uncertain_fields:
        mapped = UNCERTAIN_TO_FIELD.get(str(item).strip().lower())
        if mapped and mapped not in names:
            names.append(mapped)
    return names


def _equals_previous(canon: CanonResult, previous_record: Any) -> bool:
    """True when the canonical binding fields all equal the previous record's."""
    if previous_record is None:
        return False
    for name in BINDING_FIELDS:
        prev = record_field(previous_record, name)
        current = getattr(canon, name)
        if prev is None or current is None:
            return False
        if name == "currency":
            prev, current = str(prev).upper(), str(current).upper()
        if prev != current:
            return False
    return True


def _utterance_candidates(utterance: str | None, canon: CanonResult, amount_parser: AmountParser | None) -> list[int] | None:
    """Distinct amounts the utterance itself contains, or None when it was not given or cannot be parsed."""
    if utterance is None or not str(utterance).strip():
        return None
    parser = amount_parser
    if parser is None:
        try:
            from tasdiq.money import parse_amount as parser
        except ImportError:
            return None
    try:
        parsed = parser(str(utterance), canon.currency)
    except Exception:
        return None
    candidates = record_field(parsed, "candidates") or []
    amounts: list[int] = []
    for item in candidates:
        try:
            amount = int(item[0])
        except (TypeError, ValueError, IndexError):
            continue
        if amount not in amounts:
            amounts.append(amount)
    single = record_field(parsed, "amount_minor")
    if single is not None and int(single) not in amounts:
        amounts.append(int(single))
    return amounts


def negation_lacks_replacement(canon: CanonResult, raw: RawExtraction, utterance: str | None, amount_parser: AmountParser | None) -> bool:
    """True when a negation that refers to an earlier record states no replacement amount.

    The amount of a negated referring utterance came either from the utterance fallback
    (the model gave no amount text; the fallback cannot tell the rejected value from a
    replacement) or from the model's amount text, in which case the utterance must hold a
    second value: when its only distinct amount equals the canonical amount, that amount
    is the rejected one.
    """
    if not (canon.negated and canon.had_previous) or canon.amount_minor is None:
        return False
    if canon.amount_source == "utterance":
        return True
    if canon.amount_source == "amount_text" and raw.amount_text:
        amounts = _utterance_candidates(utterance, canon, amount_parser)
        return amounts is not None and len(amounts) == 1 and amounts[0] == int(canon.amount_minor)
    return False


def ambiguity_gate(canon: CanonResult, raw: RawExtraction, previous_record: Any = None, utterance: str | None = None, amount_parser: AmountParser | None = None) -> AskOrRecord:
    """Decide between a record, a clarification and no record (see module docstring).

    ``previous_record`` is used to tell a negation that restates the earlier record (no
    replacement given) from one that replaces it; the reference check itself uses
    ``canon.had_previous``. ``utterance`` lets the gate check that a negated referring
    utterance holds a replacement value besides the rejected one. A negated utterance with
    no previous record is taken at its resolved fields, since the model reports the
    replacement values, not the rejected ones.
    """
    uncertain = uncertain_binding_fields(raw)
    if "action_type" in uncertain:
        return ask("action_type", "uncertain:action_type")
    action = canon.action_type
    if action in NO_RECORD_ACTIONS:
        return no_record(action)
    if canon.refers_to_previous and not canon.had_previous:
        return ask("reference", "reference_without_previous")

    if action == "cancellation":
        if canon.payee_id is not None:
            return _record(canon)
        if len(canon.payee_matches) > 1:
            return ask("payee_id", "payee_ambiguous")
        if raw.payee_mention and not canon.payee_matches:
            return ask("payee_id", "payee_unknown")
        return ask("reference", "cancellation_target_unresolved")

    if canon.payee_id is None:
        if len(canon.payee_matches) > 1:
            return ask("payee_id", "payee_ambiguous")
        if raw.payee_mention and not canon.payee_matches:
            return ask("payee_id", "payee_unknown")
        return ask("payee_id", "payee_missing")
    if canon.amount_minor is None:
        if "multiple_amounts" in canon.amount_flags:
            return ask("amount_minor", "amount_multiple")
        if "unrecognised_fraction" in canon.amount_flags:
            return ask("amount_minor", "amount_fraction")
        return ask("amount_minor", "amount_missing")
    if canon.currency is None:
        return ask("currency", "currency_missing")
    if "currency_conflict" in canon.notes:
        return ask("currency", "currency_conflict")
    for name in uncertain:
        if name in BINDING_FIELDS:
            return ask(name, f"uncertain:{name}")
    if canon.negated and _equals_previous(canon, previous_record):
        if raw.amount_text:
            return ask("amount_minor", "negated_without_replacement")
        if raw.payee_mention:
            return ask("payee_id", "negated_without_replacement")
        return ask("amount_minor", "negated_without_replacement")
    if negation_lacks_replacement(canon, raw, utterance, amount_parser):
        return ask("amount_minor", "negated_without_replacement")
    return _record(canon)


def no_gate(canon: CanonResult, raw: RawExtraction, default_currency: str | None = None) -> AskOrRecord:
    """Bind without ambiguity checks: first payee match, first amount candidate, default
    currency. Inquiries and unrelated utterances still produce no record; a clarification is
    returned only when a field cannot be filled at all."""
    action = canon.action_type
    if action in NO_RECORD_ACTIONS:
        return no_record(action)
    payee_id = canon.payee_id
    if payee_id is None and canon.payee_matches:
        payee_id = canon.payee_matches[0]
    amount = canon.amount_minor
    candidate_currency: str | None = None
    if amount is None and canon.amount_candidates:
        amount, candidate_currency = canon.amount_candidates[0]
    currency = canon.currency or candidate_currency or default_currency
    if currency is not None:
        currency = str(currency).upper()
    if action == "cancellation":
        if payee_id is None and not canon.had_previous:
            return ask("reference", "incomplete")
        return AskOrRecord(
            kind="record",
            record=CaptureRecord(action, payee_id, amount, currency),
            reason="ok",
        )
    if payee_id is None:
        return ask("payee_id", "incomplete")
    if amount is None:
        return ask("amount_minor", "incomplete")
    if currency is None:
        return ask("currency", "incomplete")
    return AskOrRecord(
        kind="record",
        record=CaptureRecord(action, payee_id, int(amount), currency),
        reason="ok",
    )
