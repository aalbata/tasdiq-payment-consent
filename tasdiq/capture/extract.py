"""Structured extraction of a payment request from one user utterance.

The model receives a concise bilingual system prompt, the saved payees as display names and
aliases (never payee identifiers), the dialogue context and the latest utterance, and
returns a ``RawExtraction`` object: an action type, the payee mention and amount text copied
from the utterance, a currency text, negation and cross-turn reference flags, and the list
of fields it is unsure about. A second schema variant, used by the ``noncanon`` ablation,
asks the model for the integer amount in minor units and the ISO currency code directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .model import Message, parse_json_object, strip_think

ACTION_TYPES = ("payment", "inquiry", "revision", "cancellation", "other")
CANON_UNCERTAIN_FIELDS = ("payee_mention", "amount_text", "currency_text", "action_type")
NONCANON_UNCERTAIN_FIELDS = ("payee_mention", "amount_minor", "currency", "action_type")
SCHEMA_VARIANTS = ("canon", "noncanon")

LANG_HINTS: dict[str, str] = {
    "en": "English",
    "ar": "Modern Standard Arabic",
    "mix": "Arabic and English mixed in one message",
}


class ExtractionParseError(ValueError):
    """The model output could not be read as a RawExtraction."""


@dataclass
class RawExtraction:
    """Fields returned by the extraction model before canonicalisation.

    ``amount_minor`` and ``currency`` are filled only by the ``noncanon`` schema variant.
    """

    action_type: str = "other"
    payee_mention: str | None = None
    amount_text: str | None = None
    currency_text: str | None = None
    negated: bool = False
    refers_to_previous: bool = False
    uncertain_fields: list[str] = field(default_factory=list)
    amount_minor: int | None = None
    currency: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RawExtraction":
        """Build a RawExtraction from a decoded JSON object, coercing loose values."""
        if not isinstance(data, Mapping):
            raise ExtractionParseError("extraction is not a JSON object")
        if "action_type" not in data:
            raise ExtractionParseError("extraction lacks action_type")
        action = str(data.get("action_type") or "").strip().lower()
        if action not in ACTION_TYPES:
            action = "other"
        uncertain_raw = data.get("uncertain_fields") or []
        if isinstance(uncertain_raw, str):
            uncertain_raw = [uncertain_raw]
        uncertain = [str(x).strip() for x in uncertain_raw if str(x).strip()]
        return cls(
            action_type=action,
            payee_mention=_opt_str(data.get("payee_mention")),
            amount_text=_opt_str(data.get("amount_text")),
            currency_text=_opt_str(data.get("currency_text")),
            negated=_as_bool(data.get("negated")),
            refers_to_previous=_as_bool(data.get("refers_to_previous")),
            uncertain_fields=uncertain,
            amount_minor=_opt_int(data.get("amount_minor")),
            currency=_opt_code(data.get("currency")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "payee_mention": self.payee_mention,
            "amount_text": self.amount_text,
            "currency_text": self.currency_text,
            "negated": self.negated,
            "refers_to_previous": self.refers_to_previous,
            "uncertain_fields": list(self.uncertain_fields),
            "amount_minor": self.amount_minor,
            "currency": self.currency,
        }


def _opt_str(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("null", "none"):
        return None
    return text


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text.lstrip("-").isdigit():
            return int(text)
    return None


def _opt_code(value: Any) -> str | None:
    text = _opt_str(value)
    if text is None:
        return None
    code = text.upper()
    return code if len(code) == 3 and code.isalpha() else text


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return False


RAW_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_type": {"type": "string", "enum": list(ACTION_TYPES)},
        "payee_mention": {"type": ["string", "null"]},
        "amount_text": {"type": ["string", "null"]},
        "currency_text": {"type": ["string", "null"]},
        "negated": {"type": "boolean"},
        "refers_to_previous": {"type": "boolean"},
        "uncertain_fields": {
            "type": "array",
            "items": {"type": "string", "enum": list(CANON_UNCERTAIN_FIELDS)},
        },
    },
    "required": [
        "action_type",
        "payee_mention",
        "amount_text",
        "currency_text",
        "negated",
        "refers_to_previous",
        "uncertain_fields",
    ],
    "additionalProperties": False,
}

NONCANON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_type": {"type": "string", "enum": list(ACTION_TYPES)},
        "payee_mention": {"type": ["string", "null"]},
        "amount_minor": {"type": ["integer", "null"]},
        "currency": {"type": ["string", "null"]},
        "negated": {"type": "boolean"},
        "refers_to_previous": {"type": "boolean"},
        "uncertain_fields": {
            "type": "array",
            "items": {"type": "string", "enum": list(NONCANON_UNCERTAIN_FIELDS)},
        },
    },
    "required": [
        "action_type",
        "payee_mention",
        "amount_minor",
        "currency",
        "negated",
        "refers_to_previous",
        "uncertain_fields",
    ],
    "additionalProperties": False,
}


def schema_for(variant: str) -> dict[str, Any]:
    """JSON schema for a schema variant name ("canon" or "noncanon")."""
    if variant == "canon":
        return RAW_EXTRACTION_SCHEMA
    if variant == "noncanon":
        return NONCANON_SCHEMA
    raise ValueError(f"unknown schema variant: {variant}")


_SYSTEM_TASK_EN = (
    "Task: read the latest user message in a banking chat and extract the payment request "
    "it contains as structured fields. Reply with one JSON object and nothing else."
)

_SYSTEM_TASK_AR = (
    "المهمة: اقرأ آخر رسالة للمستخدم في محادثة مصرفية واستخرج طلب الدفع فيها كحقول منظمة. "
    "أجب بكائن JSON واحد فقط ولا شيء غيره."
)

_FIELDS_CANON_EN = (
    "Fields:\n"
    "- action_type: \"payment\" (the user asks to send money), \"inquiry\" (the user asks a "
    "question, for example about a balance, a fee, a rate or a past payment, without asking "
    "to pay), \"revision\" (the user changes an earlier payment request), \"cancellation\" "
    "(the user withdraws an earlier payment request), or \"other\".\n"
    "- payee_mention: the recipient's name copied exactly as the user wrote it, or null.\n"
    "- amount_text: the amount copied exactly as the user wrote it (digits in their original "
    "script, number words, and the currency word when it is attached), or null.\n"
    "- currency_text: the currency word, symbol or code copied as written, or null.\n"
    "- negated: true when the message rejects or withdraws a stated payment detail.\n"
    "- refers_to_previous: true when the message refers to an earlier payment in the "
    "conversation instead of stating every detail itself.\n"
    "- uncertain_fields: names of the fields whose value is unclear, chosen from "
    "payee_mention, amount_text, currency_text, action_type; empty when everything is clear."
)

_FIELDS_NONCANON_EN = (
    "Fields:\n"
    "- action_type: \"payment\" (the user asks to send money), \"inquiry\" (the user asks a "
    "question, for example about a balance, a fee, a rate or a past payment, without asking "
    "to pay), \"revision\" (the user changes an earlier payment request), \"cancellation\" "
    "(the user withdraws an earlier payment request), or \"other\".\n"
    "- payee_mention: exactly one display name from the saved payee list below, spelled "
    "exactly as listed, or null.\n"
    "- amount_minor: the amount as an integer in minor units (AED, SAR, USD, EUR, GBP, QAR, "
    "EGP, INR: 100 per unit; KWD, BHD, OMR, JOD: 1000 per unit), converting any digit "
    "script or number words yourself, or null.\n"
    "- currency: the ISO 4217 code (AED, SAR, USD, EUR, GBP, KWD, ...), or null.\n"
    "- negated: true when the message rejects or withdraws a stated payment detail.\n"
    "- refers_to_previous: true when the message refers to an earlier payment in the "
    "conversation instead of stating every detail itself.\n"
    "- uncertain_fields: names of the fields whose value is unclear, chosen from "
    "payee_mention, amount_minor, currency, action_type; empty when everything is clear."
)

_RULES_CANON_EN = (
    "Rules: copy text, never translate it and never convert numbers. Do not guess: a detail "
    "the user did not state is null. A question is an inquiry even when it names a payee and "
    "an amount. Only the saved payees below can receive money; copy the name as the user "
    "wrote it even when it is an alias or a different spelling."
)

_RULES_NONCANON_EN = (
    "Rules: do not guess: a detail the user did not state is null. A question is an inquiry "
    "even when it names a payee and an amount. Only the saved payees below can receive money."
)

_RULES_AR = (
    "القواعد: لا تخمّن، فأي تفصيل لم يذكره المستخدم يكون null. السؤال عن الرصيد أو الرسوم أو "
    "دفعة سابقة هو inquiry وليس payment حتى لو ذُكر اسم ومبلغ. تعديل طلب سابق هو revision، "
    "وسحبه هو cancellation. ضع اسم كل حقل غير واضح في uncertain_fields."
)

_RULES_AR_CANON_EXTRA = "انسخ النص كما كتبه المستخدم دون ترجمة ودون تحويل الأرقام."


def _name_groups(directory_names: Iterable[Any]) -> list[list[str]]:
    """Normalise the payee name input to a list of groups of display strings."""
    groups: list[list[str]] = []
    for item in directory_names:
        if isinstance(item, str):
            names = [item]
        elif isinstance(item, Mapping):
            names = []
            for key in ("name_en", "name_ar"):
                if item.get(key):
                    names.append(str(item[key]))
            names.extend(str(a) for a in item.get("aliases", []) or [])
        else:
            names = [str(x) for x in item]
        cleaned: list[str] = []
        for name in names:
            name = name.strip()
            if name and name not in cleaned:
                cleaned.append(name)
        if cleaned:
            groups.append(cleaned)
    return groups


def directory_display_names(directory: Any) -> list[list[str]]:
    """Display names and aliases of every payee in a directory, one group per payee.

    The directory is read through ``payees`` (list or mapping), ``all()`` or iteration; each
    payee exposes ``name_en``, ``name_ar`` and ``aliases``. Identifiers are never included.
    """
    payees: Iterable[Any]
    source = getattr(directory, "payees", None)
    if source is None and hasattr(directory, "all"):
        source = directory.all()
    if source is None:
        source = getattr(directory, "_payees", None)
    if source is None:
        source = directory
    if isinstance(source, Mapping):
        payees = list(source.values())
    else:
        payees = list(source)
    groups: list[list[str]] = []
    for payee in payees:
        names: list[str] = []
        for attr in ("name_en", "name_ar"):
            value = getattr(payee, attr, None)
            if value:
                names.append(str(value))
        names.extend(str(a) for a in (getattr(payee, "aliases", None) or []))
        groups.extend(_name_groups([names]))
    return groups


def _turn_pairs(context_turns: Iterable[Any] | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for turn in context_turns or []:
        if isinstance(turn, Mapping):
            speaker = turn.get("speaker", turn.get("role", "user"))
            text = turn.get("text", turn.get("content", ""))
        else:
            speaker, text = turn[0], turn[1]
        text = str(text or "").strip()
        if text:
            pairs.append((str(speaker), text))
    return pairs


def build_system_prompt(directory_names: Iterable[Any], variant: str = "canon") -> str:
    """Bilingual system prompt for the extraction task with the payee display names."""
    if variant not in SCHEMA_VARIANTS:
        raise ValueError(f"unknown schema variant: {variant}")
    fields = _FIELDS_CANON_EN if variant == "canon" else _FIELDS_NONCANON_EN
    rules_en = _RULES_CANON_EN if variant == "canon" else _RULES_NONCANON_EN
    rules_ar = _RULES_AR + (" " + _RULES_AR_CANON_EXTRA if variant == "canon" else "")
    groups = _name_groups(directory_names)
    if groups:
        listing = "\n".join("- " + " | ".join(group) for group in groups)
    else:
        listing = "- (none)"
    return "\n\n".join(
        [
            _SYSTEM_TASK_EN,
            _SYSTEM_TASK_AR,
            fields,
            rules_en,
            rules_ar,
            "Saved payees, one per line with display names and aliases "
            "(المستفيدون المحفوظون، أسماء العرض والأسماء البديلة):\n" + listing,
        ]
    )


def build_extraction_messages(
    utterance: str,
    context_turns: Iterable[Any] | None,
    directory_names: Iterable[Any],
    lang_hint: str | None,
    variant: str = "canon",
) -> list[Message]:
    """Chat messages for one extraction call.

    ``context_turns`` holds earlier (speaker, text) pairs or mappings with ``speaker``/``role``
    and ``text``/``content``; ``directory_names`` holds display names and aliases (flat
    strings, groups, or payee mappings); ``lang_hint`` is "en", "ar", "mix" or None.
    """
    system = build_system_prompt(directory_names, variant)
    parts: list[str] = []
    turns = _turn_pairs(context_turns)
    if turns:
        transcript = "\n".join(f"{speaker}: {text}" for speaker, text in turns)
        parts.append("Conversation so far (المحادثة حتى الآن):\n" + transcript)
    else:
        parts.append("Conversation so far (المحادثة حتى الآن): none.")
    if lang_hint:
        parts.append("Language of the message: " + LANG_HINTS.get(lang_hint, str(lang_hint)) + ".")
    parts.append("Latest user message (آخر رسالة للمستخدم):\n" + str(utterance).strip())
    parts.append("Reply with the JSON object only.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_raw_extraction(text: str) -> RawExtraction:
    """Decode model output into a RawExtraction; raises ExtractionParseError."""
    try:
        data = parse_json_object(strip_think(text))
    except ValueError as exc:
        raise ExtractionParseError(str(exc)) from exc
    return RawExtraction.from_dict(data)
