"""Canonical records of the consent protocol.

The module defines the consent record held by the store, the execution request issued by the agent, the decision
returned by a monitor, the intended record used by the simulated user and the reference policy, and the hashing and
rendering functions that bind a consent to one transaction. Money is an integer number of minor units with an ISO 4217
currency code; time is an integer number of simulated seconds.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Status = Literal["draft", "pending", "active", "consumed", "revoked", "expired", "superseded"]
Verdict = Literal["ALLOW", "DENY", "ASK"]

STATUSES: tuple[str, ...] = ("draft", "pending", "active", "consumed", "revoked", "expired", "superseded")
TERMINAL_STATUSES: frozenset[str] = frozenset({"consumed", "revoked", "expired", "superseded"})
ACTION_TYPES: tuple[str, ...] = ("payment",)
LANGS: tuple[str, ...] = ("en", "ar", "mix")

BINDING_FIELDS: tuple[str, ...] = (
    "principal_id",
    "action_type",
    "payee_id",
    "amount_minor",
    "currency",
    "total_max_minor",
    "revision",
)
FINGERPRINT_FIELDS: tuple[str, ...] = ("principal_id", "payee_id", "amount_minor", "currency")

REASONS_FIXED: frozenset[str] = frozenset(
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
    }
)
REASON_PREFIXES: tuple[str, ...] = ("field_mismatch:", "ambiguous:")


def is_reason(reason: str) -> bool:
    """Return True when the string belongs to the reason vocabulary of the protocol."""
    if reason in REASONS_FIXED:
        return True
    return any(reason.startswith(p) and len(reason) > len(p) for p in REASON_PREFIXES)


@dataclass(frozen=True)
class IntendedRecord:
    """The transaction the principal currently intends, as stated by the scenario.

    total_max_minor is the cap on amount plus fees; None means the scenario states no cap.
    """

    payee_id: str
    amount_minor: int
    currency: str
    total_max_minor: int | None = None


@dataclass(frozen=True)
class Ambiguous:
    """Marks an utterance whose intended record cannot be determined in the named field."""

    field: str


@dataclass
class ConsentRecord:
    """One consent record held by the consent store.

    The binding fields (BINDING_FIELDS) never change after creation; the store mutates only the state fields
    (status, uses, issued_at, expires_at, supersedes, confirmation_event_id, surface_hash, lang_shown).
    total_max_minor is the cap on amount plus fees in minor units. binding_hash is computed in __post_init__ when it is
    not supplied.
    """

    cr_id: str
    principal_id: str
    session_id: str
    action_type: str
    payee_id: str
    amount_minor: int
    currency: str
    total_max_minor: int
    revision: int = 1
    issued_at: int | None = None
    expires_at: int | None = None
    allowed_uses: int = 1
    uses: int = 0
    status: str = "draft"
    supersedes: str | None = None
    confirmation_event_id: str | None = None
    binding_hash: str = ""
    surface_hash: str | None = None
    lang_shown: str = ""

    def __post_init__(self) -> None:
        if not self.binding_hash:
            self.binding_hash = binding_hash(self)

    def binding_fields(self) -> dict[str, Any]:
        """Return the binding fields as a plain mapping."""
        return {name: getattr(self, name) for name in BINDING_FIELDS}

    def snapshot(self) -> dict[str, Any]:
        """Return every field as a plain mapping for logging."""
        return asdict(self)


@dataclass
class ExecutionRequest:
    """An execution request from the agent. Every field is untrusted input to the monitor."""

    req_id: str
    session_id: str
    principal_id: str
    payee_id: str
    amount_minor: int
    currency: str
    fee_minor: int
    consent_ref: str | None = None
    idempotency_key: str | None = None
    agent_claims_user_confirmed: bool = False
    t: int = 0

    def snapshot(self) -> dict[str, Any]:
        """Return every field as a plain mapping for logging."""
        return asdict(self)


@dataclass
class Decision:
    """A monitor decision on a consent request or an execution request."""

    verdict: str
    reason: str
    consent_id: str | None = None
    receipt: Any = None
    draft: ConsentRecord | None = None
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class NameRenderer(Protocol):
    """The part of the payee directory used by the confirmation surface."""

    def render_name(self, payee_id: str, lang: str) -> str:
        """Return the display name of a payee in a language ("ar" selects the Arabic name)."""
        ...


def canonical_bytes(fields: Mapping[str, Any]) -> bytes:
    """Serialise a mapping as compact JSON with sorted keys, non-ASCII preserved, encoded as UTF-8."""
    return json.dumps(dict(fields), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _field_values(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return {name: obj[name] for name in names}
    return {name: getattr(obj, name) for name in names}


def binding_hash(record: Any) -> str:
    """Return the SHA-256 hex digest of the canonical serialisation of the binding fields of a record or mapping."""
    return _sha256_hex(canonical_bytes(_field_values(record, BINDING_FIELDS)))


def fingerprint(req: Any) -> str:
    """Return the SHA-256 hex digest over (principal_id, payee_id, amount_minor, currency).

    The argument may be an ExecutionRequest, a ConsentRecord or a mapping with those keys, so the fingerprint of a
    request can be compared with the fingerprint of the record it references.
    """
    return _sha256_hex(canonical_bytes(_field_values(req, FINGERPRINT_FIELDS)))


def text_hash(text: str) -> str:
    """Return the SHA-256 hex digest of a text encoded as UTF-8."""
    return _sha256_hex(text.encode("utf-8"))


_EASTERN_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def eastern_digits(text: str) -> str:
    """Map ASCII digits in a text to Arabic-Indic digits (U+0660 to U+0669)."""
    return text.translate(_EASTERN_DIGITS)


def _format_amount(amount_minor: int, currency: str, lang: str) -> str:
    """Format an amount for the confirmation surface through tasdiq.money.format_amount."""
    from tasdiq.money import format_amount

    return format_amount(amount_minor, currency, lang)


def _duration_en(seconds: int) -> str:
    if seconds % 60 == 0:
        n = seconds // 60
        return f"{n} minute" if n == 1 else f"{n} minutes"
    return f"{seconds} second" if seconds == 1 else f"{seconds} seconds"


def _duration_ar(seconds: int) -> str:
    if seconds % 60 == 0:
        n = seconds // 60
        singular, dual, plural, counted = "دقيقة واحدة", "دقيقتين", "دقائق", "دقيقة"
    else:
        n = seconds
        singular, dual, plural, counted = "ثانية واحدة", "ثانيتين", "ثوانٍ", "ثانية"
    if n == 1:
        return singular
    if n == 2:
        return dual
    if 3 <= n <= 10:
        return f"{eastern_digits(str(n))} {plural}"
    return f"{eastern_digits(str(n))} {counted}"


def render_confirmation(record: ConsentRecord, directory: NameRenderer | None, lang: str) -> str:
    """Return the human-readable confirmation text shown on the trusted surface.

    The text names the amount, the payee display name in the requested language together with the stable payee
    identifier, the fee cap implied by total_max_minor, and the validity window when issued_at and expires_at are both
    set. Only the difference expires_at minus issued_at enters the text, so a record rendered before and after
    confirmation with the same window produces the same text. Languages other than "ar" render in English; the payee
    identifier is written as is in both languages. Amounts are formatted by tasdiq.money.format_amount.
    """
    name = directory.render_name(record.payee_id, lang) if directory is not None else record.payee_id
    amount = _format_amount(record.amount_minor, record.currency, lang)
    fee_cap = record.total_max_minor - record.amount_minor
    window = None
    if record.issued_at is not None and record.expires_at is not None:
        window = record.expires_at - record.issued_at
    if lang == "ar":
        parts = [f"ادفع {amount} إلى {name} ({record.payee_id})."]
        if fee_cap > 0:
            fee = _format_amount(fee_cap, record.currency, lang)
            parts.append(f"رسوم بحد أقصى {fee}.")
        else:
            parts.append("بدون رسوم.")
        if window is not None:
            parts.append(f"صالح لمدة {_duration_ar(window)}.")
        return " ".join(parts)
    parts = [f"Pay {amount} to {name} ({record.payee_id})."]
    if fee_cap > 0:
        fee = _format_amount(fee_cap, record.currency, lang)
        parts.append(f"Fees up to {fee}.")
    else:
        parts.append("No fees.")
    if window is not None:
        parts.append(f"Valid for {_duration_en(window)}.")
    return " ".join(parts)


def surface_hash(record: ConsentRecord, directory: NameRenderer | None, lang: str) -> str:
    """Return the SHA-256 hex digest of the rendered confirmation text in the given language."""
    return text_hash(render_confirmation(record, directory, lang))
