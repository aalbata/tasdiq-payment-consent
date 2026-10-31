"""Tests for the canonical records: hash stability, binding independence and surface hash language dependence."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace

import pytest

from tasdiq import records
from tasdiq.records import (
    BINDING_FIELDS,
    Ambiguous,
    ConsentRecord,
    Decision,
    ExecutionRequest,
    IntendedRecord,
    binding_hash,
    canonical_bytes,
    eastern_digits,
    fingerprint,
    is_reason,
    render_confirmation,
    surface_hash,
    text_hash,
)

NAMES = {
    "P-114": {"en": "Omar Khalid", "ar": "عمر خالد"},
    "P-200": {"en": "Abu Dhabi Water", "ar": "مياه أبوظبي"},
}


class FakeDirectory:
    """Directory stub that renders display names per language."""

    def render_name(self, payee_id: str, lang: str) -> str:
        return NAMES[payee_id]["ar" if lang == "ar" else "en"]


def fake_format_amount(amount_minor: int, currency: str, lang: str) -> str:
    """Formatter stub with two minor digits and Arabic-Indic digits for Arabic."""
    text = f"{amount_minor // 100}.{amount_minor % 100:02d}"
    if lang == "ar":
        return eastern_digits(text).replace(".", "٫") + " درهم"
    return f"{text} {currency}"


@pytest.fixture
def formatter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(records, "_format_amount", fake_format_amount)


def make_record(**overrides) -> ConsentRecord:
    fields = dict(
        cr_id="cr-1",
        principal_id="U-1",
        session_id="S-1",
        action_type="payment",
        payee_id="P-114",
        amount_minor=25000,
        currency="AED",
        total_max_minor=25250,
        revision=1,
    )
    fields.update(overrides)
    return ConsentRecord(**fields)


def test_canonical_bytes_is_sorted_compact_utf8() -> None:
    data = canonical_bytes({"b": 1, "a": "عمر", "c": None})
    assert data == '{"a":"عمر","b":1,"c":null}'.encode("utf-8")
    assert canonical_bytes({"x": 1, "y": 2}) == canonical_bytes({"y": 2, "x": 1})


def test_binding_hash_known_answer() -> None:
    record = make_record()
    payload = {name: getattr(record, name) for name in BINDING_FIELDS}
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert binding_hash(record) == expected
    assert record.binding_hash == expected
    assert len(expected) == 64


def test_binding_hash_stable_across_state_and_identity_fields() -> None:
    base = make_record()
    changed = replace(
        base,
        cr_id="cr-9",
        session_id="S-2",
        status="active",
        uses=1,
        issued_at=10,
        expires_at=610,
        allowed_uses=3,
        confirmation_event_id="ev-1",
        surface_hash="abc",
        lang_shown="ar",
        supersedes="cr-0",
    )
    assert binding_hash(changed) == binding_hash(base)
    assert binding_hash(base.binding_fields()) == binding_hash(base)


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("principal_id", "U-2"),
        ("payee_id", "P-200"),
        ("amount_minor", 25001),
        ("currency", "USD"),
        ("total_max_minor", 25251),
        ("revision", 2),
    ],
)
def test_binding_hash_changes_with_each_binding_field(field_name: str, value) -> None:
    base = make_record()
    assert binding_hash(replace(base, **{field_name: value})) != binding_hash(base)


def test_display_name_never_alters_binding_but_payee_id_does(formatter: None) -> None:
    class RenamedDirectory(FakeDirectory):
        def render_name(self, payee_id: str, lang: str) -> str:
            return "Omar K."

    base = make_record()
    assert surface_hash(base, FakeDirectory(), "en") != surface_hash(base, RenamedDirectory(), "en")
    assert binding_hash(base) == binding_hash(base)
    other = replace(base, payee_id="P-200")
    assert binding_hash(other) != binding_hash(base)


def test_fingerprint_matches_between_request_and_record() -> None:
    record = make_record()
    req = ExecutionRequest(
        req_id="r-1",
        session_id="S-1",
        principal_id="U-1",
        payee_id="P-114",
        amount_minor=25000,
        currency="AED",
        fee_minor=0,
        consent_ref="cr-1",
        idempotency_key="k-1",
        t=5,
    )
    assert fingerprint(req) == fingerprint(record)
    assert fingerprint(replace(req, amount_minor=25001)) != fingerprint(record)
    assert fingerprint(replace(req, payee_id="P-200")) != fingerprint(record)
    assert fingerprint(replace(req, principal_id="U-2")) != fingerprint(record)
    assert fingerprint(replace(req, fee_minor=500)) == fingerprint(record)
    expected = hashlib.sha256(
        b'{"amount_minor":25000,"currency":"AED","payee_id":"P-114","principal_id":"U-1"}'
    ).hexdigest()
    assert fingerprint(req) == expected


def test_render_confirmation_english_text(formatter: None) -> None:
    record = make_record(issued_at=100, expires_at=700)
    text = render_confirmation(record, FakeDirectory(), "en")
    assert text == "Pay 250.00 AED to Omar Khalid (P-114). Fees up to 2.50 AED. Valid for 10 minutes."


def test_render_confirmation_arabic_text(formatter: None) -> None:
    record = make_record(issued_at=100, expires_at=700)
    text = render_confirmation(record, FakeDirectory(), "ar")
    assert text.startswith("ادفع ٢٥٠٫٠٠ درهم إلى عمر خالد (P-114).")
    assert "٢٫٥٠ درهم" in text
    assert text.endswith("١٠ دقائق.")
    assert not any("0" <= ch <= "9" for ch in text.replace("P-114", ""))


def test_render_confirmation_without_fees_and_without_window(formatter: None) -> None:
    record = make_record(total_max_minor=25000)
    assert render_confirmation(record, FakeDirectory(), "en") == "Pay 250.00 AED to Omar Khalid (P-114). No fees."
    assert render_confirmation(record, FakeDirectory(), "ar").endswith("بدون رسوم.")
    assert render_confirmation(record, None, "en") == "Pay 250.00 AED to P-114 (P-114). No fees."


def test_render_confirmation_window_depends_only_on_difference(formatter: None) -> None:
    before = make_record(issued_at=0, expires_at=600)
    after = make_record(issued_at=4242, expires_at=4842, status="active", confirmation_event_id="ev-1")
    assert render_confirmation(before, FakeDirectory(), "en") == render_confirmation(after, FakeDirectory(), "en")
    odd = make_record(issued_at=0, expires_at=90)
    assert render_confirmation(odd, FakeDirectory(), "en").endswith("Valid for 90 seconds.")
    one = make_record(issued_at=0, expires_at=60)
    assert render_confirmation(one, FakeDirectory(), "en").endswith("Valid for 1 minute.")


@pytest.mark.parametrize(
    "seconds, ending",
    [
        (60, "دقيقة واحدة."),
        (120, "دقيقتين."),
        (180, "٣ دقائق."),
        (900, "١٥ دقيقة."),
        (5, "٥ ثوانٍ."),
    ],
)
def test_render_confirmation_arabic_duration_forms(formatter: None, seconds: int, ending: str) -> None:
    record = make_record(issued_at=0, expires_at=seconds)
    assert render_confirmation(record, FakeDirectory(), "ar").endswith(ending)


def test_surface_hash_differs_by_language_while_binding_hash_is_identical(formatter: None) -> None:
    record = make_record(issued_at=0, expires_at=600)
    en = surface_hash(record, FakeDirectory(), "en")
    ar = surface_hash(record, FakeDirectory(), "ar")
    assert en != ar
    assert en == text_hash(render_confirmation(record, FakeDirectory(), "en"))
    assert binding_hash(record) == binding_hash(replace(record, lang_shown="ar"))
    assert surface_hash(record, FakeDirectory(), "mix") == en


def test_surface_hash_language_dependence_with_money_module() -> None:
    try:
        importlib.import_module("tasdiq.money")
    except Exception as exc:
        pytest.skip(f"tasdiq.money is not importable: {exc}")
    record = make_record(issued_at=0, expires_at=600)
    en_text = render_confirmation(record, FakeDirectory(), "en")
    ar_text = render_confirmation(record, FakeDirectory(), "ar")
    assert en_text == "Pay 250.00 AED to Omar Khalid (P-114). Fees up to 2.50 AED. Valid for 10 minutes."
    assert en_text != ar_text
    assert "عمر خالد" in ar_text and "P-114" in ar_text and "درهم" in ar_text
    assert "٢٥٠" in ar_text and not any("0" <= ch <= "9" for ch in ar_text.replace("P-114", ""))
    assert text_hash(en_text) != text_hash(ar_text)
    assert binding_hash(record) == binding_hash(replace(record, lang_shown="ar", surface_hash=text_hash(ar_text)))


def test_eastern_digits_mapping() -> None:
    assert eastern_digits("250.50") == "٢٥٠.٥٠"
    assert eastern_digits("P-114") == "P-١١٤"


def test_reason_vocabulary() -> None:
    for reason in ("no_consent", "expired", "duplicate_replayed", "field_mismatch:payee_id", "ambiguous:amount"):
        assert is_reason(reason)
    for reason in ("", "field_mismatch:", "mismatch", "ok"):
        assert not is_reason(reason)


def test_intended_record_and_ambiguous_are_frozen() -> None:
    intended = IntendedRecord(payee_id="P-114", amount_minor=25000, currency="AED")
    assert intended.total_max_minor is None
    with pytest.raises(AttributeError):
        intended.amount_minor = 1  # type: ignore[misc]
    assert Ambiguous("amount").field == "amount"
    assert hash(intended) == hash(IntendedRecord("P-114", 25000, "AED"))


def test_decision_and_snapshots() -> None:
    record = make_record()
    decision = Decision(verdict="DENY", reason="no_consent", draft=record)
    assert decision.details == {} and decision.consent_id is None and decision.receipt is None
    snap = record.snapshot()
    assert snap["binding_hash"] == record.binding_hash and snap["status"] == "draft"
    req = ExecutionRequest("r", "S", "U", "P-114", 1, "AED", 0)
    assert req.snapshot()["agent_claims_user_confirmed"] is False and req.snapshot()["t"] == 0
