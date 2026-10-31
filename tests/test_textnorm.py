"""Tests for tasdiq.textnorm: matching normalisation, its scope, and script profiling."""

from __future__ import annotations

import inspect
import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tasdiq import money
from tasdiq.textnorm import TATWEEL, normalize_for_matching, script_profile

HARAKAT = "ًٌٍَؘُؙِؚّْٰؐؑؒؓؔؕؖؗ"
ARABIC_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهويةىأإآء"


# Each normalisation step


def test_nfc_composition() -> None:
    decomposed = "Café"
    assert normalize_for_matching(decomposed) == unicodedata.normalize("NFC", decomposed).lower()
    assert normalize_for_matching(decomposed) == "café"


def test_tatweel_removed() -> None:
    assert normalize_for_matching("عـمـر خـالـد") == "عمر خالد"
    assert TATWEEL == "ـ"


def test_harakat_removed() -> None:
    assert normalize_for_matching("عُمَر خَالِد") == "عمر خالد"
    assert normalize_for_matching("رحمٰن") == "رحمن"
    for mark in HARAKAT:
        assert normalize_for_matching("ع" + mark + "مر") == "عمر", hex(ord(mark))


def test_alef_variants_unified() -> None:
    assert normalize_for_matching("أحمد") == normalize_for_matching("إحمد") == normalize_for_matching("آحمد") == "احمد"


def test_taa_marbuta_and_alef_maqsura_folded() -> None:
    assert normalize_for_matching("سارة") == "ساره"
    assert normalize_for_matching("مصطفى") == "مصطفي"


def test_whitespace_collapsed_and_stripped() -> None:
    assert normalize_for_matching("  Omar \t\n  Khalid  ") == "omar khalid"
    assert normalize_for_matching(" Omar Khalid") == "omar khalid"


def test_latin_lowercased() -> None:
    assert normalize_for_matching("OMAR KHALID") == "omar khalid"
    assert normalize_for_matching("ÉTISALAT") == "étisalat"


def test_punctuation_and_symbols_become_spaces() -> None:
    assert normalize_for_matching("Omar Khalid.") == "omar khalid"
    assert normalize_for_matching("Omar, Khalid!") == "omar khalid"
    assert normalize_for_matching("عمر، خالد؟") == "عمر خالد"
    assert normalize_for_matching("Omar & Sons") == "omar sons"
    assert normalize_for_matching("(Omar) [Khalid]") == "omar khalid"
    assert normalize_for_matching("\"Omar\" 'Khalid'") == "omar khalid"
    assert normalize_for_matching("$£€") == ""


def test_hyphen_inside_identifier_is_kept() -> None:
    assert normalize_for_matching("al-ain") == "al-ain"
    assert normalize_for_matching("Al-Ain Water") == "al-ain water"
    assert normalize_for_matching("P-114") == "p-114"
    assert normalize_for_matching("- Omar -") == "omar"
    assert normalize_for_matching("Omar -Khalid") == "omar khalid"
    assert normalize_for_matching("a--b") == "a b"


def test_empty_and_blank() -> None:
    assert normalize_for_matching("") == ""
    assert normalize_for_matching("   ") == ""
    assert normalize_for_matching("...") == ""


def test_both_sides_of_a_match_fold_to_the_same_key() -> None:
    assert normalize_for_matching("عُمَر خالِد") == normalize_for_matching("عمر خالد")
    assert normalize_for_matching("3OMAR") == normalize_for_matching("3omar")
    assert normalize_for_matching("Sara  Ahmed.") == normalize_for_matching("sara ahmed")


_ALPHABET = ARABIC_LETTERS + HARAKAT + TATWEEL + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -.,!?()'\"،؛؟$"


@settings(max_examples=300, derandomize=True, deadline=None)
@given(st.text(alphabet=_ALPHABET, max_size=40))
def test_normalize_for_matching_is_idempotent(text: str) -> None:
    once = normalize_for_matching(text)
    assert normalize_for_matching(once) == once
    assert once == once.strip()
    assert "  " not in once
    assert not any(ch in HARAKAT or ch == TATWEEL for ch in once)


# Scope: never applied to identifiers, amounts or currencies


def test_normalisation_would_alter_amounts_and_identifiers() -> None:
    amount_text = "٢٥٠٫٥٠ د.إ"
    assert normalize_for_matching(amount_text) != amount_text
    assert normalize_for_matching("250.50") != "250.50"
    assert normalize_for_matching("P-114") != "P-114"


def test_money_module_does_not_use_matching_normalisation() -> None:
    source = inspect.getsource(money)
    assert "textnorm" not in source
    assert "normalize_for_matching" not in source
    assert not hasattr(money, "normalize_for_matching")
    parsed = money.parse_amount("٢٥٠٫٥٠ د.إ")
    assert (parsed.amount_minor, parsed.currency) == (25050, "AED")


def test_directory_compares_identifiers_verbatim() -> None:
    from tasdiq.payees import Payee, PayeeDirectory

    directory = PayeeDirectory([Payee("P-114", "Omar Khalid", "عمر خالد", ("omar",))])
    assert directory.by_id("P-114") is not None
    assert directory.by_id("p-114") is None
    assert directory.by_id(" P-114") is None
    assert directory.by_id("P-114 ") is None
    assert directory.resolve("P-114") == []


def test_binding_fields_pass_through_unnormalised() -> None:
    records = pytest.importorskip("tasdiq.records")
    base = {
        "principal_id": "U-1",
        "action_type": "payment",
        "payee_id": "P-114",
        "amount_minor": 25000,
        "currency": "AED",
        "total_max_minor": 26000,
        "revision": 1,
    }
    lowered = dict(base, payee_id="p-114")
    assert records.binding_hash(base) != records.binding_hash(lowered)
    assert records.canonical_bytes(base).decode("utf-8").count('"payee_id":"P-114"') == 1
    assert records.binding_hash(base) == records.binding_hash(dict(base))


# Script profile


def test_script_profile_keys_and_pure_latin() -> None:
    profile = script_profile("Omar Khalid")
    assert set(profile) == {"arabic_letters", "latin_letters", "eastern_digits", "ascii_digits", "n_chars", "dominant"}
    assert profile["latin_letters"] == 1.0
    assert profile["arabic_letters"] == 0.0
    assert profile["n_chars"] == 10
    assert profile["dominant"] == "en"


def test_script_profile_pure_arabic() -> None:
    profile = script_profile("عمر خالد")
    assert profile["arabic_letters"] == 1.0
    assert profile["n_chars"] == 7
    assert profile["dominant"] == "ar"


def test_script_profile_mixed_text_and_digits() -> None:
    profile = script_profile("Pay ٢٥٠ to عمر")
    assert profile["n_chars"] == 11
    assert profile["latin_letters"] == pytest.approx(5 / 11)
    assert profile["arabic_letters"] == pytest.approx(3 / 11)
    assert profile["eastern_digits"] == pytest.approx(3 / 11)
    assert profile["ascii_digits"] == 0.0
    assert profile["dominant"] == "en"
    assert script_profile("ab عم")["dominant"] == "mix"
    assert script_profile("۲۵۰")["eastern_digits"] == 1.0


def test_script_profile_without_letters() -> None:
    assert script_profile("")["dominant"] == "none"
    assert script_profile("")["n_chars"] == 0
    digits = script_profile("250 - 300")
    assert digits["ascii_digits"] == 1.0
    assert digits["n_chars"] == 6
    assert digits["dominant"] == "none"


def test_script_profile_ignores_punctuation_and_harakat() -> None:
    plain = script_profile("عمر خالد")
    marked = script_profile("عُمَر، خَالِد!")
    assert marked["n_chars"] == plain["n_chars"]
    assert marked["arabic_letters"] == 1.0


@settings(max_examples=200, derandomize=True, deadline=None)
@given(st.text(alphabet=_ALPHABET + "٠١٢٣٤٥٦٧٨٩", max_size=60))
def test_script_profile_fractions_sum_to_one_or_zero(text: str) -> None:
    profile = script_profile(text)
    total = profile["arabic_letters"] + profile["latin_letters"] + profile["eastern_digits"] + profile["ascii_digits"]
    if profile["n_chars"]:
        assert total == pytest.approx(1.0)
    else:
        assert total == 0.0
    assert profile["dominant"] in {"ar", "en", "mix", "none"}


def test_format_characters_and_presentation_forms_are_folded() -> None:
    assert normalize_for_matching("\u200fعمر خالد") == normalize_for_matching("عمر خالد")
    assert normalize_for_matching("عمر\u200d خالد\u200c") == normalize_for_matching("عمر خالد")
    assert normalize_for_matching("ﻋﻤﺮ ﺧﺎﻟﺪ") == normalize_for_matching("عمر خالد")
    assert normalize_for_matching("Ｏｍａｒ") == "omar"
    assert normalize_for_matching("\u200e") == ""
