"""Tests for tasdiq.money: digit scripts, separators, number words, currencies, fractions and formatting."""

from __future__ import annotations

import dataclasses
import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tasdiq.money import lookup_minor_unit
from tasdiq.money import (
    CURRENCIES,
    FLAGS,
    AmountParse,
    CurrencyInfo,
    eastern_digits,
    format_amount,
    lookup_currency,
    normalize_digits,
    parse_amount,
)

EXPECTED_CODES = {"AED", "SAR", "USD", "EUR", "GBP", "KWD", "BHD", "OMR", "QAR", "JOD", "EGP", "INR", "PKR", "PHP"}
THREE_DIGIT_CODES = {"KWD", "BHD", "OMR", "JOD"}


# Currency table


def test_currency_table_covers_the_listed_codes() -> None:
    assert set(CURRENCIES) == EXPECTED_CODES
    for code, info in CURRENCIES.items():
        assert isinstance(info, CurrencyInfo)
        assert info.code == code
        assert info.minor_digits == (3 if code in THREE_DIGIT_CODES else 2)
        assert info.scale == 10**info.minor_digits
        assert info.names_en and info.names_ar
        assert all(isinstance(x, str) and x for x in (*info.names_en, *info.names_ar, *info.symbols))


@pytest.mark.parametrize(
    ("code", "labels"),
    [
        ("AED", ["dirham", "dirhams", "dh", "dhs", "aed", "درهم", "دراهم", "د.إ", "درهماً", "درهمًا"]),
        ("SAR", ["riyal", "ريال", "ر.س"]),
        ("USD", ["dollar", "dollars", "$", "دولار", "دولارات"]),
        ("EUR", ["euro", "euros", "€", "يورو"]),
        ("GBP", ["pound", "pounds", "£", "جنيه", "جنيه استرليني"]),
        ("KWD", ["dinar", "دينار كويتي", "د.ك"]),
        ("QAR", ["ريال قطري"]),
        ("JOD", ["دينار أردني"]),
        ("EGP", ["جنيه مصري"]),
        ("INR", ["rupee", "روبية"]),
        ("PHP", ["peso", "بيزو"]),
    ],
)
def test_spec_labels_are_present_and_resolve(code: str, labels: list[str]) -> None:
    info = CURRENCIES[code]
    table = {*info.names_en, *info.names_ar, *info.symbols}
    for label in labels:
        assert label in table, label
        assert lookup_currency(label) == code, label


def test_minor_unit_names_in_table() -> None:
    assert "fils" in CURRENCIES["AED"].minor_en
    assert {"فلس", "فلوس"} <= set(CURRENCIES["AED"].minor_ar)


def test_lookup_currency_handles_case_abbreviations_and_unknowns() -> None:
    assert lookup_currency("Dirhams") == "AED"
    assert lookup_currency("AED") == "AED"
    assert lookup_currency("$") == "USD"
    assert lookup_currency("د.إ") == "AED"
    assert lookup_currency("د.أ") == "JOD"
    assert lookup_currency("درهمين") == "AED"
    assert lookup_currency("xyz") is None
    assert lookup_currency("") is None


def _all_labels() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for code, info in CURRENCIES.items():
        for label in (*info.names_en, *info.names_ar, *info.symbols):
            if any(ch.isalpha() for ch in label) and any(not ch.isalnum() and not ch.isspace() and ch != "." for ch in label):
                continue
            pairs.append((code, label))
    return pairs


@pytest.mark.parametrize(("code", "label"), _all_labels())
def test_every_currency_label_is_recognised_after_and_before_the_number(code: str, label: str) -> None:
    scale = CURRENCIES[code].scale
    after = parse_amount(f"5 {label}")
    assert (after.amount_minor, after.currency) == (5 * scale, code)
    assert "no_currency" not in after.flags
    before = parse_amount(f"{label} 5")
    assert (before.amount_minor, before.currency) == (5 * scale, code)


# Digit normalisation


def test_normalize_digits_maps_both_arabic_digit_sets_and_separators() -> None:
    assert normalize_digits("٠١٢٣٤٥٦٧٨٩") == "0123456789"
    assert normalize_digits("۰۱۲۳۴۵۶۷۸۹") == "0123456789"
    assert normalize_digits("٢٬٥٠٠٫٥٠") == "2,500.50"
    assert normalize_digits("١2۳") == "123"


def test_normalize_digits_keeps_everything_else() -> None:
    text = "ادفع 250 درهم to Omar, P-114; 2,500.50 AED!"
    assert normalize_digits(text) == text
    assert normalize_digits("") == ""


@settings(max_examples=300, derandomize=True, deadline=None)
@given(st.text())
def test_normalize_digits_is_idempotent(text: str) -> None:
    once = normalize_digits(text)
    assert normalize_digits(once) == once
    assert len(once) == len(text) - sum(1 for ch in text if unicodedata.category(ch) == "Cf")


def test_eastern_digits_is_the_inverse_on_digits() -> None:
    assert eastern_digits("250.50") == "٢٥٠.٥٠"
    assert normalize_digits(eastern_digits("1234567890")) == "1234567890"


# Amount parsing: digits, scripts and separators


def test_result_type_and_flag_vocabulary() -> None:
    result = parse_amount("250.50 AED")
    assert isinstance(result, AmountParse)
    assert {f.name for f in dataclasses.fields(AmountParse)} == {"amount_minor", "currency", "matched_text", "candidates", "flags"}
    assert result.flags <= set(FLAGS)
    assert set(FLAGS) == {
        "multiple_amounts", "no_currency", "unrecognised_fraction", "number_words",
        "digit_script_ar", "thousands_separator", "decimal",
        "unrecognised_number_word", "negative", "mixed_digit_scripts", "secondary_unlabelled_amount",
    }


@pytest.mark.parametrize(
    ("text", "amount", "code", "flags"),
    [
        ("250.50 AED", 25050, "AED", {"decimal"}),
        ("٢٥٠ درهم", 25000, "AED", {"digit_script_ar"}),
        ("۲۵۰ درهم", 25000, "AED", {"digit_script_ar"}),
        ("ألفان وخمسمئة درهم", 250000, "AED", {"number_words"}),
        ("250 dirhams and 50 fils", 25050, "AED", set()),
        ("ادفع 250 درهم و50 فلساً", 25050, "AED", set()),
        ("ادفع ٢٥٠ درهماً لعمر", 25000, "AED", {"digit_script_ar"}),
        ("2,500.50 AED", 250050, "AED", {"decimal", "thousands_separator"}),
        ("٢٬٥٠٠٫٥٠ د.إ", 250050, "AED", {"decimal", "thousands_separator", "digit_script_ar"}),
        ("transfer 1,250 dirhams", 125000, "AED", {"thousands_separator"}),
        ("250,50 AED", 25050, "AED", {"decimal"}),
        ("٢٥٠٫٥ درهم", 25050, "AED", {"decimal", "digit_script_ar"}),
        ("1.000,50 EUR", 100050, "EUR", {"decimal", "thousands_separator"}),
        ("1.5 KWD", 1500, "KWD", {"decimal"}),
        ("1.000 KWD", 1000, "KWD", {"decimal"}),
        ("1,000 KWD", 1000000, "KWD", {"thousands_separator"}),
        ("$250", 25000, "USD", set()),
        ("250$", 25000, "USD", set()),
        ("€100", 10000, "EUR", set()),
        ("2500AED", 250000, "AED", set()),
        ("250aed", 25000, "AED", set()),
        ("250 AED to omar", 25000, "AED", set()),
    ],
)
def test_digit_amounts(text: str, amount: int, code: str, flags: set[str]) -> None:
    result = parse_amount(text)
    assert (result.amount_minor, result.currency) == (amount, code)
    assert result.flags == flags
    assert result.candidates == [(amount, code)]
    assert result.matched_text


def test_matched_text_is_the_digit_normalised_span() -> None:
    assert parse_amount("٢٥٠ درهم").matched_text == "250 درهم"
    assert parse_amount("send 250.50 AED now").matched_text == "250.50 AED"
    assert parse_amount("ألفان وخمسمئة درهم").matched_text == "ألفان وخمسمئة درهم"


# Number words


@pytest.mark.parametrize(
    ("text", "amount", "code"),
    [
        ("two hundred and fifty dirhams", 25000, "AED"),
        ("one thousand two hundred and fifty dirhams", 125000, "AED"),
        ("two thousand five hundred dirhams", 250000, "AED"),
        ("a hundred dirhams", 10000, "AED"),
        ("ten dirhams", 1000, "AED"),
        ("ninety-nine dollars", 9900, "USD"),
        ("one million euros", 100000000, "EUR"),
        ("half a dirham", 50, "AED"),
        ("quarter of a million dirhams", 25000000, "AED"),
        ("ten and a half dirhams", 1050, "AED"),
        ("one and a half thousand dirhams", 150000, "AED"),
        ("half a dinar", 500, "KWD"),
    ],
)
def test_english_number_words(text: str, amount: int, code: str) -> None:
    result = parse_amount(text)
    assert (result.amount_minor, result.currency) == (amount, code)
    assert "number_words" in result.flags
    assert "no_currency" not in result.flags


@pytest.mark.parametrize(
    ("text", "amount", "code"),
    [
        ("مئتان وخمسون درهماً", 25000, "AED"),
        ("مئتين وخمسين درهماً", 25000, "AED"),
        ("خمسة وعشرين ريال", 2500, "SAR"),
        ("ألف دولار", 100000, "USD"),
        ("ثلاثة آلاف وخمسمائة ريال", 350000, "SAR"),
        ("أربعمئة وخمسة وعشرون درهما", 42500, "AED"),
        ("ادفع مئة وخمسين درهماً", 15000, "AED"),
        ("ادفع ألفين وثلاثمئة درهم", 230000, "AED"),
        ("اثني عشر درهماً", 1200, "AED"),
        ("أحد عشر درهماً", 1100, "AED"),
        ("تسعة عشر درهماً", 1900, "AED"),
        ("عشرة دراهم", 1000, "AED"),
        ("واحد درهم", 100, "AED"),
        ("درهمين", 200, "AED"),
        ("مليون درهم", 100000000, "AED"),
        ("ثلاثة ملايين درهم", 300000000, "AED"),
        ("خمسة عشر ألف درهم", 1500000, "AED"),
        ("مئة ألف درهم", 10000000, "AED"),
        ("مليونان وخمسمئة ألف درهم", 250000000, "AED"),
        ("نصف درهم", 50, "AED"),
        ("ربع درهم", 25, "AED"),
        ("ربع مليون درهم", 25000000, "AED"),
        ("ألف ونصف درهم", 150000, "AED"),
        ("عشرين درهماً وخمسين فلساً", 2050, "AED"),
        ("خمسة دنانير كويتية", 5000, "KWD"),
    ],
)
def test_arabic_number_words(text: str, amount: int, code: str) -> None:
    result = parse_amount(text)
    assert (result.amount_minor, result.currency) == (amount, code)
    assert "number_words" in result.flags
    assert "no_currency" not in result.flags


def test_harakat_on_number_words_are_ignored() -> None:
    assert parse_amount("خَمْسَةٌ وَعِشْرُونَ دِرْهَمًا").amount_minor == 2500
    assert parse_amount("ألفانِ وخمسمئةِ درهمٍ").amount_minor == 250000


def test_number_word_flag_absent_for_digit_amounts() -> None:
    assert "number_words" not in parse_amount("250 AED").flags
    assert "number_words" in parse_amount("fifty AED").flags


def test_bare_single_number_word_is_not_an_amount() -> None:
    for text in ("one", "half", "نصف", "ten"):
        result = parse_amount(text, "AED")
        assert result.amount_minor is None and result.candidates == []


# Currency resolution and the no_currency flag


def test_default_currency_applies_only_without_an_adjacent_label() -> None:
    bare = parse_amount("pay 250 to omar", "AED")
    assert (bare.amount_minor, bare.currency) == (25000, "AED")
    assert bare.flags == {"no_currency"}
    lowercase_default = parse_amount("pay 250", "aed")
    assert lowercase_default.currency == "AED"
    labelled = parse_amount("pay 250 USD to omar", "AED")
    assert (labelled.amount_minor, labelled.currency) == (25000, "USD")
    assert "no_currency" not in labelled.flags


def test_no_default_and_no_label_gives_amount_without_currency() -> None:
    result = parse_amount("pay 250 to omar")
    assert (result.amount_minor, result.currency) == (25000, None)
    assert result.flags == {"no_currency"}
    assert result.candidates == [(25000, None)]


def test_unknown_default_currency_is_ignored() -> None:
    result = parse_amount("pay 250", "XXX")
    assert result.currency is None and "no_currency" in result.flags


def test_minor_unit_phrases_resolve_through_the_default() -> None:
    assert parse_amount("250 fils", "AED").amount_minor == 250
    assert parse_amount("250 fils", "AED").currency == "AED"
    bare = parse_amount("250 fils")
    assert bare.amount_minor == 250 and bare.currency is None and "no_currency" in bare.flags
    assert parse_amount("250 fils", "KWD").currency == "KWD"


def test_minor_digits_of_the_currency_set_the_scale() -> None:
    assert parse_amount("5 KWD").amount_minor == 5000
    assert parse_amount("5 BHD").amount_minor == 5000
    assert parse_amount("5 OMR").amount_minor == 5000
    assert parse_amount("5 JOD").amount_minor == 5000
    assert parse_amount("5 AED").amount_minor == 500
    assert parse_amount("5", "KWD").amount_minor == 5000


def test_arabic_abbreviations_for_dirham_and_jordanian_dinar_stay_distinct() -> None:
    assert (parse_amount("50 د.إ").currency, parse_amount("50 د.أ").currency) == ("AED", "JOD")


def test_longest_label_wins() -> None:
    assert parse_amount("100 جنيه مصري").currency == "EGP"
    assert parse_amount("100 جنيه").currency == "GBP"
    assert parse_amount("50 ريال عماني").currency == "OMR"
    assert parse_amount("50 ريال").currency == "SAR"
    assert parse_amount("50 دينار أردني").currency == "JOD"
    assert parse_amount("50 روبية باكستانية").currency == "PKR"
    assert parse_amount("50 روبية").currency == "INR"


# Multiple amounts, fractions and noise


def test_two_distinct_amounts_are_flagged_and_unresolved() -> None:
    result = parse_amount("250 AED and 300 AED")
    assert result.amount_minor is None
    assert result.currency == "AED"
    assert "multiple_amounts" in result.flags
    assert result.candidates == [(25000, "AED"), (30000, "AED")]
    assert result.matched_text == "250 AED | 300 AED"


def test_two_amounts_in_different_currencies_leave_currency_unresolved() -> None:
    result = parse_amount("250 AED or 70 USD")
    assert result.amount_minor is None and result.currency is None
    assert "multiple_amounts" in result.flags
    assert result.candidates == [(25000, "AED"), (7000, "USD")]


def test_negated_alternative_is_still_two_amounts() -> None:
    result = parse_amount("send 250 AED to Omar, not 300 AED")
    assert result.amount_minor is None and "multiple_amounts" in result.flags


def test_two_bare_numbers_with_a_default_are_two_amounts() -> None:
    result = parse_amount("pay omar 250 and sara 300", "AED")
    assert result.amount_minor is None
    assert result.flags == {"multiple_amounts", "no_currency"}


def test_repeated_identical_amount_is_one_amount() -> None:
    result = parse_amount("250 AED and 250 AED")
    assert (result.amount_minor, result.currency) == (25000, "AED")
    assert "multiple_amounts" not in result.flags


@pytest.mark.parametrize(
    ("text", "default"),
    [("250.505 AED", None), ("1.2345 KWD", None), ("نصف فلس", "AED"), ("0.001 AED", None), ("ربع درهم و نصف فلس", None)],
)
def test_unrecognised_fraction(text: str, default: str | None) -> None:
    result = parse_amount(text, default)
    assert result.amount_minor is None
    assert "unrecognised_fraction" in result.flags
    assert "multiple_amounts" not in result.flags


def test_identifiers_dates_and_times_are_not_amounts() -> None:
    for text in ("the P-114 payee gets 250 AED", "250 AED on 2026-10-06", "at 10:30 pay 250 AED", "P-114 250 AED"):
        result = parse_amount(text)
        assert (result.amount_minor, result.currency) == (25000, "AED"), text
        assert "multiple_amounts" not in result.flags


def test_empty_and_amountless_text() -> None:
    for text in ("", "hello there", "pay omar please"):
        result = parse_amount(text, "AED")
        assert result == AmountParse(None, None, "", [], set())


def test_parse_is_pure_and_repeatable() -> None:
    text = "ادفع ٢٥٠ درهماً لعمر"
    first = parse_amount(text)
    second = parse_amount(text)
    assert first == second
    first.flags.add("decimal")
    assert "decimal" not in parse_amount(text).flags


# Formatting


def test_format_amount_english() -> None:
    assert format_amount(25000, "AED", "en") == "250.00 AED"
    assert format_amount(5, "AED", "en") == "0.05 AED"
    assert format_amount(1500, "KWD", "en") == "1.500 KWD"
    assert format_amount(123456789, "USD", "en") == "1234567.89 USD"
    assert format_amount(0, "EUR", "en") == "0.00 EUR"
    assert format_amount(-250, "USD", "en") == "-2.50 USD"
    assert format_amount(25000, "aed", "en") == "250.00 AED"
    assert format_amount(25000, "AED", "mix") == "250.00 AED"


def test_format_amount_arabic_uses_eastern_digits_and_the_arabic_name() -> None:
    assert format_amount(25000, "AED", "ar") == "٢٥٠٫٠٠ درهم"
    assert format_amount(1500, "KWD", "ar") == "١٫٥٠٠ دينار كويتي"
    assert format_amount(0, "EUR", "ar") == "٠٫٠٠ يورو"
    assert format_amount(250050, "AED", "ar") == "٢٥٠٠٫٥٠ درهم"
    for code in CURRENCIES:
        text = format_amount(123456, code, "ar")
        assert not any(ch in "0123456789" for ch in text), text
        assert text.endswith(" " + CURRENCIES[code].names_ar[0])
        assert "٫" in text


def test_format_amount_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        format_amount(100, "XXX", "en")
    with pytest.raises(TypeError):
        format_amount(100.0, "AED", "en")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        format_amount(True, "AED", "en")


@settings(max_examples=400, derandomize=True, deadline=None)
@given(
    amount=st.integers(min_value=0, max_value=10**15),
    code=st.sampled_from(sorted(CURRENCIES)),
    lang=st.sampled_from(["en", "ar"]),
)
def test_parse_amount_round_trips_format_amount(amount: int, code: str, lang: str) -> None:
    text = format_amount(amount, code, lang)
    result = parse_amount(text)
    assert result.amount_minor == amount
    assert result.currency == code
    assert result.candidates == [(amount, code)]
    assert not result.flags & {"multiple_amounts", "unrecognised_fraction", "no_currency", "number_words"}
    assert ("digit_script_ar" in result.flags) == (lang == "ar")


@settings(max_examples=200, derandomize=True, deadline=None)
@given(amount=st.integers(min_value=0, max_value=10**12), code=st.sampled_from(sorted(CURRENCIES)))
def test_round_trip_survives_a_default_currency_of_another_code(amount: int, code: str) -> None:
    other = "USD" if code != "USD" else "AED"
    result = parse_amount(format_amount(amount, code, "en"), other)
    assert (result.amount_minor, result.currency) == (amount, code)


# Bare multipliers, interrupted phrases, singular nouns, digit scripts, signs, identifiers, format characters


def test_bare_multiplier_and_hundreds_words_are_not_amounts() -> None:
    for text in ("thanks a million", "ألف شكر", "million", "hundred", "خمسمئة"):
        result = parse_amount(text, "AED")
        assert result.amount_minor is None and result.candidates == [], text
    assert parse_amount("a hundred dirhams", "AED").amount_minor == 10000
    assert parse_amount("خمسمئة درهم", "AED").amount_minor == 50000
    assert parse_amount("one million AED").amount_minor == 100000000


def test_number_phrase_interrupted_by_an_unrecognised_word_is_flagged() -> None:
    for text in ("ألفان وخمسمأة درهم", "ألفين وخمسمية درهم", "two thousand and fivehundred dirhams"):
        result = parse_amount(text, "AED")
        assert result.amount_minor is None and result.candidates == [], text
        assert "unrecognised_number_word" in result.flags
    assert parse_amount("ألفان وخمسمئة درهم", "AED").amount_minor == 250000
    assert parse_amount("two thousand and five hundred dirhams").amount_minor == 250000
    assert parse_amount("ألفان درهم وشكرا", "AED").amount_minor == 200000
    assert parse_amount("250 AED and thanks").amount_minor == 25000


def test_singular_currency_noun_before_a_minor_phrase_means_one_unit() -> None:
    assert parse_amount("درهم وخمسون فلساً").amount_minor == 150
    assert parse_amount("one dirham and fifty fils").amount_minor == 150
    assert parse_amount("درهم واحد").amount_minor == 100
    assert parse_amount("درهمان").amount_minor == 200
    assert parse_amount("dirhams and fifty fils", "AED").amount_minor == 50


def test_only_the_supported_digit_scripts_are_numbers() -> None:
    for text in ("१२३ AED", "２５０ AED", "๒๕๐ AED"):
        assert parse_amount(text).amount_minor is None, text
    mixed = parse_amount("٢5٠ درهم")
    assert mixed.amount_minor == 25000 and "mixed_digit_scripts" in mixed.flags and "digit_script_ar" in mixed.flags
    assert "mixed_digit_scripts" not in parse_amount("٢٥٠ درهم").flags
    assert "mixed_digit_scripts" not in parse_amount("٢٥٠ درهم و 50 فلس").flags


def test_spelled_out_sign_invalidates_the_amount() -> None:
    for text in ("minus 250 AED", "negative 250 AED", "ناقص 250 درهم", "سالب ٢٥٠ درهم", "negative two hundred dirhams"):
        result = parse_amount(text, "AED")
        assert result.amount_minor is None and result.candidates == [], text
        assert "negative" in result.flags
    assert parse_amount("send 250 AED minus nothing").amount_minor == 25000


def test_long_integers_are_identifiers_and_secondary_integers_are_flagged() -> None:
    account = parse_amount("account 1234567890 pay 250 AED")
    assert (account.amount_minor, account.currency) == (25000, "AED") and "multiple_amounts" not in account.flags
    assert parse_amount("pay 1000000 AED").amount_minor == 100000000
    assert parse_amount("AED 1000000").amount_minor == 100000000
    year = parse_amount("في 2026 ادفع 250 درهم", "AED")
    assert year.amount_minor is None and "multiple_amounts" in year.flags and "secondary_unlabelled_amount" in year.flags
    assert "secondary_unlabelled_amount" not in parse_amount("250 AED and 300 AED").flags
    assert "secondary_unlabelled_amount" not in parse_amount("pay omar 250 and sara 300", "AED").flags


def test_format_characters_are_removed_before_parsing() -> None:
    assert normalize_digits("\u200e250") == "250"
    assert normalize_digits("٢\u200f٥٠") == "250"
    assert parse_amount("\u200e250 AED").amount_minor == 25000
    assert parse_amount("\u200f٢٥٠ درهم").amount_minor == 25000
    assert lookup_minor_unit("fils") == frozenset({"AED", "KWD", "BHD", "JOD"})
    assert lookup_minor_unit("فلس") == frozenset({"AED", "KWD", "BHD", "JOD"})
    assert lookup_minor_unit("cents") == frozenset({"USD", "EUR"})
    assert lookup_minor_unit("dirham") == frozenset()
