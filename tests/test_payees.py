"""Tests for tasdiq.payees: exact alias resolution, Arabizi aliases, shared names and rendering."""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tasdiq.payees import KINDS, Payee, PayeeDirectory
from tasdiq.textnorm import normalize_for_matching

OMAR_KHALID = Payee("P-114", "Omar Khalid", "عمر خالد", ("Omar", "omar khalid", "3omar", "عمر"), "domestic", "AE** **** 4471")
OMAR_KHALIL = Payee("P-115", "Omar Khalil", "عمر خليل", ("omar khalil", "3omar khalil"), "domestic", "AE** **** 4489")
OMAR_SAEED = Payee("P-118", "Omar Saeed", "عمر سعيد", ("Omar",), "international", "GB** **** 9901")
SARA = Payee("P-207", "Sara Ahmed", "سارة أحمد", ("sara", "سارة"), "domestic", "AE** **** 1200")
TELECOM = Payee("P-301", "Gulf Telecom", "اتصالات الخليج", ("etisalat bill", "my phone bill"), "merchant", "")
AL_AIN = Payee("P-302", "Al-Ain Water", "مياه العين", (), "merchant", "")


@pytest.fixture
def directory() -> PayeeDirectory:
    return PayeeDirectory([OMAR_KHALID, OMAR_KHALIL, OMAR_SAEED, SARA, TELECOM, AL_AIN])


# Payee dataclass


def test_payee_fields_and_kinds() -> None:
    names = [f.name for f in dataclasses.fields(Payee)]
    assert names == ["payee_id", "name_en", "name_ar", "aliases", "kind", "iban_masked"]
    assert KINDS == ("domestic", "international", "merchant")
    assert OMAR_KHALID.kind == "domestic" and OMAR_SAEED.kind == "international" and TELECOM.kind == "merchant"


def test_payee_validation() -> None:
    with pytest.raises(ValueError):
        Payee("", "x", "y")
    with pytest.raises(ValueError):
        Payee("   ", "x", "y")
    with pytest.raises(ValueError):
        Payee("P-1", "x", "y", (), "corporate")


def test_payee_aliases_accept_a_list_and_become_a_tuple() -> None:
    payee = Payee("P-1", "Omar", "عمر", ["omar", "3omar"])
    assert payee.aliases == ("omar", "3omar")
    assert isinstance(payee.aliases, tuple)


def test_payee_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        OMAR_KHALID.name_en = "Other"  # type: ignore[misc]


def test_payee_labels_in_order_without_duplicates() -> None:
    assert OMAR_KHALID.labels() == ("Omar Khalid", "عمر خالد", "Omar", "omar khalid", "3omar", "عمر")
    assert Payee("P-9", "Same", "Same", ("Same", "Other")).labels() == ("Same", "Other")
    assert Payee("P-8", "", "عمر", ("",)).labels() == ("عمر",)


def test_payee_dict_round_trip() -> None:
    row = OMAR_KHALID.to_dict()
    assert row == {
        "payee_id": "P-114",
        "name_en": "Omar Khalid",
        "name_ar": "عمر خالد",
        "aliases": ["Omar", "omar khalid", "3omar", "عمر"],
        "kind": "domestic",
        "iban_masked": "AE** **** 4471",
    }
    assert Payee.from_dict(row) == OMAR_KHALID
    minimal = Payee.from_dict({"payee_id": "P-1"})
    assert minimal == Payee("P-1", "", "", (), "domestic", "")


# Directory construction


def test_directory_construction_and_container_protocol(directory: PayeeDirectory) -> None:
    assert len(directory) == 6
    assert list(directory) == [OMAR_KHALID, OMAR_KHALIL, OMAR_SAEED, SARA, TELECOM, AL_AIN]
    assert directory.all() == list(directory)
    assert directory.ids() == ["P-114", "P-115", "P-118", "P-207", "P-301", "P-302"]
    assert "P-114" in directory and "P-999" not in directory


def test_directory_rejects_duplicates_and_wrong_types() -> None:
    with pytest.raises(ValueError):
        PayeeDirectory([OMAR_KHALID, Payee("P-114", "Other", "آخر")])
    with pytest.raises(TypeError):
        PayeeDirectory([{"payee_id": "P-1"}])  # type: ignore[list-item]


def test_directory_from_dicts(directory: PayeeDirectory) -> None:
    rebuilt = PayeeDirectory.from_dicts(p.to_dict() for p in directory)
    assert rebuilt.all() == directory.all()
    assert [p.payee_id for p in rebuilt.resolve("Omar")] == ["P-114", "P-118"]


def test_empty_directory() -> None:
    empty = PayeeDirectory([])
    assert len(empty) == 0 and empty.resolve("Omar") == [] and empty.by_id("P-114") is None
    assert empty.render_name("P-114", "en") == "P-114"


# Resolution


def test_resolve_exact_english_name(directory: PayeeDirectory) -> None:
    assert directory.resolve("Omar Khalid") == [OMAR_KHALID]
    assert directory.resolve("omar khalid") == [OMAR_KHALID]
    assert directory.resolve("OMAR KHALID.") == [OMAR_KHALID]
    assert directory.resolve("  Omar   Khalid ") == [OMAR_KHALID]


def test_resolve_exact_arabic_name_with_and_without_marks(directory: PayeeDirectory) -> None:
    assert directory.resolve("عمر خالد") == [OMAR_KHALID]
    assert directory.resolve("عُمَر خَالِد") == [OMAR_KHALID]
    assert directory.resolve("عـمـر خـالـد") == [OMAR_KHALID]
    assert directory.resolve("عمر خليل") == [OMAR_KHALIL]
    assert directory.resolve("سارة أحمد") == [SARA]
    assert directory.resolve("ساره احمد") == [SARA]
    assert directory.resolve("ساره") == [SARA]


def test_resolve_arabizi_aliases(directory: PayeeDirectory) -> None:
    assert directory.resolve("3omar") == [OMAR_KHALID]
    assert directory.resolve("3OMAR") == [OMAR_KHALID]
    assert directory.resolve("3omar khalil") == [OMAR_KHALIL]
    assert directory.resolve("3omar khalid") == []


def test_resolve_merchant_aliases(directory: PayeeDirectory) -> None:
    assert directory.resolve("etisalat bill") == [TELECOM]
    assert directory.resolve("My Phone Bill") == [TELECOM]
    assert directory.resolve("اتصالات الخليج") == [TELECOM]
    assert directory.resolve("Al-Ain Water") == [AL_AIN]
    assert directory.resolve("al-ain water") == [AL_AIN]
    assert directory.resolve("al ain water") == []


def test_shared_alias_returns_every_match_in_directory_order(directory: PayeeDirectory) -> None:
    matches = directory.resolve("Omar")
    assert matches == [OMAR_KHALID, OMAR_SAEED]
    assert directory.resolve("omar") == matches
    assert directory.resolve("عمر") == [OMAR_KHALID]


def test_shared_display_name_returns_both_payees() -> None:
    first = Payee("P-1", "Omar Khalid", "عمر خالد")
    second = Payee("P-2", "Omar Khalid", "عمر خالد", ("the other omar",))
    directory = PayeeDirectory([first, second])
    assert directory.resolve("Omar Khalid") == [first, second]
    assert directory.resolve("عمر خالد") == [first, second]
    assert directory.resolve("the other omar") == [second]


def test_no_fuzzy_prefix_or_substring_matching(directory: PayeeDirectory) -> None:
    for mention in ("Oma", "Omar K", "Khalid", "Omar Khalidd", "Omar Khalid Ali", "Omar Khaled", "omr", "عمر خ", "خالد"):
        assert directory.resolve(mention) == [], mention


def test_resolve_empty_mentions(directory: PayeeDirectory) -> None:
    assert directory.resolve("") == []
    assert directory.resolve("   ") == []
    assert directory.resolve("...") == []
    assert directory.resolve(None) == []  # type: ignore[arg-type]


def test_resolve_does_not_match_identifiers(directory: PayeeDirectory) -> None:
    assert directory.resolve("P-114") == []
    assert directory.resolve("p-114") == []


def test_resolution_uses_the_matching_normalisation_on_both_sides(directory: PayeeDirectory) -> None:
    for payee in directory:
        for label in payee.labels():
            assert payee in directory.resolve(label), label
            assert payee in directory.resolve(normalize_for_matching(label)), label


# Identifier lookup and rendering


def test_by_id_is_verbatim(directory: PayeeDirectory) -> None:
    assert directory.by_id("P-114") == OMAR_KHALID
    assert directory.by_id("p-114") is None
    assert directory.by_id("P-114 ") is None
    assert directory.by_id("P-999") is None


def test_render_name(directory: PayeeDirectory) -> None:
    assert directory.render_name("P-114", "en") == "Omar Khalid"
    assert directory.render_name("P-114", "ar") == "عمر خالد"
    assert directory.render_name("P-114", "mix") == "Omar Khalid"
    assert directory.render_name("P-999", "en") == "P-999"
    assert directory.render_name("P-999", "ar") == "P-999"


def test_render_name_falls_back_to_the_other_language() -> None:
    directory = PayeeDirectory([Payee("P-1", "", "عمر"), Payee("P-2", "Omar", "")])
    assert directory.render_name("P-1", "en") == "عمر"
    assert directory.render_name("P-2", "ar") == "Omar"
    assert PayeeDirectory([Payee("P-3", "", "")]).render_name("P-3", "ar") == "P-3"


# Property: any label of a payee resolves to it, whatever marks, case or spacing the mention carries

_LATIN_WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8)
_ARABIC_WORD = st.text(alphabet="ابتثجحخدذرزسشصضطظعغفقكلمنهوي", min_size=1, max_size=8)
_MARKS = "ًٌٍَُِّْـ"


def _decorate(word: str, picks: list[int]) -> str:
    out: list[str] = []
    for i, ch in enumerate(word):
        pick = picks[i % len(picks)] if picks else 0
        out.append(ch.upper() if pick % 3 == 1 else ch)
        if pick % 4 == 2:
            out.append(_MARKS[pick % len(_MARKS)])
    return "".join(out)


@settings(max_examples=150, derandomize=True, deadline=None)
@given(
    words=st.lists(st.one_of(_LATIN_WORD, _ARABIC_WORD), min_size=1, max_size=3),
    picks=st.lists(st.integers(min_value=0, max_value=11), min_size=1, max_size=6),
    spacing=st.sampled_from([" ", "  ", " \t"]),
)
def test_any_alias_resolves_under_decoration(words: list[str], picks: list[int], spacing: str) -> None:
    alias = " ".join(words)
    payee = Payee("P-1", "Some-One", "شخص-ما", (alias,))
    other = Payee("P-2", "Other-Party", "طرف-آخر", ("other-party-alias",))
    directory = PayeeDirectory([payee, other])
    mention = spacing.join(_decorate(w, picks) for w in words)
    assert directory.resolve(mention) == [payee]
    assert directory.resolve(alias + " extra") == []
