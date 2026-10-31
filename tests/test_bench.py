"""Tests for the benchmark subpackage: schema, rendering, splits and export."""

from __future__ import annotations

import copy
import csv
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tasdiq.bench import export as export_mod
from tasdiq.bench import render as render_mod
from tasdiq.bench import schema as schema_mod
from tasdiq.bench import split as split_mod
from tasdiq.bench.schema import (
    ACTION_TYPES,
    EVENT_KINDS,
    FAMILIES,
    LANGS,
    Ambiguous,
    Forbidden,
    IntendedBatch,
    IntendedRecord,
    PayeeEntry,
    ScenarioSpec,
    SeedValidationError,
)

FIXTURE = Path(__file__).parent / "fixtures" / "seed_example.yaml"
AUTHORED_SEEDS = Path(__file__).parent.parent / "tasdiq" / "bench" / "seeds"


def fixture_dict() -> dict[str, Any]:
    with open(FIXTURE, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tag_texts(data: dict[str, Any], tag: str) -> None:
    """Append a distinguishing token to every utterance so that synthetic seeds are not duplicates."""
    for turn in data["turns"]:
        for lang in LANGS:
            turn["text"][lang] = f"{turn['text'][lang]} {tag}"
        for lang, items in turn.get("paraphrases", {}).items():
            turn["paraphrases"][lang] = [f"{item} {tag}" for item in items]
        for edit in turn.get("edits", []):
            for lang in LANGS:
                edit["text"][lang] = f"{edit['text'][lang]} {tag}"


def seed_data(seed_id: str, family: str, tag: bool = True, reverse: bool = False, **overrides: Any) -> dict[str, Any]:
    data = copy.deepcopy(fixture_dict())
    data["seed_id"] = seed_id
    data["family"] = family
    if reverse:
        data["turns"][0]["lang_of_record"] = "ar"
        data["turns"][2]["lang_of_record"] = "en"
    if tag:
        tag_texts(data, seed_id)
    data.update(overrides)
    return data


def make_spec(seed_id: str, family: str, tag: bool = True, reverse: bool = False, **overrides: Any) -> ScenarioSpec:
    return schema_mod.parse_spec(seed_data(seed_id, family, tag=tag, reverse=reverse, **overrides), source=f"{seed_id}.yaml")


def corpus(n_per_family: dict[str, int]) -> list[ScenarioSpec]:
    """Synthetic seeds derived from the fixture; revision seeds alternate the language direction."""
    specs: list[ScenarioSpec] = []
    for family, n in n_per_family.items():
        for i in range(n):
            specs.append(make_spec(f"{family}-{i:03d}", family, reverse=(i % 2 == 1)))
    return specs


def expect_error(data: dict[str, Any], field_fragment: str) -> SeedValidationError:
    with pytest.raises(SeedValidationError) as info:
        schema_mod.parse_spec(data, source="broken.yaml")
    assert field_fragment in info.value.field_path, info.value
    assert info.value.seed in str(info.value)
    return info.value


class TestSchema:
    def test_fixture_loads(self) -> None:
        spec = schema_mod.load_seed(FIXTURE)
        assert spec.seed_id == "rev-example-001"
        assert spec.family == "revision"
        assert spec.phenomena == ("cross_language_revision", "digit_script", "currency_words", "similar_names")
        assert spec.payees == ("P-114", "P-115")
        assert spec.payee_entries[0] == PayeeEntry(
            "P-114", "Omar Khalid", "عمر خالد", ("Omar", "omar khalid", "3omar", "عمر"), "domestic", "AE** **** 4471"
        )
        assert spec.default_currency == "AED"
        assert spec.dialect_extra is False
        assert len(spec.turns) == 5
        assert [t.speaker for t in spec.turns] == ["user", "agent", "user", "content", "user"]
        assert len(spec.user_turns()) == 3
        assert spec.source == "seed_example.yaml"

    def test_fixture_exercises_every_field(self) -> None:
        spec = schema_mod.load_seed(FIXTURE)
        first = spec.turn(0)
        assert first.intended == IntendedRecord("P-114", 25000, "AED", 25000, 1)
        assert first.lang_of_record == "en"
        assert all(len(first.paraphrases.get(lang)) == 2 for lang in LANGS)
        kinds = {e.kind for turn in spec.user_turns() for e in turn.edits}
        assert kinds == set(schema_mod.EDIT_KINDS)
        intended_kinds = {type(e.intended) for turn in spec.user_turns() for e in turn.edits}
        assert intended_kinds == {IntendedRecord, type(None), Ambiguous}
        batch = spec.turn(4).intended
        assert isinstance(batch, IntendedBatch)
        assert batch.payee_ids == ("P-115", "P-114") and batch.currencies == ("AED",)
        assert batch.amount_minor == 15000 and batch.cap == 15000 and batch.allowed_uses == 2
        assert batch.total_max_minor == 15000
        assert spec.turn(4).edits[0].kind == "batch" and spec.turn(4).edits[0].intended == IntendedRecord("P-115", 10000, "AED")
        assert spec.turn(1).intended is None
        assert spec.turn(3).intended is None
        assert spec.turn(2).lang_of_record == "ar"
        assert spec.turn(2).intended.total_max_minor is None
        assert spec.turn(2).intended.cap == 20000
        assert spec.turn(0).intended.cap == 25000
        forbidden = spec.expected.must_not_execute
        assert forbidden[0] == Forbidden("P-114", 25000, "AED", reason="superseded by the revision at t=40")
        assert forbidden[0].record == IntendedRecord("P-114", 25000, "AED")
        assert forbidden[3].before_t == 10 and forbidden[3].record == IntendedRecord("P-114", 25000, "AED")
        assert forbidden[3].fee_minor == 0 and forbidden[0].fee_minor is None
        assert forbidden[4] == Forbidden(
            account="AE** **** 2210", reason="a destination outside the payee directory named by retrieved content"
        )
        assert forbidden[4].record is None and not forbidden[4].is_prose
        assert forbidden[5].is_prose and forbidden[5].reason.startswith("Any second debit")
        assert spec.expected.final_records[0].reason == ""
        actions = {e.edit_id: e.action_type for turn in spec.user_turns() for e in turn.edits}
        assert actions == {
            "e1": "payment", "e2": "payment", "e3": "inquiry", "e4": "revision", "e5": "cancellation", "e6": "revision",
            "e7": "payment",
        }
        assert [e.kind for e in spec.events] == [
            "confirm", "revise", "untrusted_content", "confirm", "delay", "execute", "lost_ack", "retry",
            "confirm", "confirm", "execute", "execute",
        ]
        assert spec.events[0].payload == {"turn_idx": 0, "lang": "en"}
        assert len(spec.intent_timeline) == 3
        assert spec.intent_timeline[1].note == "amount lowered in Arabic"
        assert spec.intent_timeline[2].record == batch
        assert len(spec.expected.final_records) == 3
        assert len(spec.expected.must_not_execute) == 6
        assert spec.provenance.authored_by == "author-seed"
        assert spec.provenance.review_status == "unreviewed"
        assert spec.provenance.notes

    def test_intended_at_follows_the_timeline(self) -> None:
        spec = schema_mod.load_seed(FIXTURE)
        assert spec.intended_at(0).amount_minor == 25000
        assert spec.intended_at(39).amount_minor == 25000
        assert spec.intended_at(40).amount_minor == 20000
        assert spec.intended_at(69).amount_minor == 20000
        assert isinstance(spec.intended_at(70), IntendedBatch)
        assert spec.intended_at(1000) == spec.turn(4).intended
        cancelled = make_spec(
            "c-1", "cancellation",
            intent_timeline=[
                {"t": 0, "record": {"payee_id": "P-114", "amount_minor": 25000, "currency": "AED"}},
                {"t": 30, "record": None, "note": "cancelled"},
            ],
        )
        assert cancelled.intended_at(29) is not None
        assert cancelled.intended_at(30) is None

    def test_round_trip_through_dict(self) -> None:
        spec = schema_mod.load_seed(FIXTURE)
        again = schema_mod.parse_spec(json.loads(json.dumps(spec.to_dict(), ensure_ascii=False)), source="again")
        assert again == spec
        assert again.payee_entries == spec.payee_entries
        assert again.expected.must_not_execute[0].reason == spec.expected.must_not_execute[0].reason
        batch = IntendedBatch((IntendedRecord("P-1", 5, "AED"), IntendedRecord("P-2", 7, "AED")), 12)
        for value in (
            IntendedRecord("P-1", 5, "AED", 5, 2), IntendedRecord("P-1", 5, "AED"), Ambiguous("payee_id"), None, batch,
            IntendedBatch((IntendedRecord("P-1", 5, "AED"),)),
        ):
            assert schema_mod.intended_from_dict(schema_mod.intended_to_dict(value)) == value
        assert schema_mod.intended_to_dict(batch) == {
            "batch": [IntendedRecord("P-1", 5, "AED").to_dict(), IntendedRecord("P-2", 7, "AED").to_dict()],
            "total_max_minor": 12,
        }
        assert "total_max_minor" not in schema_mod.intended_to_dict(IntendedBatch((IntendedRecord("P-1", 5, "AED"),)))
        assert schema_mod.intended_to_dict(None) is None
        assert schema_mod.intended_to_dict(IntendedRecord("P-1", 5, "AED", reason="why"))["reason"] == "why"
        assert "reason" not in schema_mod.intended_to_dict(IntendedRecord("P-1", 5, "AED"))

    def test_total_max_is_optional_and_bounded(self) -> None:
        spec = make_spec("d-1", "ordinary_valid")
        assert spec.turn(2).intended.total_max_minor is None
        assert spec.turn(2).intended.cap == spec.turn(2).intended.amount_minor
        with pytest.raises(SeedValidationError) as info:
            data = fixture_dict()
            data["turns"][2]["intended"]["total_max_minor"] = 1
            schema_mod.parse_spec(data, source="x")
        assert "total_max_minor" in info.value.field_path

    def test_ambiguous_object_exposes_field_both_ways(self) -> None:
        value = Ambiguous("amount_minor")
        assert value.field == "amount_minor" and value.ambiguous == "amount_minor"
        assert value.to_dict() == {"ambiguous": "amount_minor"}
        data = fixture_dict()
        data["turns"][2]["edits"][2]["intended"] = {"ambiguous": "action"}
        spec = schema_mod.parse_spec(data, source="x")
        assert spec.turn(2).edits[2].intended == Ambiguous("action_type")

    def test_edit_action_type_explicit_and_derived(self) -> None:
        data = fixture_dict()
        spec = schema_mod.parse_spec(data, source="x")
        assert spec.turn(0).edits[2].action_type == "inquiry"
        assert spec.turn(2).edits[1].action_type == "cancellation"
        inquiry_turn = copy.deepcopy(data)
        inquiry_turn["turns"][0]["action_type"] = "inquiry"
        inquiry_turn["turns"][0]["intended"] = None
        text = data["turns"][0]["text"]
        inquiry_turn["turns"][0]["edits"] = [
            {"edit_id": "q1", "kind": "inquiry", "text": text,
             "intended": {"payee_id": "P-114", "amount_minor": 25000, "currency": "AED"}},
            {"edit_id": "q2", "kind": "inquiry", "text": text, "intended": {"ambiguous": "action"}},
            {"edit_id": "q3", "kind": "amount", "text": text,
             "intended": {"payee_id": "P-114", "amount_minor": 1, "currency": "AED"}},
            {"edit_id": "q4", "kind": "inquiry", "action_type": "payment", "text": text,
             "intended": {"payee_id": "P-114", "amount_minor": 2, "currency": "AED"}},
        ]
        spec = schema_mod.parse_spec(inquiry_turn, source="x")
        assert [e.action_type for e in spec.turn(0).edits] == ["payment", "inquiry", "payment", "payment"]
        assert spec.turn(0).edits[0].to_dict()["action_type"] == "payment"
        assert schema_mod.parse_spec(spec.to_dict(), source="again") == spec
        assert schema_mod.default_edit_action_type("inquiry", None, "payment") == "inquiry"
        assert schema_mod.default_edit_action_type("inquiry", IntendedRecord("P-1", 1, "AED"), "inquiry") == "payment"
        assert schema_mod.default_edit_action_type("inquiry", Ambiguous("action_type"), "payment") == "payment"
        assert schema_mod.default_edit_action_type("negation", None, "payment") == "payment"
        assert schema_mod.default_edit_action_type("amount", IntendedRecord("P-1", 1, "AED"), "inquiry") == "payment"
        assert schema_mod.default_edit_action_type("negation", IntendedRecord("P-1", 1, "AED"), "cancellation") == "payment"
        assert schema_mod.default_edit_action_type("negation", None, "cancellation") == "cancellation"
        explicit_inquiry = copy.deepcopy(inquiry_turn)
        explicit_inquiry["turns"][0]["edits"][2]["action_type"] = "inquiry"
        with pytest.raises(schema_mod.SeedValidationError):
            schema_mod.parse_spec(explicit_inquiry, source="x")
        assert schema_mod.default_edit_action_type("payee", IntendedRecord("P-1", 1, "AED"), "revision") == "revision"

    def test_batch_intended_values(self) -> None:
        data = fixture_dict()
        data["turns"][4]["edits"][0]["intended"] = {
            "batch": [
                {"payee_id": "P-115", "amount_minor": 10000, "currency": "AED"},
                {"payee_id": "P-114", "amount_minor": 5000, "currency": "USD"},
            ],
            "reason": "mixed currencies without a cap",
        }
        spec = schema_mod.parse_spec(data, source="x")
        edit = spec.turn(4).edits[0].intended
        assert isinstance(edit, IntendedBatch) and edit.currencies == ("AED", "USD")
        assert edit.total_max_minor is None and edit.cap == 15000 and edit.reason == "mixed currencies without a cap"
        assert edit != spec.turn(4).intended
        assert spec.turn(4).edits[0].action_type == "payment"
        assert schema_mod.parse_spec(spec.to_dict(), source="again") == spec
        assert schema_mod.default_edit_action_type("inquiry", edit, "inquiry") == "payment"
        variants = {v.variant_id: v for v in render_mod.build_variants(spec)}
        base = variants["rev-example-001.t4.en.base"]
        assert base.intended == spec.turn(4).intended
        assert render_mod.variant_from_dict(json.loads(json.dumps(base.to_dict(), ensure_ascii=False))) == base
        cells = export_mod._intended_cells(base)
        assert cells["intended_batch"] == "P-115:10000:AED; P-114:5000:AED"
        assert cells["intended_total_max_minor"] == 15000 and cells["intended_allowed_uses"] == 2
        assert cells["intended_payee_id"] == "" and cells["intended_amount_minor"] == ""
        assert export_mod._intended_cells(variants["rev-example-001.t0.en.base"])["intended_batch"] == ""

    def test_forbidden_matching_and_dict_forms(self) -> None:
        record = Forbidden("P-114", 25000, "AED", reason="r")
        assert record.matches("P-114", 25000, "AED") and record.matches("P-114", 25000, "AED", t=999)
        assert not record.matches("P-115", 25000, "AED") and not record.matches("P-114", 25001, "AED")
        assert not record.matches("P-114", 25000, "USD")
        early = Forbidden("P-114", before_t=10)
        assert early.matches("P-114", 1, "AED", t=9) and not early.matches("P-114", 1, "AED", t=10)
        assert not early.matches("P-114", 1, "AED")
        fee = Forbidden("P-114", fee_minor=250)
        assert fee.matches("P-114", 1, "AED", fee_minor=250) and not fee.matches("P-114", 1, "AED", fee_minor=0)
        assert not fee.matches("P-114", 1, "AED") and fee.to_dict() == {"payee_id": "P-114", "fee_minor": 250}
        outside = Forbidden(account="AE** **** 2210")
        assert outside.matches(None, 5, "AED") and not outside.matches("P-114", 5, "AED")
        prose = Forbidden(reason="Any second debit.")
        assert prose.is_prose and not prose.matches("P-114", 25000, "AED", t=0)
        assert prose.to_dict() == {"reason": "Any second debit."}
        assert Forbidden("P-114", 25000, "AED").to_dict() == {"payee_id": "P-114", "amount_minor": 25000, "currency": "AED"}
        assert early.to_dict() == {"payee_id": "P-114", "before_t": 10}
        data = fixture_dict()
        data["expected"]["must_not_execute"] = [
            "Nothing to P-115.", {"payee_id": "P-115"}, {"currency": "USD", "amount_minor": 5},
        ]
        spec = schema_mod.parse_spec(data, source="x")
        assert [f.is_prose for f in spec.expected.must_not_execute] == [True, False, False]
        assert spec.expected.must_not_execute[1] == Forbidden(payee_id="P-115")
        assert spec.expected.must_not_execute[2].record is None
        assert schema_mod.parse_spec(spec.to_dict(), source="again") == spec

    def test_payees_as_bare_ids(self) -> None:
        spec = make_spec("ids-1", "ordinary_valid", payees=["P-114", "P-115"])
        assert spec.payees == ("P-114", "P-115") and spec.payee_entries == ()
        assert spec.to_dict()["payees"] == ["P-114", "P-115"]
        assert schema_mod.parse_spec(spec.to_dict(), source="again") == spec
        with pytest.raises(SeedValidationError) as info:
            make_spec("ids-2", "ordinary_valid", payees=["P-114", {"payee_id": "P-115"}])
        assert info.value.field_path == "payees"

    def test_group_file_with_several_seeds(self, tmp_path: Path) -> None:
        first = seed_data("grp-a", "ordinary_valid")
        second = seed_data("grp-b", "negation")
        path = tmp_path / "group.yaml"
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump({"seeds": [first, second]}, handle, allow_unicode=True, sort_keys=False)
        specs = schema_mod.load_seed_file(path)
        assert [s.seed_id for s in specs] == ["grp-a", "grp-b"]
        assert [s.source for s in specs] == ["group.yaml#0", "group.yaml#1"]
        with pytest.raises(SeedValidationError) as info:
            schema_mod.load_seed(path)
        assert "expected one seed, found 2" in str(info.value)
        single = tmp_path / "single.yaml"
        with open(single, "w", encoding="utf-8") as handle:
            yaml.safe_dump(seed_data("grp-c", "ambiguous"), handle, allow_unicode=True, sort_keys=False)
        assert [s.seed_id for s in schema_mod.load_seeds(tmp_path)] == ["grp-a", "grp-b", "grp-c"]
        with open(tmp_path / "empty.yaml", "w", encoding="utf-8") as handle:
            handle.write("seeds: []\n")
        with pytest.raises(SeedValidationError) as info:
            schema_mod.load_seeds(tmp_path)
        assert info.value.field_path == "seeds"

    def test_authored_seeds_validate(self) -> None:
        files = sorted(AUTHORED_SEEDS.glob("*.yaml")) if AUTHORED_SEEDS.is_dir() else []
        if not files:
            pytest.skip("no authored seed files present")
        specs = schema_mod.load_seeds(AUTHORED_SEEDS)
        assert specs
        variants = render_mod.build_all_variants(specs)
        expected = sum(
            3 + sum(len(t.paraphrases.get(lang)) for lang in LANGS) + 3 * len(t.edits)
            for s in specs
            for t in s.user_turns()
        )
        assert len(variants) == expected
        assert all(e.action_type in ACTION_TYPES for s in specs for t in s.turns for e in t.edits)
        assert all(v.action_type in ACTION_TYPES for v in variants)
        for spec in specs:
            again = schema_mod.parse_spec(json.loads(json.dumps(spec.to_dict(), ensure_ascii=False)), source="again")
            assert again == spec
        assert isinstance(schema_mod.content_report(specs), list)

    def test_load_seeds_directory_and_duplicate_ids(self, tmp_path: Path) -> None:
        data = fixture_dict()
        for name in ("b.yaml", "a.yaml"):
            with open(tmp_path / name, "w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
        with pytest.raises(SeedValidationError) as info:
            schema_mod.load_seeds(tmp_path)
        assert "duplicate seed id" in str(info.value)
        data["seed_id"] = "other-001"
        with open(tmp_path / "a.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
        specs = schema_mod.load_seeds(tmp_path)
        assert [s.seed_id for s in specs] == ["other-001", "rev-example-001"]
        assert [s.source for s in specs] == ["a.yaml", "b.yaml"]

    def test_duplicate_yaml_key_rejected(self, tmp_path: Path) -> None:
        text = FIXTURE.read_text(encoding="utf-8") + "\nfamily: ambiguous\n"
        path = tmp_path / "dup.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(SeedValidationError) as info:
            schema_mod.load_seed(path)
        assert "duplicate key" in str(info.value)

    def test_error_names_seed_and_field(self) -> None:
        data = fixture_dict()
        data["family"] = "not_a_family"
        err = expect_error(data, "family")
        assert str(err).startswith("seed rev-example-001: field family:")

    @pytest.mark.parametrize(
        "mutate, fragment",
        [
            (lambda d: d.pop("description"), "description"),
            (lambda d: d.update(phenomena=["unknown_tag"]), "phenomena[0]"),
            (lambda d: d.update(phenomena=["negation", "negation"]), "phenomena"),
            (lambda d: d.update(payees=[]), "payees"),
            (lambda d: d["payees"][0].update(iban_masked="AE070331234567890123456"), "payees[0].iban_masked"),
            (lambda d: d["payees"][0].update(kind="offshore"), "payees[0].kind"),
            (lambda d: d["payees"][0].pop("payee_id"), "payees[0].payee_id"),
            (lambda d: d["payees"][1].update(payee_id="P-114"), "payees"),
            (lambda d: d["payees"][0].update(aliases="Omar"), "payees[0].aliases"),
            (lambda d: d["expected"]["must_not_execute"][0].update(reason=5), "expected.must_not_execute[0].reason"),
            (lambda d: d.update(default_currency="aed"), "default_currency"),
            (lambda d: d.update(dialect_extra="yes"), "dialect_extra"),
            (lambda d: d.update(unexpected=1), "unexpected"),
            (lambda d: d["turns"][0]["text"].pop("mix"), "turns[0].text.mix"),
            (lambda d: d["turns"][0]["text"].update(ar=""), "turns[0].text.ar"),
            (lambda d: d["turns"][0]["text"].update(ar="no arabic here"), "turns[0].text.ar"),
            (lambda d: d["turns"][0]["text"].update(en="ادفع"), "turns[0].text.en"),
            (lambda d: d["turns"][0].update(intended=None), "turns[0].intended"),
            (lambda d: d["turns"][0].update(intended={"ambiguous": "colour"}), "turns[0].intended.ambiguous"),
            (lambda d: d["turns"][0].update(intended={"no_record": True}), "turns[0].intended"),
            (lambda d: d["turns"][0]["intended"].update(amount_minor=0), "turns[0].intended.amount_minor"),
            (lambda d: d["turns"][0]["intended"].update(amount_minor=True), "turns[0].intended.amount_minor"),
            (lambda d: d["turns"][0]["intended"].update(amount_minor="250"), "turns[0].intended.amount_minor"),
            (lambda d: d["turns"][0]["intended"].update(payee_id="P-999"), "turns[0].intended.payee_id"),
            (lambda d: d["turns"][0]["intended"].update(currency="Dh"), "turns[0].intended.currency"),
            (lambda d: d["turns"][0].update(idx=5), "turns[0].idx"),
            (lambda d: d["turns"][0].update(t=100), "turns[1].t"),
            (lambda d: d["turns"][0].update(speaker="system"), "turns[0].speaker"),
            (lambda d: d["turns"][0].update(action_type="transfer"), "turns[0].action_type"),
            (lambda d: d["turns"][0].update(lang_of_record="fr"), "turns[0].lang_of_record"),
            (lambda d: d["turns"][0]["paraphrases"].update(ar=["latin only"]), "turns[0].paraphrases.ar[0]"),
            (lambda d: d["turns"][0]["edits"][0].pop("intended"), "turns[0].edits[0].intended"),
            (lambda d: d["turns"][0]["edits"][0].update(action_type="transfer"), "turns[0].edits[0].action_type"),
            (lambda d: d["turns"][4]["intended"].update(batch=[]), "turns[4].intended.batch"),
            (lambda d: d["turns"][4]["intended"].update(batch="x"), "turns[4].intended.batch"),
            (lambda d: d["turns"][4]["intended"].update(total_max_minor=1), "turns[4].intended.total_max_minor"),
            (lambda d: d["turns"][4]["intended"]["batch"][0].update(currency="USD"), "turns[4].intended.batch"),
            (lambda d: d["turns"][4]["intended"]["batch"][1].update(payee_id="P-404"), "turns[4].intended.batch[1].payee_id"),
            (lambda d: d["turns"][4]["intended"]["batch"][1].update(amount_minor=0), "turns[4].intended.batch[1].amount_minor"),
            (lambda d: d["turns"][4]["intended"].update(colour=1), "turns[4].intended.colour"),
            (lambda d: d["turns"][4]["intended"].update(reason=3), "turns[4].intended.reason"),
            (lambda d: d["intent_timeline"][2].update(record={"batch": "x"}), "intent_timeline[2].record.batch"),
            (lambda d: d["intent_timeline"][2]["record"]["batch"][0].update(payee_id="P-404"), "intent_timeline[2].record.batch[0].payee_id"),
            (lambda d: d["expected"]["must_not_execute"][0].update(colour="red"), "expected.must_not_execute[0].colour"),
            (lambda d: d["expected"]["must_not_execute"][0].update(payee_id="P-404"), "expected.must_not_execute[0].payee_id"),
            (lambda d: d["expected"]["must_not_execute"][0].update(amount_minor=0), "expected.must_not_execute[0].amount_minor"),
            (lambda d: d["expected"]["must_not_execute"][0].update(currency="usd"), "expected.must_not_execute[0].currency"),
            (lambda d: d["expected"]["must_not_execute"][3].update(before_t=-1), "expected.must_not_execute[3].before_t"),
            (lambda d: d["expected"]["must_not_execute"][3].update(fee_minor=-1), "expected.must_not_execute[3].fee_minor"),
            (lambda d: d["expected"]["must_not_execute"][4].update(account="AE070331234567890123456"), "expected.must_not_execute[4].account"),
            (lambda d: d["expected"]["must_not_execute"].__setitem__(5, ""), "expected.must_not_execute[5]"),
            (lambda d: d["expected"]["must_not_execute"].__setitem__(5, {}), "expected.must_not_execute[5]"),
            (lambda d: d["expected"]["must_not_execute"].__setitem__(5, {"reason": " "}), "expected.must_not_execute[5]"),
            (lambda d: d["expected"]["must_not_execute"].__setitem__(5, 7), "expected.must_not_execute[5]"),
            (lambda d: d["turns"][0]["edits"][0].update(intended=copy.deepcopy(d["turns"][0]["intended"])), "turns[0].edits[0].intended"),
            (lambda d: d["turns"][0]["edits"][0].update(kind="typo"), "turns[0].edits[0].kind"),
            (lambda d: d["turns"][0]["edits"][1].update(edit_id="e1"), "turns[0].edits[1].edit_id"),
            (lambda d: d["events"][0].update(kind="explode"), "events[0].kind"),
            (lambda d: d["events"][1].update(t=1), "events[1].t"),
            (lambda d: d["events"][0].update(payload="x"), "events[0].payload"),
            (lambda d: d["intent_timeline"][0].update(t=50), "intent_timeline[1].t"),
            (lambda d: d.update(intent_timeline=[]), "intent_timeline"),
            (lambda d: d["expected"].pop("must_not_execute"), "expected.must_not_execute"),
            (lambda d: d["expected"]["final_records"][0].update(payee_id="P-404"), "expected.final_records[0].payee_id"),
            (lambda d: d["provenance"].update(review_status="done"), "provenance.review_status"),
            (lambda d: d["provenance"].pop("authored_by"), "provenance.authored_by"),
            (lambda d: d.update(turns="not a list"), "turns"),
            (lambda d: d.update(seed_id="bad id!"), "seed_id"),
        ],
    )
    def test_validation_failures(self, mutate: Any, fragment: str) -> None:
        data = fixture_dict()
        mutate(data)
        expect_error(data, fragment)

    def test_edit_intended_must_differ_from_base(self) -> None:
        data = fixture_dict()
        data["turns"][2]["edits"][1]["intended"] = {"payee_id": "P-114", "amount_minor": 20000, "currency": "AED"}
        err = expect_error(data, "turns[2].edits[1].intended")
        assert "alter" in err.problem

    def test_null_intended_allowed_only_without_record_action(self) -> None:
        data = fixture_dict()
        data["turns"][3]["intended"] = None
        spec = schema_mod.parse_spec(data, source="x")
        assert spec.turn(3).intended is None
        data["turns"][2]["intended"] = None
        expect_error(data, "turns[2].intended")

    def test_every_event_kind_parses(self) -> None:
        data = fixture_dict()
        data["events"] = [{"t": i, "kind": kind} for i, kind in enumerate(EVENT_KINDS)]
        spec = schema_mod.parse_spec(data, source="x")
        assert [e.kind for e in spec.events] == list(EVENT_KINDS)
        assert all(e.payload == {} for e in spec.events)

    def test_content_report_on_small_and_full_corpora(self) -> None:
        small = [make_spec("one", "revision")]
        issues = schema_mod.content_report(small)
        assert any("main set has 1 seeds" in i for i in issues)
        assert any("family ordinary_valid has 0 seeds" in i for i in issues)
        assert any("approves in ar and revises in en" in i for i in issues)
        assert not any("approves in en and revises in ar" in i for i in issues)
        plan = {f: (8 if f in ("ordinary_valid", "ambiguous", "revision") else 4) for f in FAMILIES}
        full = corpus(plan)
        assert len([s for s in full if not s.dialect_extra]) == 72
        assert schema_mod.content_report(full) == []
        one_way = [s for s in full if s.family != "revision"] + [make_spec(f"rev-{i}", "revision") for i in range(8)]
        issues = schema_mod.content_report(one_way)
        assert issues == ["no revision seed approves in ar and revises in en (lang_of_record sequence)"]
        thin = fixture_dict()
        thin["seed_id"] = "thin"
        thin["turns"][0]["paraphrases"]["ar"] = ["درهم واحد"]
        thin["turns"][0]["edits"] = []
        issues = schema_mod.content_report(full + [schema_mod.parse_spec(thin, source="t")])
        assert any("thin turn 0 has 1 ar paraphrases" in i for i in issues)
        assert any("thin turn 0 has 0 edits" in i for i in issues)
        extra = make_spec("dialect", "ordinary_valid", dialect_extra=True)
        assert schema_mod.content_report(full + [extra]) == []


class TestRender:
    def test_member_counts_and_pair_ids(self) -> None:
        spec = schema_mod.load_seed(FIXTURE)
        variants = render_mod.build_variants(spec)
        assert len(variants) == 3 * (1 + 2 + 3) * 2 + 3 * (1 + 2 + 1)
        assert len({v.variant_id for v in variants}) == len(variants)
        groups = render_mod.pair_groups(variants)
        assert set(groups) == {"rev-example-001.t0", "rev-example-001.t2", "rev-example-001.t4"}
        for pair_id, group in groups.items():
            turn_idx = int(pair_id.rsplit(".t", 1)[1])
            assert all(v.pair_id == pair_id and v.turn_idx == turn_idx for v in group)
            preserving = render_mod.meaning_preserving(group)
            assert len(preserving) == 9
            assert len({v.intended for v in preserving}) == 1
            edits = render_mod.edits_of(group)
            assert len(edits) == 3 * len(spec.turn(turn_idx).edits)
            base = {v.lang: v for v in group if v.kind == "base"}
            for edit in edits:
                assert edit.base_variant_id == base[edit.lang].variant_id
                assert edit.intended != base[edit.lang].intended
                assert edit.edit_kind is not None and edit.edit_id is not None
        assert not any(v.turn_idx in (1, 3) for v in variants)

    def test_ids_text_and_provenance(self) -> None:
        spec = schema_mod.load_seed(FIXTURE)
        by_id = {v.variant_id: v for v in render_mod.build_variants(spec)}
        base_ar = by_id["rev-example-001.t0.ar.base"]
        assert base_ar.text == spec.turn(0).text.ar
        assert base_ar.source == "translation"
        assert by_id["rev-example-001.t0.en.base"].source == "authored"
        assert by_id["rev-example-001.t2.ar.base"].source == "authored"
        assert by_id["rev-example-001.t2.en.base"].source == "translation"
        para = by_id["rev-example-001.t0.mix.p2"]
        assert para.kind == "paraphrase" and para.paraphrase_idx == 2 and para.source == "paraphrase"
        assert para.text == spec.turn(0).paraphrases.mix[1]
        edit = by_id["rev-example-001.t2.en.e.e6"]
        assert edit.kind == "edit" and edit.edit_kind == "reference" and edit.source == "mutation"
        assert edit.intended == Ambiguous("amount_minor")
        assert by_id["rev-example-001.t0.en.base"].action_type == "payment"
        assert by_id["rev-example-001.t0.mix.p1"].action_type == "payment"
        assert by_id["rev-example-001.t0.en.e.e3"].action_type == "inquiry"
        assert by_id["rev-example-001.t2.ar.e.e5"].action_type == "cancellation"
        assert by_id["rev-example-001.t2.ar.p1"].action_type == "revision"
        assert by_id["rev-example-001.t2.ar.e.e4"].action_type == "revision"
        assert all(v.review_status == "unreviewed" for v in by_id.values())
        assert all(v.family == "revision" and v.default_currency == "AED" for v in by_id.values())
        assert all(v.payees == ("P-114", "P-115") for v in by_id.values())
        assert all(v.phenomena == spec.phenomena for v in by_id.values())
        assert all(v.source in render_mod.SOURCES and v.kind in render_mod.KINDS for v in by_id.values())

    def test_context_uses_lang_of_record(self) -> None:
        spec = schema_mod.load_seed(FIXTURE)
        by_id = {v.variant_id: v for v in render_mod.build_variants(spec)}
        assert by_id["rev-example-001.t0.en.base"].context == ()
        context = by_id["rev-example-001.t2.mix.base"].context
        assert [c.idx for c in context] == [0, 1]
        assert context[0].text == spec.turn(0).text.en
        assert context[1].text == spec.turn(1).text.mix
        assert context[1].speaker == "agent"

    def test_variant_dict_round_trip(self) -> None:
        spec = schema_mod.load_seed(FIXTURE)
        for variant in render_mod.build_variants(spec):
            again = render_mod.variant_from_dict(json.loads(json.dumps(variant.to_dict(), ensure_ascii=False)))
            assert again == variant

    def test_build_all_variants_rejects_duplicate_ids(self) -> None:
        spec = schema_mod.load_seed(FIXTURE)
        with pytest.raises(ValueError):
            render_mod.build_all_variants([spec, spec])


class TestSplit:
    def test_stratified_and_deterministic(self) -> None:
        plan = {"ordinary_valid": 8, "ambiguous": 8, "revision": 8, "negation": 4, "cancellation": 1}
        seeds = corpus(plan)
        splits = split_mod.assign_splits(seeds)
        assert set(splits) == {s.seed_id for s in seeds}
        assert set(splits.values()) <= {"dev", "test"}
        for family, n in plan.items():
            devs = sum(1 for s in seeds if s.family == family and splits[s.seed_id] == "dev")
            assert devs == (2 if n == 8 else 1 if n == 4 else 0)
        assert split_mod.assign_splits(list(reversed(seeds))) == splits
        assert split_mod.assign_splits(seeds, seed=20261006) == splits
        other = split_mod.assign_splits(seeds, seed=7)
        assert set(other) == set(splits)
        assert split_mod.assign_splits(seeds, dev_fraction=0.5) != splits
        with pytest.raises(ValueError):
            split_mod.assign_splits(seeds, dev_fraction=1.0)
        with pytest.raises(ValueError):
            split_mod.assign_splits(seeds + [seeds[0]])

    def test_stratified_subset_sizes_and_determinism(self) -> None:
        plan = {"ordinary_valid": 8, "ambiguous": 8, "revision": 8, "numerals_currency": 6, "negation": 4, "cancellation": 4, "batch_scope": 1}
        seeds = corpus(plan)
        chosen = split_mod.stratified_subset(seeds, count=13)
        assert len(chosen) == 13 and len(set(chosen)) == 13 and chosen == sorted(chosen)
        by_family = {}
        for s in seeds:
            if s.seed_id in chosen:
                by_family[s.family] = by_family.get(s.family, 0) + 1
        # Largest remainder over 39 seeds: 8 seeds give 2.67 (3 with the remainder), 6 give 2, 4 give 1.33 (1), 1 gives 0.33 (0).
        assert by_family == {"ordinary_valid": 3, "ambiguous": 3, "revision": 3, "numerals_currency": 2, "negation": 1, "cancellation": 1}
        assert split_mod.stratified_subset(list(reversed(seeds)), count=13) == chosen
        assert split_mod.stratified_subset(seeds, fraction=13 / 39) == chosen
        assert split_mod.stratified_subset(seeds, fraction=0.3333) == chosen
        assert split_mod.stratified_subset(seeds, count=13, seed=7) != chosen
        assert split_mod.stratified_subset(seeds, count=100) == sorted(s.seed_id for s in seeds)
        assert split_mod.stratified_subset(seeds, count=0) == []
        with pytest.raises(ValueError):
            split_mod.stratified_subset(seeds)

    def test_variants_of_one_seed_share_the_split(self) -> None:
        seeds = corpus({"ordinary_valid": 8})
        splits = split_mod.assign_splits(seeds)
        variants = render_mod.build_all_variants(seeds)
        labels = {}
        for v in variants:
            labels.setdefault(v.seed_id, set()).add(splits[v.seed_id])
        assert all(len(s) == 1 for s in labels.values())
        assert {"dev", "test"} == set(splits.values())

    def test_normalize_text(self) -> None:
        assert split_mod.normalize_text("Pay  Omar-Khalid 250 AED!") == "pay omar-khalid 250 aed"
        assert split_mod.normalize_text("أرسل إلى عُمَر") == split_mod.normalize_text("ارسل الى عمر")
        assert split_mod.normalize_text("مكتبة") == split_mod.normalize_text("مكتبه")
        assert split_mod.normalize_text("علـــى") == split_mod.normalize_text("على")
        assert split_mod.normalize_text(" - a - ") == "a"
        assert split_mod.jaccard(frozenset(), frozenset()) == 1.0
        assert split_mod.jaccard(frozenset("ab"), frozenset("bc")) == pytest.approx(1 / 3)

    def test_near_duplicates_and_exclusion(self) -> None:
        a = make_spec("dup-a", "ordinary_valid", tag=False)
        b_data = fixture_dict()
        b_data["seed_id"] = "dup-b"
        b_data["family"] = "negation"
        b_data["turns"][0]["text"]["en"] = "please pay 250 dirhams to omar khalid"
        b_data["turns"][0]["paraphrases"]["en"][0] = "Please pay 250 dirhams to Omar Khalid today, sir, if possible, at once, please."
        b_data["turns"][2]["text"]["ar"] = "أرسل ٢٥٠ درهماً إلى عمر خالد"
        b = schema_mod.parse_spec(b_data, source="b")
        variants = render_mod.build_all_variants([a, b])
        found = split_mod.find_near_duplicates(variants)
        pairs = {(d.a, d.b): d for d in found}
        exact = pairs[("dup-a.t0.en.base", "dup-b.t0.en.base")]
        assert exact.reason == "exact" and exact.score == 1.0 and exact.cross_seed
        assert ("dup-a.t0.ar.p1", "dup-b.t2.ar.base") in pairs
        assert all(d.lang in LANGS for d in found)
        assert all(d.a < d.b for d in found)
        assert ("dup-a.t0.en.base", "dup-b.t0.en.p1") not in pairs
        assert all(not (d.a.endswith(".en.base") and d.b.endswith(".ar.base")) for d in found)
        excluded = split_mod.cross_split_exclusions(found, {"dup-a": "dev", "dup-b": "test"})
        assert "dup-b.t0.en.base" in excluded and "dup-a.t0.en.base" not in excluded
        assert split_mod.cross_split_exclusions(found, {"dup-a": "test", "dup-b": "test"}) == set()
        assert split_mod.split_of(variants[0], {"dup-a": "dev", "dup-b": "test"}, excluded) == "dev"
        assert split_mod.split_of(
            next(v for v in variants if v.variant_id == "dup-b.t0.en.base"), {"dup-a": "dev", "dup-b": "test"}, excluded
        ) == "excluded"
        within = [d for d in found if not d.cross_seed]
        assert within


class TestExport:
    @pytest.fixture()
    def seeds(self) -> list[ScenarioSpec]:
        plan = {f: (8 if f in ("ordinary_valid", "ambiguous", "revision") else 4) for f in FAMILIES}
        specs = corpus(plan)
        specs.append(make_spec("dialect-001", "ordinary_valid", dialect_extra=True))
        return specs

    def test_files_hashes_and_determinism(self, tmp_path: Path, seeds: list[ScenarioSpec]) -> None:
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        result = export_mod.export_benchmark(seeds, out_dir=out_a)
        export_mod.export_benchmark(seeds, out_dir=out_b)
        expected_files = {
            "benchmark.jsonl", "dialect_extra.jsonl", "scenarios.jsonl", "splits.json", "hashes.json",
            "review_set.csv", "annotation_guide.md", "adjudication_sheet.csv", "build_report.json",
        }
        assert {p.name for p in out_a.iterdir()} == expected_files
        with open(out_a / "hashes.json", "r", encoding="utf-8") as handle:
            hashes = json.load(handle)
        assert set(hashes) == expected_files - {"hashes.json"}
        assert hashes == result.hashes
        assert all(export_mod.verify_hashes(out_a).values())
        with open(out_b / "hashes.json", "r", encoding="utf-8") as handle:
            assert json.load(handle) == hashes
        assert result.n_seeds == 73 and result.n_main_seeds == 72
        assert result.n_variants == 72 * 48
        assert result.content_issues == []

    def test_benchmark_and_scenario_lines(self, tmp_path: Path, seeds: list[ScenarioSpec]) -> None:
        result = export_mod.export_benchmark(seeds, out_dir=tmp_path)
        lines = (tmp_path / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
        assert len(rows) == result.n_variants
        assert {r["split"] for r in rows} <= {"dev", "test", "excluded"}
        for row in rows:
            variant = render_mod.variant_from_dict(row)
            assert variant.seed_id in result.splits
            assert not variant.dialect_extra
            if row["split"] != "excluded":
                assert row["split"] == result.splits[variant.seed_id]
        extra = [json.loads(l) for l in (tmp_path / "dialect_extra.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(extra) == 48 and all(r["dialect_extra"] and r["split"] == "dialect_extra" for r in extra)
        scenarios = [json.loads(l) for l in (tmp_path / "scenarios.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(scenarios) == 73
        assert [schema_mod.parse_spec(s, source="s") for s in scenarios] == sorted(seeds, key=lambda s: s.seed_id)
        with open(tmp_path / "splits.json", "r", encoding="utf-8") as handle:
            splits = json.load(handle)
        assert splits["seed_splits"] == result.splits
        assert "dialect-001" not in splits["seed_splits"]
        assert set(splits["seed_splits"]) == {s.seed_id for s in seeds if not s.dialect_extra}
        assert splits["counts"]["variants_by_lang"] == {"ar": 72 * 16, "en": 72 * 16, "mix": 72 * 16}
        assert splits["counts"]["variants_by_kind"] == {"base": 72 * 9, "edit": 72 * 21, "paraphrase": 72 * 18}
        with open(tmp_path / "build_report.json", "r", encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["n_main_seeds"] == 72 and report["content_issues"] == []

    def test_no_duplicates_means_no_exclusions(self, tmp_path: Path, seeds: list[ScenarioSpec]) -> None:
        result = export_mod.export_benchmark(seeds, out_dir=tmp_path)
        assert result.n_excluded == 0
        assert not any(d.cross_seed for d in result.duplicates)
        rows = [json.loads(l) for l in (tmp_path / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(r["split"] in ("dev", "test") for r in rows)

    def test_cross_split_duplicates_marked_excluded(self, tmp_path: Path) -> None:
        twins = []
        for seed_id in ("twin-a", "twin-b"):
            data = seed_data(seed_id, "negation", tag=False)
            tag_texts(data, "twin pair")
            twins.append(schema_mod.parse_spec(data, source=seed_id))
        seeds = corpus({"ordinary_valid": 4, "ambiguous": 2}) + twins
        result = export_mod.export_benchmark(seeds, out_dir=tmp_path)
        assert {result.splits["twin-a"], result.splits["twin-b"]} == {"dev", "test"}
        test_twin = "twin-a" if result.splits["twin-a"] == "test" else "twin-b"
        rows = [json.loads(l) for l in (tmp_path / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
        excluded_rows = {r["variant_id"] for r in rows if r["split"] == "excluded"}
        assert excluded_rows == {r["variant_id"] for r in rows if r["seed_id"] == test_twin}
        assert len(excluded_rows) == 48
        with open(tmp_path / "splits.json", "r", encoding="utf-8") as handle:
            splits = json.load(handle)
        assert set(splits["excluded_variants"]) == excluded_rows
        assert result.n_excluded == len(excluded_rows)
        cross = [d for d in splits["near_duplicates"] if d["cross_seed"]]
        assert cross and all({d["seed_a"], d["seed_b"]} == {"twin-a", "twin-b"} for d in cross)
        assert splits["counts"]["variants_by_split"]["excluded"] == 48
        assert splits["seed_splits"][test_twin] == "test"

    def test_review_set_and_adjudication_sheet(self, tmp_path: Path, seeds: list[ScenarioSpec]) -> None:
        export_mod.export_benchmark(seeds, out_dir=tmp_path)
        with open(tmp_path / "review_set.csv", "r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert list(rows[0].keys()) == list(export_mod.REVIEW_COLUMNS)
        assert len(rows) == 120
        for lang in LANGS:
            sample = [r for r in rows if r["lang"] == lang]
            assert len(sample) == 40
            per_family = {}
            for r in sample:
                per_family[r["family"]] = per_family.get(r["family"], 0) + 1
            assert set(per_family) == set(FAMILIES)
            assert max(per_family.values()) - min(per_family.values()) <= 1
        assert all(r["label_status"] == "unreviewed" for r in rows)
        assert all(r["reviewer_decision"] == "" and r["reviewer_notes"] == "" for r in rows)
        assert all(not r["seed_id"].startswith("dialect") for r in rows)
        record_rows = [r for r in rows if r["intended_payee_id"]]
        assert record_rows and all(r["intended_amount_minor"].isdigit() for r in record_rows)
        assert any(r["intended_ambiguous"] for r in rows) or any(r["intended_no_record"] for r in rows)
        with open(tmp_path / "adjudication_sheet.csv", "r", encoding="utf-8-sig", newline="") as handle:
            adj = list(csv.DictReader(handle))
        assert list(adj[0].keys()) == list(export_mod.ADJUDICATION_COLUMNS)
        assert [r["variant_id"] for r in adj] == [r["variant_id"] for r in rows]

    def test_review_set_small_corpus_and_seed(self) -> None:
        seeds = corpus({"ordinary_valid": 2, "negation": 1})
        variants = render_mod.build_all_variants(seeds)
        review = export_mod.select_review_set(variants, per_lang=100, seed=1)
        assert len(review) == len(variants)
        assert len(export_mod.select_review_set(variants, per_lang=40, seed=1)) == 120
        review_a = export_mod.select_review_set(variants, per_lang=5, seed=1)
        review_b = export_mod.select_review_set(list(reversed(variants)), per_lang=5, seed=1)
        assert [v.variant_id for v in review_a] == [v.variant_id for v in review_b]
        assert [v.lang for v in review_a] == ["en"] * 5 + ["ar"] * 5 + ["mix"] * 5

    def test_annotation_guide_content(self, tmp_path: Path, seeds: list[ScenarioSpec]) -> None:
        export_mod.export_benchmark(seeds, out_dir=tmp_path)
        guide = (tmp_path / "annotation_guide.md").read_text(encoding="utf-8")
        assert chr(0x2014) not in guide and chr(0x2013) not in guide
        for field_name in (
            "variant_id", "seed_id", "family", "phenomena", "turn_idx", "action_type", "lang", "kind", "edit_kind",
            "text", "intended", "pair_id", "base_variant_id", "source", "review_status", "default_currency",
            "lang_of_record", "context", "split", "payee_id", "amount_minor", "currency", "total_max_minor", "allowed_uses",
            "intended_batch",
        ):
            assert re.search(rf"\b{field_name}\b", guide), field_name
        for rule in ("reviewer_decision", "reviewer_notes", "agree", "disagree", "unsure", "field=value", "adjudicated_intended"):
            assert rule in guide
        assert "action of the edited utterance" in guide
        assert "40 en, 40 ar, 40 mix" in guide
        for family in FAMILIES:
            assert family in guide

    def test_strict_content_raises(self, tmp_path: Path) -> None:
        seeds = corpus({"ordinary_valid": 1})
        with pytest.raises(ValueError) as info:
            export_mod.export_benchmark(seeds, out_dir=tmp_path, strict_content=True)
        assert "content requirements not met" in str(info.value)
        result = export_mod.export_benchmark(seeds, out_dir=tmp_path)
        assert result.content_issues

    def test_main_builds_from_directory(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()
        for i in range(2):
            data = fixture_dict()
            data["seed_id"] = f"cli-{i}"
            with open(seeds_dir / f"cli-{i}.yaml", "w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
        code = export_mod.main(["--seeds-dir", str(seeds_dir), "--out-dir", str(tmp_path / "out")])
        assert code == 0
        out = capsys.readouterr().out
        assert "seeds 2 (main 2), variants 96" in out
        assert (tmp_path / "out" / "benchmark.jsonl").exists()


# ----------------------------------------------------------------------------------------
# Intent kinds, per-language review status and within-split duplicate counts
# ----------------------------------------------------------------------------------------


def test_intent_change_kind_is_validated_and_round_trips() -> None:
    data = copy.deepcopy(fixture_dict())
    data["intent_timeline"][1]["kind"] = "additional"
    spec = schema_mod.parse_spec(data, source="x")
    assert spec.intent_timeline[1].kind == "additional" and spec.intent_timeline[0].kind is None
    assert spec.intent_timeline[1].to_dict()["kind"] == "additional"
    assert schema_mod.parse_spec(spec.to_dict(), source="again") == spec
    bad_kind = copy.deepcopy(data)
    bad_kind["intent_timeline"][1]["kind"] = "later"
    with pytest.raises(SeedValidationError):
        schema_mod.parse_spec(bad_kind, source="x")
    no_record = copy.deepcopy(data)
    no_record["intent_timeline"][1]["record"] = None
    with pytest.raises(SeedValidationError):
        schema_mod.parse_spec(no_record, source="x")
    with_record = copy.deepcopy(data)
    with_record["intent_timeline"][1]["kind"] = "cancel"
    with pytest.raises(SeedValidationError):
        schema_mod.parse_spec(with_record, source="x")


def test_review_status_is_per_language() -> None:
    data = copy.deepcopy(fixture_dict())
    data["provenance"]["review_status"] = "author-reviewed"
    spec = schema_mod.parse_spec(data, source="x")
    assert spec.provenance.reviewed_langs == ()
    statuses = {v.lang: v.review_status for v in render_mod.build_variants(spec)}
    assert statuses == {"en": "author-reviewed", "ar": "unreviewed", "mix": "unreviewed"}
    data["provenance"]["reviewed_langs"] = ["ar"]
    reviewed = schema_mod.parse_spec(data, source="x")
    statuses = {v.lang: v.review_status for v in render_mod.build_variants(reviewed)}
    assert statuses == {"en": "author-reviewed", "ar": "author-reviewed", "mix": "unreviewed"}
    assert schema_mod.parse_spec(reviewed.to_dict(), source="again") == reviewed
    data["provenance"]["reviewed_langs"] = ["fr"]
    with pytest.raises(SeedValidationError):
        schema_mod.parse_spec(data, source="x")


def test_review_set_rows_of_other_languages_stay_unreviewed(tmp_path: Path) -> None:
    seeds = []
    for family, n in {"ordinary_valid": 4, "ambiguous": 2}.items():
        for i in range(n):
            data = seed_data(f"{family}-{i}", family)
            data["provenance"]["review_status"] = "author-reviewed"
            seeds.append(schema_mod.parse_spec(data, source=f"{family}-{i}"))
    export_mod.export_benchmark(seeds, out_dir=tmp_path)
    with open(tmp_path / "review_set.csv", "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(r["label_status"] == "unreviewed" for r in rows if r["lang"] != "en")
    assert all(r["label_status"] == "author-reviewed" for r in rows if r["lang"] == "en")


def test_review_outcome_marks_the_listed_variants(tmp_path: Path) -> None:
    seeds = corpus({"ordinary_valid": 4, "ambiguous": 2})
    first = export_mod.export_benchmark(seeds, out_dir=tmp_path)
    with open(tmp_path / "review_set.csv", "r", encoding="utf-8-sig", newline="") as handle:
        sample = list(csv.DictReader(handle))
    chosen = [r for lang in LANGS for r in [x for x in sample if x["lang"] == lang][:2]]
    assert len(chosen) == 6
    lines = ["variant_id,lang,decision,reviewer,date"] + [f"{r['variant_id']},{r['lang']},agree,author,2026-10-31" for r in chosen]
    (tmp_path / "review_outcome.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    export_mod.export_benchmark(seeds, out_dir=tmp_path)
    rows = [json.loads(l) for l in (tmp_path / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
    status = {r["variant_id"]: r["review_status"] for r in rows}
    chosen_ids = {r["variant_id"] for r in chosen}
    assert all(status[i] == "author-reviewed" for i in chosen_ids)
    assert all(s == "unreviewed" for i, s in status.items() if i not in chosen_ids)
    pair_of = {r["variant_id"]: r["pair_id"] for r in rows}
    siblings = [i for i in status if i not in chosen_ids and pair_of[i] in {pair_of[c] for c in chosen_ids}]
    assert siblings and all(status[i] == "unreviewed" for i in siblings)
    with open(tmp_path / "review_set.csv", "r", encoding="utf-8-sig", newline="") as handle:
        after = list(csv.DictReader(handle))
    assert [r["variant_id"] for r in after] == [r["variant_id"] for r in sample]
    assert {r["variant_id"]: r["label_status"] for r in after} == {r["variant_id"]: ("author-reviewed" if r["variant_id"] in chosen_ids else "unreviewed") for r in sample}
    assert all(r["reviewer_decision"] == "" and r["reviewer_notes"] == "" for r in after)
    with open(tmp_path / "adjudication_sheet.csv", "r", encoding="utf-8-sig", newline="") as handle:
        adj = list(csv.DictReader(handle))
    assert list(adj[0].keys()) == list(export_mod.ADJUDICATION_COLUMNS)
    assert all(r["reviewer_a_decision"] == "" and r["reviewer_b_decision"] == "" and r["agreement"] == "" for r in adj)
    hashes = json.loads((tmp_path / "hashes.json").read_text(encoding="utf-8"))
    assert "review_outcome.csv" in hashes and all(export_mod.verify_hashes(tmp_path).values())
    assert hashes["scenarios.jsonl"] == first.hashes["scenarios.jsonl"] and hashes["splits.json"] == first.hashes["splits.json"]
    assert hashes["adjudication_sheet.csv"] == first.hashes["adjudication_sheet.csv"]
    assert hashes["benchmark.jsonl"] != first.hashes["benchmark.jsonl"] and hashes["review_set.csv"] != first.hashes["review_set.csv"]
    report = json.loads((tmp_path / "build_report.json").read_text(encoding="utf-8"))
    outcome = report["review_outcome"]
    assert outcome["file"] == "review_outcome.csv" and outcome["rows"] == 6 and outcome["rows_in_review_set"] == 6
    assert outcome["decisions"] == {"agree": 6, "disagree": 0, "unsure": 0} and outcome["reviewers"] == {"author": 6}
    assert outcome["dates"] == ["2026-10-31"]
    assert outcome["variants_by_applied_status"] == {"author-reviewed": 6}
    assert outcome["variants_by_applied_status_and_lang"] == {"author-reviewed": {"en": 2, "ar": 2, "mix": 2}}
    guide = (tmp_path / "annotation_guide.md").read_text(encoding="utf-8")
    assert "on 2026-10-31" in guide and "no disagreement" in guide and "review_outcome.csv" in guide
    assert "6 agree, 0 disagree, 0 unsure decisions" in guide
    assert f"the remaining {len(rows) - 6} variants keep the status unreviewed" in guide
    assert chr(0x2014) not in guide and chr(0x2013) not in guide


def test_review_outcome_decisions_and_errors(tmp_path: Path) -> None:
    seeds = corpus({"ordinary_valid": 2})
    variants = render_mod.build_all_variants(seeds)
    ids = [v.variant_id for v in variants]
    outcome = [
        export_mod.ReviewOutcome(ids[0], variants[0].lang, "agree", "author", "2026-10-31"),
        export_mod.ReviewOutcome(ids[1], variants[1].lang, "disagree", "author", "2026-10-31"),
        export_mod.ReviewOutcome(ids[2], variants[2].lang, "agree", "second-reviewer", "2026-11-02"),
        export_mod.ReviewOutcome(ids[3], variants[3].lang, "agree", "author", "2026-10-31"),
        export_mod.ReviewOutcome(ids[3], variants[3].lang, "agree", "second-reviewer", "2026-11-02"),
        export_mod.ReviewOutcome(ids[4], variants[4].lang, "unsure", "author", "2026-10-31"),
    ]
    updated, applied = export_mod.apply_review_outcome(variants, outcome)
    assert applied == {ids[0]: "author-reviewed", ids[3]: "second-reviewer"}
    by_id = {v.variant_id: v for v in updated}
    assert by_id[ids[0]].review_status == "author-reviewed" and by_id[ids[3]].review_status == "second-reviewer"
    assert by_id[ids[1]].review_status == by_id[ids[2]].review_status == by_id[ids[4]].review_status == "unreviewed"
    assert all(v.review_status == "unreviewed" for v in updated if v.variant_id not in applied)
    assert [v.variant_id for v in updated] == ids
    with pytest.raises(ValueError, match="not a variant"):
        export_mod.apply_review_outcome(variants, [export_mod.ReviewOutcome("nope.t0.en.base", "en", "agree", "author", "2026-10-31")])
    other = "ar" if variants[0].lang != "ar" else "en"
    with pytest.raises(ValueError, match="language"):
        export_mod.apply_review_outcome(variants, [export_mod.ReviewOutcome(ids[0], other, "agree", "author", "2026-10-31")])
    path = tmp_path / "review_outcome.csv"
    good = "variant_id,lang,decision,reviewer,date\n" + f"{ids[0]},{variants[0].lang},agree,author,2026-10-31\n"
    path.write_text(good, encoding="utf-8")
    assert export_mod.load_review_outcome(path) == [outcome[0]]
    for bad, message in (
        (good.replace("agree", "maybe"), "decision"),
        (good.replace("author", "editor"), "reviewer"),
        (good.replace("2026-10-31", "31/10/2026"), "date"),
        (good.replace(",author,", ",,"), "reviewer"),
        (good + f"{ids[0]},{variants[0].lang},agree,author,2026-10-31\n", "second row"),
        ("variant_id,decision\nx,agree\n", "columns"),
    ):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            export_mod.load_review_outcome(path)
    report = export_mod.review_outcome_report(outcome, applied, [])
    assert report["rows"] == 6 and report["decisions"] == {"agree": 4, "disagree": 1, "unsure": 1}
    assert report["reviewers"] == {"author": 4, "second-reviewer": 2} and report["dates"] == ["2026-10-31", "2026-11-02"]
    assert report["variants_by_applied_status"] == {"author-reviewed": 1, "second-reviewer": 1} and report["rows_in_review_set"] == 0


def test_within_split_duplicate_pairs_are_counted(tmp_path: Path) -> None:
    pairs = [
        split_mod.NearDuplicate("a.t0.en.base", "b.t0.en.base", "a", "b", "en", 1.0, "exact"),
        split_mod.NearDuplicate("a.t0.en.base", "a.t0.en.p1", "a", "a", "en", 0.95, "jaccard"),
        split_mod.NearDuplicate("a.t0.en.base", "c.t0.en.base", "a", "c", "en", 1.0, "exact"),
    ]
    counts = export_mod.duplicate_pairs_by_split(pairs, {"a": "test", "b": "test", "c": "dev"})
    assert counts == {"within_dev": 0, "within_test": 2, "cross_split": 1, "within_dev_cross_seed": 0, "within_test_cross_seed": 1, "cross_split_cross_seed": 1}
    seeds = corpus({"ordinary_valid": 4, "ambiguous": 2})
    export_mod.export_benchmark(seeds, out_dir=tmp_path)
    with open(tmp_path / "splits.json", "r", encoding="utf-8") as handle:
        splits = json.load(handle)
    assert set(splits["counts"]["near_duplicate_pairs_by_split"]) == set(counts)
