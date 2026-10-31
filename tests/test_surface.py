"""Tests for the simulated confirmation surface, the intent state and the two simulated readers."""

from __future__ import annotations

import pytest

from tasdiq import records
from tasdiq.clock import SimClock
from tasdiq.records import IntendedRecord, eastern_digits, render_confirmation, text_hash
from tasdiq.store import ConsentStore
from tasdiq.surface import (
    READERS,
    ConfirmationSurface,
    ExactIntentUser,
    InattentiveUser,
    IntentState,
    IntentTimeline,
    StaticIntent,
    SurfaceEvent,
    UserModel,
    first_mismatch,
    make_user_model,
    reader_name,
)

NAMES = {
    "P-114": {"en": "Omar Khalid", "ar": "عمر خالد"},
    "P-115": {"en": "Omar Saeed", "ar": "عمر سعيد"},
    "P-200": {"en": "Abu Dhabi Water", "ar": "مياه أبوظبي"},
}


class FakeDirectory:
    def render_name(self, payee_id: str, lang: str) -> str:
        return NAMES[payee_id]["ar" if lang == "ar" else "en"]


def fake_format_amount(amount_minor: int, currency: str, lang: str) -> str:
    text = f"{amount_minor // 100}.{amount_minor % 100:02d}"
    if lang == "ar":
        return eastern_digits(text).replace(".", "٫") + " درهم"
    return f"{text} {currency}"


@pytest.fixture(autouse=True)
def formatter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(records, "_format_amount", fake_format_amount)


def make_store_and_draft(clock: SimClock, **overrides):
    store = ConsentStore(clock)
    fields = dict(
        principal_id="U-1",
        session_id="S-1",
        payee_id="P-114",
        amount_minor=25000,
        currency="AED",
        total_max_minor=25250,
        issued_at=0,
        expires_at=600,
    )
    fields.update(overrides)
    return store, store.create_draft(fields)


INTENT = IntendedRecord(payee_id="P-114", amount_minor=25000, currency="AED", total_max_minor=25250)


def test_static_intent_and_timeline_satisfy_protocol() -> None:
    assert isinstance(StaticIntent(INTENT), IntentState)
    timeline = IntentTimeline([(50, None), (10, INTENT)])
    assert isinstance(timeline, IntentState)
    assert timeline.intended_at(0) is None
    assert timeline.intended_at(10) == INTENT
    assert timeline.intended_at(49) == INTENT
    assert timeline.intended_at(50) is None
    assert timeline.intended_at(1000) is None


def test_approves_matching_draft_and_logs_simulated_event() -> None:
    clock = SimClock(7)
    store, draft = make_store_and_draft(clock)
    surface = ConfirmationSurface(clock, FakeDirectory())
    event = surface.present(draft, "en", StaticIntent(INTENT))
    assert isinstance(event, SurfaceEvent)
    assert event.decision == "approved" and event.reason == "match"
    assert event.t == 7 and event.lang == "en" and event.cr_id == draft.cr_id and event.simulated is True
    assert event.reader == "exact" and surface.reader == "exact"
    assert event.shown_text == render_confirmation(draft, FakeDirectory(), "en")
    assert event.shown_text == "Pay 250.00 AED to Omar Khalid (P-114). Fees up to 2.50 AED. Valid for 10 minutes."
    assert event.event_id.startswith("sev-")
    assert surface.presentations() == 1 and surface.presentations("S-1") == 1 and surface.presentations("S-9") == 0
    line = surface.log[0]
    assert line["simulated"] is True and line["decision"] == "approved" and line["event_id"] == event.event_id
    assert line["surface_hash"] == text_hash(event.shown_text) and line["binding_hash"] == draft.binding_hash
    assert line["session_id"] == "S-1" and line["principal_id"] == "U-1" and line["revision"] == 1
    assert line["reader"] == "exact"


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"payee_id": "P-200"}, "field_mismatch:payee_id"),
        ({"amount_minor": 25001, "total_max_minor": 25250}, "field_mismatch:amount_minor"),
        ({"currency": "USD"}, "field_mismatch:currency"),
        ({"total_max_minor": 25251}, "cap_exceeded"),
    ],
)
def test_declines_with_first_mismatching_field(overrides: dict, reason: str) -> None:
    clock = SimClock(0)
    store, draft = make_store_and_draft(clock, **overrides)
    surface = ConfirmationSurface(clock, FakeDirectory())
    event = surface.present(draft, "en", StaticIntent(INTENT))
    assert event.decision == "declined" and event.reason == reason
    assert surface.log[0]["reason"] == reason and surface.log[0]["simulated"] is True


def test_first_mismatch_reports_payee_before_amount() -> None:
    clock = SimClock(0)
    store, draft = make_store_and_draft(clock, payee_id="P-200", amount_minor=1, total_max_minor=1)
    assert first_mismatch(draft, INTENT) == "field_mismatch:payee_id"
    assert first_mismatch(draft, None) == "no_intent"
    store, exact = make_store_and_draft(clock, total_max_minor=25250)
    assert first_mismatch(exact, INTENT) is None
    uncapped = IntendedRecord(payee_id="P-114", amount_minor=25000, currency="AED")
    store, high_cap = make_store_and_draft(clock, total_max_minor=99999)
    assert first_mismatch(high_cap, uncapped) is None


def test_declines_when_no_intent_or_after_cancellation() -> None:
    clock = SimClock(0)
    store, draft = make_store_and_draft(clock)
    surface = ConfirmationSurface(clock, FakeDirectory())
    timeline = IntentTimeline([(0, INTENT), (100, None)])
    assert surface.present(draft, "en", timeline).decision == "approved"
    clock.set(100)
    event = surface.present(draft, "en", timeline)
    assert event.decision == "declined" and event.reason == "no_intent"
    assert surface.presentations() == 2
    assert len({e.event_id for e in surface.events}) == 2


def test_arabic_presentation_has_different_text_and_hash_but_same_binding() -> None:
    clock = SimClock(0)
    store, draft = make_store_and_draft(clock)
    surface = ConfirmationSurface(clock, FakeDirectory())
    en = surface.present(draft, "en", StaticIntent(INTENT))
    ar = surface.present(draft, "ar", StaticIntent(INTENT))
    assert en.decision == "approved" and ar.decision == "approved"
    assert en.shown_text != ar.shown_text
    assert "عمر خالد" in ar.shown_text and "P-114" in ar.shown_text
    assert surface.log[0]["surface_hash"] != surface.log[1]["surface_hash"]
    assert surface.log[0]["binding_hash"] == surface.log[1]["binding_hash"]


def test_custom_user_model_is_used() -> None:
    class AlwaysDecline:
        def decide(self, draft, intended, t, lang):
            return "declined", "scripted"

    clock = SimClock(0)
    store, draft = make_store_and_draft(clock)
    surface = ConfirmationSurface(clock, FakeDirectory(), user_model=AlwaysDecline())
    event = surface.present(draft, "en", StaticIntent(INTENT))
    assert event.decision == "declined" and event.reason == "scripted"
    assert event.reader == "AlwaysDecline" and surface.log[0]["reader"] == "AlwaysDecline"
    assert ExactIntentUser().decide(draft, INTENT, 0, "en") == ("approved", "match")


def test_event_ids_are_deterministic_for_equal_histories() -> None:
    def run() -> list[str]:
        clock = SimClock(3)
        store, draft = make_store_and_draft(clock)
        surface = ConfirmationSurface(clock, FakeDirectory())
        surface.present(draft, "en", StaticIntent(INTENT))
        surface.present(draft, "ar", StaticIntent(INTENT))
        return [e.event_id for e in surface.events]

    assert run() == run()
    assert len(set(run())) == 2


def test_reader_is_selected_by_name() -> None:
    clock = SimClock(0)
    assert READERS == ("exact", "inattentive")
    assert ConfirmationSurface(clock, FakeDirectory()).reader == "exact"
    assert ConfirmationSurface(clock, FakeDirectory(), "exact").reader == "exact"
    assert ConfirmationSurface(clock, FakeDirectory(), "inattentive").reader == "inattentive"
    with pytest.raises(ValueError):
        ConfirmationSurface(clock, FakeDirectory(), "careless")
    assert isinstance(make_user_model(None, None), ExactIntentUser)
    assert isinstance(make_user_model("inattentive", None), InattentiveUser)
    assert isinstance(make_user_model("exact", None), UserModel)
    custom = ExactIntentUser()
    assert make_user_model(custom, None) is custom
    assert reader_name(ExactIntentUser()) == "exact" and reader_name(InattentiveUser(None)) == "inattentive"
    assert reader_name(object()) == "object"


def test_inattentive_reader_approves_on_shared_first_token_of_display_name() -> None:
    clock = SimClock(0)
    store, draft = make_store_and_draft(clock, payee_id="P-115")
    exact = ConfirmationSurface(clock, FakeDirectory(), "exact")
    inattentive = ConfirmationSurface(clock, FakeDirectory(), "inattentive")
    assert exact.present(draft, "en", StaticIntent(INTENT)).reason == "field_mismatch:payee_id"
    for lang in ("en", "ar", "mix"):
        event = inattentive.present(draft, lang, StaticIntent(INTENT))
        assert event.decision == "approved" and event.reason == "match" and event.reader == "inattentive"
    assert all(line["reader"] == "inattentive" for line in inattentive.log)
    assert all(line["reader"] == "exact" for line in exact.log)
    store, other = make_store_and_draft(clock, payee_id="P-200")
    event = inattentive.present(other, "en", StaticIntent(INTENT))
    assert event.decision == "declined" and event.reason == "field_mismatch:payee_id"


@pytest.mark.parametrize(
    "amount, decision",
    [
        (25000, "approved"),
        (26250, "approved"),
        (26251, "declined"),
        (23750, "approved"),
        (23749, "declined"),
    ],
)
def test_inattentive_reader_tolerates_five_percent_of_the_intended_amount(amount: int, decision: str) -> None:
    clock = SimClock(0)
    store, draft = make_store_and_draft(clock, amount_minor=amount, total_max_minor=amount)
    surface = ConfirmationSurface(clock, FakeDirectory(), "inattentive")
    event = surface.present(draft, "en", StaticIntent(INTENT))
    assert event.decision == decision
    assert event.reason == ("match" if decision == "approved" else "field_mismatch:amount_minor")
    exact = ExactIntentUser().decide(draft, INTENT, 0, "en")
    assert exact == (("approved", "match") if amount == 25000 else ("declined", "field_mismatch:amount_minor"))


def test_inattentive_reader_currency_cap_and_missing_intent() -> None:
    clock = SimClock(0)
    reader = InattentiveUser(FakeDirectory())
    store, usd = make_store_and_draft(clock, currency="USD")
    assert reader.decide(usd, INTENT, 0, "en") == ("declined", "field_mismatch:currency")
    store, high_cap = make_store_and_draft(clock, total_max_minor=99999)
    assert reader.decide(high_cap, INTENT, 0, "en") == ("approved", "match")
    assert ExactIntentUser().decide(high_cap, INTENT, 0, "en") == ("declined", "cap_exceeded")
    store, draft = make_store_and_draft(clock)
    assert reader.decide(draft, None, 0, "en") == ("declined", "no_intent")
    store, wrong_name_and_amount = make_store_and_draft(clock, payee_id="P-200", amount_minor=1, total_max_minor=1)
    assert reader.decide(wrong_name_and_amount, INTENT, 0, "en") == ("declined", "field_mismatch:payee_id")


def test_inattentive_reader_without_directory_and_tolerance_argument() -> None:
    clock = SimClock(0)
    store, same_id = make_store_and_draft(clock)
    store, other_id = make_store_and_draft(clock, payee_id="P-115")
    reader = InattentiveUser(None)
    assert reader.decide(same_id, INTENT, 0, "en") == ("approved", "match")
    assert reader.decide(other_id, INTENT, 0, "en") == ("declined", "field_mismatch:payee_id")
    strict = InattentiveUser(FakeDirectory(), tolerance_percent=0)
    store, close = make_store_and_draft(clock, amount_minor=25001, total_max_minor=25250)
    assert strict.decide(close, INTENT, 0, "en") == ("declined", "field_mismatch:amount_minor")
    assert strict.decide(same_id, INTENT, 0, "en") == ("approved", "match")
    with pytest.raises(ValueError):
        InattentiveUser(FakeDirectory(), tolerance_percent=-1)
    with pytest.raises(ValueError):
        InattentiveUser(FakeDirectory(), tolerance_percent=True)


def test_readers_are_distinguished_in_the_log_of_the_same_draft() -> None:
    clock = SimClock(0)
    store, draft = make_store_and_draft(clock, payee_id="P-115")
    decisions = {}
    for name in READERS:
        surface = ConfirmationSurface(clock, FakeDirectory(), name)
        event = surface.present(draft, "ar", StaticIntent(INTENT))
        decisions[name] = (event.decision, surface.log[0]["reader"], surface.log[0]["simulated"])
    assert decisions == {"exact": ("declined", "exact", True), "inattentive": ("approved", "inattentive", True)}

def _execution(payee_id: str = "P-114", amount_minor: int = 25000, currency: str = "AED") -> dict:
    return {"payee_id": payee_id, "amount_minor": amount_minor, "currency": currency}


def test_declines_presentation_after_the_intended_execution() -> None:
    clock = SimClock(0)
    store, draft = make_store_and_draft(clock)
    executions: list[dict] = []
    surface = ConfirmationSurface(clock, FakeDirectory(), executions=lambda: executions)
    assert surface.present(draft, "en", StaticIntent(INTENT)).decision == "approved"
    executions.append(_execution())
    clock.set(30)
    event = surface.present(draft, "en", StaticIntent(INTENT))
    assert event.decision == "declined" and event.reason == "already_executed" and event.reader == "exact"
    assert surface.log[-1]["decision"] == "declined" and surface.log[-1]["reason"] == "already_executed"
    assert surface.presentations() == 2
    # an execution of another tuple does not count against the intended record
    executions[:] = [_execution(amount_minor=25001)]
    assert surface.present(draft, "en", StaticIntent(INTENT)).decision == "approved"
    # a record the reader does not accept keeps its field reason after an execution of the intended record
    executions[:] = [_execution()]
    store, wrong = make_store_and_draft(clock, payee_id="P-200")
    assert surface.present(wrong, "en", StaticIntent(INTENT)).reason == "field_mismatch:payee_id"
    # without read access to the executions the rule is inactive
    plain = ConfirmationSurface(clock, FakeDirectory())
    assert plain.present(draft, "en", StaticIntent(INTENT)).decision == "approved"
    assert surface.executed_count(INTENT, "en") == 1 and plain.executed_count(INTENT, "en") == 0


def test_additional_intent_of_the_same_record_approves_twice() -> None:
    key = ("P-114", 25000, "AED")
    other = IntendedRecord(payee_id="P-200", amount_minor=25000, currency="AED")
    timeline = IntentTimeline([(0, INTENT), (300, INTENT, "additional")])
    assert timeline.changes == [(0, INTENT, None), (300, INTENT, "additional")]
    assert timeline.intended_count(key, 0) == 1 and timeline.intended_count(key, 299) == 1 and timeline.intended_count(key, 300) == 2
    assert timeline.intended_count(("P-200", 25000, "AED"), 300) == 0
    assert IntentTimeline([(0, INTENT), (300, other, "revise")]).intended_count(key, 300) == 0
    assert IntentTimeline([(0, INTENT), (300, None)]).intended_count(key, 300) == 0
    assert IntentTimeline([(0, INTENT), (300, other, "additional")]).intended_count(key, 300) == 1
    assert StaticIntent(INTENT).intended_count(key, 5) == 1 and StaticIntent(None).intended_count(key, 5) == 0
    with pytest.raises(ValueError):
        IntentTimeline([(0,)])
    clock = SimClock(10)
    store, draft = make_store_and_draft(clock)
    executions: list[dict] = []
    surface = ConfirmationSurface(clock, FakeDirectory(), executions=lambda: executions)
    assert surface.present(draft, "en", timeline).decision == "approved"
    executions.append(_execution())
    clock.set(20)
    assert surface.present(draft, "en", timeline).reason == "already_executed"
    clock.set(312)
    assert surface.present(draft, "en", timeline).decision == "approved"
    executions.append(_execution())
    clock.set(335)
    assert surface.present(draft, "en", timeline).reason == "already_executed"
    # the expected final records bound the count from below
    bounded = IntentTimeline([(0, INTENT)], required={key: 2})
    assert bounded.intended_count(key, 0) == 2
    executions[:] = [_execution()]
    assert surface.present(draft, "en", bounded).decision == "approved"
    executions.append(_execution())
    assert surface.present(draft, "en", bounded).reason == "already_executed"


def test_inattentive_reader_follows_the_already_executed_rule() -> None:
    clock = SimClock(0)
    executions: list[dict] = []
    surface = ConfirmationSurface(clock, FakeDirectory(), "inattentive", executions=lambda: executions)
    store, close = make_store_and_draft(clock, payee_id="P-115", amount_minor=25500, total_max_minor=25500)
    assert surface.present(close, "en", StaticIntent(INTENT)).decision == "approved"
    # the execution of the approved draft counts as the intended payment under the reader's own tests
    executions.append(_execution("P-115", 25500))
    clock.set(30)
    event = surface.present(close, "en", StaticIntent(INTENT))
    assert event.decision == "declined" and event.reason == "already_executed" and event.reader == "inattentive"
    # an execution the reader would not accept as the intended payment does not count
    executions[:] = [_execution(amount_minor=30000)]
    assert surface.present(close, "en", StaticIntent(INTENT)).decision == "approved"
    executions[:] = [_execution()]
    assert surface.present(close, "en", StaticIntent(INTENT)).reason == "already_executed"
    reader = InattentiveUser(FakeDirectory())
    assert reader.same_payment(_execution("P-115", 26000), INTENT, "en") is True
    assert reader.same_payment(_execution("P-200"), INTENT, "en") is False
    assert reader.same_payment(_execution(currency="USD"), INTENT, "en") is False
    assert ExactIntentUser().same_payment(_execution(), INTENT, "en") is True
    assert ExactIntentUser().same_payment(_execution("P-115"), INTENT, "en") is False
