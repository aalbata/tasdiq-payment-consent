"""Tests for the simulated bank rail: fee quotes, receipts and fault injection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tasdiq.clock import SimClock
from tasdiq.rail import BankRail, FaultPlan, FeePolicy, RailError, RailTimeout, Receipt
from tasdiq.records import ExecutionRequest


@dataclass(frozen=True)
class PayeeStub:
    payee_id: str
    kind: str


def make_req(req_id: str = "r-1", amount: int = 25000, key: str | None = "k-1", fee: int = 0) -> ExecutionRequest:
    return ExecutionRequest(
        req_id=req_id,
        session_id="S-1",
        principal_id="U-1",
        payee_id="P-114",
        amount_minor=amount,
        currency="AED",
        fee_minor=fee,
        consent_ref="cr-1",
        idempotency_key=key,
        t=0,
    )


def test_quote_fee_by_kind() -> None:
    rail = BankRail(SimClock(0))
    assert rail.quote_fee(PayeeStub("P-1", "domestic"), 25000, "AED") == 0
    assert rail.quote_fee(PayeeStub("P-2", "merchant"), 25000, "AED") == 0
    assert rail.quote_fee("international", 25000, "AED") == 1000
    assert rail.quote_fee("international", 500000, "AED") == 5000
    assert rail.quote_fee("international", 123456, "AED") == 1235
    assert rail.quote_fee("international", 99999, "AED") == 1000
    assert rail.quote_fee("international", 100001, "AED") == 1001
    assert rail.quote_fee("international", 0, "AED") == 1000


def test_quote_fee_rejects_bad_inputs() -> None:
    rail = BankRail(SimClock(0))
    with pytest.raises(ValueError):
        rail.quote_fee("offshore", 100, "AED")
    with pytest.raises(ValueError):
        rail.quote_fee("domestic", -1, "AED")
    with pytest.raises(ValueError):
        rail.quote_fee("domestic", 100, "")


def test_custom_fee_policy() -> None:
    rail = BankRail(SimClock(0), FeePolicy(domestic_fee_minor=50, international_rate_bp=250, international_min_minor=10))
    assert rail.quote_fee("domestic", 1, "AED") == 50
    assert rail.quote_fee("international", 1000, "AED") == 25
    assert rail.quote_fee("international", 10, "AED") == 10


def test_execute_records_receipt_with_clock_time() -> None:
    clock = SimClock(40)
    rail = BankRail(clock)
    receipt = rail.execute(make_req(fee=250))
    assert isinstance(receipt, Receipt)
    assert receipt.receipt_id == "rcpt-000001"
    assert receipt.payee_id == "P-114" and receipt.amount_minor == 25000 and receipt.currency == "AED"
    assert receipt.fee_minor == 250 and receipt.t == 40 and receipt.idempotency_key == "k-1"
    assert rail.executions == [receipt]
    clock.advance(5)
    second = rail.execute(make_req(req_id="r-2", key=None))
    assert second.receipt_id == "rcpt-000002" and second.t == 45 and second.idempotency_key is None
    assert rail.count() == 2
    assert rail.debits_to("P-114") == [receipt, second]
    assert rail.debits_to("P-999") == []


def test_timeout_after_debit_records_debit_and_raises() -> None:
    rail = BankRail(SimClock(0), faults=FaultPlan({"r-1": "timeout_after_debit"}))
    with pytest.raises(RailTimeout):
        rail.execute(make_req())
    assert rail.count() == 1
    assert rail.executions[0].amount_minor == 25000
    receipt = rail.execute(make_req())
    assert rail.count() == 2 and receipt.receipt_id == "rcpt-000002"


def test_timeout_before_debit_and_error_record_nothing() -> None:
    rail = BankRail(SimClock(0), faults=FaultPlan({"r-1": "timeout_before_debit", "r-2": "error"}))
    with pytest.raises(RailTimeout):
        rail.execute(make_req("r-1"))
    with pytest.raises(RailError):
        rail.execute(make_req("r-2"))
    assert rail.executions == []
    assert rail.execute(make_req("r-1")).receipt_id == "rcpt-000001"


def test_fault_plan_repeat_and_add() -> None:
    plan = FaultPlan(repeat=True)
    plan.add("r-1", "error")
    rail = BankRail(SimClock(0), faults=plan)
    for _ in range(3):
        with pytest.raises(RailError):
            rail.execute(make_req("r-1"))
    assert rail.executions == []
    with pytest.raises(ValueError):
        plan.add("r-2", "explode")
    with pytest.raises(ValueError):
        FaultPlan({"r-3": "unknown"})


def test_execute_rejects_invalid_requests() -> None:
    rail = BankRail(SimClock(0))
    with pytest.raises(RailError):
        rail.execute(make_req(amount=0))
    with pytest.raises(RailError):
        rail.execute(make_req(amount=-5))
    with pytest.raises(RailError):
        rail.execute(make_req(fee=-1))
    bad_currency = make_req()
    bad_currency.currency = ""
    with pytest.raises(RailError):
        rail.execute(bad_currency)
    bad_payee = make_req()
    bad_payee.payee_id = ""
    with pytest.raises(RailError):
        rail.execute(bad_payee)
    assert rail.executions == []
