"""Simulated bank rail.

The rail quotes fees, records every debit it performs and injects faults from a plan keyed by request id. The list of
receipts in `executions` is the ground truth of executed payments for labelling.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from tasdiq.clock import SimClock
from tasdiq.records import ExecutionRequest

FAULT_KINDS: tuple[str, ...] = ("timeout_after_debit", "timeout_before_debit", "error")
PAYEE_KINDS: tuple[str, ...] = ("domestic", "international", "merchant")


class RailTimeout(Exception):
    """Raised when the rail does not acknowledge a request; the caller cannot tell whether the debit took place."""


class RailError(Exception):
    """Raised when the rail rejects a request without debiting."""


@dataclass(frozen=True)
class Receipt:
    """Acknowledgement of one debit."""

    receipt_id: str
    payee_id: str
    amount_minor: int
    currency: str
    fee_minor: int
    t: int
    idempotency_key: str | None


@dataclass(frozen=True)
class FeePolicy:
    """Fee rule of the rail: a rate in basis points with a minimum for international payees, flat fees otherwise."""

    domestic_fee_minor: int = 0
    merchant_fee_minor: int = 0
    international_rate_bp: int = 100
    international_min_minor: int = 1000


@dataclass
class FaultPlan:
    """Faults to inject, keyed by request id. Each fault fires once unless repeat is True."""

    faults: dict[str, str] = field(default_factory=dict)
    repeat: bool = False
    fired: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        for req_id, kind in self.faults.items():
            if kind not in FAULT_KINDS:
                raise ValueError(f"unknown fault kind for {req_id}: {kind}")

    def add(self, req_id: str, kind: str) -> None:
        """Plan a fault for a request id."""
        if kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind: {kind}")
        self.faults[req_id] = kind
        self.fired.discard(req_id)

    def take(self, req_id: str) -> str | None:
        """Return the fault planned for a request id, or None; mark it fired."""
        kind = self.faults.get(req_id)
        if kind is None:
            return None
        if req_id in self.fired and not self.repeat:
            return None
        self.fired.add(req_id)
        return kind


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class BankRail:
    """Simulated execution service that debits exactly what it is asked and keeps every receipt."""

    def __init__(self, clock: SimClock, fee_policy: FeePolicy | None = None, faults: FaultPlan | None = None) -> None:
        self._clock = clock
        self.fee_policy = fee_policy if fee_policy is not None else FeePolicy()
        self.faults = faults if faults is not None else FaultPlan()
        self.executions: list[Receipt] = []
        self._lock = threading.Lock()
        self._counter = 0

    def quote_fee(self, payee: Any, amount_minor: int, currency: str) -> int:
        """Return the fee in minor units for a payee (an object with a kind attribute, or the kind itself)."""
        kind = payee if isinstance(payee, str) else getattr(payee, "kind", None)
        if kind not in PAYEE_KINDS:
            raise ValueError(f"unknown payee kind: {kind}")
        if not _is_int(amount_minor) or amount_minor < 0:
            raise ValueError("amount_minor must be a non-negative integer")
        if not isinstance(currency, str) or not currency:
            raise ValueError("currency must be a non-empty string")
        policy = self.fee_policy
        if kind == "domestic":
            return policy.domestic_fee_minor
        if kind == "merchant":
            return policy.merchant_fee_minor
        proportional = -(-amount_minor * policy.international_rate_bp // 10000)
        return max(proportional, policy.international_min_minor)

    def execute(self, req: ExecutionRequest) -> Receipt:
        """Debit the request and return the receipt, applying any fault planned for req.req_id.

        timeout_after_debit records the debit and raises RailTimeout; timeout_before_debit raises RailTimeout without a
        debit; error raises RailError without a debit. A request with a non-positive amount, an empty currency or an
        empty payee identifier raises RailError without a debit.
        """
        if not isinstance(req.payee_id, str) or not req.payee_id:
            raise RailError("invalid_request:payee_id")
        if not _is_int(req.amount_minor) or req.amount_minor <= 0:
            raise RailError("invalid_request:amount_minor")
        if not isinstance(req.currency, str) or not req.currency:
            raise RailError("invalid_request:currency")
        if not _is_int(req.fee_minor) or req.fee_minor < 0:
            raise RailError("invalid_request:fee_minor")
        with self._lock:
            fault = self.faults.take(req.req_id)
            if fault == "timeout_before_debit":
                raise RailTimeout(req.req_id)
            if fault == "error":
                raise RailError(f"rail_error:{req.req_id}")
            self._counter += 1
            receipt = Receipt(
                receipt_id=f"rcpt-{self._counter:06d}",
                payee_id=req.payee_id,
                amount_minor=req.amount_minor,
                currency=req.currency,
                fee_minor=req.fee_minor,
                t=self._clock.now(),
                idempotency_key=req.idempotency_key,
            )
            self.executions.append(receipt)
            if fault == "timeout_after_debit":
                raise RailTimeout(req.req_id)
            return receipt

    def count(self) -> int:
        """Return the number of debits performed."""
        with self._lock:
            return len(self.executions)

    def debits_to(self, payee_id: str) -> list[Receipt]:
        """Return the receipts of every debit to a payee."""
        with self._lock:
            return [r for r in self.executions if r.payee_id == payee_id]
