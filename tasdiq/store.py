"""Consent store and idempotency ledger.

The store holds consent records and applies the state machine draft -> pending -> active -> consumed, with revoked,
expired and superseded as the other terminal states. Every method runs under one lock, so the check-and-consume in
try_consume is atomic with respect to every other transition. A revocation that arrives while a use is in flight
(status consumed) is deferred: the record stays consumed, and a later release moves it to revoked instead of active.
The idempotency ledger keeps one entry per idempotency key with the payload fingerprint, the identity of the request
that opened the entry and the receipt of a completed execution.
"""

from __future__ import annotations

import copy
import hashlib
import threading
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from tasdiq.clock import SimClock
from tasdiq.records import ACTION_TYPES, ConsentRecord, TERMINAL_STATUSES, binding_hash, fingerprint

_STATE_KEYS: frozenset[str] = frozenset({"status", "uses", "binding_hash", "surface_hash", "confirmation_event_id"})
_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "cr_id",
        "principal_id",
        "session_id",
        "action_type",
        "payee_id",
        "amount_minor",
        "currency",
        "total_max_minor",
        "revision",
        "allowed_uses",
        "supersedes",
        "issued_at",
        "expires_at",
        "lang_shown",
    }
)


class InvalidTransition(Exception):
    """Raised when a transition is requested from a status that does not permit it."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_str(fields: Mapping[str, Any], name: str) -> str:
    value = fields.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_int(fields: Mapping[str, Any], name: str) -> int | None:
    value = fields.get(name)
    if value is None:
        return None
    if not _is_int(value):
        raise TypeError(f"{name} must be an integer or None")
    return value


class ConsentStore:
    """Thread-safe store of consent records with an appended history of every transition."""

    def __init__(self, clock: SimClock) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._records: dict[str, ConsentRecord] = {}
        self._history: list[dict[str, Any]] = []
        self._revoke_requested: set[str] = set()
        self._counter = 0

    def _append(self, kind: str, cr_id: str, src: str | None, dst: str | None, reason: str, t: int, **detail: Any) -> None:
        entry: dict[str, Any] = {"kind": kind, "t": t, "cr_id": cr_id, "from": src, "to": dst, "reason": reason}
        if detail:
            entry["detail"] = detail
        self._history.append(entry)

    def _next_id(self, session_id: str, principal_id: str) -> str:
        while True:
            self._counter += 1
            digest = hashlib.sha256(f"{session_id}|{principal_id}|{self._counter}".encode("utf-8")).hexdigest()
            cr_id = f"cr-{digest[:12]}"
            if cr_id not in self._records:
                return cr_id

    def create_draft(self, record_fields: Mapping[str, Any]) -> ConsentRecord:
        """Create a draft record from the given fields and return a copy of it.

        Required keys: principal_id, session_id, payee_id, amount_minor (positive integer), currency. Optional keys:
        action_type (default "payment"), total_max_minor (default amount_minor; must not be below amount_minor),
        revision (default 1), allowed_uses (default 1), supersedes, issued_at, expires_at, lang_shown and cr_id (assigned
        by the store when absent). State keys (status, uses, binding_hash, surface_hash, confirmation_event_id) are
        ignored so that the snapshot of an earlier record can seed a revision; any other key raises ValueError.
        """
        unknown = set(record_fields) - _INPUT_KEYS - _STATE_KEYS
        if unknown:
            raise ValueError(f"unknown record fields: {sorted(unknown)}")
        principal_id = _require_str(record_fields, "principal_id")
        session_id = _require_str(record_fields, "session_id")
        payee_id = _require_str(record_fields, "payee_id")
        currency = _require_str(record_fields, "currency")
        action_type = record_fields.get("action_type", "payment")
        if action_type not in ACTION_TYPES:
            raise ValueError(f"action_type must be one of {ACTION_TYPES}")
        amount_minor = record_fields.get("amount_minor")
        if not _is_int(amount_minor) or amount_minor <= 0:
            raise ValueError("amount_minor must be a positive integer")
        total_max_minor = _optional_int(record_fields, "total_max_minor")
        if total_max_minor is None:
            total_max_minor = amount_minor
        if total_max_minor < amount_minor:
            raise ValueError("total_max_minor must not be below amount_minor")
        revision = record_fields.get("revision", 1)
        if not _is_int(revision) or revision < 1:
            raise ValueError("revision must be an integer not below 1")
        allowed_uses = record_fields.get("allowed_uses", 1)
        if not _is_int(allowed_uses) or allowed_uses < 1:
            raise ValueError("allowed_uses must be an integer not below 1")
        supersedes = record_fields.get("supersedes")
        if supersedes is not None and not isinstance(supersedes, str):
            raise TypeError("supersedes must be a string or None")
        issued_at = _optional_int(record_fields, "issued_at")
        expires_at = _optional_int(record_fields, "expires_at")
        if issued_at is not None and expires_at is not None and expires_at <= issued_at:
            raise ValueError("expires_at must be after issued_at")
        lang_shown = record_fields.get("lang_shown", "")
        if not isinstance(lang_shown, str):
            raise TypeError("lang_shown must be a string")
        given_id = record_fields.get("cr_id")
        if given_id is not None and (not isinstance(given_id, str) or not given_id):
            raise ValueError("cr_id must be a non-empty string when given")
        with self._lock:
            if given_id is not None:
                if given_id in self._records:
                    raise ValueError(f"cr_id already exists: {given_id}")
                cr_id = given_id
            else:
                cr_id = self._next_id(session_id, principal_id)
            record = ConsentRecord(
                cr_id=cr_id,
                principal_id=principal_id,
                session_id=session_id,
                action_type=action_type,
                payee_id=payee_id,
                amount_minor=amount_minor,
                currency=currency,
                total_max_minor=total_max_minor,
                revision=revision,
                issued_at=issued_at,
                expires_at=expires_at,
                allowed_uses=allowed_uses,
                uses=0,
                status="draft",
                supersedes=supersedes,
                lang_shown=lang_shown,
            )
            record.binding_hash = binding_hash(record)
            self._records[cr_id] = record
            self._append("transition", cr_id, None, "draft", "create_draft", self._clock.now(), revision=revision)
            return replace(record)

    def _get_locked(self, cr_id: str) -> ConsentRecord:
        try:
            return self._records[cr_id]
        except KeyError:
            raise KeyError(f"unknown consent record: {cr_id}") from None

    def present(self, cr_id: str) -> ConsentRecord:
        """Move a draft to pending (a pending record stays pending) and return a copy."""
        with self._lock:
            record = self._get_locked(cr_id)
            if record.status not in ("draft", "pending"):
                raise InvalidTransition(f"cannot present a record in status {record.status}")
            src = record.status
            record.status = "pending"
            self._append("transition", cr_id, src, "pending", "present", self._clock.now())
            return replace(record)

    def confirm(
        self,
        cr_id: str,
        event_id: str,
        lang_shown: str,
        window_s: int,
        surface_hash: str | None = None,
    ) -> ConsentRecord:
        """Activate a draft or pending record for window_s seconds from now and return a copy.

        Sets issued_at to the current time, expires_at to issued_at plus window_s, the confirmation event, the language
        shown and, when given, the surface hash of the text that was shown.
        """
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be a non-empty string")
        if not _is_int(window_s) or window_s <= 0:
            raise ValueError("window_s must be a positive integer")
        with self._lock:
            record = self._get_locked(cr_id)
            if record.status not in ("draft", "pending"):
                raise InvalidTransition(f"cannot confirm a record in status {record.status}")
            src = record.status
            now = self._clock.now()
            record.status = "active"
            record.issued_at = now
            record.expires_at = now + window_s
            record.confirmation_event_id = event_id
            record.lang_shown = lang_shown
            record.surface_hash = surface_hash
            self._append(
                "transition", cr_id, src, "active", "confirm", now, event_id=event_id, window_s=window_s, lang=lang_shown
            )
            return replace(record)

    def revoke(self, cr_id: str) -> bool:
        """Move an active or pending record to revoked; return False when the status does not permit it.

        A revocation of a consumed record (a use is in flight) is deferred: the record keeps its status, the
        request is recorded in the history as deferred, and a later release moves the record to revoked.
        """
        with self._lock:
            record = self._get_locked(cr_id)
            if record.status == "consumed":
                self._revoke_requested.add(cr_id)
                self._append("deferred", cr_id, record.status, None, "revoke", self._clock.now())
                return False
            if record.status not in ("active", "pending"):
                self._append("ignored", cr_id, record.status, None, "revoke", self._clock.now())
                return False
            src = record.status
            record.status = "revoked"
            self._append("transition", cr_id, src, "revoked", "revoke", self._clock.now())
            return True

    def supersede(self, old_id: str, new_id: str) -> bool:
        """Mark the old record superseded by the new one when the old record is active or pending.

        The new record's supersedes field is set to old_id when it is unset and the transition takes place. Returns
        False, with no record changed, when the old record's status does not permit the transition.
        """
        if old_id == new_id:
            raise ValueError("a record cannot supersede itself")
        with self._lock:
            old = self._get_locked(old_id)
            new = self._get_locked(new_id)
            if old.status not in ("active", "pending"):
                self._append("ignored", old_id, old.status, None, "supersede", self._clock.now(), by=new_id)
                return False
            if new.supersedes is None:
                new.supersedes = old_id
            src = old.status
            old.status = "superseded"
            self._append("transition", old_id, src, "superseded", "supersede", self._clock.now(), by=new_id)
            return True

    def sweep(self, t: int) -> list[str]:
        """Expire every active record whose expires_at is at or before t; return the expired ids."""
        expired: list[str] = []
        with self._lock:
            for cr_id, record in self._records.items():
                if record.status == "active" and record.expires_at is not None and record.expires_at <= t:
                    record.status = "expired"
                    self._append("transition", cr_id, "active", "expired", "sweep", t)
                    expired.append(cr_id)
        return expired

    def try_consume(self, cr_id: str, req_fingerprint: str, t: int) -> tuple[bool, str]:
        """Atomically check a record and consume one use of it.

        Under the lock the record must be active, t must be before expires_at, uses must be below allowed_uses and the
        request fingerprint must equal the record's fingerprint. On success uses is incremented and the status becomes
        consumed when uses reaches allowed_uses. An active record found at or past its expiry is moved to expired.
        Returns (True, "ok") or (False, reason) with reason among no_consent, expired, revoked, superseded, consumed and
        field_mismatch:fingerprint.
        """
        with self._lock:
            record = self._records.get(cr_id)
            if record is None or record.status in ("draft", "pending"):
                self._append("denied", cr_id, None if record is None else record.status, None, "no_consent", t)
                return False, "no_consent"
            if record.status in TERMINAL_STATUSES:
                self._append("denied", cr_id, record.status, None, record.status, t)
                return False, record.status
            if record.expires_at is None or t >= record.expires_at:
                record.status = "expired"
                self._append("transition", cr_id, "active", "expired", "expired", t)
                return False, "expired"
            if record.uses >= record.allowed_uses:
                self._append("denied", cr_id, record.status, None, "consumed", t)
                return False, "consumed"
            if req_fingerprint != fingerprint(record):
                self._append("denied", cr_id, record.status, None, "field_mismatch:fingerprint", t)
                return False, "field_mismatch:fingerprint"
            record.uses += 1
            if record.uses == record.allowed_uses:
                record.status = "consumed"
                self._append("transition", cr_id, "active", "consumed", "consume", t, uses=record.uses)
            else:
                self._append("transition", cr_id, "active", "active", "consume", t, uses=record.uses)
            return True, "ok"

    def release(self, cr_id: str) -> bool:
        """Return one use to a record after the rail reported that no debit took place.

        A consumed record becomes active again, unless a revocation was deferred while the use was in flight (the
        record becomes revoked) or its validity has ended (the record becomes expired). Returns False when the record
        has no use to return.
        """
        with self._lock:
            record = self._get_locked(cr_id)
            if record.status not in ("active", "consumed") or record.uses <= 0:
                self._append("ignored", cr_id, record.status, None, "release", self._clock.now())
                return False
            src = record.status
            now = self._clock.now()
            record.uses -= 1
            detail: dict[str, Any] = {"uses": record.uses}
            if cr_id in self._revoke_requested:
                self._revoke_requested.discard(cr_id)
                record.status = "revoked"
                detail["deferred_revocation"] = True
            elif record.expires_at is not None and record.expires_at <= now:
                record.status = "expired"
            else:
                record.status = "active"
            self._append("transition", cr_id, src, record.status, "release", now, **detail)
            return True

    def revocation_pending(self, cr_id: str) -> bool:
        """Return True when a revocation of the record was deferred because a use was in flight."""
        with self._lock:
            return cr_id in self._revoke_requested

    def get(self, cr_id: str) -> ConsentRecord:
        """Return a copy of the record; raise KeyError when it does not exist."""
        with self._lock:
            return replace(self._get_locked(cr_id))

    def contains(self, cr_id: str) -> bool:
        """Return True when a record with this id exists."""
        with self._lock:
            return cr_id in self._records

    def active_for(self, session_id: str) -> list[ConsentRecord]:
        """Return copies of the active records whose session_id matches, in creation order."""
        with self._lock:
            return [replace(r) for r in self._records.values() if r.status == "active" and r.session_id == session_id]

    def all_records(self) -> list[ConsentRecord]:
        """Return copies of every record in creation order."""
        with self._lock:
            return [replace(r) for r in self._records.values()]

    def history(self) -> list[dict[str, Any]]:
        """Return a deep copy of the transition history, one entry per transition, denial, deferral or ignored request."""
        with self._lock:
            return copy.deepcopy(self._history)


class IdempotencyLedger:
    """Ledger of idempotency keys with the payload fingerprint and the receipt of the completed execution."""

    def __init__(self, clock: SimClock, retention_s: int = 86400) -> None:
        if not _is_int(retention_s) or retention_s <= 0:
            raise ValueError("retention_s must be a positive integer")
        self._clock = clock
        self._retention_s = retention_s
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}

    def begin(self, key: str, fingerprint_value: str, owner: str | None = None) -> tuple[str, Any]:
        """Register the start of an execution under a key.

        Returns ("new", None) when the key is unseen and is now in flight, ("mismatch", None) when the key is known
        with a different fingerprint, ("in_flight", None) when an execution under the key has not completed, and
        ("replay", receipt) when an execution under the key completed. owner identifies the request that opened the
        entry (its req_id), so that a request re-issued after a crash can recognise its own in-flight entry.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        if not isinstance(fingerprint_value, str) or not fingerprint_value:
            raise ValueError("fingerprint must be a non-empty string")
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._entries[key] = {
                    "fingerprint": fingerprint_value,
                    "state": "in_flight",
                    "receipt": None,
                    "t": self._clock.now(),
                    "owner": owner,
                }
                return "new", None
            if entry["fingerprint"] != fingerprint_value:
                return "mismatch", None
            if entry["state"] == "in_flight":
                return "in_flight", None
            return "replay", entry["receipt"]

    def complete(self, key: str, receipt: Any) -> None:
        """Store the receipt for a key that is in flight."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry["state"] != "in_flight":
                raise KeyError(f"key is not in flight: {key}")
            entry["state"] = "complete"
            entry["receipt"] = receipt
            entry["t"] = self._clock.now()

    def fail(self, key: str) -> None:
        """Remove an in-flight entry so that the key can be used again; a completed entry is kept."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry["state"] == "in_flight":
                del self._entries[key]

    def purge(self, t: int) -> list[str]:
        """Remove completed entries older than the retention period at time t; return the removed keys."""
        removed: list[str] = []
        with self._lock:
            for key, entry in list(self._entries.items()):
                if entry["state"] == "complete" and entry["t"] + self._retention_s <= t:
                    del self._entries[key]
                    removed.append(key)
        return removed

    def state_of(self, key: str) -> str | None:
        """Return "in_flight", "complete" or None when the key is unknown."""
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry["state"]

    def owner_of(self, key: str) -> str | None:
        """Return the identity of the request that opened the entry under key, or None."""
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry.get("owner")

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
