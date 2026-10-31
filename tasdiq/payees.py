"""Payee directory with exact alias matching.

A payee is identified by a stable identifier (``payee_id``) that is part of every consent
binding. Display names and aliases, including Arabizi spellings such as "3omar", exist only to
resolve a user's mention to identifiers. Resolution is an exact comparison after
``tasdiq.textnorm.normalize_for_matching`` on both sides; there is no fuzzy matching, so two
payees that share a display name are both returned and the caller must ask the user.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from tasdiq.textnorm import normalize_for_matching

KINDS: tuple[str, ...] = ("domestic", "international", "merchant")


@dataclass(frozen=True)
class Payee:
    """One directory entry.

    ``payee_id`` is the stable identifier used in consent records and compared verbatim.
    ``name_en`` and ``name_ar`` are the display names in each language; ``aliases`` are
    further spellings, transliterations and Arabizi forms that resolve to this payee.
    ``kind`` selects the fee rule of the rail and ``iban_masked`` is the masked account
    reference shown to the user.
    """

    payee_id: str
    name_en: str
    name_ar: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    kind: str = "domestic"
    iban_masked: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.payee_id, str) or not self.payee_id.strip():
            raise ValueError("payee_id must be a non-empty string")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not isinstance(self.aliases, tuple):
            object.__setattr__(self, "aliases", tuple(str(a) for a in self.aliases))

    def labels(self) -> tuple[str, ...]:
        """Every display name and alias of this payee, raw, in a stable order."""
        out: list[str] = []
        for label in (self.name_en, self.name_ar, *self.aliases):
            if label and label not in out:
                out.append(label)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping of the entry, for JSON and YAML serialisation."""
        return {
            "payee_id": self.payee_id,
            "name_en": self.name_en,
            "name_ar": self.name_ar,
            "aliases": list(self.aliases),
            "kind": self.kind,
            "iban_masked": self.iban_masked,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Payee":
        """Build an entry from a mapping with the field names of the dataclass."""
        return cls(
            payee_id=str(row["payee_id"]),
            name_en=str(row.get("name_en", "") or ""),
            name_ar=str(row.get("name_ar", "") or ""),
            aliases=tuple(str(a) for a in (row.get("aliases") or [])),
            kind=str(row.get("kind", "domestic") or "domestic"),
            iban_masked=str(row.get("iban_masked", "") or ""),
        )


class PayeeDirectory:
    """Directory of payees with exact, normalised alias lookup.

    ``resolve`` returns every payee whose English name, Arabic name or alias equals the
    mention after ``normalize_for_matching``; the list is empty when nothing matches and has
    several entries when the mention is shared. ``by_id`` compares identifiers verbatim.
    ``render_name`` returns the display name for the confirmation surface.
    """

    def __init__(self, payees: Iterable[Payee]) -> None:
        self.payees: list[Payee] = []
        self._by_id: dict[str, Payee] = {}
        self._index: dict[str, list[str]] = {}
        for payee in payees:
            if not isinstance(payee, Payee):
                raise TypeError(f"expected Payee, got {type(payee).__name__}")
            if payee.payee_id in self._by_id:
                raise ValueError(f"duplicate payee_id {payee.payee_id!r}")
            self._by_id[payee.payee_id] = payee
            self.payees.append(payee)
            for label in payee.labels():
                key = normalize_for_matching(label)
                if not key:
                    continue
                ids = self._index.setdefault(key, [])
                if payee.payee_id not in ids:
                    ids.append(payee.payee_id)

    @classmethod
    def from_dicts(cls, rows: Iterable[Mapping[str, Any]]) -> "PayeeDirectory":
        """Build a directory from mappings such as those loaded from a YAML seed."""
        return cls(Payee.from_dict(row) for row in rows)

    def __iter__(self) -> Iterator[Payee]:
        return iter(self.payees)

    def __len__(self) -> int:
        return len(self.payees)

    def __contains__(self, payee_id: object) -> bool:
        return payee_id in self._by_id

    def all(self) -> list[Payee]:
        """Every payee in insertion order."""
        return list(self.payees)

    def ids(self) -> list[str]:
        """Every payee identifier in insertion order."""
        return [p.payee_id for p in self.payees]

    def resolve(self, mention: str) -> list[Payee]:
        """Return all payees whose name or alias equals the mention after normalisation.

        The comparison is exact after ``normalize_for_matching`` on both the mention and the
        stored labels; no fuzzy, prefix or substring matching is performed. An empty or
        whitespace-only mention resolves to nothing. Payees are returned in directory order.
        """
        if mention is None:
            return []
        key = normalize_for_matching(str(mention))
        if not key:
            return []
        return [self._by_id[pid] for pid in self._index.get(key, [])]

    def by_id(self, payee_id: str) -> Payee | None:
        """Return the payee with exactly this identifier, or None. No normalisation is applied."""
        return self._by_id.get(payee_id)

    def render_name(self, payee_id: str, lang: str) -> str:
        """Display name of a payee in the given language ("ar" selects the Arabic name).

        Any language other than "ar" renders the English name. When the requested name is
        empty the other language's name is used, and an identifier that is not in the
        directory is rendered as the identifier itself so that a confirmation text can still
        be produced for it.
        """
        payee = self._by_id.get(payee_id)
        if payee is None:
            return payee_id
        if lang == "ar":
            return payee.name_ar or payee.name_en or payee_id
        return payee.name_en or payee.name_ar or payee_id
