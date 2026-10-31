"""Variant construction for the Tasdiq benchmark.

Every user turn of a seed yields a group of variants that share one pair_id: the
three language members of the authored text (kind base), the paraphrases of each
language (kind paraphrase, same intended value as the base) and the edits of each
language (kind edit, their own intended value). Each variant carries the earlier
turns of its dialogue as context, the seed-level metadata needed to drive the
simulator, and provenance fields that state where its text came from and whether
it has been reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from tasdiq.bench.schema import (
    LANGS,
    Intended,
    ScenarioSpec,
    Turn,
    intended_from_dict,
    intended_to_dict,
)

KINDS: tuple[str, ...] = ("base", "paraphrase", "edit")
SOURCES: tuple[str, ...] = ("authored", "paraphrase", "translation", "mutation")
DEFAULT_AUTHORED_LANG = "en"


@dataclass(frozen=True)
class ContextTurn:
    """An earlier turn of the dialogue, rendered in one language."""

    idx: int
    speaker: str
    action_type: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"idx": self.idx, "speaker": self.speaker, "action_type": self.action_type, "text": self.text}


@dataclass(frozen=True)
class Variant:
    """One benchmark item: a single utterance with its intended value and provenance.

    action_type is the action of this utterance: the action of the turn for base and
    paraphrase members, and the action of the edited utterance for an edit.
    """

    variant_id: str
    seed_id: str
    family: str
    phenomena: tuple[str, ...]
    turn_idx: int
    t: int
    action_type: str
    lang: str
    kind: str
    edit_kind: str | None
    edit_id: str | None
    paraphrase_idx: int | None
    text: str
    intended: Intended
    pair_id: str
    base_variant_id: str
    source: str
    review_status: str
    default_currency: str
    payees: tuple[str, ...]
    lang_of_record: str | None
    context: tuple[ContextTurn, ...]
    dialect_extra: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "seed_id": self.seed_id,
            "family": self.family,
            "phenomena": list(self.phenomena),
            "turn_idx": self.turn_idx,
            "t": self.t,
            "action_type": self.action_type,
            "lang": self.lang,
            "kind": self.kind,
            "edit_kind": self.edit_kind,
            "edit_id": self.edit_id,
            "paraphrase_idx": self.paraphrase_idx,
            "text": self.text,
            "intended": intended_to_dict(self.intended),
            "pair_id": self.pair_id,
            "base_variant_id": self.base_variant_id,
            "source": self.source,
            "review_status": self.review_status,
            "default_currency": self.default_currency,
            "payees": list(self.payees),
            "lang_of_record": self.lang_of_record,
            "context": [c.to_dict() for c in self.context],
            "dialect_extra": self.dialect_extra,
        }


def variant_from_dict(data: Mapping[str, Any]) -> Variant:
    """Rebuild a Variant from the mapping written by Variant.to_dict."""
    return Variant(
        variant_id=str(data["variant_id"]),
        seed_id=str(data["seed_id"]),
        family=str(data["family"]),
        phenomena=tuple(data["phenomena"]),
        turn_idx=int(data["turn_idx"]),
        t=int(data["t"]),
        action_type=str(data["action_type"]),
        lang=str(data["lang"]),
        kind=str(data["kind"]),
        edit_kind=data.get("edit_kind"),
        edit_id=data.get("edit_id"),
        paraphrase_idx=data.get("paraphrase_idx"),
        text=str(data["text"]),
        intended=intended_from_dict(data["intended"]),
        pair_id=str(data["pair_id"]),
        base_variant_id=str(data["base_variant_id"]),
        source=str(data["source"]),
        review_status=str(data["review_status"]),
        default_currency=str(data["default_currency"]),
        payees=tuple(data["payees"]),
        lang_of_record=data.get("lang_of_record"),
        context=tuple(
            ContextTurn(idx=int(c["idx"]), speaker=str(c["speaker"]), action_type=str(c["action_type"]), text=str(c["text"]))
            for c in data.get("context", [])
        ),
        dialect_extra=bool(data.get("dialect_extra", False)),
    )


def pair_id_for(seed_id: str, turn_idx: int) -> str:
    return f"{seed_id}.t{turn_idx}"


def base_variant_id_for(seed_id: str, turn_idx: int, lang: str) -> str:
    return f"{pair_id_for(seed_id, turn_idx)}.{lang}.base"


def context_for(spec: ScenarioSpec, turn_idx: int, lang: str) -> tuple[ContextTurn, ...]:
    """Earlier turns rendered in the variant's language, or in each turn's lang_of_record when set."""
    context: list[ContextTurn] = []
    for turn in spec.turns:
        if turn.idx >= turn_idx:
            break
        turn_lang = turn.lang_of_record or lang
        context.append(
            ContextTurn(idx=turn.idx, speaker=turn.speaker, action_type=turn.action_type, text=turn.text.get(turn_lang))
        )
    return tuple(context)


def build_variants(spec: ScenarioSpec) -> list[Variant]:
    """Build every variant of a seed: base members, paraphrases and edits of each user turn."""
    variants: list[Variant] = []
    for turn in spec.user_turns():
        variants.extend(_turn_variants(spec, turn))
    ids = [v.variant_id for v in variants]
    if len(set(ids)) != len(ids):
        raise ValueError(f"seed {spec.seed_id}: variant ids are not unique")
    return variants


def _turn_variants(spec: ScenarioSpec, turn: Turn) -> list[Variant]:
    pair_id = pair_id_for(spec.seed_id, turn.idx)
    authored_lang = turn.lang_of_record or DEFAULT_AUTHORED_LANG
    common = dict(
        seed_id=spec.seed_id,
        family=spec.family,
        phenomena=spec.phenomena,
        turn_idx=turn.idx,
        t=turn.t,
        pair_id=pair_id,
        default_currency=spec.default_currency,
        payees=spec.payees,
        lang_of_record=turn.lang_of_record,
        dialect_extra=spec.dialect_extra,
    )
    out: list[Variant] = []
    for lang in LANGS:
        base_id = base_variant_id_for(spec.seed_id, turn.idx, lang)
        context = context_for(spec, turn.idx, lang)
        out.append(
            Variant(
                variant_id=base_id,
                lang=lang,
                kind="base",
                action_type=turn.action_type,
                edit_kind=None,
                edit_id=None,
                paraphrase_idx=None,
                text=turn.text.get(lang),
                intended=turn.intended,
                base_variant_id=base_id,
                source="authored" if lang == authored_lang else "translation",
                context=context,
                **common,
                review_status=spec.provenance.status_for(lang),
            )
        )
        for n, text in enumerate(turn.paraphrases.get(lang), start=1):
            out.append(
                Variant(
                    variant_id=f"{pair_id}.{lang}.p{n}",
                    lang=lang,
                    kind="paraphrase",
                    action_type=turn.action_type,
                    edit_kind=None,
                    edit_id=None,
                    paraphrase_idx=n,
                    text=text,
                    intended=turn.intended,
                    base_variant_id=base_id,
                    source="paraphrase",
                    context=context,
                    **common,
                    review_status=spec.provenance.status_for(lang),
                )
            )
        for edit in turn.edits:
            out.append(
                Variant(
                    variant_id=f"{pair_id}.{lang}.e.{edit.edit_id}",
                    lang=lang,
                    kind="edit",
                    action_type=edit.action_type,
                    edit_kind=edit.kind,
                    edit_id=edit.edit_id,
                    paraphrase_idx=None,
                    text=edit.text.get(lang),
                    intended=edit.intended,
                    base_variant_id=base_id,
                    source="mutation",
                    context=context,
                    **common,
                    review_status=spec.provenance.status_for(lang),
                )
            )
    return out


def build_all_variants(specs: Iterable[ScenarioSpec]) -> list[Variant]:
    """Build the variants of several seeds in order; variant ids must be unique across seeds."""
    variants: list[Variant] = []
    for spec in specs:
        variants.extend(build_variants(spec))
    ids = [v.variant_id for v in variants]
    if len(set(ids)) != len(ids):
        raise ValueError("variant ids are not unique across seeds")
    return variants


def pair_groups(variants: Sequence[Variant]) -> dict[str, list[Variant]]:
    """Group variants by pair_id, preserving order within each group."""
    groups: dict[str, list[Variant]] = {}
    for variant in variants:
        groups.setdefault(variant.pair_id, []).append(variant)
    return groups


def meaning_preserving(group: Sequence[Variant]) -> list[Variant]:
    """The base and paraphrase members of a pair group."""
    return [v for v in group if v.kind in ("base", "paraphrase")]


def edits_of(group: Sequence[Variant]) -> list[Variant]:
    """The edit members of a pair group."""
    return [v for v in group if v.kind == "edit"]
