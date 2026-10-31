"""Seed-level split assignment and near-duplicate detection.

The split is stratified by family at the level of seeds, so that every variant of a
seed shares one split and the seed remains the independent statistical unit. Near
duplicates are pairs of variants in the same language whose normalised texts are
equal or whose token sets have a Jaccard index at or above a threshold; a pair that
straddles the dev and test splits is reported and its test-side member is excluded
from the test set.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from tasdiq.bench.render import Variant
from tasdiq.bench.schema import ScenarioSpec

DEFAULT_DEV_FRACTION = 0.25
DEFAULT_SPLIT_SEED = 20261006
DEFAULT_JACCARD_THRESHOLD = 0.9
SPLITS: tuple[str, ...] = ("dev", "test")

_TATWEEL = "ـ"
_HARAKAT = re.compile(r"[ً-ْٰؐ-ؚ]")
_ALEF_FORMS = str.maketrans({"آ": "ا", "أ": "ا", "إ": "ا", "ة": "ه", "ى": "ي"})
_PUNCTUATION = re.compile(r"[^\w\s-]|_", re.UNICODE)
_INNER_HYPHEN = re.compile(r"(?<!\w)-|-(?!\w)")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalise an utterance for duplicate detection.

    Applies NFC, removes tatweel and Arabic harakat, unifies alef forms, maps teh
    marbuta to heh and alef maksura to yeh, lowercases Latin letters, strips
    punctuation except hyphens inside identifiers, and collapses whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_TATWEEL, "")
    text = _HARAKAT.sub("", text)
    text = text.translate(_ALEF_FORMS)
    text = text.lower()
    text = _PUNCTUATION.sub(" ", text)
    text = _INNER_HYPHEN.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def token_set(text: str) -> frozenset[str]:
    """Whitespace tokens of the normalised text."""
    return frozenset(normalize_text(text).split())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard index of two token sets; 1.0 when both are empty."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def assign_splits(
    seeds: Sequence[ScenarioSpec],
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, str]:
    """Assign every seed to dev or test, stratified by family, with a stated random seed.

    Within each family the seed ids are sorted, shuffled with the given seed and the
    first round(n * dev_fraction) go to dev; a family with two or more seeds always
    keeps at least one seed on each side. The result does not depend on input order.
    """
    if not 0.0 <= dev_fraction < 1.0:
        raise ValueError("dev_fraction must lie in [0, 1)")
    by_family: dict[str, list[str]] = {}
    seen: set[str] = set()
    for spec in seeds:
        if spec.seed_id in seen:
            raise ValueError(f"duplicate seed id {spec.seed_id!r}")
        seen.add(spec.seed_id)
        by_family.setdefault(spec.family, []).append(spec.seed_id)
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for family in sorted(by_family):
        ids = sorted(by_family[family])
        rng.shuffle(ids)
        n = len(ids)
        n_dev = int(round(n * dev_fraction))
        if n >= 2:
            n_dev = min(max(n_dev, 1), n - 1)
        else:
            n_dev = 0
        for i, seed_id in enumerate(ids):
            assignment[seed_id] = "dev" if i < n_dev else "test"
    return assignment


def stratified_subset(
    seeds: Sequence[ScenarioSpec],
    fraction: float | None = None,
    count: int | None = None,
    seed: int = DEFAULT_SPLIT_SEED,
) -> list[str]:
    """Draw a subset of seed ids stratified by family with a stated random seed.

    The subset size is ``count`` or round(n * fraction) over the given seeds; it is
    allocated to the families in proportion to their sizes by the largest-remainder rule
    (ties broken by family name), never beyond a family's size. Within each family the
    seed ids are sorted and shuffled with the given seed, in sorted family order, and the
    first quota are taken, so the same seeds, size and random seed always yield the same
    subset and equal fractions yield equal subsets. The result is sorted.
    """
    by_family: dict[str, list[str]] = {}
    for spec in seeds:
        by_family.setdefault(spec.family, []).append(spec.seed_id)
    total = sum(len(ids) for ids in by_family.values())
    if count is None:
        if fraction is None or not 0.0 <= fraction <= 1.0:
            raise ValueError("stratified_subset needs a count or a fraction in [0, 1]")
        count = int(round(total * float(fraction)))
    count = max(0, min(int(count), total))
    families = sorted(by_family)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for family in families:
        share = count * len(by_family[family]) / total if total else 0.0
        quotas[family] = min(int(share), len(by_family[family]))
        remainders.append((share - int(share), family))
    short = count - sum(quotas.values())
    for _, family in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if short <= 0:
            break
        if quotas[family] < len(by_family[family]):
            quotas[family] += 1
            short -= 1
    rng = random.Random(seed)
    chosen: list[str] = []
    for family in families:
        ids = sorted(by_family[family])
        rng.shuffle(ids)
        chosen.extend(ids[: quotas[family]])
    return sorted(chosen)


@dataclass(frozen=True)
class NearDuplicate:
    """A pair of variants in one language judged near duplicates."""

    a: str
    b: str
    seed_a: str
    seed_b: str
    lang: str
    score: float
    reason: str

    @property
    def cross_seed(self) -> bool:
        return self.seed_a != self.seed_b

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "seed_a": self.seed_a,
            "seed_b": self.seed_b,
            "lang": self.lang,
            "score": self.score,
            "reason": self.reason,
            "cross_seed": self.cross_seed,
        }


def find_near_duplicates(
    variants: Sequence[Variant],
    threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> list[NearDuplicate]:
    """Find near-duplicate variant pairs within each language.

    Two variants are near duplicates when their normalised texts are equal (reason
    "exact") or when the Jaccard index of their token sets is at least the threshold
    (reason "jaccard"). Each unordered pair is reported once, ordered by variant id.
    """
    found: list[NearDuplicate] = []
    by_lang: dict[str, list[Variant]] = {}
    for variant in variants:
        by_lang.setdefault(variant.lang, []).append(variant)
    for lang in sorted(by_lang):
        items = sorted(by_lang[lang], key=lambda v: v.variant_id)
        normalised = [normalize_text(v.text) for v in items]
        tokens = [frozenset(n.split()) for n in normalised]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if normalised[i] == normalised[j]:
                    score, reason = 1.0, "exact"
                else:
                    score = jaccard(tokens[i], tokens[j])
                    if score < threshold:
                        continue
                    reason = "jaccard"
                found.append(
                    NearDuplicate(
                        a=items[i].variant_id,
                        b=items[j].variant_id,
                        seed_a=items[i].seed_id,
                        seed_b=items[j].seed_id,
                        lang=lang,
                        score=score,
                        reason=reason,
                    )
                )
    return found


def cross_split_exclusions(
    duplicates: Iterable[NearDuplicate],
    splits: Mapping[str, str],
) -> set[str]:
    """Variant ids to exclude from the test set: the test-side member of every cross-split pair."""
    excluded: set[str] = set()
    for pair in duplicates:
        split_a = splits.get(pair.seed_a)
        split_b = splits.get(pair.seed_b)
        if split_a is None or split_b is None or split_a == split_b:
            continue
        excluded.add(pair.a if split_a == "test" else pair.b)
    return excluded


def split_of(variant: Variant, splits: Mapping[str, str], excluded: Iterable[str] = ()) -> str:
    """The split label of a variant: dev, test, or excluded for a test-side near duplicate."""
    if variant.variant_id in set(excluded):
        return "excluded"
    return splits[variant.seed_id]
