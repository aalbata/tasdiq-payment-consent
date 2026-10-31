"""Benchmark export: data files, review set, annotation guide and adjudication sheet.

export_benchmark builds the variants of every seed, assigns the seed-level split,
detects near duplicates, applies the recorded review decisions of review_outcome.csv to
the review status of the variants they name, and writes the benchmark files into one
directory. Every file is written deterministically (sorted keys, uniform line endings),
and hashes.json records the SHA-256 digest of every other file in the directory, the
review outcome file included.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tasdiq.bench.render import Variant, build_all_variants
from tasdiq.bench.schema import (
    AMBIGUOUS_FIELDS,
    FAMILIES,
    LANGS,
    REVIEW_STATUSES,
    Ambiguous,
    IntendedBatch,
    IntendedRecord,
    ScenarioSpec,
    content_report,
    load_seeds,
)
from tasdiq.bench.split import (
    DEFAULT_DEV_FRACTION,
    DEFAULT_JACCARD_THRESHOLD,
    DEFAULT_SPLIT_SEED,
    NearDuplicate,
    assign_splits,
    cross_split_exclusions,
    find_near_duplicates,
)

DEFAULT_SEEDS_DIR = Path("tasdiq") / "bench" / "seeds"
DEFAULT_OUT_DIR = Path("data")
DEFAULT_REVIEW_PER_LANG = 40

BENCHMARK_FILE = "benchmark.jsonl"
DIALECT_EXTRA_FILE = "dialect_extra.jsonl"
SCENARIOS_FILE = "scenarios.jsonl"
SPLITS_FILE = "splits.json"
HASHES_FILE = "hashes.json"
REVIEW_SET_FILE = "review_set.csv"
ANNOTATION_GUIDE_FILE = "annotation_guide.md"
ADJUDICATION_SHEET_FILE = "adjudication_sheet.csv"
BUILD_REPORT_FILE = "build_report.json"
REVIEW_OUTCOME_FILE = "review_outcome.csv"

INTENDED_COLUMNS: tuple[str, ...] = (
    "intended_payee_id",
    "intended_amount_minor",
    "intended_currency",
    "intended_total_max_minor",
    "intended_allowed_uses",
    "intended_batch",
    "intended_ambiguous",
    "intended_no_record",
)

REVIEW_COLUMNS: tuple[str, ...] = (
    "variant_id",
    "seed_id",
    "family",
    "kind",
    "lang",
    "text",
    *INTENDED_COLUMNS,
    "label_status",
    "reviewer_decision",
    "reviewer_notes",
)

ADJUDICATION_COLUMNS: tuple[str, ...] = (
    "variant_id",
    "seed_id",
    "family",
    "kind",
    "lang",
    "text",
    *INTENDED_COLUMNS,
    "reviewer_a_decision",
    "reviewer_a_notes",
    "reviewer_b_decision",
    "reviewer_b_notes",
    "agreement",
    "adjudicated_intended",
    "adjudication_notes",
)

REVIEWER_DECISIONS: tuple[str, ...] = ("agree", "disagree", "unsure")

REVIEW_OUTCOME_COLUMNS: tuple[str, ...] = ("variant_id", "lang", "decision", "reviewer", "date")

REVIEWER_STATUS: dict[str, str] = {"author": "author-reviewed", "second-reviewer": "second-reviewer"}
"""The reviewer named in a review outcome row and the review status that the reviewer's agreement confers."""

_STATUS_RANK: dict[str, int] = {status: rank for rank, status in enumerate(REVIEW_STATUSES)}


@dataclass(frozen=True)
class ReviewOutcome:
    """One recorded review decision: a reviewer's decision on the label of one variant on one date."""

    variant_id: str
    lang: str
    decision: str
    reviewer: str
    date: str

    def to_dict(self) -> dict[str, str]:
        return {"variant_id": self.variant_id, "lang": self.lang, "decision": self.decision, "reviewer": self.reviewer, "date": self.date}


def load_review_outcome(path: Path | str) -> list[ReviewOutcome]:
    """Read the review outcome file: one row per reviewed variant and reviewer.

    The columns are variant_id, lang, decision (agree, disagree or unsure), reviewer (author
    or second-reviewer) and date (an ISO 8601 calendar date). A ValueError names the first
    malformed row; a variant may carry at most one row per reviewer.
    """
    path = Path(path)
    rows: list[ReviewOutcome] = []
    seen: set[tuple[str, str]] = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REVIEW_OUTCOME_COLUMNS:
            raise ValueError(f"{path.name}: the columns must be {', '.join(REVIEW_OUTCOME_COLUMNS)}")
        for line_no, raw in enumerate(reader, start=2):
            row = ReviewOutcome(**{column: (raw.get(column) or "").strip() for column in REVIEW_OUTCOME_COLUMNS})
            if not row.variant_id:
                raise ValueError(f"{path.name} line {line_no}: empty variant_id")
            if row.lang not in LANGS:
                raise ValueError(f"{path.name} line {line_no}: lang {row.lang!r} is not one of {', '.join(LANGS)}")
            if row.decision not in REVIEWER_DECISIONS:
                raise ValueError(f"{path.name} line {line_no}: decision {row.decision!r} is not one of {', '.join(REVIEWER_DECISIONS)}")
            if row.reviewer not in REVIEWER_STATUS:
                raise ValueError(f"{path.name} line {line_no}: reviewer {row.reviewer!r} is not one of {', '.join(REVIEWER_STATUS)}")
            try:
                datetime.date.fromisoformat(row.date)
            except ValueError as error:
                raise ValueError(f"{path.name} line {line_no}: date {row.date!r} is not an ISO 8601 calendar date") from error
            key = (row.variant_id, row.reviewer)
            if key in seen:
                raise ValueError(f"{path.name} line {line_no}: a second row for {row.variant_id} by {row.reviewer}")
            seen.add(key)
            rows.append(row)
    return rows


def apply_review_outcome(variants: Sequence[Variant], outcome: Sequence[ReviewOutcome]) -> tuple[list[Variant], dict[str, str]]:
    """Return the variants with the review status the recorded decisions give them, and the statuses applied.

    A variant whose label the author agreed with becomes author-reviewed, and one that the
    author and the second reviewer both agreed with becomes second-reviewer; a status that
    the provenance of the seed already gives is never lowered. A disagree or unsure decision
    leaves the status unchanged, since the label is confirmed only once the seed has been
    corrected and reviewed again. Every row must name a variant of the benchmark in the
    language the row states.
    """
    by_id = {v.variant_id: v for v in variants}
    agreed: dict[str, set[str]] = {}
    for row in outcome:
        variant = by_id.get(row.variant_id)
        if variant is None:
            raise ValueError(f"review outcome names {row.variant_id}, which is not a variant of the benchmark")
        if variant.lang != row.lang:
            raise ValueError(f"review outcome row {row.variant_id} states language {row.lang}; the variant is in {variant.lang}")
        if row.decision == "agree":
            agreed.setdefault(row.variant_id, set()).add(row.reviewer)
    applied: dict[str, str] = {}
    for variant_id, reviewers in agreed.items():
        if "author" not in reviewers:
            continue
        status = "second-reviewer" if "second-reviewer" in reviewers else "author-reviewed"
        if _STATUS_RANK[status] > _STATUS_RANK[by_id[variant_id].review_status]:
            applied[variant_id] = status
    updated = [replace(v, review_status=applied[v.variant_id]) if v.variant_id in applied else v for v in variants]
    return updated, applied


def review_outcome_report(outcome: Sequence[ReviewOutcome], applied: Mapping[str, str], review: Sequence[Variant]) -> dict[str, Any]:
    """Counts of the recorded decisions and of the statuses they conferred, for build_report.json."""
    review_ids = {v.variant_id for v in review}
    lang_of = {row.variant_id: row.lang for row in outcome}
    by_status_lang: dict[str, dict[str, int]] = {}
    for variant_id, status in applied.items():
        by_status_lang.setdefault(status, {})
        lang = lang_of[variant_id]
        by_status_lang[status][lang] = by_status_lang[status].get(lang, 0) + 1
    return {
        "file": REVIEW_OUTCOME_FILE if outcome else None,
        "rows": len(outcome),
        "decisions": {decision: sum(1 for r in outcome if r.decision == decision) for decision in REVIEWER_DECISIONS},
        "reviewers": {reviewer: sum(1 for r in outcome if r.reviewer == reviewer) for reviewer in sorted({r.reviewer for r in outcome})},
        "dates": sorted({r.date for r in outcome}),
        "rows_in_review_set": sum(1 for r in outcome if r.variant_id in review_ids),
        "variants_by_applied_status": dict(sorted(Counter(applied.values()).items())),
        "variants_by_applied_status_and_lang": {status: {lang: by_status_lang[status].get(lang, 0) for lang in LANGS} for status in sorted(by_status_lang)},
    }


@dataclass
class ExportResult:
    """What export_benchmark wrote and found."""

    out_dir: Path
    hashes: dict[str, str]
    n_seeds: int
    n_main_seeds: int
    n_variants: int
    n_excluded: int
    splits: dict[str, str]
    duplicates: list[NearDuplicate]
    content_issues: list[str]
    review_rows: int
    counts: dict[str, Any] = field(default_factory=dict)


def export_benchmark(
    seeds: Sequence[ScenarioSpec],
    out_dir: Path | str = DEFAULT_OUT_DIR,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
    review_per_lang: int = DEFAULT_REVIEW_PER_LANG,
    review_seed: int = DEFAULT_SPLIT_SEED,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    strict_content: bool = False,
    review_outcome: Path | str | None = None,
) -> ExportResult:
    """Write the benchmark files for the given seeds into out_dir.

    Seeds flagged dialect_extra are written to scenarios.jsonl and dialect_extra.jsonl
    only; they receive no split and enter no result file. With strict_content the
    corpus-level content requirements must all hold, otherwise a ValueError lists
    the unmet ones. The review decisions of review_outcome (by default the file
    review_outcome.csv in out_dir, when it exists) set the review status of the
    variants they name; a file given from elsewhere is copied into out_dir under that
    name, so that the directory stays self-contained and hashes.json covers it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted(seeds, key=lambda s: s.seed_id)
    issues = content_report(seeds)
    if strict_content and issues:
        raise ValueError("content requirements not met:\n" + "\n".join(issues))

    main = [s for s in seeds if not s.dialect_extra]
    extra = [s for s in seeds if s.dialect_extra]
    splits = assign_splits(main, dev_fraction=dev_fraction, seed=split_seed)
    variants = build_all_variants(main)
    extra_variants = build_all_variants(extra)
    outcome_path = out_dir / REVIEW_OUTCOME_FILE
    if review_outcome is not None:
        given = Path(review_outcome)
        if given.resolve() != outcome_path.resolve():
            shutil.copyfile(given, outcome_path)
    outcome = load_review_outcome(outcome_path) if outcome_path.is_file() else []
    variants, applied = apply_review_outcome(variants, outcome)
    duplicates = find_near_duplicates(variants, threshold=jaccard_threshold)
    excluded = cross_split_exclusions(duplicates, splits)

    def split_label(v: Variant) -> str:
        return "excluded" if v.variant_id in excluded else splits[v.seed_id]

    files: dict[str, str] = {}
    files[BENCHMARK_FILE] = _write_jsonl(
        out_dir / BENCHMARK_FILE, ({**v.to_dict(), "split": split_label(v)} for v in variants)
    )
    files[DIALECT_EXTRA_FILE] = _write_jsonl(
        out_dir / DIALECT_EXTRA_FILE, ({**v.to_dict(), "split": "dialect_extra"} for v in extra_variants)
    )
    files[SCENARIOS_FILE] = _write_jsonl(out_dir / SCENARIOS_FILE, (s.to_dict() for s in seeds))

    counts = _counts(variants, splits, excluded)
    counts["near_duplicate_pairs_by_split"] = duplicate_pairs_by_split(duplicates, splits)
    splits_doc = {
        "dev_fraction": dev_fraction,
        "seed": split_seed,
        "jaccard_threshold": jaccard_threshold,
        "seed_splits": dict(sorted(splits.items())),
        "excluded_variants": sorted(excluded),
        "near_duplicates": [d.to_dict() for d in duplicates],
        "counts": counts,
    }
    files[SPLITS_FILE] = _write_json(out_dir / SPLITS_FILE, splits_doc)

    review = select_review_set(variants, per_lang=review_per_lang, seed=review_seed)
    by_seed = {s.seed_id: s for s in main}
    for v in review:
        if v.variant_id in applied:
            continue
        if v.lang != "en" and v.lang not in by_seed[v.seed_id].provenance.reviewed_langs and v.review_status != "unreviewed":
            raise ValueError(f"review row {v.variant_id} of language {v.lang} is marked {v.review_status} although the language is not listed as reviewed")
    files[REVIEW_SET_FILE] = _write_csv(out_dir / REVIEW_SET_FILE, REVIEW_COLUMNS, (_review_row(v) for v in review))
    files[ADJUDICATION_SHEET_FILE] = _write_csv(
        out_dir / ADJUDICATION_SHEET_FILE, ADJUDICATION_COLUMNS, (_adjudication_row(v) for v in review)
    )
    if outcome:
        files[REVIEW_OUTCOME_FILE] = sha256_of(outcome_path)
    files[ANNOTATION_GUIDE_FILE] = _write_text(
        out_dir / ANNOTATION_GUIDE_FILE, annotation_guide(review, review_per_lang, outcome=outcome, applied=applied, n_variants=len(variants))
    )

    report = {
        "n_seeds": len(seeds),
        "n_main_seeds": len(main),
        "n_dialect_extra_seeds": len(extra),
        "n_variants": len(variants),
        "n_excluded_variants": len(excluded),
        "n_near_duplicate_pairs": len(duplicates),
        "n_cross_seed_duplicate_pairs": sum(1 for d in duplicates if d.cross_seed),
        "review_rows": len(review),
        "review_outcome": review_outcome_report(outcome, applied, review),
        "content_issues": issues,
        "counts": counts,
    }
    files[BUILD_REPORT_FILE] = _write_json(out_dir / BUILD_REPORT_FILE, report)
    _write_json(out_dir / HASHES_FILE, dict(sorted(files.items())))

    return ExportResult(
        out_dir=out_dir,
        hashes=dict(sorted(files.items())),
        n_seeds=len(seeds),
        n_main_seeds=len(main),
        n_variants=len(variants),
        n_excluded=len(excluded),
        splits=splits,
        duplicates=duplicates,
        content_issues=issues,
        review_rows=len(review),
        counts=counts,
    )


def select_review_set(variants: Sequence[Variant], per_lang: int = DEFAULT_REVIEW_PER_LANG, seed: int = DEFAULT_SPLIT_SEED) -> list[Variant]:
    """Draw a stratified review sample: up to per_lang variants per language, balanced across families.

    For each language the variants are grouped by family, each group is shuffled with
    the given seed, and the groups are visited in round robin until the quota is met
    or every group is exhausted. The result is ordered by language, family and id.
    """
    rng = random.Random(seed)
    chosen: list[Variant] = []
    for lang in LANGS:
        groups: dict[str, list[Variant]] = {}
        for v in sorted((v for v in variants if v.lang == lang), key=lambda v: v.variant_id):
            groups.setdefault(v.family, []).append(v)
        for family in sorted(groups):
            rng.shuffle(groups[family])
        picked: list[Variant] = []
        families = sorted(groups)
        while len(picked) < per_lang and any(groups[f] for f in families):
            for family in families:
                if len(picked) >= per_lang:
                    break
                if groups[family]:
                    picked.append(groups[family].pop())
        chosen.extend(picked)
    order = {lang: i for i, lang in enumerate(LANGS)}
    chosen.sort(key=lambda v: (order[v.lang], v.family, v.variant_id))
    return chosen


def _intended_cells(variant: Variant) -> dict[str, Any]:
    cells: dict[str, Any] = {column: "" for column in INTENDED_COLUMNS}
    intended = variant.intended
    if isinstance(intended, IntendedRecord):
        cells["intended_payee_id"] = intended.payee_id
        cells["intended_amount_minor"] = intended.amount_minor
        cells["intended_currency"] = intended.currency
        cells["intended_total_max_minor"] = intended.total_max_minor
        cells["intended_allowed_uses"] = intended.allowed_uses
    elif isinstance(intended, IntendedBatch):
        cells["intended_batch"] = batch_cell(intended)
        cells["intended_total_max_minor"] = intended.total_max_minor
        cells["intended_allowed_uses"] = intended.allowed_uses
    elif isinstance(intended, Ambiguous):
        cells["intended_ambiguous"] = intended.field
    elif intended is None:
        cells["intended_no_record"] = "true"
    return cells


def batch_cell(batch: IntendedBatch) -> str:
    """The review-sheet rendering of a batch: payee_id:amount_minor:currency per item, separated by semicolons."""
    return "; ".join(f"{item.payee_id}:{item.amount_minor}:{item.currency}" for item in batch.items)


def _review_row(variant: Variant) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant_id": variant.variant_id,
        "seed_id": variant.seed_id,
        "family": variant.family,
        "kind": variant.kind,
        "lang": variant.lang,
        "text": variant.text,
        "label_status": variant.review_status,
        "reviewer_decision": "",
        "reviewer_notes": "",
    }
    row.update(_intended_cells(variant))
    return row


def _adjudication_row(variant: Variant) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant_id": variant.variant_id,
        "seed_id": variant.seed_id,
        "family": variant.family,
        "kind": variant.kind,
        "lang": variant.lang,
        "text": variant.text,
        "reviewer_a_decision": "",
        "reviewer_a_notes": "",
        "reviewer_b_decision": "",
        "reviewer_b_notes": "",
        "agreement": "",
        "adjudicated_intended": "",
        "adjudication_notes": "",
    }
    row.update(_intended_cells(variant))
    return row


def duplicate_pairs_by_split(duplicates: Sequence[NearDuplicate], splits: dict[str, str]) -> dict[str, int]:
    """Near-duplicate pairs counted by the splits of their seeds.

    Within-split pairs (both seeds in dev, or both in test) are retained in the benchmark
    and reported here; cross-split pairs lose their test-side member. Cross-seed pairs are
    counted separately from pairs inside one seed.
    """
    out: dict[str, int] = {"within_dev": 0, "within_test": 0, "cross_split": 0, "within_dev_cross_seed": 0, "within_test_cross_seed": 0, "cross_split_cross_seed": 0}
    for pair in duplicates:
        split_a = splits.get(pair.seed_a)
        split_b = splits.get(pair.seed_b)
        if split_a is None or split_b is None:
            continue
        key = "cross_split" if split_a != split_b else f"within_{split_a}"
        out[key] += 1
        if pair.cross_seed:
            out[f"{key}_cross_seed"] += 1
    return out


def _counts(variants: Sequence[Variant], splits: dict[str, str], excluded: set[str]) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_split: dict[str, int] = {}
    seeds_by_split: dict[str, int] = {}
    for v in variants:
        by_family[v.family] = by_family.get(v.family, 0) + 1
        by_lang[v.lang] = by_lang.get(v.lang, 0) + 1
        by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
        label = "excluded" if v.variant_id in excluded else splits[v.seed_id]
        by_split[label] = by_split.get(label, 0) + 1
    for split in splits.values():
        seeds_by_split[split] = seeds_by_split.get(split, 0) + 1
    return {
        "variants_by_family": dict(sorted(by_family.items())),
        "variants_by_lang": dict(sorted(by_lang.items())),
        "variants_by_kind": dict(sorted(by_kind.items())),
        "variants_by_split": dict(sorted(by_split.items())),
        "seeds_by_split": dict(sorted(seeds_by_split.items())),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return sha256_of(path)


def _write_json(path: Path, document: dict[str, Any]) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2))
        handle.write("\n")
    return sha256_of(path)


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return sha256_of(path)


def _write_text(path: Path, text: str) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return sha256_of(path)


def sha256_of(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(out_dir: Path | str) -> dict[str, bool]:
    """Recompute the digest of every file listed in hashes.json and report which match."""
    out_dir = Path(out_dir)
    with open(out_dir / HASHES_FILE, "r", encoding="utf-8") as handle:
        recorded = json.load(handle)
    return {name: sha256_of(out_dir / name) == digest for name, digest in recorded.items()}


def review_record(review: Sequence[Variant], outcome: Sequence[ReviewOutcome], applied: Mapping[str, str], n_variants: int) -> str:
    """The paragraph of the annotation guide that records the review decisions taken so far."""
    if not outcome:
        return (
            "The review set has not yet been reviewed, so every item carries the review status of its seed. "
            "When the author has reviewed the items, the file review_outcome.csv records one row per item "
            "(variant_id, lang, decision, reviewer, date) and the benchmark is rebuilt, which marks the listed "
            "variants author-reviewed. The adjudication sheet is for a second reviewer, if one is engaged, who "
            "reviews the same items with this guide."
        )
    author_rows = [r for r in outcome if r.reviewer == "author"]
    rows = author_rows or list(outcome)
    who = "The author" if author_rows else "A reviewer"
    by_lang = {lang: sum(1 for r in rows if r.lang == lang) for lang in LANGS}
    per_lang = ", ".join(f"{by_lang[lang]} {lang}" for lang in LANGS)
    dates = sorted({r.date for r in rows})
    when = f"on {dates[0]}" if len(dates) == 1 else f"between {dates[0]} and {dates[-1]}"
    decisions = {d: sum(1 for r in rows if r.decision == d) for d in REVIEWER_DECISIONS}
    in_set = sum(1 for r in rows if r.variant_id in {v.variant_id for v in review})
    scope = "the items of this review set" if in_set == len(rows) == len(review) else f"{len(rows)} items, {in_set} of them in this review set"
    outcome_text = ", ".join(f"{decisions[d]} {d}" for d in REVIEWER_DECISIONS)
    verdict = " and therefore no disagreement" if decisions["disagree"] == 0 and decisions["unsure"] == 0 else ""
    marked = Counter(applied.values())
    marked_text = " and ".join(f"{marked[s]} variants {s}" for s in REVIEW_STATUSES if marked.get(s))
    rest = n_variants - sum(marked.values())
    return (
        f"{who} reviewed {scope} ({per_lang}) against the labelling rules above {when} and recorded "
        f"{outcome_text} decisions{verdict}. The file review_outcome.csv records the decision of every reviewed "
        f"item (variant_id, lang, decision, reviewer, date); the export reads it and marks {marked_text}, while "
        f"the remaining {rest} variants keep the status unreviewed. The adjudication sheet is for a second "
        f"reviewer, if one is engaged, who reviews the same items with this guide."
    )


def annotation_guide(review: Sequence[Variant], per_lang: int, outcome: Sequence[ReviewOutcome] = (), applied: Mapping[str, str] | None = None, n_variants: int | None = None) -> str:
    """The annotation guide shipped with the review set."""
    n_by_lang = {lang: sum(1 for v in review if v.lang == lang) for lang in LANGS}
    families = ", ".join(FAMILIES)
    ambiguous_fields = ", ".join(AMBIGUOUS_FIELDS)
    counts = ", ".join(f"{n_by_lang[lang]} {lang}" for lang in LANGS)
    record = review_record(review, outcome, applied or {}, len(review) if n_variants is None else n_variants)
    return f"""# Annotation guide for the Tasdiq benchmark review set

## Purpose

The benchmark pairs every payment utterance with the transaction record the user
intends. The review set in review_set.csv is a stratified sample of the benchmark
(up to {per_lang} items per language, balanced across the families; this build
holds {counts}). The reviewer reads each item and states whether the recorded
intended value is the one a careful reader of the utterance would form. The review
does not evaluate any system; it checks the labels that systems are later scored
against.

## Files

- benchmark.jsonl holds one variant per line.
- scenarios.jsonl holds one seed scenario per line.
- splits.json records the seed-level dev and test assignment, the near-duplicate
  pairs found, and the variants excluded from the test split.
- review_set.csv is the sample to review; adjudication_sheet.csv is the sheet used
  when a second reviewer has reviewed the same sample.
- review_outcome.csv records the decision of every reviewed item (variant_id, lang,
  decision, reviewer, date); the export reads it and sets the review_status of the
  variants it names.
- hashes.json records the SHA-256 digest of every other file in the directory.

## Fields of a variant

- variant_id is the unique identifier of the item. It is built from the seed id,
  the turn index, the language and the kind, for example ov-001.t0.ar.p1.
- seed_id names the scenario the item belongs to. All items of one seed are
  dependent observations and share one split.
- family names the scenario family. The families are {families}.
- phenomena lists the linguistic phenomena the seed exercises, such as digit_script,
  currency_words, number_words, negation, inquiry_vs_approval, cross_turn_reference,
  cross_language_revision, code_switching, arabizi and similar_names.
- turn_idx and t give the position and the simulated time of the turn, in seconds
  from the start of the dialogue.
- action_type states what the utterance does in the dialogue (payment, inquiry,
  revision, cancellation or other). For a base or paraphrase member it is the action
  of the turn. For an edit that adds or removes a record it is the
  action of the edited utterance, which may differ from the turn (an inquiry edit of a
  payment turn that carries no record is an inquiry; an edit of an inquiry turn that
  carries a record is a payment). An edit whose intended value is an ambiguous marker
  keeps the action_type of its turn, so an inquiry turn edited into an instruction that
  omits the amount keeps action_type inquiry.
- lang is the language member (en for English, ar for Modern Standard Arabic, mix
  for the code-switched member).
- kind is base for the authored text, paraphrase for a meaning-preserving rewording
  with the same intended value as the base, and edit for a permission-altering
  rewording with its own intended value. edit_kind names what the edit changes
  (payee, amount, currency, negation, inquiry, reference or batch); edit_id and
  paraphrase_idx identify the member within its turn.
- text is the utterance itself.
- intended is the value a correct capture must produce. It takes one of the forms
  described in the next section.
- pair_id links every member of one turn of one seed. The base and paraphrase
  members of a pair share one intended value; each edit differs from the base in
  exactly the way its edit_kind states. base_variant_id names the base member in the
  same language.
- source records where the text came from (authored for the base member in the
  language the turn was written in, translation for the other base members,
  paraphrase, or mutation for an edit).
- review_status is the review state of the variant (unreviewed, author-reviewed or
  second-reviewer). It is the status that the provenance of its seed gives its
  language, raised for a variant whose label review_outcome.csv records as agreed by
  the author (author-reviewed) or by the author and a second reviewer
  (second-reviewer).
- default_currency is the currency assumed when the utterance names none; payees
  lists the directory identifiers the scenario may refer to. A payee identifier is
  stable within a seed, and the same identifier may denote a different synthetic
  payee in another seed, so identifiers are resolved and compared within one seed
  only.
- lang_of_record, when set, is the language in which this turn enters the simulated
  dialogue; it marks cross-language revisions.
- context holds the earlier turns of the dialogue in the item's language, so that a
  reference such as "the same amount" can be resolved.
- split is dev, test, or excluded for a test-side member of a near-duplicate pair
  that straddles the two splits.

## Forms of the intended value

- A record with payee_id (the directory identifier of the payee within the seed,
  never a display name), amount_minor (an integer in minor units, so 250.50 AED is
  25050), currency (an ISO 4217 code), total_max_minor (the cap on amount plus fees,
  filled under the cap convention below) and allowed_uses (1 unless the utterance
  states a batch with an explicit count). In the review sheets these appear as the
  columns intended_payee_id, intended_amount_minor, intended_currency,
  intended_total_max_minor and intended_allowed_uses.
- A batch, listing the records that one instruction pays, in the order stated, with
  a cap on the sum of the amounts plus fees when the utterance states a total; each
  item carries its own total_max_minor under the cap convention below. In the review
  sheets the items appear in the column intended_batch as
  payee_id:amount_minor:currency separated by semicolons, the cap in
  intended_total_max_minor and the number of items in intended_allowed_uses; the
  single-record columns stay empty.
- An ambiguous marker naming the field a correct capture must ask about (column
  intended_ambiguous). The allowed fields are {ambiguous_fields}. The field
  reference stands for an unresolved reference to an earlier turn, and action_type
  for an utterance in which approval and inquiry cannot be told apart.
- No record (column intended_no_record set to true; the value null in
  benchmark.jsonl) for an utterance that must not yield a record at all, such as a
  clear inquiry, a cancellation or an unrelated remark.

Cap convention. For a payee of kind international, total_max_minor holds the cap the
utterance states and otherwise the amount plus the scheduled fee of the simulated
rail, which is one percent of the amount with a minimum of 1000 minor units (10.00 in
a two-decimal currency); 2,000.00 USD sent to an international payee with no stated
cap is therefore recorded with total_max_minor 202000, and 320.00 EUR with 33000.
For a payee of kind domestic or merchant the field is empty unless the utterance
states a cap. The schema reads an empty field as a cap equal to the amount, and the
scheduled fee of these two kinds is zero, so for them a cap equal to the amount and
an empty field describe the same record. The cap is compared by the monitor at
execution time and enters the scenario-level labels (an execution whose amount plus
quoted fee exceeds it is a violation of class cap_exceeded); the capture-stage
scoring compares payee_id, amount_minor and currency and never the cap. A reviewer
therefore marks a disagreement on intended_total_max_minor only when the utterance
states a cap that the column does not carry, or when the column carries a value that
neither the utterance nor this convention gives.

## Labelling rules

The rules below restate the reference policy from which every label is derived.
The system under test is never used to produce a label.

- The canonical record is the tuple (principal, action_type, payee_id, amount_minor,
  currency, total_max_minor, revision). Display names, aliases and transliterations
  are not part of it. Two utterances that name the same payee by different names or
  the same amount in different scripts carry the same record.
- A capture is correct when the produced record equals the intended record on
  payee_id, amount_minor and currency (the cap is left to the monitor, which adds the
  quoted fee), or when the pipeline asks about exactly the ambiguous field. Asking
  about a field that the utterance determines is an unnecessary clarification;
  producing a record for an utterance marked ambiguous or no-record is an error.
- Meaning-preserving members of one turn (base and paraphrases, in all three
  languages) carry the same intended value. A permission-altering edit carries the
  edited value. A reviewer who finds a paraphrase that changes the meaning, or an
  edit that does not, marks a disagreement.
- An inquiry asks about a payment without instructing it; an approval instructs it.
  When the utterance does not settle which one it is, the intended value is the
  ambiguous marker on action_type. Inquiry and approval members deliberately share
  vocabulary, so the decision rests on the whole utterance.
- A negated instruction without a clear replacement yields no record or an
  ambiguous marker, never the negated record. A negation with a replacement yields
  the replacement.
- An edit that turns a payment instruction into an inquiry carries no record. An edit
  that leaves a binding field undetermined (an amount the utterance omits, a payee
  name that two directory entries match) carries the ambiguous marker on that field.
  In both cases the action_type of the edit follows the rule stated under Fields of a
  variant; the review checks the intended value, not the action_type.
- A batch instruction authorises each listed item once and nothing else. An item
  the utterance excludes, or an item added from outside the utterance, lies outside
  the scope of the batch. A batch edit changes which items the instruction contains;
  a payee or amount edit of a batch changes one item and keeps the others.
- An utterance that refers to an earlier turn ("the same amount", "that payment")
  yields the record that the context determines; when the context does not
  determine it, the intended value is the ambiguous marker on reference.
- Amounts written in Eastern Arabic digits, in number words, or with an Arabic
  decimal or thousands separator denote the same amount as their ASCII form. A
  currency named by a word, a symbol or an Arabic name is the named currency; an
  amount without a currency takes the default currency of the scenario.
- A scenario-level execution is authorised only when a consent event for the exact
  record exists inside its validity window, has not been consumed, the principal's
  intent at execution time still includes the record (no later revision or
  cancellation), and the request is not a duplicate. A confirmation request is
  necessary when no valid consent for the exact record exists, and unnecessary when
  one does or when the utterance is unambiguous and the request is a clarification.
  These labels are computed from the scenario; the review set checks only the
  intended values on which they rest.

## How to review

For each row of review_set.csv, read the text together with its family, kind and the
earlier turns of the seed (context in benchmark.jsonl), then fill two columns:

- reviewer_decision takes one of three values: agree when the intended columns
  state the value you would form from the text; disagree when they do not; unsure
  when the text itself does not let you decide.
- reviewer_notes is empty for agree. For disagree, write the corrected value in the
  form field=value, one per item separated by semicolons, for example
  intended_amount_minor=20000; intended_currency=AED. For a text defect (dialect,
  an untranslatable phrase, a typographical error) write text=<the corrected text>.
  For unsure, state what the text leaves open.

Judge the Arabic and code-switched members on their own wording, not on the English
member; the English member is visible in benchmark.jsonl for comparison only. Do
not change the text or the intended columns in review_set.csv; the author applies
corrections to the seed files and rebuilds the benchmark, so that every file stays
derived from the seeds.

## Adjudication

When a second reviewer has reviewed the same sample, adjudication_sheet.csv holds
one row per item with both decisions and notes. The agreement column is agree when
both decisions are agree, disagree when both are disagree with the same corrected
value, and split otherwise. For every row that is not agree, the adjudicated_intended
column records the value adopted after discussion, in the same field=value form,
and adjudication_notes records the reason. Items whose adjudicated value differs
from the seed are corrected in the seed files and the benchmark is rebuilt. A variant
reaches review_status second-reviewer only after this step; until then the paper
reports the labels as reviewed by the author alone.

## Review record

{record}

## Review status values

- unreviewed means that no one has reviewed the label of the variant since it was
  authored. Every variant starts in this state.
- author-reviewed means that the author has reviewed the label of the variant, as an
  item of the review set recorded in review_outcome.csv or as a member of a seed whose
  provenance lists its language as reviewed, and has applied any correction.
- second-reviewer means that a second qualified reviewer has also reviewed the item
  and the adjudication has been applied.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Build the benchmark files from a seed directory."""
    parser = argparse.ArgumentParser(description="Build the Tasdiq benchmark files from the seed scenarios.")
    parser.add_argument("--seeds-dir", default=str(DEFAULT_SEEDS_DIR), help="directory holding the seed YAML files")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="directory to write the benchmark files into")
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--review-per-lang", type=int, default=DEFAULT_REVIEW_PER_LANG)
    parser.add_argument("--strict-content", action="store_true", help="fail when the corpus content requirements are not met")
    parser.add_argument("--review-outcome", default=None, help="review outcome file to apply (default: review_outcome.csv in the output directory, when present)")
    args = parser.parse_args(argv)
    seeds = load_seeds(Path(args.seeds_dir))
    result = export_benchmark(
        seeds,
        out_dir=Path(args.out_dir),
        dev_fraction=args.dev_fraction,
        split_seed=args.split_seed,
        review_per_lang=args.review_per_lang,
        strict_content=args.strict_content,
        review_outcome=args.review_outcome,
    )
    print(
        f"seeds {result.n_seeds} (main {result.n_main_seeds}), variants {result.n_variants}, "
        f"excluded {result.n_excluded}, near-duplicate pairs {len(result.duplicates)}, review rows {result.review_rows}"
    )
    for issue in result.content_issues:
        print(f"content: {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
