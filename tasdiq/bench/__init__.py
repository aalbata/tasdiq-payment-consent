"""Benchmark construction: seed schema, variant rendering, splits and export.

The export names (ExportResult, export_benchmark, select_review_set and
verify_hashes) are resolved on first access, so that running the export module as
a script does not import it twice.
"""

from typing import Any

from tasdiq.bench.render import ContextTurn, Variant, build_all_variants, build_variants, pair_groups, variant_from_dict
from tasdiq.bench.schema import (
    ACTION_TYPES,
    AMBIGUOUS_FIELD_ALIASES,
    AMBIGUOUS_FIELDS,
    EDIT_KINDS,
    EVENT_KINDS,
    FAMILIES,
    LANGS,
    PAYEE_KINDS,
    PHENOMENA,
    REVIEW_STATUSES,
    SPEAKERS,
    Ambiguous,
    Edit,
    Event,
    ExpectedOutcome,
    Forbidden,
    IntendedBatch,
    IntendedRecord,
    IntentChange,
    LangText,
    LiveTask,
    LiveText,
    Followup,
    Paraphrases,
    PayeeEntry,
    Provenance,
    ScenarioSpec,
    SeedValidationError,
    Turn,
    content_report,
    intended_from_dict,
    intended_to_dict,
    load_seed,
    load_seed_file,
    load_seeds,
    load_live_tasks,
    parse_live_task,
    parse_spec,
    default_edit_action_type,
)
from tasdiq.bench.split import (
    NearDuplicate,
    assign_splits,
    cross_split_exclusions,
    find_near_duplicates,
    jaccard,
    normalize_text,
    token_set,
)

_EXPORT_NAMES: frozenset[str] = frozenset({"ExportResult", "export_benchmark", "select_review_set", "verify_hashes"})


def __getattr__(name: str) -> Any:
    if name in _EXPORT_NAMES:
        from tasdiq.bench import export as _export

        return getattr(_export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACTION_TYPES",
    "AMBIGUOUS_FIELD_ALIASES",
    "AMBIGUOUS_FIELDS",
    "EDIT_KINDS",
    "EVENT_KINDS",
    "FAMILIES",
    "LANGS",
    "PAYEE_KINDS",
    "PHENOMENA",
    "REVIEW_STATUSES",
    "SPEAKERS",
    "Ambiguous",
    "ContextTurn",
    "Edit",
    "Event",
    "ExpectedOutcome",
    "ExportResult",
    "Forbidden",
    "IntendedBatch",
    "IntendedRecord",
    "IntentChange",
    "LangText",
    "LiveTask",
    "LiveText",
    "Followup",
    "NearDuplicate",
    "Paraphrases",
    "PayeeEntry",
    "Provenance",
    "ScenarioSpec",
    "SeedValidationError",
    "Turn",
    "Variant",
    "assign_splits",
    "build_all_variants",
    "build_variants",
    "content_report",
    "cross_split_exclusions",
    "default_edit_action_type",
    "export_benchmark",
    "find_near_duplicates",
    "intended_from_dict",
    "intended_to_dict",
    "jaccard",
    "load_seed",
    "load_seed_file",
    "load_seeds",
    "load_live_tasks",
    "parse_live_task",
    "normalize_text",
    "pair_groups",
    "parse_spec",
    "select_review_set",
    "token_set",
    "variant_from_dict",
    "verify_hashes",
]
