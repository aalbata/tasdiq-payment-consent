"""Capture track: run the capture pipelines over the benchmark variants with a local model.

For every selected variant, capture pipeline and capture model the runner builds the
dialogue context of the variant (earlier turns in the variant's language, the record the
dialogue refers to, the English member that corresponds to the variant) and runs the named
pipeline. The record the dialogue refers to comes from one of two sources, chosen by the
configuration key previous_record_source: "oracle" (default) takes the scenario's intended
record of the latest earlier user turn that states one; "captured" takes the record the
same pipeline and model produced for the earlier turn's base member in the variant's
language, run on demand in the same worker. Every row records the source and the record
used. The output row carries the variant's identification fields, the intended record (or
the ambiguous field, or no record), the produced record or the field asked about, the
parse failure if any, the per-stage timings and the token counts, in the shape read by
analysis.metrics.

Rows are written to results/capture/<run_id>/captures.jsonl and, per model and pipeline,
to <model>__<pipeline>.jsonl; capture summaries per model, pipeline, language, family and
phenomenon go to aggregates.json. One worker pool per model loads that model once per
process; the model runtime is imported inside the worker initialiser only. Turns whose
intended value is a batch are outside the single-record capture schema and are skipped
(their number is reported in the manifest).

The selection of work units reads, besides the pipelines, kinds, languages and splits
(capture_splits, falling back to splits): capture_pipeline_members, a mapping from a
pipeline to the languages and kinds it runs on; capture_seed_subsets, a mapping from a
pipeline to the fraction of the selected seeds it runs on, drawn stratified by family
with capture_subset_seed (default split_seed); and per model the subset keys
capture_kinds, capture_pipelines, capture_languages and capture_seed_fraction or
capture_seed_count (a stratified subset of the selected seeds). Every drawn seed subset
is written to the run manifest. Variants the near-duplicate rule excluded are left out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tasdiq.analysis.metrics import capture_correct, capture_summary_by, explode_phenomena
from tasdiq.bench.render import Variant
from tasdiq.bench.schema import Ambiguous, IntendedBatch, IntendedRecord, ScenarioSpec
from tasdiq.bench.split import stratified_subset
from tasdiq.capture.pipelines import PIPELINE_NAMES, CaptureContext, run_pipeline
from tasdiq.runners.config import ModelSpec, RunConfig, append_jsonl, load_config, make_config, prepare_run_dir, read_jsonl, write_json
from tasdiq.runners.trace import directory_for, excluded_variants, load_benchmark
from tasdiq.runners.workers import Progress, make_key, plan_workers, run_in_workers, worker_state

AGGREGATE_KEYS: tuple[tuple[str, ...], ...] = (
    ("model", "pipeline"),
    ("model", "pipeline", "lang"),
    ("model", "pipeline", "family"),
    ("model", "pipeline", "lang", "family"),
    ("model", "pipeline", "kind"),
    ("model", "pipeline", "lang", "edit_kind"),
)


@dataclass(frozen=True)
class CaptureItem:
    """One unit of capture work: a variant under one pipeline and model."""

    seed_id: str
    variant_id: str
    pipeline: str
    model: str

    @property
    def key(self) -> str:
        return make_key(self.variant_id, self.pipeline, self.model)

    def to_dict(self) -> dict[str, Any]:
        return {"seed_id": self.seed_id, "variant_id": self.variant_id, "pipeline": self.pipeline, "model": self.model}


PREVIOUS_RECORD_SOURCES: tuple[str, ...] = ("oracle", "captured")


def previous_record_for(spec: ScenarioSpec, turn_idx: int) -> dict[str, Any] | None:
    """The record the dialogue refers to before turn_idx: the intended record of the latest earlier user turn."""
    previous: dict[str, Any] | None = None
    for turn in spec.turns:
        if turn.idx >= turn_idx:
            break
        if turn.speaker != "user":
            continue
        if isinstance(turn.intended, IntendedRecord):
            previous = {"payee_id": turn.intended.payee_id, "amount_minor": turn.intended.amount_minor, "currency": turn.intended.currency, "total_max_minor": turn.intended.total_max_minor}
    return previous


def english_text_for(spec: ScenarioSpec, variant: Variant) -> str | None:
    """The English member that corresponds to a variant.

    An edit takes the English text of the same edit, a paraphrase the English paraphrase
    with the same index, and a base member the authored English text of the turn.
    """
    try:
        turn = spec.turn(variant.turn_idx)
    except KeyError:
        return None
    if variant.kind == "edit":
        for edit in turn.edits:
            if edit.edit_id == variant.edit_id:
                return edit.text.en
        return None
    if variant.kind == "paraphrase" and variant.paraphrase_idx is not None:
        english = list(turn.paraphrases.get("en"))
        position = int(variant.paraphrase_idx) - 1
        return english[position] if 0 <= position < len(english) else None
    return turn.text.en


def context_for_variant(spec: ScenarioSpec, variant: Variant, previous_record: Any = None, previous_source: str = "oracle") -> CaptureContext:
    """Build the pipeline context of a variant from the scenario.

    With previous_source "oracle" the record the dialogue refers to is the scenario's
    intended record of the latest earlier user turn; with "captured" it is the
    previous_record given by the caller (the output of the same pipeline on the earlier
    turn).
    """
    if previous_source not in PREVIOUS_RECORD_SOURCES:
        raise ValueError(f"unknown previous_record_source: {previous_source}")
    previous = previous_record_for(spec, variant.turn_idx) if previous_source == "oracle" else previous_record
    return CaptureContext(
        turns=[(c.speaker, c.text) for c in variant.context],
        previous_record=previous,
        lang=variant.lang,
        english_text=english_text_for(spec, variant),
    )


def earlier_user_turns(spec: ScenarioSpec, turn_idx: int) -> list[int]:
    """Indices of the user turns before turn_idx, latest first."""
    return [turn.idx for turn in reversed(spec.turns) if turn.idx < turn_idx and turn.speaker == "user"]


def captured_previous_record(spec: ScenarioSpec, variant: Variant, item: "CaptureItem", runner: Any, directory: Any, call_cfg: Mapping[str, Any], variants_by_id: Mapping[str, Variant], cache: dict[str, Any]) -> dict[str, Any] | None:
    """The record the same pipeline and model produced for the latest earlier turn that yields one.

    Earlier turns are run on their base member in the variant's language, latest first,
    each with its own captured previous record; outputs are cached per worker process.
    """
    for idx in earlier_user_turns(spec, variant.turn_idx):
        base_id = f"{spec.seed_id}.t{idx}.{variant.lang}.base"
        base = variants_by_id.get(base_id)
        if base is None:
            continue
        key = f"{base_id}|{item.pipeline}|{item.model}"
        if key not in cache:
            previous = captured_previous_record(spec, base, item, runner, directory, call_cfg, variants_by_id, cache)
            context = context_for_variant(spec, base, previous, "captured")
            cache[key] = run_pipeline(item.pipeline, base.text, context, runner, directory, call_cfg)
        output = cache[key]
        if output.kind == "record" and output.record is not None:
            return {"payee_id": output.record.payee_id, "amount_minor": output.record.amount_minor, "currency": output.record.currency, "total_max_minor": output.record.total_max_minor}
    return None


def capture_applicable(variant: Variant) -> bool:
    """True for variants the single-record capture schema can label (no batch intents)."""
    return not isinstance(variant.intended, IntendedBatch)


def capture_splits_of(cfg: RunConfig) -> set[str]:
    """The splits the capture track runs on: capture_splits when set, else splits."""
    return set(cfg.get("capture_splits") or cfg.get("splits", ["test"]) or ["test"])


def capture_seed_subsets(cfg: RunConfig, specs: Sequence[ScenarioSpec], splits: Mapping[str, str]) -> dict[str, dict[str, list[str]]]:
    """The stratified seed subsets the configuration draws, per pipeline and per model.

    The pipeline subsets come from capture_seed_subsets (pipeline to fraction) and the
    model subsets from the model's capture_seed_fraction or capture_seed_count; both are
    drawn from the seeds of the capture splits with capture_subset_seed (default
    split_seed). A pipeline or model without a subset rule is absent from the result.
    """
    wanted = capture_splits_of(cfg)
    pool = [s for s in specs if not s.dialect_extra and splits.get(s.seed_id) in wanted]
    seed = int(cfg.get("capture_subset_seed") or cfg.get("split_seed", 20261006) or 20261006)
    out: dict[str, dict[str, list[str]]] = {"pipelines": {}, "models": {}}
    for pipeline, fraction in dict(cfg.get("capture_seed_subsets") or {}).items():
        out["pipelines"][str(pipeline)] = stratified_subset(pool, fraction=float(fraction), seed=seed)
    for spec in cfg.models_for("capture"):
        if spec.subset.get("capture_seed_count") is not None:
            out["models"][spec.name] = stratified_subset(pool, count=int(spec.subset["capture_seed_count"]), seed=seed)
        elif spec.subset.get("capture_seed_fraction") is not None:
            out["models"][spec.name] = stratified_subset(pool, fraction=float(spec.subset["capture_seed_fraction"]), seed=seed)
    return out


def pipeline_members(cfg: RunConfig, pipeline: str) -> tuple[set[str], set[str]]:
    """The (languages, kinds) a pipeline is restricted to by capture_pipeline_members; empty sets mean no restriction."""
    rules = dict((cfg.get("capture_pipeline_members") or {}).get(pipeline) or {})
    return set(rules.get("languages") or []), set(rules.get("kinds") or [])


def enumerate_capture_items(cfg: RunConfig, specs: Sequence[ScenarioSpec], variants: Sequence[Variant], splits: Mapping[str, str]) -> list[CaptureItem]:
    """List the capture work units selected by the configuration."""
    wanted_splits = capture_splits_of(cfg)
    langs = set(cfg.get("languages", ["en", "ar", "mix"]) or ["en", "ar", "mix"])
    kinds = set(cfg.get("capture_kinds", ["base", "paraphrase", "edit"]) or ["base", "paraphrase", "edit"])
    pipelines = [str(p) for p in (cfg.get("capture_pipelines") or ["direct"])]
    for name in pipelines:
        if name not in PIPELINE_NAMES:
            raise ValueError(f"unknown capture pipeline: {name}")
    members = {name: pipeline_members(cfg, name) for name in pipelines}
    subsets = capture_seed_subsets(cfg, specs, splits)
    pipeline_seeds = {name: set(ids) for name, ids in subsets["pipelines"].items()}
    excluded = excluded_variants(cfg)
    by_seed = {s.seed_id: s for s in specs}
    items: list[CaptureItem] = []
    for spec in cfg.models_for("capture"):
        subset_kinds = set(spec.subset.get("capture_kinds", []) or [])
        subset_pipelines = set(spec.subset.get("capture_pipelines", []) or [])
        subset_langs = set(spec.subset.get("capture_languages", []) or [])
        model_seeds = set(subsets["models"][spec.name]) if spec.name in subsets["models"] else None
        for variant in variants:
            if variant.seed_id not in by_seed or variant.dialect_extra or not capture_applicable(variant) or variant.variant_id in excluded:
                continue
            if splits.get(variant.seed_id) not in wanted_splits or variant.lang not in langs or variant.kind not in kinds:
                continue
            if subset_kinds and variant.kind not in subset_kinds:
                continue
            if subset_langs and variant.lang not in subset_langs:
                continue
            if model_seeds is not None and variant.seed_id not in model_seeds:
                continue
            for pipeline in pipelines:
                if subset_pipelines and pipeline not in subset_pipelines:
                    continue
                if pipeline == "english_reference" and variant.lang == "en":
                    continue
                member_langs, member_kinds = members[pipeline]
                if member_langs and variant.lang not in member_langs:
                    continue
                if member_kinds and variant.kind not in member_kinds:
                    continue
                if pipeline in pipeline_seeds and variant.seed_id not in pipeline_seeds[pipeline]:
                    continue
                items.append(CaptureItem(variant.seed_id, variant.variant_id, pipeline, spec.name))
    limit = cfg.get("limit")
    if limit:
        items = items[: int(limit)]
    return items


def filter_by_model(items: Sequence[CaptureItem], cfg: RunConfig) -> list[CaptureItem]:
    """Keep the items of the model named by model_filter (a model name), or every item when unset."""
    wanted = cfg.get("model_filter")
    if not wanted:
        return list(items)
    names = {m.name for m in cfg.models}
    if str(wanted) not in names:
        raise KeyError(f"model_filter names an unknown model: {wanted!r}; configured models are {sorted(names)}")
    return [it for it in items if it.model == str(wanted)]


def capture_counts(cfg: RunConfig | Path | str) -> dict[str, Any]:
    """Count the capture work units of a configuration per model and pipeline, with the pending ones.

    Nothing is run and nothing is written; the pending count reads the done keys of the
    run directory the configuration names, so that a resumed run reports the work left.
    """
    if not isinstance(cfg, RunConfig):
        cfg = load_config(cfg)
    specs, variants, splits = load_benchmark(cfg)
    items = filter_by_model(enumerate_capture_items(cfg, specs, variants, splits), cfg)
    progress = Progress(cfg.results_dir / "capture" / cfg.run_id / "done.jsonl")
    subsets = capture_seed_subsets(cfg, specs, splits)
    per_model: dict[str, dict[str, Any]] = {}
    for it in items:
        entry = per_model.setdefault(it.model, {"items": 0, "pending": 0, "pipelines": {}})
        entry["items"] += 1
        entry["pipelines"][it.pipeline] = entry["pipelines"].get(it.pipeline, 0) + 1
        if it.key not in progress:
            entry["pending"] += 1
    return {
        "track": "capture",
        "run_id": cfg.run_id,
        "model_filter": cfg.get("model_filter"),
        "n_seeds": len(specs),
        "n_variants": len(variants),
        "items": len(items),
        "pending": sum(e["pending"] for e in per_model.values()),
        "models": per_model,
        "seed_subsets": {kind: {name: len(ids) for name, ids in group.items()} for kind, group in subsets.items()},
    }


def capture_row(cfg: RunConfig, spec: ScenarioSpec, variant: Variant, item: CaptureItem, output: Any, context: CaptureContext | None = None, previous_source: str = "oracle") -> dict[str, Any]:
    """Assemble the capture record of one pipeline output in the shape read by analysis.metrics.

    intended_kind states whether the turn intends a record, is ambiguous in one field, or
    must yield no record; previous_record_source and previous_record state the dialogue
    state the pipeline was given.
    """
    intended = variant.intended
    intended_dict: dict[str, Any] | None = None
    ambiguous_field: str | None = None
    intended_kind = "no_record"
    if isinstance(intended, IntendedRecord):
        intended_dict = {"payee_id": intended.payee_id, "amount_minor": intended.amount_minor, "currency": intended.currency, "total_max_minor": intended.total_max_minor}
        intended_kind = "record"
    elif isinstance(intended, Ambiguous):
        ambiguous_field = intended.field
        intended_kind = "ambiguous"
    produced = None
    if output.record is not None and output.kind == "record":
        produced = {"payee_id": output.record.payee_id, "amount_minor": output.record.amount_minor, "currency": output.record.currency, "total_max_minor": output.record.total_max_minor, "action_type": output.record.action_type}
    row: dict[str, Any] = {
        "record": "capture",
        "run_id": cfg.run_id,
        "config_hash": cfg.config_hash,
        "code_hash": cfg.code_hash,
        "scenario_id": spec.seed_id,
        "seed_id": spec.seed_id,
        "variant_id": variant.variant_id,
        "pair_id": variant.pair_id,
        "base_variant_id": variant.base_variant_id,
        "family": spec.family,
        "phenomena": list(spec.phenomena),
        "lang": variant.lang,
        "kind": variant.kind,
        "edit_kind": variant.edit_kind,
        "turn_idx": variant.turn_idx,
        "action_type": variant.action_type,
        "text": variant.text,
        "pipeline": item.pipeline,
        "model": item.model,
        "intended": intended_dict,
        "ambiguous_field": ambiguous_field,
        "intended_kind": intended_kind,
        "intended_action_type": variant.action_type,
        "previous_record_source": previous_source,
        "previous_record": None if context is None else context.previous_record,
        "produced": produced,
        "ask_field": output.ask_field,
        "outcome_kind": output.kind,
        "reason": output.reason,
        "parse_failure": output.parse_failure,
        "translation": output.translation,
        "extracted_from": output.extracted_from,
        "model_text": output.model_text,
        "raw": output.raw.to_dict() if output.raw else None,
        "timings": dict(output.timings),
        "tokens": dict(output.tokens),
        "model_identity": dict(output.model),
    }
    row["correct"] = capture_correct(row)
    return row


def _load_runner(model_values: dict[str, Any]) -> dict[str, Any]:
    """Worker initialiser: load one model per process (the runtime is imported here only)."""
    from tasdiq.capture.model import LlamaRunner

    return {"runner": LlamaRunner.from_config(model_values)}


def _run_capture_batch(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entry: run the capture items of a batch with the process's loaded model."""
    runner = worker_state().get("runner")
    if runner is None:
        raise RuntimeError("no model loaded in this worker")
    cfg = make_config(payload["values"], source=Path(payload["source"]) if payload.get("source") else None)
    specs, variants, _ = load_benchmark(cfg)
    by_seed = {s.seed_id: s for s in specs}
    by_variant = {v.variant_id: v for v in variants}
    previous_source = str(cfg.get("previous_record_source", "oracle") or "oracle")
    if previous_source not in PREVIOUS_RECORD_SOURCES:
        raise ValueError(f"unknown previous_record_source: {previous_source}")
    cache: dict[str, Any] = worker_state().setdefault("previous_outputs", {})
    rows: list[dict[str, Any]] = []
    for raw in payload["items"]:
        item = CaptureItem(**raw)
        spec = by_seed[item.seed_id]
        variant = by_variant[item.variant_id]
        context = getattr(runner, "context", None)
        if isinstance(context, dict):
            context.update({"variant_id": variant.variant_id, "scenario_id": spec.seed_id, "pipeline": item.pipeline, "track": "capture", **cfg.meta()})
        call_cfg = dict(cfg.values)
        call_cfg.update({"default_currency": variant.default_currency, "max_tokens": int(cfg.get("capture_max_tokens", 256) or 256), "temperature": float(cfg.get("capture_temperature", 0.0) or 0.0), "seed": int(cfg.get("capture_seed", 1) or 1)})
        directory = directory_for(spec)
        previous = None
        if previous_source == "captured":
            previous = captured_previous_record(spec, variant, item, runner, directory, call_cfg, by_variant, cache)
        dialogue = context_for_variant(spec, variant, previous, previous_source)
        output = run_pipeline(item.pipeline, variant.text, dialogue, runner, directory, call_cfg)
        rows.append(capture_row(cfg, spec, variant, item, output, dialogue, previous_source))
    return {"rows": rows}


def aggregate_captures(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Capture summaries per model, pipeline, language, family, kind, edit kind and phenomenon."""
    out: dict[str, Any] = {"n_rows": len(rows), "groups": []}
    for keys in AGGREGATE_KEYS:
        for values, summary in capture_summary_by(list(rows), keys).items():
            out["groups"].append({"keys": dict(zip(keys, values)), "summary": summary.as_dict()})
    exploded = explode_phenomena(list(rows))
    for values, summary in capture_summary_by(exploded, ("model", "pipeline", "lang", "phenomenon")).items():
        out["groups"].append({"keys": dict(zip(("model", "pipeline", "lang", "phenomenon"), values)), "summary": summary.as_dict()})
    return out


def run_capture(
    cfg: RunConfig | Path | str,
    log: Callable[[str], None] | None = print,
    runner_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> Path:
    """Run the capture track for a configuration and return the run directory.

    runner_factory, when given, builds the model runner in the calling process (a
    scripted stand-in for tests) and the run proceeds with one worker.
    """
    if not isinstance(cfg, RunConfig):
        cfg = load_config(cfg)
    resume = bool(cfg.get("resume", True))
    run_dir = prepare_run_dir(cfg.results_dir / "capture" / cfg.run_id, resume)
    progress = Progress(run_dir / "done.jsonl") if resume else Progress(None)
    specs, variants, splits = load_benchmark(cfg)
    items = enumerate_capture_items(cfg, specs, variants, splits)
    selected = filter_by_model(items, cfg)
    pending = [it for it in selected if it.key not in progress]
    if log:
        log(f"capture: {len(specs)} seeds, {len(variants)} variants, {len(items)} items, {len(selected)} selected ({len(pending)} pending)")
    started = time.time()
    plan = plan_workers(cfg, log)
    workers = 1 if runner_factory is not None else int(cfg.get("capture_workers", plan.workers) or 1)
    peak = 0
    n_done = 0
    by_model: dict[str, list[CaptureItem]] = {}
    for it in pending:
        by_model.setdefault(it.model, []).append(it)
    batch_size = int(cfg.get("capture_batch_size", 8) or 8)
    for model_name, model_items in by_model.items():
        spec: ModelSpec = cfg.model(model_name)
        if runner_factory is None:
            model_values = spec.runner_config(int(cfg.get("n_threads", 3) or 3), int(cfg.get("capture_seed", 1) or 1), cfg.manifest_path, run_dir / "model_calls.jsonl")
            initializer: Callable[..., Any] | None = _load_runner
            initargs: tuple[Any, ...] = (model_values,)
        else:
            initializer = lambda values: {"runner": runner_factory(values)}  # noqa: E731
            initargs = (spec.to_dict(),)
        payloads = []
        for start in range(0, len(model_items), batch_size):
            batch = model_items[start : start + batch_size]
            payloads.append({"values": cfg.values, "source": cfg.source.as_posix() if cfg.source else None, "items": [it.to_dict() for it in batch]})
        for result in run_in_workers(_run_capture_batch, payloads, workers, initializer=initializer, initargs=initargs, plan=plan, log=log):
            peak = max(peak, int(result.get("peak_rss_bytes", 0)))
            rows = result["rows"]
            for row in rows:
                row["worker"] = result.get("worker")
                row["peak_rss_bytes"] = result.get("peak_rss_bytes")
            append_jsonl(run_dir / "captures.jsonl", rows)
            by_file: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                by_file.setdefault(f"{row['model']}__{row['pipeline']}.jsonl", []).append(row)
            for name, group in by_file.items():
                append_jsonl(run_dir / name, group)
            for row in rows:
                progress.mark(make_key(row["variant_id"], row["pipeline"], row["model"]))
                n_done += 1
            if log:
                log(f"capture: {n_done}/{len(pending)} items written")
    rows = read_jsonl(run_dir / "captures.jsonl")
    write_json(run_dir / "aggregates.json", aggregate_captures(rows))
    elapsed = time.time() - started
    manifest = {
        "track": "capture",
        "run_id": cfg.run_id,
        "config": cfg.to_dict(),
        "environment": cfg.environment(),
        "workers": plan.to_dict(),
        "peak_rss_bytes": peak,
        "n_seeds": len(specs),
        "n_variants": len(variants),
        "n_items": len(items),
        "n_items_selected": len(selected),
        "model_filter": cfg.get("model_filter"),
        "n_rows_written": len(rows),
        "n_skipped_batch_variants": sum(1 for v in variants if not capture_applicable(v)),
        "n_excluded_variants": len(excluded_variants(cfg)),
        "capture_splits": sorted(capture_splits_of(cfg)),
        "seed_subsets": capture_seed_subsets(cfg, specs, splits),
        "splits": dict(sorted(splits.items())),
        "elapsed_seconds": elapsed,
        "started": started,
        "ended": started + elapsed,
    }
    write_json(run_dir / "manifest.json", manifest)
    if log:
        log(f"capture: run {cfg.run_id} complete in {elapsed:.1f} s; {len(rows)} rows in {run_dir}")
    return run_dir
