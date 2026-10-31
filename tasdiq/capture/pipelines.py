"""Capture pipelines: extraction, canonicalisation and gating in named configurations.

``direct`` runs extract, canonicalise and gate; ``noncanon`` takes the model's own integer
amount and currency code and matches the display name exactly; ``nogate`` binds the first
candidates without asking; ``translate`` translates a non-English utterance to English with
the same model and then runs ``direct``; ``translate_mt`` does the same through an open
machine-translation model (Helsinki-NLP opus-mt-ar-en, loaded lazily from the Hugging Face
cache); ``english_reference`` runs ``direct`` on the English member of the variant. Every
pipeline returns a CaptureOutput with the record or the ask field, the raw extraction,
per-stage timings and token counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping, Sequence

from .canonicalise import CanonResult, canonicalise, canonicalise_noncanon
from .extract import (
    ExtractionParseError,
    RawExtraction,
    build_extraction_messages,
    directory_display_names,
    parse_raw_extraction,
    schema_for,
)
from .gate import AskOrRecord, CaptureRecord, ambiguity_gate, no_gate
from .model import DEFAULT_MT_MODEL, ModelResult, MarianTranslator, Runner, Translator

PIPELINE_NAMES = ("direct", "noncanon", "nogate", "translate", "translate_mt", "english_reference")
TRANSLATE_PIPELINES = ("translate", "translate_mt")
STAGES = ("translate", "model", "canonicalise", "gate")

_MT_TRANSLATORS: dict[tuple[str, int, bool], MarianTranslator] = {}


@dataclass
class CaptureContext:
    """Dialogue state handed to a pipeline.

    ``turns`` are earlier (speaker, text) pairs in the variant's language;
    ``previous_record`` is the record the dialogue currently refers to, if any; ``lang`` is
    "en", "ar" or "mix"; ``english_text`` is the English member of the same seed turn, used
    by the ``english_reference`` pipeline.
    """

    turns: list[tuple[str, str]] = field(default_factory=list)
    previous_record: Any = None
    lang: str = "en"
    english_text: str | None = None

    @classmethod
    def coerce(cls, value: Any) -> "CaptureContext":
        """Accept a CaptureContext, a mapping with its keys, or None."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            turns = [(str(s), str(t)) for s, t in (value.get("turns") or [])]
            return cls(
                turns=turns,
                previous_record=value.get("previous_record"),
                lang=str(value.get("lang") or "en"),
                english_text=value.get("english_text"),
            )
        return cls(
            turns=[(str(s), str(t)) for s, t in (getattr(value, "turns", None) or [])],
            previous_record=getattr(value, "previous_record", None),
            lang=str(getattr(value, "lang", None) or "en"),
            english_text=getattr(value, "english_text", None),
        )


@dataclass
class CaptureOutput:
    """Result of one pipeline run.

    ``kind`` is "record", "ask", "no_record" or "parse_failure". ``extracted_from`` is the
    text the extraction model saw (the translation for ``translate``, the English member for
    ``english_reference``). ``timings`` holds seconds per stage and ``tokens`` the summed
    prompt and completion tokens over all model calls.
    """

    pipeline: str
    kind: str
    record: CaptureRecord | None
    ask_field: str | None
    reason: str
    raw: RawExtraction | None
    canon: CanonResult | None
    model_text: str
    extracted_from: str
    translation: str | None
    timings: dict[str, float]
    tokens: dict[str, int]
    parse_failure: str | None
    model: dict[str, Any]
    lang: str
    source_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "kind": self.kind,
            "record": self.record.to_dict() if self.record else None,
            "ask_field": self.ask_field,
            "reason": self.reason,
            "raw": self.raw.to_dict() if self.raw else None,
            "canon": self.canon.to_dict() if self.canon else None,
            "model_text": self.model_text,
            "extracted_from": self.extracted_from,
            "translation": self.translation,
            "timings": dict(self.timings),
            "tokens": dict(self.tokens),
            "parse_failure": self.parse_failure,
            "model": dict(self.model),
            "lang": self.lang,
            "source_text": self.source_text,
        }


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _identity(runner: Any) -> dict[str, Any]:
    ident = getattr(runner, "identity", None)
    if callable(ident):
        try:
            return dict(ident())
        except Exception:
            return {}
    return {}


def _add_tokens(tokens: dict[str, int], result: ModelResult) -> None:
    tokens["prompt_tokens"] += int(result.prompt_tokens)
    tokens["completion_tokens"] += int(result.completion_tokens)


def resolve_mt_translator(cfg: Any, runner: Any = None) -> Translator:
    """Translator for the ``translate_mt`` pipeline.

    ``cfg`` may carry ``mt_translator`` (an object with ``translate`` and ``identity``, used
    as given); otherwise a MarianTranslator is built from ``mt_model`` (default
    Helsinki-NLP/opus-mt-ar-en), ``mt_n_threads`` (else ``n_threads``, else the runner's
    thread count), ``mt_seed`` (else ``seed``, else 1), ``mt_local_files_only`` (True),
    ``mt_manifest_path`` and ``mt_log_path``, and kept per process so that one loaded model
    serves every call with the same settings.
    """
    given = _cfg(cfg, "mt_translator", None)
    if given is not None:
        return given
    model_name = str(_cfg(cfg, "mt_model", DEFAULT_MT_MODEL))
    n_threads = int(_cfg(cfg, "mt_n_threads", _cfg(cfg, "n_threads", getattr(runner, "n_threads", 1))))
    local_only = bool(_cfg(cfg, "mt_local_files_only", True))
    key = (model_name, n_threads, local_only)
    translator = _MT_TRANSLATORS.get(key)
    if translator is None:
        seed = _cfg(cfg, "mt_seed", _cfg(cfg, "seed", None))
        translator = MarianTranslator(
            model_name,
            n_threads=n_threads,
            seed=1 if seed is None else int(seed),
            manifest_path=_cfg(cfg, "mt_manifest_path", None),
            log_path=_cfg(cfg, "mt_log_path", None),
            local_files_only=local_only,
        )
        _MT_TRANSLATORS[key] = translator
    return translator


def run_pipeline(
    name: str,
    utterance: str,
    ctx: Any,
    runner: Runner,
    directory: Any,
    cfg: Any,
) -> CaptureOutput:
    """Run one named pipeline on an utterance.

    ``cfg`` (mapping or object) supplies ``default_currency``, ``max_tokens`` (256),
    ``temperature`` (0.0), ``seed`` (None), the ``mt_*`` keys read by
    ``resolve_mt_translator`` and, for tests, ``amount_parser`` (a replacement for
    ``tasdiq.money.parse_amount``) and ``currencies`` (a replacement for
    ``tasdiq.money.CURRENCIES``).
    """
    if name not in PIPELINE_NAMES:
        raise ValueError(f"unknown pipeline: {name}")
    context = CaptureContext.coerce(ctx)
    default_currency = _cfg(cfg, "default_currency", None)
    max_tokens = int(_cfg(cfg, "max_tokens", 256))
    temperature = float(_cfg(cfg, "temperature", 0.0))
    seed = _cfg(cfg, "seed", None)
    amount_parser = _cfg(cfg, "amount_parser", None)
    currencies = _cfg(cfg, "currencies", None)
    timings: dict[str, float] = {stage: 0.0 for stage in STAGES}
    tokens: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    model_text = ""
    translation: str | None = None
    text = str(utterance)
    lang = context.lang
    translator_identity: dict[str, Any] | None = None

    if name in TRANSLATE_PIPELINES and context.lang != "en":
        translator: Any = runner if name == "translate" else resolve_mt_translator(cfg, runner)
        translated = translator.translate(text, context.lang, "en")
        timings["translate"] = float(translated.seconds)
        _add_tokens(tokens, translated)
        translation = translated.text
        text = translated.text
        lang = "en"
        translator_identity = _identity(translator)
    elif name == "english_reference":
        if context.english_text is not None:
            text = str(context.english_text)
        elif context.lang != "en":
            raise ValueError("english_reference needs the English member in ctx.english_text")
        lang = "en"

    variant = "noncanon" if name == "noncanon" else "canon"
    messages = build_extraction_messages(
        text, context.turns, directory_display_names(directory), lang, variant=variant
    )
    result = runner.chat(
        messages,
        json_schema=schema_for(variant),
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    timings["model"] = float(result.seconds)
    _add_tokens(tokens, result)
    model_text = result.text
    model_identity = _identity(runner)
    if translator_identity is not None:
        model_identity["translator"] = translator_identity

    def finish(
        kind: str,
        record: CaptureRecord | None,
        ask_field: str | None,
        reason: str,
        raw: RawExtraction | None,
        canon: CanonResult | None,
        parse_failure: str | None,
    ) -> CaptureOutput:
        timings["total"] = sum(timings[stage] for stage in STAGES)
        return CaptureOutput(
            pipeline=name,
            kind=kind,
            record=record,
            ask_field=ask_field,
            reason=reason,
            raw=raw,
            canon=canon,
            model_text=model_text,
            extracted_from=text,
            translation=translation,
            timings=timings,
            tokens=tokens,
            parse_failure=parse_failure,
            model=model_identity,
            lang=context.lang,
            source_text=str(utterance),
        )

    try:
        raw = parse_raw_extraction(result.text)
    except ExtractionParseError as exc:
        return finish("parse_failure", None, None, "parse_failure", None, None, str(exc))

    started = perf_counter()
    if variant == "noncanon":
        canon = canonicalise_noncanon(raw, directory, default_currency, context.previous_record)
    else:
        canon = canonicalise(
            raw,
            directory,
            default_currency,
            context.previous_record,
            utterance=text,
            amount_parser=amount_parser,
            currencies=currencies,
        )
    timings["canonicalise"] = perf_counter() - started

    started = perf_counter()
    outcome: AskOrRecord
    if name == "nogate":
        outcome = no_gate(canon, raw, default_currency)
    else:
        outcome = ambiguity_gate(canon, raw, context.previous_record, utterance=text, amount_parser=amount_parser)
    timings["gate"] = perf_counter() - started
    return finish(outcome.kind, outcome.record, outcome.ask_field, outcome.reason, raw, canon, None)


def run_pipelines(
    names: Sequence[str],
    utterance: str,
    ctx: Any,
    runner: Runner,
    directory: Any,
    cfg: Any,
) -> dict[str, CaptureOutput]:
    """Run several pipelines on the same utterance and return outputs keyed by name."""
    return {name: run_pipeline(name, utterance, ctx, runner, directory, cfg) for name in names}
