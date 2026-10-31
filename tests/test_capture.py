"""Tests for the capture stage: runner parsing, extraction prompts, canonicaliser, gate and pipelines.

A scripted FakeRunner returns RawExtraction JSON so that every stage after the model is
exercised without a model file, and a scripted FakeTranslator stands in for the machine
translation model. A small stand-in amount parser and currency table keep the suite
independent of tasdiq.money; the checks that need the real parser skip when that module is
absent. The check against the real opus-mt-ar-en model runs only when the environment
variable TASDIQ_MT_TESTS is set and the weights are in the Hugging Face cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tasdiq.capture.canonicalise import (
    canonicalise,
    canonicalise_noncanon,
    currency_code_from_text,
    exact_display_match,
)
from tasdiq.capture.extract import (
    NONCANON_SCHEMA,
    RAW_EXTRACTION_SCHEMA,
    ExtractionParseError,
    RawExtraction,
    build_extraction_messages,
    build_system_prompt,
    directory_display_names,
    parse_raw_extraction,
    schema_for,
)
from tasdiq.capture.gate import CaptureRecord, ambiguity_gate, no_gate
from tasdiq.capture.model import (
    DEFAULT_MT_MODEL,
    NO_THINK_SWITCH,
    LlamaRunner,
    ModelResult,
    MarianTranslator,
    clean_translation,
    load_manifest,
    model_sha256,
    mt_direction,
    mt_weights_path,
    parse_json_object,
    parse_tool_calls,
    prepare_messages,
    strip_think,
)
from tasdiq.capture.pipelines import (
    PIPELINE_NAMES,
    TRANSLATE_PIPELINES,
    CaptureContext,
    resolve_mt_translator,
    run_pipeline,
    run_pipelines,
)


# ----------------------------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------------------------

_EASTERN = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_EASTERN.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
_HARAKAT = re.compile("[\u0610-\u061a\u064b-\u0652\u0670\u0640]")
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_CURRENCY_WORDS = {
    "aed": "AED", "dirham": "AED", "dirhams": "AED", "dh": "AED", "dhs": "AED",
    "درهم": "AED", "دراهم": "AED", "درهما": "AED",
    "usd": "USD", "dollar": "USD", "dollars": "USD", "$": "USD", "دولار": "USD",
    "eur": "EUR", "euro": "EUR", "euros": "EUR", "€": "EUR",
    "kwd": "KWD", "dinar": "KWD", "دينار": "KWD",
    "sar": "SAR", "riyal": "SAR", "ريال": "SAR",
}
_MINOR_DIGITS = {"KWD": 3, "BHD": 3, "OMR": 3, "JOD": 3}
STUB_CURRENCIES = {
    "AED": SimpleNamespace(code="AED", names_en=["dirham", "dirhams"], names_ar=["درهم", "درهماً"], symbols=["د.إ"]),
    "USD": SimpleNamespace(code="USD", names_en=["dollar", "dollars"], names_ar=["دولار"], symbols=["$"]),
    "EUR": SimpleNamespace(code="EUR", names_en=["euro", "euros"], names_ar=["يورو"], symbols=["€"]),
}


def loose(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _HARAKAT.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    return " ".join(text.lower().strip(".,;:!?()[]\"'").split())


@dataclass
class FakePayee:
    payee_id: str
    name_en: str
    name_ar: str
    aliases: list[str] = field(default_factory=list)
    kind: str = "domestic"
    iban_masked: str = "AE** **** **** 1234"


class FakeDirectory:
    """Exact match after loose normalisation on names and aliases; all matches returned."""

    def __init__(self, payees: list[FakePayee]) -> None:
        self.payees = list(payees)

    def resolve(self, mention: str) -> list[FakePayee]:
        key = loose(mention)
        return [p for p in self.payees if key in {loose(x) for x in (p.name_en, p.name_ar, *p.aliases)}]

    def by_id(self, payee_id: str) -> FakePayee:
        return next(p for p in self.payees if p.payee_id == payee_id)


@dataclass
class StubParse:
    amount_minor: int | None
    currency: str | None
    matched_text: str
    candidates: list[tuple[int, str | None]]
    flags: set[str]


def stub_parse_amount(text: str, default_currency: str | None = None) -> StubParse:
    """Digits in either script, one currency word, thousands and decimal separators."""
    norm = text.translate(_EASTERN).replace("\u066b", ".").replace("\u066c", ",")
    flags: set[str] = set()
    if norm != text:
        flags.add("digit_script_ar")
    currency: str | None = None
    for token in re.findall(r"[^\s\d.,]+", norm):
        key = loose(token)
        if key in _CURRENCY_WORDS:
            currency = _CURRENCY_WORDS[key]
            break
    if currency is None:
        currency = default_currency
        flags.add("no_currency")
    minor_digits = _MINOR_DIGITS.get(currency or "", 2)
    candidates: list[tuple[int, str | None]] = []
    for match in _NUMBER.finditer(norm):
        token = match.group(0).replace(",", "")
        if "." in token:
            whole, frac = token.split(".")
            if len(frac) > minor_digits:
                flags.add("unrecognised_fraction")
                continue
            flags.add("decimal")
            minor = int(whole) * 10**minor_digits + int(frac.ljust(minor_digits, "0"))
        else:
            minor = int(token) * 10**minor_digits
        candidates.append((minor, currency))
    distinct = {c[0] for c in candidates}
    amount: int | None = None
    if len(distinct) > 1:
        flags.add("multiple_amounts")
    elif len(distinct) == 1 and "unrecognised_fraction" not in flags:
        amount = candidates[0][0]
    return StubParse(amount, currency, text, candidates, flags)


class FakeRunner:
    """Scripted stand-in for LlamaRunner returning the next RawExtraction JSON per chat call."""

    def __init__(self, extractions: list[Any], translations: dict[str, str] | None = None) -> None:
        self.script = list(extractions)
        self.translations = dict(translations or {})
        self.chat_calls: list[dict[str, Any]] = []
        self.translate_calls: list[tuple[str, str, str]] = []

    def chat(self, messages, tools=None, json_schema=None, max_tokens=256, temperature=0.0, seed=None) -> ModelResult:
        self.chat_calls.append(
            {"messages": messages, "tools": tools, "json_schema": json_schema,
             "max_tokens": max_tokens, "temperature": temperature, "seed": seed}
        )
        item = self.script.pop(0)
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return ModelResult(text=text, raw_text=text, tool_calls=[], prompt_tokens=100,
                         completion_tokens=40, seconds=0.01, finish_reason="stop", kind="chat")

    def translate(self, text: str, src_lang: str, dst_lang: str) -> ModelResult:
        self.translate_calls.append((text, src_lang, dst_lang))
        out = self.translations.get(text, text)
        return ModelResult(text=out, raw_text=out, tool_calls=[], prompt_tokens=50,
                         completion_tokens=20, seconds=0.02, finish_reason="stop", kind="translate")

    def identity(self) -> dict[str, Any]:
        return {"model_file": "scripted", "model_sha256": "0" * 64, "n_threads": 1, "seed": 1}


class FakeTranslator:
    """Scripted stand-in for MarianTranslator with the Translator interface."""

    def __init__(self, translations: dict[str, str] | None = None) -> None:
        self.translations = dict(translations or {})
        self.calls: list[tuple[str, str, str]] = []

    def translate(self, text: str, src_lang: str, dst_lang: str) -> ModelResult:
        self.calls.append((text, src_lang, dst_lang))
        out = self.translations.get(text, text)
        return ModelResult(text=out, raw_text=out, tool_calls=[], prompt_tokens=30,
                         completion_tokens=12, seconds=0.03, finish_reason="stop", kind="translate_mt")

    def identity(self) -> dict[str, Any]:
        return {"backend": "scripted", "model_name": "opus-mt-test", "model_sha256": "1" * 64}


def raw(**kwargs: Any) -> RawExtraction:
    base: dict[str, Any] = {"action_type": "payment", "payee_mention": "Omar Khalid",
                            "amount_text": "250 dirhams", "currency_text": None,
                            "negated": False, "refers_to_previous": False, "uncertain_fields": []}
    base.update(kwargs)
    return RawExtraction(**base)


def extraction(**kwargs: Any) -> dict[str, Any]:
    return raw(**kwargs).to_dict()


@pytest.fixture
def directory() -> FakeDirectory:
    return FakeDirectory([
        FakePayee("P-114", "Omar Khalid", "عمر خالد", ["omar", "3omar"]),
        FakePayee("P-118", "Omar Saeed", "عمر سعيد", ["omar"]),
        FakePayee("P-207", "Sara Ahmed", "سارة أحمد", ["sara", "سارة"]),
        FakePayee("P-301", "Etisalat", "اتصالات", ["etisalat bill"], kind="merchant"),
    ])


@pytest.fixture
def cfg() -> dict[str, Any]:
    return {"default_currency": "AED", "amount_parser": stub_parse_amount, "currencies": STUB_CURRENCIES,
            "max_tokens": 128}


PREVIOUS = {"payee_id": "P-114", "amount_minor": 25000, "currency": "AED"}


def canon(r: RawExtraction, directory: FakeDirectory, default: str | None = "AED",
          previous: Any = None, utterance: str | None = None):
    return canonicalise(r, directory, default, previous, utterance=utterance, amount_parser=stub_parse_amount,
                        currencies=STUB_CURRENCIES)


# ----------------------------------------------------------------------------------------
# model.py
# ----------------------------------------------------------------------------------------

def test_strip_think_removes_closed_and_open_blocks() -> None:
    assert strip_think("<think>\nreasoning\n</think>\n\n{\"a\": 1}") == '{"a": 1}'
    assert strip_think("<think>cut off by the token limit") == ""
    assert strip_think("plain") == "plain"
    assert strip_think("") == ""


def test_parse_json_object_tolerates_doubled_braces_and_surrounding_text() -> None:
    artefact = ('<tool_call>\n{{"name": "send_payment", "arguments": {"payee_id": "P-114", '
                '"amount_minor": 25000, "currency": "AED"}}}}\n</tool_call>')
    obj = parse_json_object(artefact)
    assert obj["name"] == "send_payment"
    assert obj["arguments"] == {"payee_id": "P-114", "amount_minor": 25000, "currency": "AED"}
    assert parse_json_object('Sure: {"a": 1} trailing') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no object here")
    with pytest.raises(ValueError):
        parse_json_object("[1, 2]")


def test_parse_tool_calls_native_field_takes_precedence() -> None:
    native = [{"id": "c1", "type": "function",
               "function": {"name": "send_payment", "arguments": '{"payee_id": "P-114"}'}}]
    calls, errors = parse_tool_calls('<tool_call>{"name": "other", "arguments": {}}</tool_call>', native)
    assert errors == []
    assert [c.name for c in calls] == ["send_payment"]
    assert calls[0].arguments == {"payee_id": "P-114"}
    assert calls[0].call_id == "c1" and calls[0].source == "native"


def test_parse_tool_calls_from_tags_and_think_block() -> None:
    content = ("<think>plan</think>\n<tool_call>\n{\"name\": \"quote_fee\", \"arguments\": "
               "{\"payee_id\": \"P-114\"}}\n</tool_call>\n<tool_call>{\"name\": \"finish\", "
               "\"parameters\": {\"summary\": \"done\"}}</tool_call>")
    calls, errors = parse_tool_calls(content, None)
    assert errors == []
    assert [(c.name, c.source) for c in calls] == [("quote_fee", "tag"), ("finish", "tag")]
    assert calls[1].arguments == {"summary": "done"}


def test_parse_tool_calls_unterminated_tag_is_recovered_and_flagged() -> None:
    content = '<tool_call>\n{"name": "send_payment", "arguments": {"amount_minor": 25000}}'
    calls, errors = parse_tool_calls(content, None)
    assert len(calls) == 1 and calls[0].arguments["amount_minor"] == 25000
    assert any("unterminated" in e for e in errors)


def test_parse_tool_calls_bare_json_only_when_tools_expected() -> None:
    content = '{"name": "get_balance", "arguments": {}}'
    assert parse_tool_calls(content, None, expect_tools=False) == ([], [])
    calls, errors = parse_tool_calls(content, None, expect_tools=True)
    assert errors == [] and calls[0].name == "get_balance" and calls[0].source == "bare"
    calls, errors = parse_tool_calls('<tool_call>{"arguments": {}}</tool_call>', None)
    assert calls == [] and errors and "name" in errors[0]


def test_prepare_messages_adds_no_think_switch_once() -> None:
    messages = [{"role": "system", "content": "Task."}, {"role": "user", "content": "hi"}]
    out = prepare_messages(messages, True)
    assert out[0]["content"] == "Task.\n" + NO_THINK_SWITCH
    assert messages[0]["content"] == "Task."
    again = prepare_messages(out, True)
    assert again[0]["content"].count(NO_THINK_SWITCH) == 1
    inserted = prepare_messages([{"role": "user", "content": "hi"}], True)
    assert inserted[0] == {"role": "system", "content": NO_THINK_SWITCH}
    assert prepare_messages(messages, False) == messages


def test_clean_translation_strips_label_quotes_and_think() -> None:
    assert clean_translation('Translation: "Pay Omar 250 dirhams."') == "Pay Omar 250 dirhams."
    assert clean_translation("<think>x</think>\n«ادفع ٢٥٠ درهم»") == "ادفع ٢٥٠ درهم"
    assert clean_translation("plain") == "plain"


def test_load_manifest_accepts_list_and_mapping_layouts(tmp_path: Path) -> None:
    listed = tmp_path / "list.json"
    listed.write_text(json.dumps([{"file": "a.gguf", "sha256": "AB"}, {"path": "models/b.gguf", "sha256": "cd"}]),
                      encoding="utf-8")
    assert load_manifest(listed) == {"a.gguf": "ab", "b.gguf": "cd"}
    mapped = tmp_path / "map.json"
    mapped.write_text(json.dumps({"a.gguf": "ef", "b.gguf": {"sha256": "01"}}), encoding="utf-8")
    assert load_manifest(mapped) == {"a.gguf": "ef", "b.gguf": "01"}
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"repo": "x/y", "files": [{"file": "pytorch_model.bin", "sha256": "E9"}]}),
                       encoding="utf-8")
    assert load_manifest(wrapped) == {"pytorch_model.bin": "e9"}
    assert load_manifest(tmp_path / "missing.json") == {}
    assert load_manifest(None) == {}


def test_model_sha256_prefers_manifest_and_computes_otherwise(tmp_path: Path) -> None:
    listed = tmp_path / "listed.gguf"
    listed.write_bytes(b"listed model bytes")
    unlisted = tmp_path / "unlisted.gguf"
    unlisted.write_bytes(b"unlisted model bytes")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"file": "listed.gguf", "sha256": "f" * 64}]), encoding="utf-8")
    assert model_sha256(listed, manifest) == "f" * 64
    assert model_sha256(unlisted, manifest) == hashlib.sha256(b"unlisted model bytes").hexdigest()
    assert model_sha256(tmp_path / "absent.gguf", manifest) == "unknown"


class FakeLlama:
    """Backend double recording constructor and completion arguments."""

    instances: list["FakeLlama"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.requests: list[dict[str, Any]] = []
        self.content = ("<think>\n\n</think>\n\n<tool_call>\n{\"name\": \"send_payment\", "
                        "\"arguments\": {\"payee_id\": \"P-114\", \"amount_minor\": 25000}}\n</tool_call>")
        FakeLlama.instances.append(self)

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        return {"choices": [{"message": {"content": self.content, "tool_calls": None},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 290, "completion_tokens": 45}}


def _runner(tmp_path: Path, name: str = "Qwen3-test-Q4_K_M.gguf", **extra: Any) -> LlamaRunner:
    model = tmp_path / name
    model.write_bytes(b"model " + name.encode())
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps([{"file": name, "sha256": "a" * 64}]), encoding="utf-8")
    return LlamaRunner(model, n_threads=3, n_ctx=2048, seed=7, manifest_path=manifest,
                       log_path=tmp_path / "logs" / "model.jsonl", llama_factory=FakeLlama, **extra)


def test_llama_runner_chat_parses_logs_and_applies_no_think(tmp_path: Path) -> None:
    FakeLlama.instances.clear()
    runner = _runner(tmp_path)
    assert runner.no_think is True
    runner.context = {"scenario_id": "S1", "variant_id": "V1", "arm": "Tasdiq"}
    tools = [{"type": "function", "function": {"name": "send_payment", "parameters": {"type": "object"}}}]
    result = runner.chat([{"role": "system", "content": "Task."}, {"role": "user", "content": "Pay Omar 250."}],
                         tools=tools, max_tokens=64, temperature=0.0)
    assert "<think>" not in result.text and result.raw_text.startswith("<think>")
    assert [c.name for c in result.tool_calls] == ["send_payment"]
    assert result.tool_calls[0].arguments["amount_minor"] == 25000
    assert (result.prompt_tokens, result.completion_tokens, result.finish_reason) == (290, 45, "stop")
    assert result.seconds >= 0.0 and result.call_id == 1
    backend = FakeLlama.instances[-1]
    assert backend.kwargs["n_threads"] == 3 and backend.kwargs["seed"] == 7 and backend.kwargs["n_ctx"] == 2048
    request = backend.requests[0]
    assert request["messages"][0]["content"].endswith(NO_THINK_SWITCH)
    assert request["tool_choice"] == "auto" and request["tools"] == tools
    assert request["seed"] == 7 and request["max_tokens"] == 64 and "response_format" not in request
    runner.chat([{"role": "user", "content": "x"}], json_schema=RAW_EXTRACTION_SCHEMA, seed=11)
    second = backend.requests[1]
    assert second["response_format"] == {"type": "json_object", "schema": RAW_EXTRACTION_SCHEMA}
    assert second["seed"] == 11 and second["messages"][0]["role"] == "system"
    assert len(FakeLlama.instances) == 1 and runner.load_seconds is not None
    lines = [json.loads(l) for l in (tmp_path / "logs" / "model.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines == runner.calls
    first = lines[0]
    assert first["model_file"] == "Qwen3-test-Q4_K_M.gguf" and first["model_sha256"] == "a" * 64
    assert first["n_threads"] == 3 and first["call_seed"] == 7 and first["temperature"] == 0.0
    assert first["scenario_id"] == "S1" and first["arm"] == "Tasdiq" and first["tools"] == ["send_payment"]
    assert first["n_tool_calls"] == 1 and first["kind"] == "chat" and lines[1]["json_schema"] is True


def test_llama_runner_no_think_follows_model_name_and_override(tmp_path: Path) -> None:
    assert _runner(tmp_path, "qwen2.5-1.5b-instruct-q4_k_m.gguf").no_think is False
    assert _runner(tmp_path, "qwen2.5-other.gguf", no_think=True).no_think is True
    with pytest.raises(ValueError):
        _runner(tmp_path, "qwen3-mismatch.gguf", expected_sha256="b" * 64)


def test_llama_runner_from_config_reads_model_and_manifest(tmp_path: Path) -> None:
    name = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    (tmp_path / name).write_bytes(b"bytes")
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps([{"file": name, "sha256": "c" * 64}]), encoding="utf-8")
    cfg = {"model_dir": str(tmp_path), "model_file": name, "n_threads": 3, "n_ctx": 1024, "seed": 5,
           "manifest_path": str(manifest), "expected_sha256": "c" * 64, "log_path": str(tmp_path / "model.jsonl")}
    runner = LlamaRunner.from_config(cfg, llama_factory=FakeLlama)
    assert runner.model_path == tmp_path / name and runner.model_sha256 == "c" * 64
    assert (runner.n_threads, runner.n_ctx, runner.seed, runner.no_think) == (3, 1024, 5, False)
    assert runner.log_path == tmp_path / "model.jsonl" and runner._factory is FakeLlama
    with pytest.raises(ValueError):
        LlamaRunner.from_config(dict(cfg, expected_sha256="d" * 64))
    with pytest.raises(ValueError):
        LlamaRunner.from_config({"n_threads": 1})


def test_llama_runner_translate_uses_translation_prompt(tmp_path: Path) -> None:
    FakeLlama.instances.clear()
    runner = _runner(tmp_path, "qwen2.5-translate.gguf")
    backend_holder: list[FakeLlama] = []

    def factory(**kwargs: Any) -> FakeLlama:
        backend = FakeLlama(**kwargs)
        backend.content = 'Translation: "Pay Omar Khalid 250 dirhams."'
        backend_holder.append(backend)
        return backend

    runner._factory = factory
    result = runner.translate("ادفع ٢٥٠ درهم لعمر خالد", "ar", "en")
    assert result.text == "Pay Omar Khalid 250 dirhams." and result.kind == "translate"
    request = backend_holder[0].requests[0]
    assert "Modern Standard Arabic" in request["messages"][0]["content"]
    assert "English" in request["messages"][0]["content"]
    assert request["messages"][1]["content"] == "ادفع ٢٥٠ درهم لعمر خالد"
    assert runner.calls[-1]["kind"] == "translate"


class FakeMarianBackend:
    """Backend double for MarianTranslator recording constructor and generate arguments."""

    instances: list["FakeMarianBackend"] = []

    def __init__(self, model_name: str, n_threads: int, local_files_only: bool) -> None:
        self.model_name, self.n_threads, self.local_files_only = model_name, n_threads, local_files_only
        self.requests: list[dict[str, Any]] = []
        self.output = "Pay 250 dirhams to Omar Khalid."
        FakeMarianBackend.instances.append(self)

    def generate(self, text: str, num_beams: int, max_new_tokens: int, max_input_tokens: int,
                 seed: int) -> tuple[str, int, int]:
        self.requests.append({"text": text, "num_beams": num_beams, "max_new_tokens": max_new_tokens,
                              "max_input_tokens": max_input_tokens, "seed": seed})
        return self.output, 10, 9


def _mt_model_dir(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "opus-mt-ar-en"
    model_dir.mkdir()
    (model_dir / "pytorch_model.bin").write_bytes(b"weights")
    manifest = tmp_path / "mt_download.json"
    manifest.write_text(json.dumps({"repo": "x/opus-mt-ar-en",
                                    "files": [{"file": "pytorch_model.bin", "sha256": "e" * 64}]}),
                        encoding="utf-8")
    return model_dir, manifest


def test_marian_translator_loads_lazily_logs_and_checks_direction(tmp_path: Path) -> None:
    FakeMarianBackend.instances.clear()
    model_dir, manifest = _mt_model_dir(tmp_path)
    translator = MarianTranslator(str(model_dir), n_threads=2, seed=3, manifest_path=manifest,
                                  log_path=tmp_path / "mt.jsonl", backend_factory=FakeMarianBackend)
    assert translator.loaded is False and FakeMarianBackend.instances == []
    assert translator.direction == ("ar", "en") and translator.model_file == "pytorch_model.bin"
    assert translator.model_sha256 == "e" * 64 and translator.model_commit is None
    translator.context = {"scenario_id": "S1", "variant_id": "V1"}
    result = translator.translate("ادفع ٢٥٠ درهم لعمر خالد", "ar", "en")
    assert result.text == "Pay 250 dirhams to Omar Khalid." and result.kind == "translate_mt"
    assert (result.prompt_tokens, result.completion_tokens, result.finish_reason) == (10, 9, "stop")
    assert translator.loaded is True and translator.load_seconds is not None
    backend = FakeMarianBackend.instances[-1]
    assert (backend.model_name, backend.n_threads, backend.local_files_only) == (str(model_dir), 2, True)
    assert backend.requests[0] == {"text": "ادفع ٢٥٠ درهم لعمر خالد", "num_beams": 4, "max_new_tokens": 128,
                                   "max_input_tokens": 512, "seed": 3}
    translator.translate("حوّل ١٠٠ درهم", "mix", "en")
    assert len(FakeMarianBackend.instances) == 1
    with pytest.raises(ValueError):
        translator.translate("Pay Omar", "en", "ar")
    lines = [json.loads(l) for l in (tmp_path / "mt.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines == translator.calls
    first = lines[0]
    assert first["event"] == "mt_call" and first["model_sha256"] == "e" * 64 and first["n_threads"] == 2
    assert first["seed"] == 3 and first["num_beams"] == 4 and first["scenario_id"] == "S1"
    assert (first["src_lang"], first["dst_lang"], first["text"]) == ("ar", "en", "Pay 250 dirhams to Omar Khalid.")
    assert translator.identity()["model_name"] == str(model_dir)


def test_marian_translator_length_finish_and_unknown_model(tmp_path: Path) -> None:
    FakeMarianBackend.instances.clear()

    def factory(model_name: str, n_threads: int, local_files_only: bool) -> FakeMarianBackend:
        backend = FakeMarianBackend(model_name, n_threads, local_files_only)
        backend.generate = lambda text, num_beams, max_new_tokens, max_input_tokens, seed: ("x " * 8, 4, 8)
        return backend

    translator = MarianTranslator("some/other-model", max_new_tokens=8, backend_factory=factory)
    assert translator.direction is None and translator.model_sha256 == "unknown" and translator.model_file is None
    result = translator.translate("نص", "ar", "en")
    assert result.finish_reason == "length" and result.text == "x x x x x x x x"
    assert mt_direction("Helsinki-NLP/opus-mt-en-ar") == ("en", "ar") and mt_direction("plain") is None
    assert mt_weights_path(str(tmp_path / "absent")) is None
    assert DEFAULT_MT_MODEL == "Helsinki-NLP/opus-mt-ar-en"


@pytest.mark.skipif(not os.environ.get("TASDIQ_MT_TESTS"),
                    reason="set TASDIQ_MT_TESTS to run the cached opus-mt model")
def test_marian_translator_real_model_from_cache(tmp_path: Path) -> None:
    if mt_weights_path(DEFAULT_MT_MODEL) is None:
        pytest.skip("opus-mt-ar-en is not in the Hugging Face cache")
    translator = MarianTranslator(DEFAULT_MT_MODEL, n_threads=2, log_path=tmp_path / "mt.jsonl")
    assert translator.model_commit is not None and len(translator.model_sha256) == 64
    result = translator.translate("أرسل 300 dirhams لـ Omar غداً", "mix", "en")
    assert "300" in result.text and "Omar" in result.text and result.completion_tokens > 0
    again = translator.translate("أرسل 300 dirhams لـ Omar غداً", "mix", "en")
    assert again.text == result.text


def test_extraction_schemas_convert_to_llama_grammar() -> None:
    llama_grammar = pytest.importorskip("llama_cpp.llama_grammar")
    for schema in (RAW_EXTRACTION_SCHEMA, NONCANON_SCHEMA):
        grammar = llama_grammar.LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
        assert grammar is not None


# ----------------------------------------------------------------------------------------
# extract.py
# ----------------------------------------------------------------------------------------

def test_messages_show_names_and_aliases_but_never_payee_ids(directory: FakeDirectory) -> None:
    names = directory_display_names(directory)
    assert names == [["Omar Khalid", "عمر خالد", "omar", "3omar"], ["Omar Saeed", "عمر سعيد", "omar"],
                     ["Sara Ahmed", "سارة أحمد", "sara", "سارة"], ["Etisalat", "اتصالات", "etisalat bill"]]
    turns = [("user", "What is my balance?"), ("agent", "Your balance is 1,000.00 AED.")]
    messages = build_extraction_messages("ادفع ٢٥٠ درهم لعمر", turns, names, "ar")
    assert [m["role"] for m in messages] == ["system", "user"]
    joined = "\n".join(m["content"] for m in messages)
    for pid in ("P-114", "P-118", "P-207", "P-301"):
        assert pid not in joined
    for name in ("Omar Khalid", "عمر خالد", "3omar", "Sara Ahmed", "اتصالات"):
        assert name in messages[0]["content"]
    assert "ادفع ٢٥٠ درهم لعمر" in messages[1]["content"]
    assert "user: What is my balance?" in messages[1]["content"]
    assert "agent: Your balance is 1,000.00 AED." in messages[1]["content"]
    assert "Modern Standard Arabic" in messages[1]["content"]
    assert "do not guess" in messages[0]["content"].lower()
    assert "uncertain_fields" in messages[0]["content"]
    assert "المهمة" in messages[0]["content"]


def test_messages_accept_mapping_turns_and_flat_names() -> None:
    turns = [{"role": "user", "content": "hello"}, {"speaker": "agent", "text": ""}]
    messages = build_extraction_messages("Pay Sara 10 AED", turns, ["Sara Ahmed", "Omar Khalid"], None)
    assert "user: hello" in messages[1]["content"]
    assert "agent:" not in messages[1]["content"]
    assert "- Sara Ahmed" in messages[0]["content"] and "Language of the message" not in messages[1]["content"]


def test_noncanon_prompt_asks_for_integer_minor_units() -> None:
    canon_prompt = build_system_prompt([["Omar Khalid"]], "canon")
    noncanon_prompt = build_system_prompt([["Omar Khalid"]], "noncanon")
    assert "amount_text" in canon_prompt and "amount_minor" not in canon_prompt
    assert "amount_minor" in noncanon_prompt and "amount_text" not in noncanon_prompt
    assert schema_for("canon") is RAW_EXTRACTION_SCHEMA and schema_for("noncanon") is NONCANON_SCHEMA
    with pytest.raises(ValueError):
        schema_for("other")


def test_parse_raw_extraction_coerces_loose_values() -> None:
    text = ('<think>done</think>{"action_type": "Payment", "payee_mention": " Omar ", "amount_text": "",'
            ' "currency_text": null, "negated": "true", "refers_to_previous": 0,'
            ' "uncertain_fields": "amount_text", "amount_minor": "25,000", "currency": "aed"}')
    parsed = parse_raw_extraction(text)
    assert parsed.action_type == "payment" and parsed.payee_mention == "Omar"
    assert parsed.amount_text is None and parsed.negated is True and parsed.refers_to_previous is False
    assert parsed.uncertain_fields == ["amount_text"] and parsed.amount_minor == 25000 and parsed.currency == "AED"
    assert parse_raw_extraction('{"action_type": "unknown"}').action_type == "other"
    assert json.loads(json.dumps(parsed.to_dict()))["action_type"] == "payment"


def test_parse_raw_extraction_failures() -> None:
    with pytest.raises(ExtractionParseError):
        parse_raw_extraction("I cannot extract that.")
    with pytest.raises(ExtractionParseError):
        parse_raw_extraction('{"payee_mention": "Omar"}')


# ----------------------------------------------------------------------------------------
# canonicalise.py
# ----------------------------------------------------------------------------------------

def test_eastern_digits_amount_text_through_canonicaliser(directory: FakeDirectory) -> None:
    result = canon(raw(payee_mention="عمر خالد", amount_text="٢٥٠ درهم"), directory)
    assert (result.payee_id, result.amount_minor, result.currency) == ("P-114", 25000, "AED")
    assert result.amount_source == "amount_text" and result.currency_source == "amount_text"
    assert "digit_script_ar" in result.amount_flags and result.ask_field is None
    decimal = canon(raw(amount_text="٢٥٠٫٥٠ درهم"), directory)
    assert decimal.amount_minor == 25050


def test_eastern_digits_and_number_words_with_real_money_module(directory: FakeDirectory) -> None:
    pytest.importorskip("tasdiq.money")
    result = canonicalise(raw(payee_mention="عمر خالد", amount_text="٢٥٠ درهم"), directory, "AED", None)
    assert (result.payee_id, result.amount_minor, result.currency) == ("P-114", 25000, "AED")
    words = canonicalise(raw(amount_text="ألفان وخمسمئة درهم"), directory, "AED", None)
    assert (words.amount_minor, words.currency) == (250000, "AED")
    fils = canonicalise(raw(amount_text="250 dirhams and 50 fils"), directory, "AED", None)
    assert fils.amount_minor == 25050


def test_amount_from_utterance_when_amount_text_missing(directory: FakeDirectory) -> None:
    result = canon(raw(amount_text=None), directory, utterance="Pay Omar Khalid 250.50 dirhams please")
    assert result.amount_minor == 25050 and result.amount_source == "utterance"
    unparsed = canon(raw(amount_text="some"), directory, utterance="Pay Omar Khalid 300 dirhams")
    assert unparsed.amount_minor == 30000 and "amount_text_unparsed" in unparsed.notes
    none = canon(raw(amount_text=None), directory, utterance="Pay Omar Khalid please")
    assert none.amount_minor is None and none.ask_field == "amount_minor"


def test_currency_precedence_explicit_previous_default(directory: FakeDirectory) -> None:
    explicit = canon(raw(amount_text="250", currency_text="USD"), directory)
    assert (explicit.currency, explicit.currency_source) == ("USD", "explicit")
    default = canon(raw(amount_text="250"), directory)
    assert (default.currency, default.currency_source) == ("AED", "default")
    previous = canon(raw(action_type="revision", payee_mention=None, amount_text="300", refers_to_previous=True),
                     directory, previous={"payee_id": "P-114", "amount_minor": 25000, "currency": "USD"})
    assert (previous.currency, previous.currency_source, previous.amount_minor) == ("USD", "previous", 30000)
    adjacent = canon(raw(amount_text="300 euros", refers_to_previous=True), directory,
                     previous={"payee_id": "P-114", "amount_minor": 25000, "currency": "USD"})
    assert (adjacent.currency, adjacent.currency_source) == ("EUR", "amount_text")
    missing = canon(raw(amount_text="250"), directory, default=None)
    assert missing.currency is None and missing.ask_field == "currency"
    conflict = canon(raw(amount_text="250 dollars", currency_text="AED"), directory)
    assert conflict.currency == "USD" and conflict.amount_minor == 25000 and "currency_conflict" in conflict.notes


def test_refers_to_previous_resolves_missing_fields(directory: FakeDirectory) -> None:
    revision = canon(raw(action_type="revision", payee_mention=None, amount_text="300", refers_to_previous=True),
                     directory, previous=PREVIOUS)
    assert (revision.payee_id, revision.payee_source) == ("P-114", "previous")
    assert (revision.amount_minor, revision.currency) == (30000, "AED")
    cancel = canon(raw(action_type="cancellation", payee_mention=None, amount_text=None, refers_to_previous=False),
                   directory, previous=PREVIOUS, utterance="Cancel that")
    assert cancel.refers_to_previous is True
    assert (cancel.payee_id, cancel.amount_minor, cancel.currency) == ("P-114", 25000, "AED")
    assert cancel.amount_source == "previous" and cancel.currency_source == "previous"
    swap = canon(raw(payee_mention="Sara", amount_text=None, refers_to_previous=True), directory,
                 previous=PREVIOUS, utterance="send it to Sara instead")
    assert (swap.payee_id, swap.amount_minor, swap.currency) == ("P-207", 25000, "AED")
    absent = canon(raw(payee_mention=None, amount_text=None, refers_to_previous=True), directory,
                   utterance="make it the same")
    assert absent.had_previous is False and absent.payee_id is None


def test_multiple_payee_matches_recorded(directory: FakeDirectory) -> None:
    result = canon(raw(payee_mention="Omar"), directory)
    assert result.payee_matches == ["P-114", "P-118"] and result.payee_id is None
    assert result.ask_field == "payee_id"
    unknown = canon(raw(payee_mention="Khalid Omar"), directory)
    assert unknown.payee_matches == [] and unknown.payee_id is None
    arabizi = canon(raw(payee_mention="3omar"), directory)
    assert arabizi.payee_id == "P-114"


def test_noncanon_takes_model_integer_and_exact_display_name(directory: FakeDirectory) -> None:
    model = raw(payee_mention="Omar Khalid", amount_text=None, amount_minor=250, currency="aed")
    result = canonicalise_noncanon(model, directory, "AED", None)
    assert (result.payee_id, result.amount_minor, result.currency, result.mode) == ("P-114", 250, "AED", "noncanon")
    assert result.amount_source == "model" and result.currency_source == "model"
    lowercase = canonicalise_noncanon(raw(payee_mention="omar khalid", amount_minor=25000), directory, "AED", None)
    assert lowercase.payee_id is None and lowercase.payee_matches == []
    assert canon(raw(payee_mention="omar khalid"), directory).payee_id == "P-114"
    assert exact_display_match(directory, "عمر خالد") == ["P-114"]
    assert exact_display_match(directory, "omar") == ["P-114", "P-118"]
    inherited = canonicalise_noncanon(raw(action_type="revision", payee_mention=None, amount_minor=30000),
                                      directory, "AED", PREVIOUS)
    assert (inherited.payee_id, inherited.amount_minor, inherited.currency) == ("P-114", 30000, "AED")
    defaulted = canonicalise_noncanon(raw(payee_mention="Sara Ahmed", amount_minor=1000, currency="dirhams"),
                                      directory, "AED", None)
    assert defaulted.currency == "AED" and "currency_code_invalid" in defaulted.notes


def test_real_payee_directory_integration() -> None:
    payees_mod = pytest.importorskip("tasdiq.payees")
    real = payees_mod.PayeeDirectory([
        payees_mod.Payee("P-114", "Omar Khalid", "عمر خالد", ("omar", "3omar")),
        payees_mod.Payee("P-118", "Omar Saeed", "عمر سعيد", ("omar",)),
        payees_mod.Payee("P-207", "Sara Ahmed", "سارة أحمد", ("sara",)),
    ])
    groups = directory_display_names(real)
    assert groups == [["Omar Khalid", "عمر خالد", "omar", "3omar"], ["Omar Saeed", "عمر سعيد", "omar"],
                      ["Sara Ahmed", "سارة أحمد", "sara"]]
    assert "P-114" not in build_system_prompt(groups)
    result = canonicalise(raw(payee_mention="عُمر خالد", amount_text="٢٥٠ درهم"), real, "AED", None,
                          amount_parser=stub_parse_amount)
    assert (result.payee_id, result.amount_minor, result.currency) == ("P-114", 25000, "AED")
    shared = canonicalise(raw(payee_mention="Omar"), real, "AED", None, amount_parser=stub_parse_amount)
    assert shared.payee_matches == ["P-114", "P-118"] and shared.payee_id is None
    assert exact_display_match(real, "3omar") == ["P-114"]
    assert canonicalise_noncanon(raw(payee_mention="Sara Ahmed", amount_minor=1000), real, "AED", None).payee_id == "P-207"


def test_currency_code_from_text_with_explicit_table() -> None:
    table = {"AED": SimpleNamespace(code="AED", names_en=["dirham", "dirhams"], names_ar=["درهم", "درهماً"],
                                    symbols=["د.إ"]),
             "USD": SimpleNamespace(code="USD", names_en=["dollar", "dollars"], names_ar=["دولار"], symbols=["$"])}
    assert currency_code_from_text("Dirhams", table) == "AED"
    assert currency_code_from_text("درهما", table) == "AED"
    assert currency_code_from_text("د.إ", table) == "AED"
    assert currency_code_from_text("$", table) == "USD"
    assert currency_code_from_text("usd", table) == "USD"
    assert currency_code_from_text("EUR", table) is None
    assert currency_code_from_text("pebbles", table) is None
    assert currency_code_from_text(None, table) is None


# ----------------------------------------------------------------------------------------
# gate.py
# ----------------------------------------------------------------------------------------

def test_multiple_payee_matches_gives_ask_payee(directory: FakeDirectory) -> None:
    r = raw(payee_mention="Omar")
    out = ambiguity_gate(canon(r, directory), r)
    assert out.kind == "ask" and out.ask_field == "payee_id" and out.reason == "payee_ambiguous"
    unknown = raw(payee_mention="Khalid Omar")
    out = ambiguity_gate(canon(unknown, directory), unknown)
    assert out.is_ask and out.ask_field == "payee_id" and out.reason == "payee_unknown"
    missing = raw(payee_mention=None)
    out = ambiguity_gate(canon(missing, directory), missing)
    assert out.is_ask and out.reason == "payee_missing"


def test_inquiry_gives_no_record(directory: FakeDirectory) -> None:
    r = raw(action_type="inquiry")
    out = ambiguity_gate(canon(r, directory), r)
    assert out.kind == "no_record" and out.record is None and out.ask_field is None
    other = raw(action_type="other", payee_mention=None, amount_text=None)
    assert ambiguity_gate(canon(other, directory), other).kind == "no_record"
    unsure = raw(action_type="inquiry", uncertain_fields=["action_type"])
    out = ambiguity_gate(canon(unsure, directory), unsure)
    assert out.is_ask and out.ask_field == "action_type"


def test_negation_without_replacement_gives_ask(directory: FakeDirectory) -> None:
    bare = raw(negated=True, amount_text=None)
    out = ambiguity_gate(canon(bare, directory, utterance="No, not to Omar Khalid"), bare)
    assert out.is_ask and out.ask_field == "amount_minor"
    restated = raw(negated=True, payee_mention=None, amount_text="250", refers_to_previous=True)
    out = ambiguity_gate(canon(restated, directory, previous=PREVIOUS), restated, PREVIOUS)
    assert out.is_ask and out.ask_field == "amount_minor" and out.reason == "negated_without_replacement"
    payee_only = raw(negated=True, payee_mention="Omar Khalid", amount_text=None, refers_to_previous=True)
    out = ambiguity_gate(canon(payee_only, directory, previous=PREVIOUS, utterance="not Omar Khalid"),
                         payee_only, PREVIOUS)
    assert out.is_ask and out.ask_field == "payee_id" and out.reason == "negated_without_replacement"
    replaced = raw(negated=True, payee_mention=None, amount_text="300", refers_to_previous=True)
    out = ambiguity_gate(canon(replaced, directory, previous=PREVIOUS), replaced, PREVIOUS)
    assert out.is_record and out.record == CaptureRecord("payment", "P-114", 30000, "AED")
    fresh = raw(negated=True, payee_mention="Sara", amount_text="100 dirhams")
    out = ambiguity_gate(canon(fresh, directory), fresh)
    assert out.is_record and out.record.payee_id == "P-207"


def test_reference_without_previous_gives_ask(directory: FakeDirectory) -> None:
    r = raw(payee_mention=None, amount_text="300", refers_to_previous=True)
    out = ambiguity_gate(canon(r, directory, utterance="make it 300"), r)
    assert out.is_ask and out.ask_field == "reference"
    revision = raw(action_type="revision", payee_mention=None, amount_text="300")
    out = ambiguity_gate(canon(revision, directory), revision)
    assert out.is_ask and out.ask_field == "reference"


def test_uncertain_binding_field_gives_ask(directory: FakeDirectory) -> None:
    r = raw(uncertain_fields=["amount_text"])
    out = ambiguity_gate(canon(r, directory), r)
    assert out.is_ask and out.ask_field == "amount_minor" and out.reason == "uncertain:amount_minor"
    r = raw(uncertain_fields=["payee_mention"])
    assert ambiguity_gate(canon(r, directory), r).ask_field == "payee_id"
    r = raw(uncertain_fields=["currency_text"])
    assert ambiguity_gate(canon(r, directory), r).ask_field == "currency"
    r = raw(uncertain_fields=["something_else"])
    assert ambiguity_gate(canon(r, directory), r).is_record


def test_amount_problems_give_ask(directory: FakeDirectory) -> None:
    several = raw(amount_text="250 or 300 dirhams")
    out = ambiguity_gate(canon(several, directory), several)
    assert out.is_ask and out.ask_field == "amount_minor" and out.reason == "amount_multiple"
    fraction = raw(amount_text="250.555 dirhams")
    out = ambiguity_gate(canon(fraction, directory), fraction)
    assert out.is_ask and out.reason == "amount_fraction"
    missing = raw(amount_text=None)
    out = ambiguity_gate(canon(missing, directory, utterance="Pay Omar Khalid"), missing)
    assert out.is_ask and out.reason == "amount_missing"
    no_currency = raw(amount_text="250")
    out = ambiguity_gate(canon(no_currency, directory, default=None), no_currency)
    assert out.is_ask and out.ask_field == "currency"


def test_cancellation_resolves_previous_or_asks(directory: FakeDirectory) -> None:
    r = raw(action_type="cancellation", payee_mention=None, amount_text=None, refers_to_previous=True)
    out = ambiguity_gate(canon(r, directory, previous=PREVIOUS, utterance="cancel it"), r)
    assert out.is_record and out.record == CaptureRecord("cancellation", "P-114", 25000, "AED")
    out = ambiguity_gate(canon(r, directory, utterance="cancel it"), r)
    assert out.is_ask and out.ask_field == "reference"
    named = raw(action_type="cancellation", payee_mention="Sara", amount_text=None)
    out = ambiguity_gate(canon(named, directory, previous=PREVIOUS, utterance="cancel the payment to Sara"), named)
    assert out.is_record and out.record.payee_id == "P-207" and out.record.amount_minor is None
    ambiguous = raw(action_type="cancellation", payee_mention="Omar", amount_text=None)
    out = ambiguity_gate(canon(ambiguous, directory, previous=PREVIOUS, utterance="cancel Omar"), ambiguous)
    assert out.is_ask and out.ask_field == "payee_id"


def test_no_gate_binds_first_candidates_and_default_currency(directory: FakeDirectory) -> None:
    two = raw(payee_mention="Omar", amount_text="250 or 300")
    result = canon(two, directory, default=None)
    gated = ambiguity_gate(result, two)
    ungated = no_gate(result, two, "AED")
    assert gated.is_ask
    assert ungated.is_record and ungated.record == CaptureRecord("payment", "P-114", 25000, "AED")
    unsure = raw(uncertain_fields=["amount_text"], negated=True)
    assert no_gate(canon(unsure, directory), unsure, "AED").is_record
    inquiry = raw(action_type="inquiry")
    assert no_gate(canon(inquiry, directory), inquiry, "AED").kind == "no_record"
    empty = raw(payee_mention=None, amount_text=None)
    out = no_gate(canon(empty, directory, utterance="pay please"), empty, "AED")
    assert out.is_ask and out.ask_field == "payee_id" and out.reason == "incomplete"
    no_amount = raw(amount_text=None)
    out = no_gate(canon(no_amount, directory, utterance="pay Omar Khalid"), no_amount, "AED")
    assert out.is_ask and out.ask_field == "amount_minor"
    cancel = raw(action_type="cancellation", payee_mention=None, amount_text=None)
    assert no_gate(canon(cancel, directory, utterance="cancel"), cancel, "AED").ask_field == "reference"


# ----------------------------------------------------------------------------------------
# pipelines.py
# ----------------------------------------------------------------------------------------

AR_UTTERANCE = "ادفع ٢٥٠ درهم لعمر خالد"
AR_EXTRACTION = extraction(payee_mention="عمر خالد", amount_text="٢٥٠ درهم")


def test_direct_pipeline_produces_record_with_timings_and_tokens(directory: FakeDirectory, cfg: dict) -> None:
    runner = FakeRunner([AR_EXTRACTION])
    ctx = CaptureContext(turns=[("user", "ما رصيدي؟"), ("agent", "١٬٠٠٠٫٠٠ درهم")], lang="ar")
    out = run_pipeline("direct", AR_UTTERANCE, ctx, runner, directory, cfg)
    assert out.kind == "record" and out.record == CaptureRecord("payment", "P-114", 25000, "AED")
    assert out.ask_field is None and out.parse_failure is None and out.pipeline == "direct"
    assert out.raw.payee_mention == "عمر خالد" and out.canon.amount_source == "amount_text"
    assert set(out.timings) == {"translate", "model", "canonicalise", "gate", "total"}
    assert out.timings["model"] == pytest.approx(0.01) and out.timings["translate"] == 0.0
    assert out.timings["total"] >= out.timings["model"]
    assert out.tokens == {"prompt_tokens": 100, "completion_tokens": 40}
    assert out.model["model_file"] == "scripted" and out.lang == "ar"
    assert out.extracted_from == AR_UTTERANCE and out.translation is None
    call = runner.chat_calls[0]
    assert call["json_schema"] is RAW_EXTRACTION_SCHEMA and call["max_tokens"] == 128
    joined = "\n".join(m["content"] for m in call["messages"])
    assert "P-114" not in joined and "عمر خالد" in joined and "user: ما رصيدي؟" in joined
    assert runner.translate_calls == []


def test_noncanon_pipeline_differs_from_direct(directory: FakeDirectory, cfg: dict) -> None:
    direct = run_pipeline("direct", AR_UTTERANCE, {"lang": "ar"}, FakeRunner([AR_EXTRACTION]), directory, cfg)
    model_output = {"action_type": "payment", "payee_mention": "Omar Khalid", "amount_minor": 250,
                    "currency": "AED", "negated": False, "refers_to_previous": False, "uncertain_fields": []}
    runner = FakeRunner([model_output])
    noncanon = run_pipeline("noncanon", AR_UTTERANCE, {"lang": "ar"}, runner, directory, cfg)
    assert direct.record.amount_minor == 25000 and noncanon.record.amount_minor == 250
    assert noncanon.canon.mode == "noncanon" and runner.chat_calls[0]["json_schema"] is NONCANON_SCHEMA
    assert "amount_minor" in runner.chat_calls[0]["messages"][0]["content"]
    spelled = dict(model_output, payee_mention="omar khalid")
    lowered = run_pipeline("noncanon", AR_UTTERANCE, {"lang": "ar"}, FakeRunner([spelled]), directory, cfg)
    assert lowered.kind == "ask" and lowered.ask_field == "payee_id"
    canon_spelled = run_pipeline("direct", AR_UTTERANCE, {"lang": "ar"},
                                 FakeRunner([extraction(payee_mention="omar khalid", amount_text="250")]),
                                 directory, cfg)
    assert canon_spelled.kind == "record" and canon_spelled.record.payee_id == "P-114"


def test_nogate_pipeline_differs_from_direct(directory: FakeDirectory, cfg: dict) -> None:
    script = extraction(payee_mention="Omar", amount_text="250 or 300")
    gated = run_pipeline("direct", "Pay Omar 250 or 300", {"lang": "en"}, FakeRunner([script]), directory, cfg)
    ungated = run_pipeline("nogate", "Pay Omar 250 or 300", {"lang": "en"}, FakeRunner([script]), directory, cfg)
    assert gated.kind == "ask" and gated.ask_field == "payee_id"
    assert ungated.kind == "record" and ungated.record == CaptureRecord("payment", "P-114", 25000, "AED")


def test_translate_pipeline_translates_non_english_only(directory: FakeDirectory, cfg: dict) -> None:
    english = "Pay 250 dirhams to Omar Khalid"
    runner = FakeRunner([extraction(payee_mention="Omar Khalid", amount_text="250 dirhams")],
                        translations={AR_UTTERANCE: english})
    out = run_pipeline("translate", AR_UTTERANCE, {"lang": "ar"}, runner, directory, cfg)
    assert runner.translate_calls == [(AR_UTTERANCE, "ar", "en")]
    assert out.translation == english and out.extracted_from == english and out.source_text == AR_UTTERANCE
    assert out.record == CaptureRecord("payment", "P-114", 25000, "AED")
    assert out.timings["translate"] == pytest.approx(0.02)
    assert out.tokens == {"prompt_tokens": 150, "completion_tokens": 60}
    assert english in runner.chat_calls[0]["messages"][1]["content"]
    assert "English" in runner.chat_calls[0]["messages"][1]["content"]
    plain = FakeRunner([extraction(payee_mention="Omar Khalid", amount_text="250 dirhams")])
    out_en = run_pipeline("translate", english, {"lang": "en"}, plain, directory, cfg)
    assert plain.translate_calls == [] and out_en.translation is None and out_en.kind == "record"


def test_translate_mt_pipeline_uses_the_mt_translator_not_the_runner(directory: FakeDirectory, cfg: dict) -> None:
    english = "Pay 250 dirhams to Omar Khalid"
    translator = FakeTranslator({AR_UTTERANCE: english})
    runner = FakeRunner([extraction(payee_mention="Omar Khalid", amount_text="250 dirhams")],
                        translations={AR_UTTERANCE: "runner translation"})
    out = run_pipeline("translate_mt", AR_UTTERANCE, {"lang": "ar"}, runner, directory,
                       dict(cfg, mt_translator=translator))
    assert translator.calls == [(AR_UTTERANCE, "ar", "en")] and runner.translate_calls == []
    assert out.pipeline == "translate_mt" and out.translation == english and out.extracted_from == english
    assert out.record == CaptureRecord("payment", "P-114", 25000, "AED") and out.lang == "ar"
    assert out.timings["translate"] == pytest.approx(0.03)
    assert out.tokens == {"prompt_tokens": 130, "completion_tokens": 52}
    assert out.model["model_file"] == "scripted" and out.model["translator"]["backend"] == "scripted"
    assert english in runner.chat_calls[0]["messages"][1]["content"]
    plain = FakeRunner([extraction(payee_mention="Omar Khalid", amount_text="250 dirhams")])
    out_en = run_pipeline("translate_mt", english, {"lang": "en"}, plain, directory,
                          dict(cfg, mt_translator=translator))
    assert len(translator.calls) == 1 and out_en.translation is None and "translator" not in out_en.model
    assert TRANSLATE_PIPELINES == ("translate", "translate_mt") and "translate_mt" in PIPELINE_NAMES


def test_resolve_mt_translator_prefers_cfg_object_and_caches_per_model(tmp_path: Path) -> None:
    given = FakeTranslator()
    assert resolve_mt_translator({"mt_translator": given}) is given
    model_dir, manifest = _mt_model_dir(tmp_path)
    cfg = {"mt_model": str(model_dir), "mt_n_threads": 2, "mt_manifest_path": str(manifest), "mt_seed": 4}
    first = resolve_mt_translator(cfg, SimpleNamespace(n_threads=3))
    second = resolve_mt_translator(cfg)
    assert first is second and isinstance(first, MarianTranslator) and first.loaded is False
    assert (first.n_threads, first.seed, first.model_sha256, first.local_files_only) == (2, 4, "e" * 64, True)
    other = resolve_mt_translator({"mt_model": str(model_dir)}, SimpleNamespace(n_threads=3))
    assert other is not first and other.n_threads == 3 and other.seed == 1


def test_english_reference_pipeline_uses_english_member(directory: FakeDirectory, cfg: dict) -> None:
    english = "Pay 250 dirhams to Omar Khalid"
    runner = FakeRunner([extraction(payee_mention="Omar Khalid", amount_text="250 dirhams")])
    ctx = CaptureContext(lang="ar", english_text=english)
    out = run_pipeline("english_reference", AR_UTTERANCE, ctx, runner, directory, cfg)
    assert out.extracted_from == english and out.source_text == AR_UTTERANCE and out.lang == "ar"
    assert english in runner.chat_calls[0]["messages"][1]["content"]
    assert AR_UTTERANCE not in runner.chat_calls[0]["messages"][1]["content"]
    assert out.record == CaptureRecord("payment", "P-114", 25000, "AED")
    with pytest.raises(ValueError):
        run_pipeline("english_reference", AR_UTTERANCE, {"lang": "ar"}, FakeRunner([AR_EXTRACTION]), directory, cfg)
    same = run_pipeline("english_reference", english, {"lang": "en"},
                        FakeRunner([extraction(payee_mention="Omar Khalid", amount_text="250 dirhams")]),
                        directory, cfg)
    assert same.extracted_from == english


def test_pipeline_reports_parse_failure(directory: FakeDirectory, cfg: dict) -> None:
    out = run_pipeline("direct", AR_UTTERANCE, {"lang": "ar"}, FakeRunner(["I am not sure."]), directory, cfg)
    assert out.kind == "parse_failure" and out.record is None and out.raw is None and out.canon is None
    assert out.parse_failure and out.model_text == "I am not sure."
    assert out.timings["canonicalise"] == 0.0 and out.timings["gate"] == 0.0


def test_pipeline_uses_previous_record_from_context(directory: FakeDirectory, cfg: dict) -> None:
    script = extraction(action_type="revision", payee_mention=None, amount_text="٣٠٠", refers_to_previous=True)
    ctx = {"lang": "ar", "previous_record": PREVIOUS, "turns": [("user", AR_UTTERANCE)]}
    out = run_pipeline("direct", "اجعلها ٣٠٠", ctx, FakeRunner([script]), directory, cfg)
    assert out.record == CaptureRecord("revision", "P-114", 30000, "AED")
    out = run_pipeline("direct", "اجعلها ٣٠٠", {"lang": "ar"}, FakeRunner([script]), directory, cfg)
    assert out.kind == "ask" and out.ask_field == "reference"


def test_unknown_pipeline_and_output_serialisation(directory: FakeDirectory, cfg: dict) -> None:
    with pytest.raises(ValueError):
        run_pipeline("mystery", AR_UTTERANCE, None, FakeRunner([AR_EXTRACTION]), directory, cfg)
    out = run_pipeline("direct", AR_UTTERANCE, None, FakeRunner([AR_EXTRACTION]), directory, cfg)
    encoded = json.dumps(out.to_dict(), ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["record"]["amount_minor"] == 25000 and decoded["canon"]["amount_flags"] == ["digit_script_ar"]
    assert decoded["raw"]["payee_mention"] == "عمر خالد" and decoded["kind"] == "record"


def test_run_pipelines_covers_every_name(directory: FakeDirectory, cfg: dict) -> None:
    english = "Pay 250 dirhams to Omar Khalid"
    noncanon_output = {"action_type": "payment", "payee_mention": "Omar Khalid", "amount_minor": 25000,
                       "currency": "AED", "negated": False, "refers_to_previous": False, "uncertain_fields": []}
    translated = extraction(payee_mention="Omar Khalid", amount_text="250 dirhams")
    scripts = [AR_EXTRACTION, noncanon_output, AR_EXTRACTION, translated, translated, translated]
    runner = FakeRunner(scripts, translations={AR_UTTERANCE: english})
    translator = FakeTranslator({AR_UTTERANCE: english})
    ctx = CaptureContext(lang="ar", english_text=english)
    outputs = run_pipelines(PIPELINE_NAMES, AR_UTTERANCE, ctx, runner, directory, dict(cfg, mt_translator=translator))
    assert list(outputs) == list(PIPELINE_NAMES)
    for name, out in outputs.items():
        assert out.pipeline == name and out.record == CaptureRecord("payment", "P-114", 25000, "AED"), name
    assert runner.script == [] and len(runner.translate_calls) == 1 and len(translator.calls) == 1


def test_capture_context_coerce_forms() -> None:
    assert CaptureContext.coerce(None) == CaptureContext()
    mapped = CaptureContext.coerce({"turns": [["user", "hi"]], "lang": "mix", "previous_record": PREVIOUS,
                                    "english_text": "hi"})
    assert mapped.turns == [("user", "hi")] and mapped.lang == "mix" and mapped.previous_record is PREVIOUS
    obj = CaptureContext.coerce(SimpleNamespace(turns=[("agent", "ok")], lang="en", previous_record=None,
                                                english_text=None))
    assert obj.turns == [("agent", "ok")] and obj.lang == "en"
    native = CaptureContext(lang="ar")
    assert CaptureContext.coerce(native) is native


# ----------------------------------------------------------------------------------------
# Negations without a replacement, conflicting currency text, minor-unit currency text
# ----------------------------------------------------------------------------------------


def test_negation_whose_only_amount_is_the_rejected_value_asks() -> None:
    from tasdiq.payees import Payee, PayeeDirectory

    real = PayeeDirectory([Payee("P-114", "Omar Khalid", "عمر خالد", ("Omar",))])
    previous = {"payee_id": "P-114", "amount_minor": 25000, "currency": "AED"}
    bare = raw(action_type="revision", negated=True, refers_to_previous=True, payee_mention=None, amount_text=None)
    canon_bare = canonicalise(bare, real, "AED", previous, utterance="no, not 200 dirhams")
    assert canon_bare.amount_minor == 20000 and canon_bare.amount_source == "utterance"
    out = ambiguity_gate(canon_bare, bare, previous, utterance="no, not 200 dirhams")
    assert out.is_ask and out.ask_field == "amount_minor" and out.reason == "negated_without_replacement"
    echoed = raw(action_type="revision", negated=True, refers_to_previous=True, payee_mention=None, amount_text="200 dirhams")
    canon_echoed = canonicalise(echoed, real, "AED", previous, utterance="no, not 200 dirhams")
    out = ambiguity_gate(canon_echoed, echoed, previous, utterance="no, not 200 dirhams")
    assert out.is_ask and out.reason == "negated_without_replacement"
    replaced = raw(action_type="revision", negated=True, refers_to_previous=True, payee_mention=None, amount_text="300")
    canon_replaced = canonicalise(replaced, real, "AED", previous, utterance="no, not 200, make it 300")
    out = ambiguity_gate(canon_replaced, replaced, previous, utterance="no, not 200, make it 300")
    assert out.is_record and out.record.amount_minor == 30000
    out = ambiguity_gate(canonicalise(replaced, real, "AED", previous), replaced, previous)
    assert out.is_record and out.record.amount_minor == 30000


def test_conflicting_currency_text_keeps_the_number_currency_and_asks() -> None:
    from tasdiq.payees import Payee, PayeeDirectory

    real = PayeeDirectory([Payee("P-114", "Omar Khalid", "عمر خالد", ("Omar",))])
    kuwaiti = raw(amount_text="1.500 دينار كويتي", currency_text="درهم")
    result = canonicalise(kuwaiti, real, "AED", None)
    assert (result.amount_minor, result.currency) == (1500, "KWD") and "currency_conflict" in result.notes
    out = ambiguity_gate(result, kuwaiti)
    assert out.is_ask and out.ask_field == "currency" and out.reason == "currency_conflict"
    dollars = raw(amount_text="$250", currency_text="AED")
    result = canonicalise(dollars, real, "AED", None)
    assert (result.amount_minor, result.currency) == (25000, "USD")
    assert ambiguity_gate(result, dollars).reason == "currency_conflict"
    agreeing = raw(amount_text="250 AED", currency_text="dirhams")
    result = canonicalise(agreeing, real, "AED", None)
    assert (result.amount_minor, result.currency) == (25000, "AED") and ambiguity_gate(result, agreeing).is_record


def test_minor_unit_currency_text_scales_the_amount() -> None:
    from tasdiq.payees import Payee, PayeeDirectory

    real = PayeeDirectory([Payee("P-114", "Omar Khalid", "عمر خالد", ("Omar",))])
    for word in ("fils", "فلس"):
        fils = raw(amount_text="50", currency_text=word)
        result = canonicalise(fils, real, "AED", None)
        assert (result.amount_minor, result.currency) == (50, "AED") and "currency_text_minor_unit" in result.notes
        assert ambiguity_gate(result, fils).is_record
    cents = raw(amount_text="50", currency_text="cents")
    result = canonicalise(cents, real, None, None)
    assert result.amount_minor == 50 and result.currency is None
    assert ambiguity_gate(result, cents).ask_field == "currency"
    result = canonicalise(raw(amount_text="50", currency_text="cents"), real, "USD", None)
    assert (result.amount_minor, result.currency) == (50, "USD")
