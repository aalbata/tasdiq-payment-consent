"""Model runner for local GGUF models through llama.cpp.

The runner wraps ``llama_cpp.Llama`` (imported only when a model is loaded, so that the rest
of the package works without the library or a model file), enforces a JSON schema through
the ``response_format`` grammar of llama.cpp, parses tool calls from the native
``tool_calls`` field or from ``<tool_call>`` tags, removes ``<think>`` blocks, disables
Qwen3 thinking with the ``/no_think`` switch, and writes one JSON line per call that names
the model file, its sha256 (from a manifest), the thread count, the seed and the
temperature. ``MarianTranslator`` offers the same ``translate`` interface through an open
machine-translation model (Helsinki-NLP opus-mt) loaded lazily from transformers, for the
``translate_mt`` pipeline.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence

Message = dict[str, Any]

LANG_NAMES: dict[str, str] = {
    "en": "English",
    "ar": "Modern Standard Arabic",
    "mix": "mixed Arabic and English",
}

NO_THINK_SWITCH = "/no_think"

TRANSLATE_SYSTEM_PROMPT = (
    "Translate the user's message from {src} into {dst}. Keep the meaning exactly. Keep every "
    "personal name, payee name, number, amount and currency: write numerals with the digits 0 "
    "to 9, keep number words as number words, keep currency words as currency words, and "
    "transliterate Arabic names into Latin letters. Do not add, omit, explain or answer "
    "anything. Reply with the translation only."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL)
_TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_TAG_OPEN = re.compile(r"<tool_call>\s*(.*)\Z", re.DOTALL)
_LEADING_LABEL = re.compile(r"^\s*(translation|الترجمة)\s*[:：]\s*", re.IGNORECASE)
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("«", "»"), ("`", "`"))


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    source: str = "native"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "source": self.source,
        }


@dataclass
class ModelResult:
    """Outcome of one model call.

    ``text`` is the content with ``<think>`` blocks removed; ``raw_text`` is the content as
    returned. ``tool_calls`` holds the parsed calls and ``tool_call_errors`` the reasons for
    any call that could not be parsed.
    """

    text: str
    raw_text: str
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    finish_reason: str
    tool_call_errors: list[str] = field(default_factory=list)
    call_id: int = 0
    kind: str = "chat"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": self.seconds,
            "finish_reason": self.finish_reason,
            "tool_call_errors": list(self.tool_call_errors),
            "call_id": self.call_id,
            "kind": self.kind,
        }


class Runner(Protocol):
    """Interface shared by LlamaRunner and scripted test doubles."""

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] | None = None,
        json_schema: Mapping[str, Any] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> ModelResult: ...

    def translate(self, text: str, src_lang: str, dst_lang: str) -> ModelResult: ...

    def identity(self) -> dict[str, Any]: ...


def strip_think(text: str) -> str:
    """Remove closed ``<think>...</think>`` blocks and any unterminated ``<think>`` tail."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _THINK_OPEN.sub("", cleaned)
    return cleaned.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text``.

    Leading text before the first brace and trailing text after the object are ignored, and
    doubled opening braces (``{{``), which small models sometimes emit inside tool-call tags,
    are collapsed before decoding. Raises ValueError when no object can be decoded.
    """
    if not isinstance(text, str):
        raise ValueError("not a string")
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    body = text[start:].lstrip()
    while body.startswith("{{"):
        body = body[1:]
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg} at position {exc.pos}") from exc
    if not isinstance(obj, dict):
        raise ValueError("JSON value is not an object")
    return obj


def _normalise_call(obj: Mapping[str, Any], call_id: str | None, source: str) -> ToolCall:
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("tool call without a name")
    args: Any = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        args = parse_json_object(args) if args.strip() else {}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("tool call arguments are not an object")
    return ToolCall(name=name, arguments=dict(args), call_id=call_id, source=source)


def parse_tool_calls(
    content: str,
    native_tool_calls: Sequence[Mapping[str, Any]] | None = None,
    expect_tools: bool = False,
) -> tuple[list[ToolCall], list[str]]:
    """Collect tool calls from the native field, from ``<tool_call>`` tags, or from bare JSON.

    The native field is taken first. Otherwise every closed ``<tool_call>`` tag in ``content``
    is decoded; an unterminated tag (output cut by the token limit) is attempted as well.
    When no tag is present and ``expect_tools`` is true, content that is itself a JSON object
    with ``name`` and ``arguments`` keys is accepted as a call. Returns the calls and the list
    of parse errors.
    """
    calls: list[ToolCall] = []
    errors: list[str] = []
    for item in native_tool_calls or []:
        fn = item.get("function") if isinstance(item, Mapping) else None
        if not isinstance(fn, Mapping):
            errors.append("native tool call without a function field")
            continue
        try:
            calls.append(_normalise_call(fn, item.get("id"), "native"))
        except ValueError as exc:
            errors.append(f"native: {exc}")
    if calls:
        return calls, errors
    text = strip_think(content or "")
    tagged = _TOOL_TAG.findall(text)
    if not tagged:
        open_match = _TOOL_TAG_OPEN.search(text)
        if open_match is not None:
            tagged = [open_match.group(1)]
            errors.append("unterminated <tool_call> tag")
    for index, body in enumerate(tagged):
        try:
            calls.append(_normalise_call(parse_json_object(body), f"tag-{index}", "tag"))
        except ValueError as exc:
            errors.append(f"tag: {exc}")
    if not tagged and expect_tools and text.startswith("{"):
        try:
            obj = parse_json_object(text)
        except ValueError:
            obj = None
        if obj is not None and "name" in obj and ("arguments" in obj or "parameters" in obj):
            try:
                calls.append(_normalise_call(obj, "bare-0", "bare"))
            except ValueError as exc:
                errors.append(f"bare: {exc}")
    return calls, errors


def prepare_messages(messages: Sequence[Message], no_think: bool) -> list[Message]:
    """Copy the messages and, when ``no_think`` is set, add the Qwen3 ``/no_think`` switch
    to the system prompt (creating one when the conversation has none)."""
    prepared = [copy.deepcopy(dict(m)) for m in messages]
    if not no_think:
        return prepared
    for message in prepared:
        if message.get("role") == "system":
            content = str(message.get("content") or "")
            if NO_THINK_SWITCH not in content:
                message["content"] = (content.rstrip() + "\n" + NO_THINK_SWITCH).strip()
            return prepared
    prepared.insert(0, {"role": "system", "content": NO_THINK_SWITCH})
    return prepared


def clean_translation(text: str) -> str:
    """Strip think blocks, a leading "Translation:" label and one pair of enclosing quotes."""
    cleaned = strip_think(text)
    cleaned = _LEADING_LABEL.sub("", cleaned).strip()
    for left, right in _QUOTE_PAIRS:
        if len(cleaned) >= 2 and cleaned.startswith(left) and cleaned.endswith(right):
            cleaned = cleaned[1:-1].strip()
            break
    return cleaned


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hex sha256 of a file read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path: Path | str | None) -> dict[str, str]:
    """Read a model manifest and return a mapping from model file name to sha256.

    Accepted layouts: a list of objects with ``file`` (or ``path``) and ``sha256`` keys, an
    object keyed by file name whose values are sha256 strings or objects with a ``sha256``
    key, or an object whose ``files`` or ``models`` key holds such a list. A missing or
    unreadable manifest yields an empty mapping.
    """
    if manifest_path is None:
        return {}
    path = Path(manifest_path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, Mapping):
        for wrapper in ("files", "models"):
            if isinstance(data.get(wrapper), list):
                data = data[wrapper]
                break
    table: dict[str, str] = {}
    entries: list[tuple[str, Any]] = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, Mapping):
                name = entry.get("file") or entry.get("path")
                if name:
                    entries.append((Path(str(name)).name, entry))
    elif isinstance(data, Mapping):
        entries = [(Path(str(k)).name, v) for k, v in data.items()]
    for name, value in entries:
        sha = value.get("sha256") if isinstance(value, Mapping) else value
        if isinstance(sha, str) and sha:
            table[name] = sha.lower()
    return table


_SHA_CACHE: dict[str, str] = {}


def model_sha256(model_path: Path | str, manifest_path: Path | str | None = None) -> str:
    """sha256 of a model file, taken from the manifest when listed there and computed
    otherwise; results are cached per process. Returns "unknown" when the file is absent
    and unlisted."""
    path = Path(model_path)
    key = str(path.resolve()) if path.exists() else str(path)
    cached = _SHA_CACHE.get(key)
    if cached:
        return cached
    sha = load_manifest(manifest_path).get(path.name)
    if not sha:
        sha = file_sha256(path) if path.is_file() else "unknown"
    _SHA_CACHE[key] = sha
    return sha


class LlamaRunner:
    """Chat and translation calls against one GGUF model on the CPU.

    ``llama_factory`` replaces the llama.cpp constructor (for tests); it receives the same
    keyword arguments and must return an object with ``create_chat_completion``.
    """

    def __init__(
        self,
        model_path: Path | str,
        n_threads: int,
        n_ctx: int = 4096,
        seed: int = 1,
        chat_format: str | None = None,
        *,
        manifest_path: Path | str | None = None,
        log_path: Path | str | None = None,
        n_batch: int = 256,
        no_think: bool | None = None,
        expected_sha256: str | None = None,
        verbose: bool = False,
        llama_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_file = self.model_path.name
        self.n_threads = int(n_threads)
        self.n_ctx = int(n_ctx)
        self.seed = int(seed)
        self.chat_format = chat_format
        self.n_batch = int(n_batch)
        self.verbose = verbose
        self.log_path = Path(log_path) if log_path is not None else None
        self.model_sha256 = model_sha256(self.model_path, manifest_path)
        if expected_sha256 and self.model_sha256 != expected_sha256.lower():
            raise ValueError(
                f"model {self.model_file} has sha256 {self.model_sha256}, expected {expected_sha256}"
            )
        self.no_think = ("qwen3" in self.model_file.lower()) if no_think is None else bool(no_think)
        self.context: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self.load_seconds: float | None = None
        self._factory = llama_factory
        self._llama: Any = None
        self._counter = 0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], **overrides: Any) -> "LlamaRunner":
        """Build a runner from a config mapping.

        Keys read: ``model_path`` (or ``model_file`` joined to ``model_dir``), ``n_threads``,
        ``n_ctx``, ``seed``, ``chat_format``, ``manifest_path``, ``log_path``,
        ``expected_sha256`` (or ``sha256``), ``no_think``, ``n_batch`` and ``verbose``.
        ``overrides`` take precedence over the config values.
        """
        values: dict[str, Any] = {}
        model_path: Any = cfg.get("model_path")
        if model_path is None and cfg.get("model_file"):
            model_dir = cfg.get("model_dir")
            model_file = str(cfg["model_file"])
            model_path = Path(str(model_dir)) / model_file if model_dir else Path(model_file)
        if model_path is None:
            raise ValueError("config names no model_path or model_file")
        values["model_path"] = model_path
        values["n_threads"] = int(cfg.get("n_threads", 1))
        for key in ("n_ctx", "seed", "chat_format", "manifest_path", "log_path", "no_think", "n_batch", "verbose"):
            if key in cfg:
                values[key] = cfg[key]
        sha = cfg.get("expected_sha256", cfg.get("sha256"))
        if sha:
            values["expected_sha256"] = str(sha)
        values.update(overrides)
        return cls(**values)

    def identity(self) -> dict[str, Any]:
        """Model identity fields carried by every log record."""
        return {
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "n_threads": self.n_threads,
            "n_ctx": self.n_ctx,
            "seed": self.seed,
            "chat_format": self.chat_format,
            "no_think": self.no_think,
        }

    def load(self) -> Any:
        """Load the model on first use and return the backend object."""
        if self._llama is None:
            kwargs = dict(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                seed=self.seed,
                chat_format=self.chat_format,
                verbose=self.verbose,
            )
            started = perf_counter()
            if self._factory is not None:
                self._llama = self._factory(**kwargs)
            else:
                from llama_cpp import Llama

                self._llama = Llama(**kwargs)
            self.load_seconds = perf_counter() - started
        return self._llama

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] | None = None,
        json_schema: Mapping[str, Any] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> ModelResult:
        """One chat completion; JSON output is grammar-constrained when ``json_schema`` is
        given, and tool calls are parsed from the response."""
        return self._call(messages, tools, json_schema, max_tokens, temperature, seed, "chat")

    def translate(self, text: str, src_lang: str, dst_lang: str, max_tokens: int = 256) -> ModelResult:
        """Translate ``text`` between languages; the translation is in ``ModelResult.text``."""
        system = TRANSLATE_SYSTEM_PROMPT.format(
            src=LANG_NAMES.get(src_lang, src_lang), dst=LANG_NAMES.get(dst_lang, dst_lang)
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        result = self._call(messages, None, None, max_tokens, 0.0, None, "translate")
        result.text = clean_translation(result.raw_text)
        return result

    def _call(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] | None,
        json_schema: Mapping[str, Any] | None,
        max_tokens: int,
        temperature: float,
        seed: int | None,
        kind: str,
    ) -> ModelResult:
        effective_seed = self.seed if seed is None else int(seed)
        kwargs: dict[str, Any] = {
            "messages": prepare_messages(messages, self.no_think),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": effective_seed,
        }
        tool_names: list[str] = []
        if tools:
            kwargs["tools"] = [dict(t) for t in tools]
            kwargs["tool_choice"] = "auto"
            tool_names = [
                str(t.get("function", {}).get("name", "")) for t in tools if isinstance(t, Mapping)
            ]
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object", "schema": dict(json_schema)}
        backend = self.load()
        started = perf_counter()
        response = backend.create_chat_completion(**kwargs)
        seconds = perf_counter() - started
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        native = message.get("tool_calls") or []
        usage = response.get("usage") or {}
        calls, errors = parse_tool_calls(content, native, expect_tools=bool(tools))
        self._counter += 1
        result = ModelResult(
            text=strip_think(content),
            raw_text=content,
            tool_calls=calls,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            seconds=seconds,
            finish_reason=str(choice.get("finish_reason") or "unknown"),
            tool_call_errors=errors,
            call_id=self._counter,
            kind=kind,
        )
        line: dict[str, Any] = {
            "event": "model_call",
            "call_id": self._counter,
            "kind": kind,
            **self.identity(),
            "call_seed": effective_seed,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "json_schema": json_schema is not None,
            "tools": tool_names,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "seconds": seconds,
            "finish_reason": result.finish_reason,
            "n_tool_calls": len(calls),
            "tool_call_errors": errors,
            "text": content,
        }
        line.update(self.context)
        self._write_log(line)
        return result

    def _write_log(self, line: dict[str, Any]) -> None:
        self.calls.append(line)
        append_log_line(self.log_path, line)


def append_log_line(log_path: Path | None, line: Mapping[str, Any]) -> None:
    """Append one JSON object to a JSON-lines file, creating parent directories; does
    nothing without a path."""
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(line), ensure_ascii=False) + "\n")


DEFAULT_MT_MODEL = "Helsinki-NLP/opus-mt-ar-en"
MT_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
MT_SOURCE_ALIASES: dict[str, str] = {"mix": "ar"}
_OPUS_DIRECTION = re.compile(r"opus-mt-([a-z]{2,3})-([a-z]{2,3})", re.IGNORECASE)
_SNAPSHOT_COMMIT = re.compile(r"snapshots[\\/]([0-9a-f]{40})")


class Translator(Protocol):
    """Interface of a translation backend: the model runner itself or a Marian model."""

    def translate(self, text: str, src_lang: str, dst_lang: str) -> ModelResult: ...

    def identity(self) -> dict[str, Any]: ...


def mt_weights_path(model_name: str) -> Path | None:
    """Path of the weights file of a Hugging Face model: inside ``model_name`` when it is a
    directory, otherwise from the local Hugging Face cache; None when not found."""
    local = Path(model_name)
    if local.is_dir():
        for filename in MT_WEIGHT_FILES:
            candidate = local / filename
            if candidate.is_file():
                return candidate
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    for filename in MT_WEIGHT_FILES:
        try:
            found = try_to_load_from_cache(model_name, filename)
        except Exception:
            return None
        if isinstance(found, str) and Path(found).is_file():
            return Path(found)
    return None


def mt_direction(model_name: str) -> tuple[str, str] | None:
    """Source and target language codes read from an opus-mt model name, or None."""
    match = _OPUS_DIRECTION.search(str(model_name))
    return (match.group(1).lower(), match.group(2).lower()) if match else None


class TransformersMarianBackend:
    """Tokeniser and model of a Marian translation model held through transformers."""

    def __init__(self, model_name: str, n_threads: int, local_files_only: bool) -> None:
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self._torch = torch
        torch.set_num_threads(int(n_threads))
        self.tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = MarianMTModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.eval()

    def generate(
        self, text: str, num_beams: int, max_new_tokens: int, max_input_tokens: int, seed: int
    ) -> tuple[str, int, int]:
        """Translate one text with beam search; returns the output text, the input length
        and the number of generated tokens (decoder start token excluded)."""
        torch = self._torch
        torch.manual_seed(int(seed))
        batch = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=int(max_input_tokens))
        with torch.no_grad():
            generated = self.model.generate(
                **batch, num_beams=int(num_beams), max_length=int(max_new_tokens) + 1, do_sample=False
            )
        output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return output, int(batch["input_ids"].shape[1]), max(int(generated.shape[1]) - 1, 0)


class MarianTranslator:
    """Translation calls through an open Marian model (Helsinki-NLP opus-mt) on the CPU.

    The model is read from the local Hugging Face cache (``local_files_only``) and loaded on
    the first call. ``backend_factory`` replaces the transformers backend (for tests); it
    receives ``(model_name, n_threads, local_files_only)`` and must return an object with
    ``generate``. Every call writes one JSON line naming the model, its weights file and
    sha256 (from a manifest when listed there), the thread count, the seed and the beam
    count.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MT_MODEL,
        n_threads: int = 1,
        seed: int = 1,
        *,
        manifest_path: Path | str | None = None,
        log_path: Path | str | None = None,
        local_files_only: bool = True,
        num_beams: int = 4,
        max_new_tokens: int = 128,
        max_input_tokens: int = 512,
        backend_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.n_threads = int(n_threads)
        self.seed = int(seed)
        self.local_files_only = bool(local_files_only)
        self.num_beams = int(num_beams)
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_tokens = int(max_input_tokens)
        self.log_path = Path(log_path) if log_path is not None else None
        self.direction = mt_direction(self.model_name)
        self.weights_path = mt_weights_path(self.model_name)
        self.model_file = self.weights_path.name if self.weights_path else None
        self.model_sha256 = model_sha256(self.weights_path, manifest_path) if self.weights_path else "unknown"
        commit = _SNAPSHOT_COMMIT.search(str(self.weights_path)) if self.weights_path else None
        self.model_commit = commit.group(1) if commit else None
        self.context: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self.load_seconds: float | None = None
        self._factory = backend_factory
        self._backend: Any = None
        self._counter = 0

    @property
    def loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._backend is not None

    def identity(self) -> dict[str, Any]:
        """Identity fields of the translation model, carried by every log record."""
        return {
            "backend": "transformers",
            "model_name": self.model_name,
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "model_commit": self.model_commit,
            "n_threads": self.n_threads,
            "seed": self.seed,
            "num_beams": self.num_beams,
            "local_files_only": self.local_files_only,
        }

    def load(self) -> Any:
        """Load the model on first use and return the backend object."""
        if self._backend is None:
            started = perf_counter()
            factory = self._factory or TransformersMarianBackend
            self._backend = factory(self.model_name, self.n_threads, self.local_files_only)
            self.load_seconds = perf_counter() - started
        return self._backend

    def accepts(self, src_lang: str, dst_lang: str) -> bool:
        """Whether the model's direction covers the requested one ("mix" counts as Arabic)."""
        if self.direction is None:
            return True
        return (MT_SOURCE_ALIASES.get(src_lang, src_lang), dst_lang) == self.direction

    def translate(self, text: str, src_lang: str, dst_lang: str) -> ModelResult:
        """Translate ``text``; the translation is in ``ModelResult.text``."""
        if not self.accepts(src_lang, dst_lang):
            assert self.direction is not None
            raise ValueError(
                f"{self.model_name} translates {self.direction[0]} to {self.direction[1]}, "
                f"not {src_lang} to {dst_lang}"
            )
        backend = self.load()
        started = perf_counter()
        output, prompt_tokens, completion_tokens = backend.generate(
            str(text), self.num_beams, self.max_new_tokens, self.max_input_tokens, self.seed
        )
        seconds = perf_counter() - started
        self._counter += 1
        finish = "length" if int(completion_tokens) >= self.max_new_tokens else "stop"
        result = ModelResult(
            text=clean_translation(str(output)),
            raw_text=str(output),
            tool_calls=[],
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            seconds=seconds,
            finish_reason=finish,
            call_id=self._counter,
            kind="translate_mt",
        )
        line: dict[str, Any] = {
            "event": "mt_call",
            "call_id": self._counter,
            "kind": "translate_mt",
            **self.identity(),
            "src_lang": src_lang,
            "dst_lang": dst_lang,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "seconds": seconds,
            "finish_reason": finish,
            "text": str(output),
        }
        line.update(self.context)
        self.calls.append(line)
        append_log_line(self.log_path, line)
        return result
