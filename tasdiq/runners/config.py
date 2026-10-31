"""Run configuration shared by the runners.

A configuration is a YAML mapping (configs/smoke.yaml, pilot.yaml, full.yaml) read into a
RunConfig that resolves paths relative to the code root, computes the configuration and
code hashes carried by every log record, and resolves model files from the Hugging Face
cache or a models directory. Nothing here loads a model.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = CODE_ROOT / "tasdiq"

DEFAULTS: dict[str, Any] = {
    "run_id": None,
    "results_dir": "results",
    "data_dir": "data",
    "seeds_dir": "tasdiq/bench/seeds",
    "live_tasks_file": "tasdiq/bench/seeds/live_tasks.yaml",
    "seed_ids": "all",
    "languages": ["en", "ar", "mix"],
    "splits": ["test"],
    "arms": ["B0", "B4", "Tasdiq"],
    "windows": [600],
    "revision_rules": ["invalidate"],
    "capture_pipelines": ["direct"],
    "capture_mode": "oracle",
    "capture_file": None,
    "user_model": "exact",
    "models": [],
    "models_dir": None,
    "manifest_path": None,
    "live_tasks": "all",
    "live_arms": ["PromptOnly", "Tasdiq"],
    "live_seeds": [1],
    "live_temperature": 0.0,
    "live_max_steps": 10,
    "live_step_seconds": 5,
    "live_dev_tasks": [],
    "max_workers": 2,
    "n_threads": 3,
    "per_worker_gb": 0.9,
    "principal_id": "U-1",
    "dev_fraction": 0.25,
    "split_seed": 20261006,
    "arm_params": {},
    "explore_depth": 10,
    "fee_policy": {},
}


def canonical_json(value: Any) -> str:
    """Return the canonical JSON text of a value (sorted keys, compact separators, non-ASCII kept)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    """Return the hex sha256 of a text encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_hash(package_root: Path | str = PACKAGE_ROOT) -> str:
    """Return the sha256 over the source files of the tasdiq package.

    Every .py and .yaml file under the package root enters the digest in sorted relative-path
    order, each prefixed by its path, so that a change to any module or seed file changes the hash.
    """
    root = Path(package_root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.suffix in (".py", ".yaml") and "__pycache__" not in p.parts):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_model_path(model: Mapping[str, Any], models_dir: Path | str | None = None) -> Path | None:
    """Return the local path of a model file, or None when it is not available.

    The model mapping names the file (``file``) and optionally the Hugging Face repository
    (``repo``). A models directory is searched first; otherwise the file is looked up in the
    Hugging Face cache through huggingface_hub.try_to_load_from_cache. Nothing is downloaded.
    """
    file_name = str(model.get("file") or "")
    if not file_name:
        return None
    candidates: list[Path] = []
    if model.get("path"):
        candidates.append(Path(str(model["path"])))
    if models_dir:
        candidates.append(Path(str(models_dir)) / file_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    repo = model.get("repo")
    if repo:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return None
        try:
            found = try_to_load_from_cache(str(repo), file_name)
        except Exception:
            return None
        if isinstance(found, str) and Path(found).is_file():
            return Path(found)
    return None


def verify_model_sha256(path: Path, expected: str | None, manifest_path: Path | str | None = None) -> bool:
    """Return True when the model file's sha256 equals the expected digest.

    The digest is taken from the manifest when it lists the file and computed otherwise; an
    absent expectation counts as verified.
    """
    if not expected:
        return True
    from tasdiq.capture.model import model_sha256

    return model_sha256(path, manifest_path) == expected.lower()


@dataclass
class ModelSpec:
    """One model of the configuration with its resolved path."""

    name: str
    file: str
    repo: str | None
    expected_sha256: str | None
    path: Path | None
    tracks: tuple[str, ...] = ("capture", "live")
    n_ctx: int = 4096
    chat_format: str | None = None
    subset: dict[str, Any] = field(default_factory=dict)
    """Restrictions of a model to part of a track (capture_kinds, capture_pipelines, capture_languages, live_tasks)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "repo": self.repo,
            "expected_sha256": self.expected_sha256,
            "path": None if self.path is None else self.path.as_posix(),
            "tracks": list(self.tracks),
            "n_ctx": self.n_ctx,
            "chat_format": self.chat_format,
            "subset": dict(self.subset),
        }

    def runner_config(self, n_threads: int, seed: int, manifest_path: Path | None, log_path: Path | None) -> dict[str, Any]:
        """Return the mapping LlamaRunner.from_config reads."""
        if self.path is None:
            raise FileNotFoundError(f"model {self.file} is not available locally")
        values: dict[str, Any] = {
            "model_path": str(self.path),
            "n_threads": int(n_threads),
            "n_ctx": int(self.n_ctx),
            "seed": int(seed),
            "chat_format": self.chat_format,
        }
        if self.expected_sha256:
            values["expected_sha256"] = self.expected_sha256
        if manifest_path is not None:
            values["manifest_path"] = str(manifest_path)
        if log_path is not None:
            values["log_path"] = str(log_path)
        return values


@dataclass
class RunConfig:
    """A loaded configuration with its hashes and resolved paths."""

    values: dict[str, Any]
    source: Path | None
    config_hash: str
    code_hash: str
    root: Path = CODE_ROOT
    models: list[ModelSpec] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    @property
    def run_id(self) -> str:
        run_id = self.values.get("run_id")
        if run_id:
            return str(run_id)
        name = self.source.stem if self.source is not None else "run"
        return f"{name}-{self.config_hash[:8]}"

    def path(self, key: str) -> Path:
        """Return the path stored under key, resolved against the code root when relative."""
        value = Path(str(self.values[key]))
        return value if value.is_absolute() else self.root / value

    @property
    def results_dir(self) -> Path:
        return self.path("results_dir")

    @property
    def data_dir(self) -> Path:
        return self.path("data_dir")

    @property
    def seeds_dir(self) -> Path:
        return self.path("seeds_dir")

    @property
    def manifest_path(self) -> Path | None:
        value = self.values.get("manifest_path")
        if not value:
            return None
        path = Path(str(value))
        return path if path.is_absolute() else self.root / path

    def model(self, name: str) -> ModelSpec:
        for spec in self.models:
            if spec.name == name:
                return spec
        raise KeyError(f"unknown model: {name}")

    def models_for(self, track: str) -> list[ModelSpec]:
        return [m for m in self.models if track in m.tracks]

    def meta(self) -> dict[str, Any]:
        """Return the run-level fields merged into every log record."""
        return {"run_id": self.run_id, "config_hash": self.config_hash, "code_hash": self.code_hash}

    def environment(self) -> dict[str, Any]:
        """Return software and platform facts for the manifest."""
        versions: dict[str, str] = {}
        for module in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml", "psutil", "llama_cpp", "huggingface_hub"):
            try:
                versions[module] = str(__import__(module).__version__)
            except Exception:
                versions[module] = "absent"
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": versions,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": None if self.source is None else self.source.as_posix(),
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "code_hash": self.code_hash,
            "values": dict(self.values),
            "models": [m.to_dict() for m in self.models],
        }


def _build_models(values: Mapping[str, Any], root: Path) -> list[ModelSpec]:
    models_dir = values.get("models_dir")
    if models_dir:
        models_dir = Path(str(models_dir))
        if not models_dir.is_absolute():
            models_dir = root / models_dir
    out: list[ModelSpec] = []
    for entry in values.get("models") or []:
        if isinstance(entry, str):
            entry = {"name": entry, "file": entry}
        name = str(entry.get("name") or Path(str(entry.get("file"))).stem)
        tracks = entry.get("tracks") or ["capture", "live"]
        out.append(
            ModelSpec(
                name=name,
                file=str(entry.get("file")),
                repo=entry.get("repo"),
                expected_sha256=(str(entry["sha256"]).lower() if entry.get("sha256") else None),
                path=resolve_model_path(entry, models_dir),
                tracks=tuple(str(t) for t in tracks),
                n_ctx=int(entry.get("n_ctx", 4096)),
                chat_format=entry.get("chat_format"),
                subset=dict(entry.get("subset") or {}),
            )
        )
    return out


def make_config(values: Mapping[str, Any], source: Path | None = None, root: Path = CODE_ROOT, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """Build a RunConfig from a mapping, applying defaults and command-line overrides."""
    merged: dict[str, Any] = dict(DEFAULTS)
    merged.update(dict(values))
    for key, value in (overrides or {}).items():
        if value is not None:
            merged[key] = value
    digest = sha256_text(canonical_json(merged))
    cfg = RunConfig(values=merged, source=source, config_hash=digest, code_hash=code_hash(root / "tasdiq"), root=root)
    cfg.models = _build_models(merged, root)
    return cfg


def load_config(path: Path | str, overrides: Mapping[str, Any] | None = None, root: Path = CODE_ROOT) -> RunConfig:
    """Read a YAML configuration file into a RunConfig."""
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    with open(source, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"configuration {source} must hold a mapping")
    return make_config(data, source=source, root=root, overrides=overrides)


def write_json(path: Path, document: Any) -> Path:
    """Write a JSON document with sorted keys, UTF-8 and a trailing newline; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, default=str))
        handle.write("\n")
    return path


def prepare_run_dir(run_dir: Path, resume: bool) -> Path:
    """Create a run directory; without resumption, remove the output files an earlier run left in it.

    The outputs of a run are the regular files directly inside its directory (the JSON-lines
    logs, the aggregates and the manifest). A resumed run keeps them and appends; a run
    that redoes every work unit starts from an empty directory so that its files hold
    exactly one record per work unit.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for path in run_dir.iterdir():
            if path.is_file():
                path.unlink()
    return run_dir


def append_jsonl(path: Path, rows: Any) -> int:
    """Append rows (mappings) to a JSON-lines file and return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file into a list of mappings; a missing file yields an empty list."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
