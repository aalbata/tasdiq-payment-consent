"""Ledger of reported numbers with their provenance.

Every number that reaches a table, a figure or the manuscript text is recorded as a
claim keyed by ``claim_id`` and carrying its value, unit, sample size, interval and
the run, file, field and function it was computed from.  The ledger is written to
``results/ledger.json`` and read back by the table and figure layers, which never
recompute a number themselves.

The second half of the module is the builder: :func:`build_ledger` reads the run
directories under ``results/{trace,capture,live,verify}`` (and the benchmark under
``data/``), computes every reported quantity through :mod:`tasdiq.analysis.metrics` and
:mod:`tasdiq.analysis.stats`, records the H1 to H5 decision quantities with their
pre-stated thresholds and verdicts, and writes ``results/manifest.json`` naming the
runs that fed the ledger.  Claim ids are stable strings built from fixed prefixes and
tokens: ``cap_<metric>_<lang>_<pipeline>_<model>``, ``enf_<metric>_<arm>[_<rule>][_<family>]_w<window>``
(``enf_<metric>_reader_<reader>_<arm>...`` for a sensitivity reader, with
``sens_reader_<reader>_minus_exact_<metric>_<arm>_w<window>`` for its paired differences),
``attr_<class>_<arm>_<lang>_w<window>``, ``rev_curve_<rule>_<metric>``,
``win_curve_<window>_<metric>[_<arm>]``, ``den_<denominator>_<window>``,
``live_<metric>_<model>_<lang>_<arm>[_<family>]``, ``sec_<metric>_lang_diff_ar_minus_en_<model>_<arm>``,
``lat_<stage>_<pipeline>_<model>_<stat>``, ``mem_peak_rss_bytes_<track>``, ``ver_<metric>_<arm>_<rule>_d<depth>``,
``bench_<count>``, ``cal_<parameter>_<arm>`` and ``h<k>_<quantity>`` with ``h<k>_verdict``.
"""

from __future__ import annotations

import csv
import json
import math
import zlib
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tasdiq.analysis import stats

CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = CODE_ROOT / "results" / "ledger.json"

CLAIM_FIELDS: tuple[str, ...] = (
    "value",
    "unit",
    "n",
    "ci",
    "source_run",
    "source_file",
    "field",
    "computed_by",
    "count",
    "description",
    "clusters",
    "threshold",
    "verdict",
    "p_value",
    "p_holm",
)


def _json_safe(v: Any) -> Any:
    """Convert numpy scalars and non-finite floats into JSON-serialisable values."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, Mapping):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, Path):
        return v.as_posix()
    return v


def normalise_ci(ci: Any, level: float = 0.95, method: str | None = None) -> dict[str, Any] | None:
    """Return an interval as ``{"lo", "hi", "level", "method"}`` or ``None``.

    Accepts a mapping with those keys, an object with ``lo`` and ``hi`` attributes
    (such as :class:`tasdiq.analysis.stats.BootstrapInterval`), a pair ``(lo, hi)`` or
    a triple ``(est, lo, hi)``. A bootstrap interval that dropped non-finite replicates
    carries ``n_dropped`` and ``n_boot`` in the result.
    """
    if ci is None:
        return None
    if isinstance(ci, (str, bytes)):
        raise TypeError("ci must be a mapping, an interval object, a pair or a triple")
    if isinstance(ci, Mapping):
        out = {
            "lo": ci.get("lo"),
            "hi": ci.get("hi"),
            "level": ci.get("level", level),
            "method": ci.get("method", method),
        }
        for key in ("n_dropped", "n_boot"):
            if ci.get(key) is not None:
                out[key] = ci[key]
    elif hasattr(ci, "lo") and hasattr(ci, "hi"):
        out = {"lo": ci.lo, "hi": ci.hi, "level": level, "method": method}
        if getattr(ci, "n_dropped", 0):
            out["n_dropped"] = int(ci.n_dropped)
            out["n_boot"] = getattr(ci, "n_boot", None)
    elif isinstance(ci, Sequence) and len(ci) == 2:
        out = {"lo": ci[0], "hi": ci[1], "level": level, "method": method}
    elif isinstance(ci, Sequence) and len(ci) == 3:
        out = {"lo": ci[1], "hi": ci[2], "level": level, "method": method}
    else:
        raise TypeError("ci must be a mapping, an interval object, a pair or a triple")
    return _json_safe(out)


@dataclass(frozen=True)
class Claim:
    """One reported number with its provenance."""

    claim_id: str
    value: float | int | None
    unit: str
    n: int | None = None
    ci: dict[str, Any] | None = None
    source_run: str | None = None
    source_file: str | None = None
    field: str | None = None
    computed_by: str | None = None
    count: int | None = None
    description: str | None = None
    clusters: int | None = None
    threshold: float | None = None
    verdict: str | None = None
    p_value: float | None = None
    p_holm: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("claim_id")
        return _json_safe(d)

    @classmethod
    def from_dict(cls, claim_id: str, d: Mapping[str, Any]) -> "Claim":
        unknown = set(d) - set(CLAIM_FIELDS)
        if unknown:
            raise ValueError(f"claim {claim_id!r} has unknown fields {sorted(unknown)}")
        if "value" not in d or "unit" not in d:
            raise ValueError(f"claim {claim_id!r} needs 'value' and 'unit'")
        return cls(
            claim_id=claim_id,
            value=d.get("value"),
            unit=str(d.get("unit")),
            n=d.get("n"),
            ci=normalise_ci(d.get("ci")),
            source_run=d.get("source_run"),
            source_file=d.get("source_file"),
            field=d.get("field"),
            computed_by=d.get("computed_by"),
            count=d.get("count"),
            description=d.get("description"),
            clusters=d.get("clusters"),
            threshold=d.get("threshold"),
            verdict=d.get("verdict"),
            p_value=d.get("p_value"),
            p_holm=d.get("p_holm"),
        )


class Ledger:
    """Collection of claims keyed by ``claim_id`` with JSON persistence."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_LEDGER_PATH
        self._claims: dict[str, Claim] = {}

    def __len__(self) -> int:
        return len(self._claims)

    def __contains__(self, claim_id: object) -> bool:
        return claim_id in self._claims

    def __iter__(self) -> Iterator[str]:
        return iter(self._claims)

    def __getitem__(self, claim_id: str) -> Claim:
        return self._claims[claim_id]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ledger):
            return NotImplemented
        return self._claims == other._claims

    @property
    def claims(self) -> dict[str, Claim]:
        return dict(self._claims)

    def get(self, claim_id: str) -> Claim:
        return self._claims[claim_id]

    def add(
        self,
        claim_id: str,
        value: float | int | None,
        unit: str,
        n: int | None = None,
        ci: Any = None,
        source_run: str | None = None,
        source_file: str | Path | None = None,
        field: str | None = None,
        computed_by: str | None = None,
        count: int | None = None,
        replace: bool = False,
        description: str | None = None,
        clusters: int | None = None,
        threshold: float | None = None,
        verdict: str | None = None,
        p_value: float | None = None,
        p_holm: float | None = None,
    ) -> Claim:
        """Record a claim; an existing ``claim_id`` is refused unless ``replace`` is set.

        ``description`` states in one sentence what the number measures; ``clusters`` is
        the number of independent units behind a clustered interval; ``threshold`` and
        ``verdict`` carry the pre-stated decision value and outcome of a hypothesis
        quantity; ``p_value`` and ``p_holm`` carry a permutation p-value and its Holm
        adjustment within the secondary family.
        """
        if not claim_id:
            raise ValueError("claim_id must be a non-empty string")
        if claim_id in self._claims and not replace:
            raise KeyError(f"claim {claim_id!r} already recorded")
        claim = Claim(
            claim_id=claim_id,
            value=_json_safe(value),
            unit=unit,
            n=None if n is None else int(n),
            ci=normalise_ci(ci),
            source_run=source_run,
            source_file=None if source_file is None else Path(source_file).as_posix(),
            field=field,
            computed_by=computed_by,
            count=None if count is None else int(count),
            description=description,
            clusters=None if clusters is None else int(clusters),
            threshold=None if threshold is None else float(threshold),
            verdict=verdict,
            p_value=None if p_value is None else float(p_value),
            p_holm=None if p_holm is None else float(p_holm),
        )
        self._claims[claim_id] = claim
        return claim

    def add_rate(
        self,
        claim_id: str,
        k: int | Any,
        n: int | None = None,
        unit: str = "fraction",
        ci_method: str = "wilson",
        alpha: float = 0.05,
        source_run: str | None = None,
        source_file: str | Path | None = None,
        field: str | None = None,
        computed_by: str | None = None,
        replace: bool = False,
        description: str | None = None,
        threshold: float | None = None,
        verdict: str | None = None,
    ) -> Claim:
        """Record a proportion ``k / n`` with a binomial interval.

        ``k`` may be a :class:`tasdiq.analysis.metrics.Rate` (or any object with ``k``
        and ``n`` attributes), in which case ``n`` is taken from it.  ``ci_method`` is
        ``"wilson"``, ``"clopper_pearson"`` or ``"rule_of_three"`` (the last only for
        ``k = 0``).  A zero denominator records ``value`` ``None`` with no interval.
        """
        if hasattr(k, "k") and hasattr(k, "n"):
            n = int(k.n)
            k = int(k.k)
        if n is None:
            raise ValueError("n is required when k is an integer")
        k = int(k)
        n = int(n)
        if n == 0:
            return self.add(claim_id, None, unit, n=0, ci=None, source_run=source_run,
                            source_file=source_file, field=field, computed_by=computed_by,
                            count=k, replace=replace, description=description,
                            threshold=threshold, verdict=verdict)
        if ci_method == "wilson":
            lo, hi = stats.wilson_ci(k, n, alpha)
        elif ci_method == "clopper_pearson":
            lo, hi = stats.clopper_pearson(k, n, alpha)
        elif ci_method == "rule_of_three":
            if k != 0:
                raise ValueError("rule_of_three applies only to k = 0")
            lo, hi = 0.0, stats.rule_of_three(n)
        else:
            raise ValueError(f"unknown ci_method {ci_method!r}")
        ci = {"lo": lo, "hi": hi, "level": 1.0 - alpha, "method": ci_method}
        return self.add(claim_id, k / n, unit, n=n, ci=ci, source_run=source_run,
                        source_file=source_file, field=field, computed_by=computed_by,
                        count=k, replace=replace, description=description,
                        threshold=threshold, verdict=verdict)

    def add_interval(
        self,
        claim_id: str,
        interval: Any,
        unit: str,
        n: int | None = None,
        method: str = "cluster_bootstrap",
        level: float = 0.95,
        source_run: str | None = None,
        source_file: str | Path | None = None,
        field: str | None = None,
        computed_by: str | None = None,
        count: int | None = None,
        replace: bool = False,
        description: str | None = None,
        clusters: int | None = None,
        threshold: float | None = None,
        verdict: str | None = None,
        p_value: float | None = None,
        p_holm: float | None = None,
    ) -> Claim:
        """Record a point estimate with an interval from the statistics layer.

        ``interval`` has ``est``, ``lo`` and ``hi`` attributes (a
        :class:`tasdiq.analysis.stats.BootstrapInterval`) or is a triple
        ``(est, lo, hi)``; ``n`` is the number of independent clusters.
        """
        if hasattr(interval, "est"):
            est, lo, hi = interval.est, interval.lo, interval.hi
        else:
            est, lo, hi = interval
        ci: dict[str, Any] = {"lo": lo, "hi": hi, "level": level, "method": method}
        if getattr(interval, "n_dropped", 0):
            ci["n_dropped"] = int(interval.n_dropped)
            ci["n_boot"] = getattr(interval, "n_boot", None)
        return self.add(claim_id, est, unit, n=n, ci=ci, source_run=source_run,
                        source_file=source_file, field=field, computed_by=computed_by,
                        count=count, replace=replace, description=description,
                        clusters=clusters, threshold=threshold, verdict=verdict,
                        p_value=p_value, p_holm=p_holm)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Mapping of ``claim_id`` to claim fields, ordered by ``claim_id``."""
        return {cid: self._claims[cid].as_dict() for cid in sorted(self._claims)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Mapping[str, Any]], path: Path | str | None = None) -> "Ledger":
        ledger = cls(path)
        for cid, d in data.items():
            ledger._claims[cid] = Claim.from_dict(cid, d)
        return ledger

    def write(self, path: Path | str | None = None) -> Path:
        """Write the ledger as sorted, indented UTF-8 JSON and return the path written."""
        target = Path(path) if path is not None else self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        target.write_text(text + "\n", encoding="utf-8")
        return target

    @classmethod
    def read(cls, path: Path | str | None = None) -> "Ledger":
        """Read a ledger written by :meth:`write`."""
        source = Path(path) if path is not None else DEFAULT_LEDGER_PATH
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("ledger file must hold a JSON object keyed by claim_id")
        return cls.from_dict(data, source)


# ---------------------------------------------------------------------------
# Ledger builder: from run directories to claims
# ---------------------------------------------------------------------------

TRACKS: tuple[str, ...] = ("trace", "capture", "live", "verify")
"""Result tracks in the order the builder reads them."""

RECORD_FILES: dict[str, str] = {
    "trace": "sessions.jsonl",
    "capture": "captures.jsonl",
    "live": "episodes.jsonl",
    "verify": "explore.json",
}
"""Record file of each track inside a run directory."""

RECORD_KEYS: dict[str, tuple[str, ...]] = {
    "trace": ("variant_id", "arm", "window_s", "rule", "reader"),
    "capture": ("variant_id", "pipeline", "model"),
    "live": ("episode_id",),
    "verify": ("arm", "rule", "depth"),
}
"""Fields that identify one record of a track; later runs do not duplicate earlier ones.

The same fields order the records before any interval is computed, so the cluster bootstrap
and the permutation tests do not depend on the order in which the workers wrote the records.
"""

PRIMARY_READER = "exact"
"""Simulated reader of the main trace results; every other reader is a sensitivity run."""


def _sort_token(value: Any) -> tuple[int, float, str]:
    """A sort key that orders missing values first, numbers numerically and everything else as text."""
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, bool):
        return (2, 0.0, str(value))
    if isinstance(value, (int, float)):
        return (1, float(value), "")
    return (2, 0.0, str(value))


def ordered_records(records: Sequence[Mapping[str, Any]], track: str) -> list[Mapping[str, Any]]:
    """The records of a track sorted on the track's record key (:data:`RECORD_KEYS`).

    Records that share every key field keep their relative order.
    """
    keys = RECORD_KEYS.get(track)
    if keys is None:
        return list(records)
    return sorted(records, key=lambda r: tuple(_sort_token(r.get(k)) for k in keys))


def reader_of(record: Mapping[str, Any]) -> str:
    """The simulated reader of a trace session record; a record without the field used the primary reader."""
    return str(record.get("reader") or PRIMARY_READER)

NAMED_PHENOMENA: tuple[str, ...] = (
    "digit_script",
    "currency_words",
    "number_words",
    "negation",
    "inquiry_vs_approval",
    "cross_turn_reference",
    "cross_language_revision",
)
"""Phenomena named by H1 as the expected concentration of invariance failures."""

CAPTURE_STAGES: tuple[str, ...] = ("model", "canonicalise", "gate", "translate")
LATENCY_STATS: tuple[str, ...] = ("n", "min", "p05", "p25", "median", "p75", "p95", "p99", "max", "mean")

THRESHOLDS: dict[str, float] = {
    "h1_invariance_diff": 0.05,
    "h1_concentration": 0.5,
    "h2_relative_reduction": 0.30,
    "h2_added_confirmations": 0.10,
    "h3_unnecessary_reconf_diff": 0.10,
    "h4_stale_budget": 0.01,
    "h4_forced_budget": 0.10,
    "h4_min_windows": 5,
    "h5_reproposal": 0.10,
    "h5_reproposal_floor": 0.05,
    "h5_completion_loss": 0.10,
}
"""Pre-stated decision values of H1 to H5."""

SUPPORTS, REFUTES, INCONCLUSIVE = "supports", "refutes", "inconclusive"

NO_WINDOW_S = 10**9
"""Windows at or above this many seconds stand for an arm that keeps no validity window."""

PRIMARY_ARM = "Tasdiq"
PRIMARY_RULE = "invalidate"
CLASSICAL_ARM = "B4"
NARROWING_RULE = "narrowing"
ELEVATION_ARM = "B3"
NO_MONITOR_ARM = "B0"
PROPOSED_PIPELINE = "direct"
SAME_PARSER_PIPELINE = "noncanon"
TRANSLATE_PIPELINE = "translate"
MT_TRANSLATE_PIPELINE = "translate_mt"
TRANSLATION_COMPARATORS: tuple[str, ...] = (TRANSLATE_PIPELINE, MT_TRANSLATE_PIPELINE)
"""The translate-then-bind comparators of H2: same-model translation and the open MT model."""
REVISION_FAMILY = "revision"
DELAY_FAMILY = "expiry_stale"
CONTROL_FAMILY = "ordinary_valid"
"""Live family without a revision or cancellation: the no-change control of H5."""
CHANGE_FAMILIES: tuple[str, ...] = ("revision", "cancellation")
LIVE_BEHAVIOUR_ARM = "PromptOnly"
LIVE_MONITOR_ARM = "Tasdiq"


def model_slug(name: str) -> str:
    """Claim-id token of a model name: the text before the first hyphen, alphanumerics only.

    ``qwen2.5-1.5b`` becomes ``qwen25`` and ``qwen3-4b`` becomes ``qwen3``.
    """
    head = str(name).split("-", 1)[0]
    slug = "".join(ch for ch in head.lower() if ch.isalnum())
    return slug or "model"


def arm_token(arm: str, rule: Any) -> str:
    """Claim-id token of an arm under a revision rule (``B4``, ``Tasdiq_narrowing``)."""
    if rule is None or str(rule) in ("none", ""):
        return str(arm)
    return f"{arm}_{rule}"


def curve_token(arm: str, rule: Any) -> str | None:
    """Token of a revision-rule operating point: the rule for the proposed system, else the arm."""
    if arm == PRIMARY_ARM:
        return str(rule) if rule not in (None, "none", "") else arm
    if arm == "TasdiqNonAtomic":
        return None
    return str(arm)


def window_token(window: Any) -> str:
    """Claim-id token of a validity window in seconds (``w600``; ``wnone`` without a window)."""
    if window is None:
        return "wnone"
    return f"w{int(window)}"


def claim_seed(base: int, claim_id: str) -> int:
    """Deterministic bootstrap seed of one claim: the base seed combined with the claim id."""
    return (int(base) + zlib.crc32(claim_id.encode("utf-8"))) % (2**31 - 1)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file; a missing file yields an empty list."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> Any:
    """Read a JSON document; a missing file yields ``None``."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(results_dir: Path | str, track: str) -> list[Path]:
    """Run directories of a track that hold its record file, in name order."""
    track_dir = Path(results_dir) / track
    if not track_dir.is_dir():
        return []
    record = RECORD_FILES[track]
    return sorted(p for p in track_dir.iterdir() if p.is_dir() and (p / record).is_file())


def select_runs(results_dir: Path | str, runs: Mapping[str, Sequence[str]] | None = None) -> dict[str, list[Path]]:
    """Run directories per track: those named in ``runs`` (a track to run-id mapping) or all."""
    out: dict[str, list[Path]] = {}
    for track in TRACKS:
        wanted = None if runs is None else runs.get(track)
        found = discover_runs(results_dir, track)
        if wanted is None:
            out[track] = found
            continue
        by_name = {p.name: p for p in found}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise FileNotFoundError(f"{track} run directories not found under {Path(results_dir) / track}: {missing}")
        out[track] = [by_name[name] for name in wanted]
    return out


def _count(v: Any) -> int:
    """Length of a list field or the integer value of a counter field; ``None`` is 0."""
    if v is None:
        return 0
    if isinstance(v, (list, tuple)):
        return len(v)
    return int(v)


def _classes(v: Any, keys: Sequence[str] = ("intent_class", "reason", "class")) -> list[str]:
    """Class names of a list of labelled entries (strings or mappings)."""
    out: list[str] = []
    for entry in v or []:
        if isinstance(entry, Mapping):
            label = next((entry[k] for k in keys if entry.get(k) is not None), None)
        else:
            label = entry
        if label is not None:
            out.append(str(label).split(":", 1)[0])
    return out


def _unconfirmed_count(r: Mapping[str, Any], intended: bool = False) -> int:
    """UNCONFIRMED executions of a session (or those also INTENDED), from the counter or the labels."""
    name = "unconfirmed_intended" if intended else "unconfirmed_executions"
    v = r.get(name)
    if v is not None:
        return _count(v)
    labels = r.get("execution_labels") or []
    if intended:
        return sum(1 for lab in labels if isinstance(lab, Mapping) and lab.get("unconfirmed_intended"))
    return sum(1 for lab in labels if isinstance(lab, Mapping) and lab.get("consent_label") == "UNCONFIRMED")


def _rel_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values], dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


class LedgerBuilder:
    """Computes the claims of one study from its run directories.

    Session-level and episode-level rates carry percentile cluster-bootstrap intervals
    (``n_boot`` resamples of the seed scenarios on the trace and capture tracks and of
    the live tasks on the live track); a rate observed as zero carries the exact
    Clopper-Pearson interval; rates with execution or attempt denominators carry Wilson
    intervals (Clopper-Pearson when zero).  Paired differences between arms or rules are
    aligned on the variant and resampled by cluster.  Every bootstrap seed derives from
    ``seed`` and the claim id.

    Two families of permutation p-values (``n_perm`` permutations each, seeded from the
    claim id) are Holm-adjusted separately: the pre-stated secondary family of the
    protocol (the H1 invariance differences of the proposed pipeline on the primary model,
    the H3 difference between the invalidate and narrowing rules and the H5 completion
    difference between the change families and the control, per model) and the
    exploratory family of the live language differences.  H4 is a curve and carries no
    p-value.  The records of every track are sorted on their record key before any
    interval is computed.
    """

    def __init__(self, ledger: Ledger, n_boot: int = 2000, seed: int = 20261006, root: Path | None = None, n_perm: int = 10000) -> None:
        self.ledger = ledger
        self.n_boot = int(n_boot)
        self.n_perm = int(n_perm)
        self.seed = int(seed)
        self.root = Path(root) if root is not None else CODE_ROOT
        self.groups: dict[str, Any] = {}
        self.secondary: list[tuple[str, float]] = []
        self.exploratory: list[tuple[str, float]] = []

    # -- interval helpers -------------------------------------------------------

    def _src(self, source: Mapping[str, Any] | None) -> dict[str, Any]:
        source = source or {}
        return {"source_run": source.get("run"), "source_file": source.get("file")}

    def count_claim(self, cid: str, value: int, description: str, source: Mapping[str, Any] | None = None, field: str | None = None, computed_by: str = "tasdiq.analysis.ledger.LedgerBuilder", unit: str = "count") -> Claim:
        return self.ledger.add(cid, int(value), unit, n=None, count=int(value), field=field, computed_by=computed_by, description=description, **self._src(source))

    def text_claim(self, cid: str, value: str, description: str, source: Mapping[str, Any] | None = None, field: str | None = None) -> Claim:
        return self.ledger.add(cid, str(value), "text", field=field, computed_by="tasdiq.analysis.ledger.LedgerBuilder", description=description, **self._src(source))

    def value_claim(self, cid: str, value: float, unit: str, description: str, source: Mapping[str, Any] | None = None, field: str | None = None, n: int | None = None) -> Claim:
        return self.ledger.add(cid, value, unit, n=n, field=field, computed_by="tasdiq.analysis.ledger.LedgerBuilder", description=description, **self._src(source))

    def binomial_rate(self, cid: str, k: int, n: int, description: str, source: Mapping[str, Any] | None = None, field: str | None = None, threshold: float | None = None, verdict: str | None = None) -> Claim:
        """A rate ``k / n`` over independent trials: Wilson, or Clopper-Pearson when ``k`` is 0 or ``n``."""
        method = "clopper_pearson" if (k == 0 or k == n) else "wilson"
        return self.ledger.add_rate(cid, int(k), int(n), ci_method=method, field=field, computed_by="tasdiq.analysis.stats." + ("clopper_pearson" if method == "clopper_pearson" else "wilson_ci"), description=description, threshold=threshold, verdict=verdict, **self._src(source))

    def cluster_rate(self, cid: str, flags: Sequence[Any], clusters: Sequence[Hashable], description: str, source: Mapping[str, Any] | None = None, field: str | None = None, threshold: float | None = None, verdict: str | None = None) -> Claim:
        """Share of rows whose flag is true (``None`` rows excluded), clustered on ``clusters``."""
        vals = [(float(bool(f)), c) for f, c in zip(flags, clusters) if f is not None]
        n = len(vals)
        k = int(sum(v for v, _ in vals))
        src = self._src(source)
        if n == 0:
            return self.ledger.add(cid, None, "fraction", n=0, count=0, field=field, computed_by="tasdiq.analysis.ledger.LedgerBuilder", description=description, threshold=threshold, verdict=verdict, **src)
        n_clusters = len({c for _, c in vals})
        if k == 0 or k == n:
            lo, hi = stats.clopper_pearson(k, n)
            ci = {"lo": lo, "hi": hi, "level": 0.95, "method": "clopper_pearson"}
            return self.ledger.add(cid, k / n, "fraction", n=n, ci=ci, count=k, clusters=n_clusters, field=field, computed_by="tasdiq.analysis.stats.clopper_pearson", description=description, threshold=threshold, verdict=verdict, **src)
        if n_clusters < 2:
            lo, hi = stats.wilson_ci(k, n)
            ci = {"lo": lo, "hi": hi, "level": 0.95, "method": "wilson"}
            return self.ledger.add(cid, k / n, "fraction", n=n, ci=ci, count=k, clusters=n_clusters, field=field, computed_by="tasdiq.analysis.stats.wilson_ci", description=description, threshold=threshold, verdict=verdict, **src)
        interval = stats.cluster_bootstrap([v for v, _ in vals], [c for _, c in vals], np.mean, self.n_boot, seed=claim_seed(self.seed, cid))
        return self.ledger.add_interval(cid, interval, "fraction", n=n, count=k, clusters=n_clusters, field=field, computed_by="tasdiq.analysis.stats.cluster_bootstrap", description=description, threshold=threshold, verdict=verdict, **src)

    def cluster_mean(self, cid: str, values: Sequence[float], clusters: Sequence[Hashable], unit: str, description: str, source: Mapping[str, Any] | None = None, field: str | None = None) -> Claim:
        """Mean of a per-row quantity with a cluster-bootstrap interval."""
        src = self._src(source)
        if len(values) == 0:
            return self.ledger.add(cid, None, unit, n=0, field=field, computed_by="tasdiq.analysis.ledger.LedgerBuilder", description=description, **src)
        interval = stats.cluster_bootstrap(list(values), list(clusters), np.mean, self.n_boot, seed=claim_seed(self.seed, cid))
        return self.ledger.add_interval(cid, interval, unit, n=len(values), clusters=len(set(clusters)), field=field, computed_by="tasdiq.analysis.stats.cluster_bootstrap", description=description, **src)

    def _register(self, cid: str, p_value: float, family: str | None) -> None:
        """Enter a p-value into the Holm family named by ``family`` (``secondary`` or ``exploratory``)."""
        if family is None:
            return
        if family == "secondary":
            self.secondary.append((cid, p_value))
        elif family == "exploratory":
            self.exploratory.append((cid, p_value))
        else:
            raise ValueError(f"unknown p-value family {family!r}")

    def cluster_stat(self, cid: str, values: np.ndarray, clusters: Sequence[Hashable], stat: Callable[[np.ndarray], float], unit: str, description: str, source: Mapping[str, Any] | None = None, field: str | None = None, threshold: float | None = None, verdict: str | None = None, permutation: str | None = None, family: str | None = None) -> Claim:
        """A statistic of a row matrix with a cluster-bootstrap interval.

        With ``permutation`` the matrix holds the observation in column 0 and the group flag
        in column 1, ``stat`` is the group difference of the means, and a two-sided cluster
        permutation p-value is attached: ``"swap"`` swaps the group labels of whole clusters
        (every cluster holds both groups) and ``"label"`` permutes the cluster-level labels
        (every cluster belongs to one group).  ``family`` names the Holm family the p-value
        joins at the end of the build.
        """
        src = self._src(source)
        if values.shape[0] == 0:
            return self.ledger.add(cid, None, unit, n=0, field=field, computed_by="tasdiq.analysis.ledger.LedgerBuilder", description=description, threshold=threshold, verdict=verdict, **src)
        interval = stats.cluster_bootstrap(values, list(clusters), stat, self.n_boot, seed=claim_seed(self.seed, cid))
        p_value = None
        if permutation is not None:
            test = {"swap": stats.cluster_swap_permutation, "label": stats.cluster_label_permutation}[permutation]
            p_value = test(values[:, 0], values[:, 1] == 1.0, list(clusters), np.mean, self.n_perm, seed=claim_seed(self.seed, cid + ":perm")).p_value
            self._register(cid, p_value, family)
        return self.ledger.add_interval(cid, interval, unit, n=int(values.shape[0]), clusters=len(set(clusters)), field=field, computed_by="tasdiq.analysis.stats.cluster_bootstrap", description=description, threshold=threshold, verdict=verdict, p_value=p_value, **src)

    def paired_diff(self, cid: str, a: Sequence[float], b: Sequence[float], clusters: Sequence[Hashable], unit: str, description: str, source: Mapping[str, Any] | None = None, field: str | None = None, threshold: float | None = None, verdict: str | None = None, permutation: bool = False, family: str | None = None) -> Claim:
        """``mean(a) - mean(b)`` on aligned rows with a paired cluster-bootstrap interval.

        With ``permutation`` the two-sided paired cluster permutation p-value is attached
        and, when ``family`` is given, the claim joins that Holm family at the end of the
        build.
        """
        src = self._src(source)
        if len(a) == 0:
            return self.ledger.add(cid, None, unit, n=0, field=field, computed_by="tasdiq.analysis.stats.paired_cluster_diff", description=description, threshold=threshold, verdict=verdict, **src)
        interval = stats.paired_cluster_diff(list(a), list(b), list(clusters), np.mean, self.n_boot, seed=claim_seed(self.seed, cid))
        p_value = None
        if permutation:
            p_value = stats.paired_cluster_permutation(list(a), list(b), list(clusters), np.mean, self.n_perm, seed=claim_seed(self.seed, cid + ":perm")).p_value
            self._register(cid, p_value, family)
        return self.ledger.add_interval(cid, interval, unit, n=len(a), clusters=len(set(clusters)), field=field, computed_by="tasdiq.analysis.stats.paired_cluster_diff", description=description, threshold=threshold, verdict=verdict, p_value=p_value, **src)

    def finish_secondary(self) -> None:
        """Holm-adjust the secondary and the exploratory families separately and write the adjusted values back."""
        for members in (self.secondary, self.exploratory):
            if not members:
                continue
            adjusted = stats.holm([p for _, p in members])
            for (cid, _), p_holm in zip(members, adjusted):
                claim = self.ledger.get(cid)
                self.ledger.add(cid, claim.value, claim.unit, n=claim.n, ci=claim.ci, source_run=claim.source_run, source_file=claim.source_file, field=claim.field, computed_by=claim.computed_by, count=claim.count, replace=True, description=claim.description, clusters=claim.clusters, threshold=claim.threshold, verdict=claim.verdict, p_value=claim.p_value, p_holm=p_holm)

    def verdict_claim(self, cid: str, verdict: str, description: str, source: Mapping[str, Any] | None = None) -> Claim:
        return self.ledger.add(cid, verdict, "verdict", verdict=verdict, computed_by="tasdiq.analysis.ledger.LedgerBuilder", description=description, **self._src(source))

    # -- benchmark composition ----------------------------------------------------

    def add_benchmark(self, data_dir: Path) -> None:
        """Seed, variant, split and review counts of the benchmark under ``data_dir``."""
        data_dir = Path(data_dir)
        scenarios = read_jsonl(data_dir / "scenarios.jsonl")
        variants = read_jsonl(data_dir / "benchmark.jsonl")
        if not scenarios and not variants:
            return
        src_s = {"run": "data", "file": _rel_path(data_dir / "scenarios.jsonl", self.root)}
        src_v = {"run": "data", "file": _rel_path(data_dir / "benchmark.jsonl", self.root)}
        main_scenarios = [s for s in scenarios if not s.get("dialect_extra")]
        main_variants = [v for v in variants if not v.get("dialect_extra")]
        families = sorted({str(s.get("family")) for s in main_scenarios} | {str(v.get("family")) for v in main_variants})
        langs = sorted({str(v.get("lang")) for v in main_variants})
        kinds = [k for k in ("base", "paraphrase", "edit") if any(v.get("kind") == k for v in main_variants)]
        self.groups["benchmark"] = {"families": families, "langs": langs, "kinds": kinds}
        self.count_claim("bench_seeds_total", len(main_scenarios), "Seed scenarios in the main benchmark set (dialect_extra excluded).", src_s, "seed_id")
        self.count_claim("bench_variants_total", len(main_variants), "Rendered variants in the main benchmark set.", src_v, "variant_id")
        for fam in families:
            self.count_claim(f"bench_seeds_{fam}", sum(1 for s in main_scenarios if s.get("family") == fam), f"Seed scenarios of family {fam}.", src_s, "family")
            self.count_claim(f"bench_variants_{fam}", sum(1 for v in main_variants if v.get("family") == fam), f"Variants of family {fam}.", src_v, "family")
        for lang in langs:
            self.count_claim(f"bench_variants_{lang}", sum(1 for v in main_variants if v.get("lang") == lang), f"Variants in language {lang}.", src_v, "lang")
            for kind in kinds:
                self.count_claim(f"bench_variants_{lang}_{kind}", sum(1 for v in main_variants if v.get("lang") == lang and v.get("kind") == kind), f"Variants of kind {kind} in language {lang}.", src_v, "kind")
        for kind in kinds:
            self.count_claim(f"bench_variants_{kind}", sum(1 for v in main_variants if v.get("kind") == kind), f"Variants of kind {kind}.", src_v, "kind")
        split_file = read_json(data_dir / "splits.json")
        splits: dict[str, str] = {}
        if isinstance(split_file, Mapping) and isinstance(split_file.get("seed_splits"), Mapping):
            splits = {str(k): str(v) for k, v in split_file["seed_splits"].items()}
            src_split = {"run": "data", "file": _rel_path(data_dir / "splits.json", self.root)}
        else:
            for v in main_variants:
                if v.get("seed_id") and v.get("split") in ("dev", "test"):
                    splits.setdefault(str(v["seed_id"]), str(v["split"]))
            src_split = src_v
        for split in ("dev", "test"):
            self.count_claim(f"bench_seeds_{split}", sum(1 for s in main_scenarios if splits.get(str(s.get("seed_id"))) == split), f"Seed scenarios assigned to the {split} split (a seed keeps its split when some of its variants are excluded as near-duplicates).", src_split, "seed_splits")
            self.count_claim(f"bench_variants_{split}", sum(1 for v in main_variants if v.get("split") == split), f"Variants assigned to the {split} split.", src_v, "split")
        self.count_claim("bench_variants_excluded", sum(1 for v in main_variants if v.get("split") == "excluded"), "Variants excluded by the near-duplicate rule.", src_v, "split")
        statuses = sorted({str(v.get("review_status")) for v in main_variants})
        for status in statuses:
            self.count_claim(f"bench_review_status_{status}", sum(1 for v in main_variants if str(v.get("review_status")) == status), f"Variants whose review status is {status}.", src_v, "review_status")
        review = data_dir / "review_set.csv"
        if review.is_file():
            with open(review, "r", encoding="utf-8-sig", newline="") as handle:
                n_rows = sum(1 for _ in csv.DictReader(handle))
            self.count_claim("bench_review_rows", n_rows, "Rows of the author review set.", {"run": "data", "file": _rel_path(review, self.root)}, "variant_id")
        report = read_json(data_dir / "build_report.json")
        if isinstance(report, Mapping):
            src_r = {"run": "data", "file": _rel_path(data_dir / "build_report.json", self.root)}
            if report.get("n_near_duplicate_pairs") is not None:
                self.count_claim("bench_near_duplicate_pairs", int(report["n_near_duplicate_pairs"]), "Near-duplicate variant pairs found by the Jaccard rule.", src_r, "n_near_duplicate_pairs")
            if report.get("n_dialect_extra_seeds") is not None:
                self.count_claim("bench_seeds_dialect_extra", int(report["n_dialect_extra_seeds"]), "Seeds of the optional dialect set, excluded from every result.", src_r, "n_dialect_extra_seeds")

    # -- capture track -------------------------------------------------------------

    def add_capture(self, rows: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> None:
        """Capture-stage claims per model, pipeline and language, with H1 and the capture part of H2."""
        if not rows:
            return
        from tasdiq.analysis import metrics

        rows = ordered_records(rows, "capture")
        annotated = metrics.annotate_capture_by_pipeline(list(rows), reference_pipeline=PROPOSED_PIPELINE)
        models = sorted({str(r.get("model")) for r in annotated})
        pipelines = sorted({str(r.get("pipeline")) for r in annotated})
        langs = sorted({str(r.get("lang")) for r in annotated})
        direct_counts = {m: sum(1 for r in annotated if r.get("model") == m and r.get("pipeline") == PROPOSED_PIPELINE) for m in models}
        primary_model = max(models, key=lambda k: (direct_counts[k], k)) if any(direct_counts.values()) else None
        self.groups["capture"] = {"models": models, "model_slugs": {m: model_slug(m) for m in models}, "pipelines": pipelines, "langs": langs, "primary_model": primary_model}
        fields = list(metrics.CAPTURE_FIELDS)
        for m in models:
            ms = model_slug(m)
            for p in pipelines:
                sub = [r for r in annotated if r.get("model") == m and r.get("pipeline") == p]
                if not sub:
                    continue
                for lang in langs + ["all"]:
                    g = sub if lang == "all" else [r for r in sub if r.get("lang") == lang]
                    if not g:
                        continue
                    tok = f"{lang}_{p}_{ms}"
                    cl = [r.get("seed_id") for r in g]
                    where = f"{lang} members under pipeline {p} and model {m}" if lang != "all" else f"all members under pipeline {p} and model {m}"
                    self.count_claim(f"cap_n_{tok}", len(g), f"Capture records: {where}.", source, "variant_id")
                    self.cluster_rate(f"cap_invariance_{tok}", [r.get("invariant") for r in g], cl, f"Matched-pair invariance rate (outcome equals the English base member's outcome) over {where}.", source, "invariant")
                    self.cluster_rate(f"cap_sensitivity_{tok}", [r.get("sensitive") for r in g], cl, f"Sensitivity rate (an edit's outcome differs from its base member's) over {where}.", source, "sensitive")
                    self.cluster_rate(f"cap_accuracy_{tok}", [r.get("correct") for r in g], cl, f"Reference-policy accuracy over {where}.", source, "correct")
                    self.cluster_rate(f"cap_error_{tok}", [not bool(r.get("correct")) for r in g], cl, f"Capture error rate (one minus accuracy) over {where}.", source, "correct")
                    rec = [r for r in g if r.get("intended_kind") == "record"]
                    amb = [r for r in g if r.get("intended_kind") == "ambiguous"]
                    self.cluster_rate(f"cap_exact_match_{tok}", [r.get("exact_match") for r in rec], [r.get("seed_id") for r in rec], f"Record exact-match rate on turns that intend a record over {where}.", source, "exact_match")
                    self.cluster_rate(f"cap_ask_unambiguous_{tok}", [r.get("asked") for r in rec], [r.get("seed_id") for r in rec], f"Clarification rate on unambiguous turns (unnecessary questions) over {where}.", source, "asked")
                    self.cluster_rate(f"cap_ask_ambiguous_{tok}", [r.get("asked") for r in amb], [r.get("seed_id") for r in amb], f"Clarification rate on ambiguous turns over {where}.", source, "asked")
                    self.cluster_rate(f"cap_ask_correct_field_{tok}", [r.get("ask_correct") for r in amb], [r.get("seed_id") for r in amb], f"Share of ambiguous turns asked about exactly the ambiguous field over {where}.", source, "ask_correct")
                    self.cluster_rate(f"cap_parse_failure_{tok}", [bool(r.get("parse_failure")) for r in g], cl, f"Parse-failure rate over {where}.", source, "parse_failure")
                    for f in fields:
                        self.cluster_rate(f"cap_field_error_{f}_{tok}", [r.get(f"field_error_{f}") for r in rec], [r.get("seed_id") for r in rec], f"Field error rate of {f} among produced records over {where}.", source, f"field_error_{f}")
                    failures = [r for r in g if r.get("invariant") is False]
                    if lang != "all":
                        self.count_claim(f"cap_invariance_failures_{tok}", len(failures), f"Members whose outcome differs from the English base member over {where}.", source, "invariant")
                        counts: dict[str, int] = {}
                        for r in failures:
                            for ph in r.get("phenomena") or []:
                                counts[str(ph)] = counts.get(str(ph), 0) + 1
                        for ph, c in sorted(counts.items()):
                            self.count_claim(f"cap_invariance_failures_{tok}_{ph}", c, f"Invariance failures carrying phenomenon {ph} over {where} (a member counts under each of its phenomena).", source, "phenomena")
                        named = sorted(((counts.get(ph, 0), ph) for ph in NAMED_PHENOMENA), reverse=True)[:2]
                        top = {ph for _, ph in named}
                        covered = sum(1 for r in failures if top & set(str(x) for x in (r.get("phenomena") or [])))
                        if lang != "en":
                            self.binomial_rate(f"h1_concentration_{tok}", covered, len(failures), f"Share of invariance failures over {where} that carry at least one of the two named phenomena with the most failures ({', '.join(sorted(top))}).", source, "phenomena", threshold=THRESHOLDS["h1_concentration"])
                en_rows = [r for r in sub if r.get("lang") == "en" and r.get("invariant") is not None]
                for lang in langs:
                    if lang == "en":
                        continue
                    other = [r for r in sub if r.get("lang") == lang and r.get("invariant") is not None]
                    cid = f"h1_invariance_diff_en_minus_{lang}_{p}_{ms}"
                    both = en_rows + other
                    if not en_rows or not other:
                        self.ledger.add(cid, None, "fraction", n=len(both), computed_by="tasdiq.analysis.stats.cluster_bootstrap", description=f"English-minus-{lang} invariance difference under pipeline {p} and model {m}; one language has no members.", threshold=THRESHOLDS["h1_invariance_diff"], **self._src(source))
                        continue
                    mat = np.asarray([[float(bool(r.get("invariant"))), 1.0 if r.get("lang") == "en" else 0.0] for r in both], dtype=float)
                    tested = p == PROPOSED_PIPELINE and m == primary_model
                    note = " Secondary family of the protocol: two-sided cluster permutation p-value (language labels swapped within whole seeds) with its Holm adjustment across the family." if tested else ""
                    self.cluster_stat(cid, mat, [r.get("seed_id") for r in both], _lang_diff_stat, "fraction", f"English-minus-{lang} invariance difference (percentage of members as a fraction) under pipeline {p} and model {m}, clustered on the seed.{note}", source, "invariant", threshold=THRESHOLDS["h1_invariance_diff"], permutation="swap" if tested else None, family="secondary" if tested else None)
            self._add_h2_capture(annotated, m, source)
        self._add_capture_latency(rows, source)

    def _add_h2_capture(self, annotated: Sequence[Mapping[str, Any]], model: str, source: Mapping[str, Any]) -> None:
        """Capture-error reductions of the proposed pipeline and of translate-then-bind against the same parser."""
        ms = model_slug(model)
        by_pipe: dict[str, dict[Any, Mapping[str, Any]]] = {}
        for r in annotated:
            if r.get("model") == model:
                by_pipe.setdefault(str(r.get("pipeline")), {})[r.get("variant_id")] = r
        for p, rows in sorted(by_pipe.items()):
            vals = list(rows.values())
            self.cluster_rate(f"h2_capture_error_{p}_{ms}", [not bool(r.get("correct")) for r in vals], [r.get("seed_id") for r in vals], f"Capture error rate of pipeline {p} under model {model} on every member.", source, "correct")
        base = by_pipe.get(SAME_PARSER_PIPELINE)
        if not base:
            return
        for p in (PROPOSED_PIPELINE, TRANSLATE_PIPELINE, MT_TRANSLATE_PIPELINE, "nogate"):
            rows = by_pipe.get(p)
            if not rows:
                continue
            keys = [k for k in rows if k in base]
            if not keys:
                continue
            mat = np.asarray([[0.0 if base[k].get("correct") else 1.0, 0.0 if rows[k].get("correct") else 1.0] for k in keys], dtype=float)
            cl = [rows[k].get("seed_id") for k in keys]
            self.cluster_stat(f"h2_capture_error_reduction_{p}_vs_{SAME_PARSER_PIPELINE}_{ms}", mat, cl, _relative_reduction_stat, "fraction", f"Relative reduction of the capture error rate of pipeline {p} against {SAME_PARSER_PIPELINE} (same parser) under model {model}, paired on the variant and clustered on the seed.", source, "correct", threshold=THRESHOLDS["h2_relative_reduction"] if p == PROPOSED_PIPELINE else None)
            self.cluster_stat(f"h2_capture_error_abs_reduction_{p}_vs_{SAME_PARSER_PIPELINE}_{ms}", mat, cl, _abs_reduction_stat, "fraction", f"Absolute reduction (fraction of members) of the capture error rate of pipeline {p} against {SAME_PARSER_PIPELINE} under model {model}.", source, "correct")
            rec_keys = [k for k in keys if rows[k].get("intended_kind") == "record"]
            self.paired_diff(f"h2_ask_unambiguous_added_{p}_minus_{SAME_PARSER_PIPELINE}_{ms}", [1.0 if rows[k].get("asked") else 0.0 for k in rec_keys], [1.0 if base[k].get("asked") else 0.0 for k in rec_keys], [rows[k].get("seed_id") for k in rec_keys], "fraction", f"Added clarification rate on unambiguous turns of pipeline {p} over {SAME_PARSER_PIPELINE} under model {model} (fraction of members).", source, "asked", threshold=THRESHOLDS["h2_added_confirmations"] if p == PROPOSED_PIPELINE else None)
        direct = by_pipe.get(PROPOSED_PIPELINE)
        for comparator in TRANSLATION_COMPARATORS:
            translate = by_pipe.get(comparator)
            if not direct or not translate:
                continue
            keys = [k for k in direct if k in translate and k in base]
            if not keys:
                continue
            a = [(0.0 if base[k].get("correct") else 1.0) - (0.0 if direct[k].get("correct") else 1.0) for k in keys]
            b = [(0.0 if base[k].get("correct") else 1.0) - (0.0 if translate[k].get("correct") else 1.0) for k in keys]
            self.paired_diff(f"h2_reduction_diff_{PROPOSED_PIPELINE}_minus_{comparator}_{ms}", a, b, [direct[k].get("seed_id") for k in keys], "fraction", f"Difference between the absolute error reductions of {PROPOSED_PIPELINE} and {comparator} against {SAME_PARSER_PIPELINE} under model {model}, paired on the variant.", source, "correct", threshold=0.0)

    def _add_capture_latency(self, rows: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> None:
        """Per-stage timing quantiles of the capture pipelines and their peak memory."""
        by: dict[tuple[str, str, str], list[float]] = {}
        peak = 0
        for r in rows:
            timings = r.get("timings") or {}
            for stage in CAPTURE_STAGES:
                v = timings.get(stage)
                if v is None:
                    continue
                if stage == "translate" and not float(v):
                    continue
                by.setdefault((stage, str(r.get("pipeline")), str(r.get("model"))), []).append(float(v))
            peak = max(peak, int(r.get("peak_rss_bytes") or 0))
        latency = self.groups.setdefault("latency", [])
        for (stage, p, m), vals in sorted(by.items()):
            tok = f"{stage}_{p}_{model_slug(m)}"
            latency.append({"stage": stage, "pipeline": p, "model": m, "token": tok})
            for stat, v in _quantiles(vals).items():
                unit = "count" if stat == "n" else "seconds"
                self.value_claim(f"lat_{tok}_{stat}", v, unit, f"Latency of stage {stage} under pipeline {p} and model {m}: {stat} over timed calls.", source, f"timings.{stage}", n=len(vals))
        if peak:
            self.value_claim("mem_peak_rss_bytes_capture", peak, "bytes", "Peak resident set size of a capture worker.", source, "peak_rss_bytes")

    # -- trace track ------------------------------------------------------------------

    def add_trace(self, sessions: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> None:
        """Enforcement claims per arm, rule, window, family and language, with H2 to H4.

        Sessions of the primary reader (``exact``) feed every claim.  Sessions of another
        reader (the inattentive sensitivity reader) feed a parallel set of enforcement
        claims whose ids carry the token ``reader_<name>`` after the ``enf_`` prefix, and
        paired differences against the primary reader at the primary window
        (``sens_reader_<name>_minus_exact_<metric>_<arm>_w<window>``).
        """
        if not sessions:
            return
        all_rows = [dict(r) for r in ordered_records(sessions, "trace")]
        for r in all_rows:
            r["rule"] = "none" if r.get("rule") in (None, "") else str(r["rule"])
            r["armtok"] = arm_token(str(r.get("arm")), r["rule"])
            r["n_intent"] = _count(r.get("intent_violations"))
            r["n_unconfirmed"] = _unconfirmed_count(r)
            r["n_unconfirmed_intended"] = _unconfirmed_count(r, intended=True)
            r["n_capture_class"] = _count(r.get("capture_class_events"))
            r["reader_name"] = reader_of(r)
        readers = sorted({r["reader_name"] for r in all_rows})
        primary_reader = PRIMARY_READER if PRIMARY_READER in readers else readers[0]
        rows = [r for r in all_rows if r["reader_name"] == primary_reader]
        combined = source
        source = _records_source(rows) or combined
        windows = sorted({r.get("window_s") for r in rows}, key=lambda w: (w is None, w if w is not None else 0))
        primary_window = 600 if 600 in windows else windows[0]
        armtoks = sorted({r["armtok"] for r in rows})
        families = sorted({str(r.get("family")) for r in rows})
        langs = sorted({str(r.get("lang")) for r in rows})
        self.groups["trace"] = {"arms": armtoks, "windows": [w for w in windows], "primary_window": primary_window, "families": families, "langs": langs, "capture_mode": str(rows[0].get("mode", "oracle")), "readers": readers, "primary_reader": primary_reader}
        for a in armtoks:
            for w in windows:
                g = [r for r in rows if r["armtok"] == a and r.get("window_s") == w]
                if g:
                    self._add_enforcement_group(g, a, w, source, families)
        self._add_attribution(rows, armtoks, langs, primary_window, source)
        self._add_calibration(rows, source)
        self._add_h2_premise(rows, source)
        self._add_revision_curves(rows, primary_window, source)
        self._add_window_curves(rows, windows, source)
        self._add_trace_latency(rows, source)
        for reader in readers:
            if reader != primary_reader:
                reader_rows = [r for r in all_rows if r["reader_name"] == reader]
                self._add_reader_sensitivity(rows, reader_rows, reader, primary_window, _records_source(reader_rows) or combined, combined)

    def _add_reader_sensitivity(self, primary: list[dict[str, Any]], rows: list[dict[str, Any]], reader: str, w: Any, source: Mapping[str, Any], combined: Mapping[str, Any]) -> None:
        """Enforcement claims of a sensitivity reader and its paired differences against the primary reader.

        The parallel claims cover every arm and window of the reader's sessions and name the
        reader's run as their source.  The differences are taken at the primary window (an
        arm that runs at exactly one window of its own is read at that window), aligned on
        the variant and clustered on the seed, for the share of sessions with an executed
        intent violation, with a declined presentation and with the intended payments
        completed; they name both runs (``combined``) as their source.
        """
        windows = sorted({r.get("window_s") for r in rows}, key=lambda x: (x is None, x if x is not None else 0))
        armtoks = sorted({r["armtok"] for r in rows})
        families = sorted({str(r.get("family")) for r in rows})
        for a in armtoks:
            for win in windows:
                g = [r for r in rows if r["armtok"] == a and r.get("window_s") == win]
                if g:
                    self._add_enforcement_group(g, a, win, source, families, reader=reader)
        for a in armtoks:
            own = sorted({r.get("window_s") for r in rows if r["armtok"] == a}, key=lambda x: (x is None, x if x is not None else 0))
            window = w if w in own else (own[0] if len(own) == 1 else w)
            wt = window_token(window)
            base = {r.get("variant_id"): r for r in primary if r["armtok"] == a and r.get("window_s") == window}
            other = {r.get("variant_id"): r for r in rows if r["armtok"] == a and r.get("window_s") == window}
            keys = [k for k in other if k in base]
            if not keys:
                continue
            for metric, flag, label in (
                ("intent_violation_sessions", lambda r: r["n_intent"] > 0, "share of sessions with at least one executed INTENT-VIOLATION"),
                ("declined_presentation_sessions", lambda r: _count(r.get("declined_presentations")) > 0, "share of sessions with at least one declined presentation"),
                ("completion", lambda r: bool(r.get("completed")), "completion rate of the intended payments"),
            ):
                cid = f"sens_reader_{reader}_minus_{PRIMARY_READER}_{metric}_{a}_{wt}"
                self.paired_diff(cid, [1.0 if flag(other[k]) else 0.0 for k in keys], [1.0 if flag(base[k]) else 0.0 for k in keys], [base[k].get("seed_id") for k in keys], "fraction", f"Sensitivity of the reader model: {label} under the {reader} reader minus the same share under the {PRIMARY_READER} reader, arm {a} at window {window} s, aligned on the variant and clustered on the seed.", combined, metric)

    def _add_trace_latency(self, rows: list[dict[str, Any]], source: Mapping[str, Any]) -> None:
        """Quantiles of the timed monitor hook calls of the trace track (monitor_seconds per session)."""
        monitor: list[float] = []
        for r in rows:
            values = r.get("monitor_seconds")
            if isinstance(values, (list, tuple)):
                monitor.extend(float(v) for v in values if v is not None)
        if not monitor:
            return
        latency = self.groups.setdefault("latency", [])
        latency.append({"stage": "monitor", "pipeline": "trace", "model": "all", "token": "monitor_trace"})
        for stat, v in _quantiles(monitor).items():
            self.value_claim(f"lat_monitor_trace_{stat}", v, "count" if stat == "n" else "seconds", f"Latency of one monitor hook call (consent request, execution or user event) on the trace track: {stat} over timed calls.", source, "monitor_seconds", n=len(monitor))

    def _add_enforcement_group(self, g: list[dict[str, Any]], a: str, w: Any, source: Mapping[str, Any], families: Sequence[str], reader: str | None = None) -> None:
        """Enforcement claims of one arm at one window; ``reader`` names a sensitivity reader whose claims carry its token."""
        wt = window_token(w)
        tok = f"{a}_{wt}" if reader is None else f"reader_{reader}_{a}_{wt}"
        cl = [r.get("seed_id") for r in g]
        where = f"arm {a} at window {w} s" if reader is None else f"arm {a} at window {w} s under the {reader} reader"
        executions = sum(_count(r.get("executions")) for r in g)
        attempts = sum(_count(r.get("execution_attempts")) for r in g)
        intent = sum(r["n_intent"] for r in g)
        self.count_claim(f"enf_sessions_{tok}", len(g), f"Sessions under {where}.", source, "variant_id")
        self.count_claim(f"enf_executions_{tok}", executions, f"Rail executions under {where}.", source, "executions")
        self.count_claim(f"enf_intent_violations_{tok}", intent, f"Executed INTENT-VIOLATION labels under {where}.", source, "intent_violations")
        self.binomial_rate(f"enf_intent_violation_rate_{tok}", intent, executions, f"INTENT-VIOLATION executions per rail execution under {where}.", source, "intent_violations")
        self.cluster_rate(f"enf_intent_violation_sessions_{tok}", [r["n_intent"] > 0 for r in g], cl, f"Share of sessions with at least one INTENT-VIOLATION execution under {where}.", source, "intent_violations")
        classes: dict[str, int] = {}
        for r in g:
            for c in _classes(r.get("intent_violations")):
                classes[c] = classes.get(c, 0) + 1
        for c, k in sorted(classes.items()):
            self.count_claim(f"enf_intent_class_{c}_{tok}", k, f"INTENT-VIOLATION executions of class {c} under {where}.", source, "intent_violations")
        primary = sum(_count(r.get("violations")) for r in g)
        self.binomial_rate(f"enf_unauthorised_rate_{tok}", primary, attempts, f"Primary-label violations per execution attempt under {where}.", source, "violations")
        self.binomial_rate(f"enf_unconfirmed_rate_{tok}", sum(r["n_unconfirmed"] for r in g), executions, f"UNCONFIRMED executions (consent layer) per rail execution under {where}.", source, "unconfirmed_executions")
        self.binomial_rate(f"enf_unconfirmed_intended_rate_{tok}", sum(r["n_unconfirmed_intended"] for r in g), executions, f"Executions that were unconfirmed but intended per rail execution under {where}.", source, "unconfirmed_intended")
        self.cluster_rate(f"enf_completion_{tok}", [bool(r.get("completed")) for r in g], cl, f"Completion rate of the intended payments (intent layer) under {where}.", source, "completed")
        self.cluster_rate(f"enf_completion_confirmed_{tok}", [r.get("completed_confirmed") for r in g], cl, f"Completion rate with every execution also CONFIRMED under {where}.", source, "completed_confirmed")
        self.cluster_rate(f"enf_unnecessary_confirmation_sessions_{tok}", [_count(r.get("unnecessary_confirmations")) > 0 for r in g], cl, f"Share of sessions with at least one unnecessary confirmation under {where}.", source, "unnecessary_confirmations")
        if any("unnecessary_confirmations_narrowing" in r for r in g):
            self.cluster_rate(f"enf_unnecessary_reconf_narrowing_sessions_{tok}", [_count(r.get("unnecessary_confirmations_narrowing")) > 0 for r in g], cl, f"Share of sessions with at least one unnecessary re-confirmation under the narrowing reading of the label (unnecessary_confirmations_narrowing) under {where}.", source, "unnecessary_confirmations_narrowing")
        self.binomial_rate(f"enf_unnecessary_per_confirmation_{tok}", sum(_count(r.get("unnecessary_confirmations")) for r in g), sum(_count(r.get("confirmations")) for r in g), f"Unnecessary confirmations per confirmation request under {where}.", source, "unnecessary_confirmations")
        self.binomial_rate(f"enf_false_block_rate_{tok}", sum(_count(r.get("false_blocks")) for r in g), sum(_count(r.get("authorised_requests")) for r in g), f"Authorised requests blocked per authorised request under {where}.", source, "false_blocks")
        for name, label in (("duplicates", "duplicate execution"), ("stale_executions", "stale execution"), ("forced_reconfirmations", "forced re-confirmation"), ("capture_class_events", "capture-class event"), ("declined_presentations", "declined presentation")):
            short = {"duplicates": "duplicate", "stale_executions": "stale", "forced_reconfirmations": "forced_reconf", "capture_class_events": "capture_class", "declined_presentations": "declined_presentation"}[name]
            self.cluster_rate(f"enf_{short}_sessions_{tok}", [_count(r.get(name)) > 0 for r in g], cl, f"Share of sessions with at least one {label} under {where}.", source, name)
        self.count_claim(f"enf_declined_presentations_{tok}", sum(_count(r.get("declined_presentations")) for r in g), f"Presentations declined by the simulated reader under {where}.", source, "declined_presentations")
        for fam in families:
            fg = [r for r in g if r.get("family") == fam]
            if not fg:
                continue
            fcl = [r.get("seed_id") for r in fg]
            ftok = f"{a}_{fam}_{wt}" if reader is None else f"reader_{reader}_{a}_{fam}_{wt}"
            self.count_claim(f"enf_sessions_{ftok}", len(fg), f"Sessions of family {fam} under {where}.", source, "variant_id")
            self.count_claim(f"enf_intent_violations_{ftok}", sum(r["n_intent"] for r in fg), f"Executed INTENT-VIOLATION labels on family {fam} under {where}.", source, "intent_violations")
            self.cluster_rate(f"enf_intent_violation_sessions_{ftok}", [r["n_intent"] > 0 for r in fg], fcl, f"Share of family {fam} sessions with at least one INTENT-VIOLATION execution under {where}.", source, "intent_violations")
            self.cluster_rate(f"enf_completion_{ftok}", [bool(r.get("completed")) for r in fg], fcl, f"Completion rate on family {fam} under {where}.", source, "completed")
            self.cluster_rate(f"enf_unnecessary_confirmation_sessions_{ftok}", [_count(r.get("unnecessary_confirmations")) > 0 for r in fg], fcl, f"Share of family {fam} sessions with at least one unnecessary confirmation under {where}.", source, "unnecessary_confirmations")

    def _add_attribution(self, rows: list[dict[str, Any]], armtoks: Sequence[str], langs: Sequence[str], w: Any, source: Mapping[str, Any]) -> None:
        """Capture-class and enforcement-class session shares per arm and language at the primary window.

        An arm without sessions at the primary window that runs at exactly one window of its
        own (B3 at its elevation window) is read at that window, under that window's token.
        """
        for a in armtoks:
            arm_rows = [r for r in rows if r["armtok"] == a]
            own = sorted({r.get("window_s") for r in arm_rows}, key=lambda x: (x is None, x if x is not None else 0))
            window = w if any(r.get("window_s") == w for r in arm_rows) else (own[0] if len(own) == 1 else w)
            wt = window_token(window)
            for lang in list(langs) + ["all"]:
                g = [r for r in arm_rows if r.get("window_s") == window and (lang == "all" or r.get("lang") == lang)]
                if not g:
                    continue
                cl = [r.get("seed_id") for r in g]
                where = f"arm {a}, language {lang}, window {window} s"
                self.count_claim(f"attr_sessions_{a}_{lang}_{wt}", len(g), f"Sessions under {where}.", source, "variant_id")
                self.cluster_rate(f"attr_capture_class_{a}_{lang}_{wt}", [r["n_capture_class"] > 0 for r in g], cl, f"Share of sessions with a capture-class error (declined presentation, unnecessary clarification or wrong unpresented record) under {where}.", source, "capture_class_events")
                self.cluster_rate(f"attr_enforcement_class_{a}_{lang}_{wt}", [r["n_intent"] > 0 for r in g], cl, f"Share of sessions with an enforcement-class error (an executed INTENT-VIOLATION) under {where}.", source, "intent_violations")

    def _add_calibration(self, rows: list[dict[str, Any]], source: Mapping[str, Any]) -> None:
        """Declared window of every arm, from the session records or the arm's default construction.

        An arm whose sessions carry a single enforced window (B3 at its elevation window, or
        an arm without a window) is read from the records.  An arm whose window the trace
        track sweeps carries one window per session, so its claim reports the window of the
        arm's default construction (the window the protocol states for the arm).
        """
        arms = sorted({str(r.get("arm")) for r in rows})
        swept = sorted({int(r["window_s"]) for r in rows if r.get("window_s") is not None})
        self.text_claim("cal_windows_swept", ", ".join(str(w) for w in swept), "Validity windows swept on the trace track, in seconds.", source, "window_s")
        for arm in arms:
            recorded = sorted({r.get("arm_window_s") for r in rows if r.get("arm") == arm and "arm_window_s" in r}, key=lambda v: (v is None, v if v is not None else 0))
            value: Any = None
            field = "arm_window_s"
            note = ""
            if len(recorded) == 1:
                value = recorded[0]
            else:
                try:
                    from tasdiq.policy.arms import make_arm

                    default_arm = make_arm(arm)
                    value = getattr(default_arm, "window_s", None)
                    if value is None:
                        value = getattr(default_arm, "interval_s", None)
                except Exception:
                    value = None
                field = "make_arm"
                note = " (the window of the arm's default construction; the trace track sweeps the window of this arm)"
            if value is None or int(value) >= NO_WINDOW_S:
                self.text_claim(f"cal_window_s_{arm}", "none", f"Validity window enforced by arm {arm}: the arm keeps no window{note}.", source, field)
            else:
                self.value_claim(f"cal_window_s_{arm}", int(value), "seconds", f"Validity window enforced by arm {arm} in seconds{note}.", source, field)

    def _add_h2_premise(self, rows: list[dict[str, Any]], source: Mapping[str, Any]) -> None:
        """Intent violations of B4 and the proposed system on oracle sessions outside the measured classes.

        The premise count of an arm covers the sessions under the premise rules and excludes
        the arm's measured classes (lapsed for every windowed arm; cancelled and superseded
        for B4), which are reported per class as separate claims. The measured classes of
        the proposed system under cap_settle (superseded and amount_change, with lapsed) are
        reported per class as well; that rule takes no part in the premise count.
        """
        oracle = [r for r in rows if str(r.get("mode", "oracle")) == "oracle"]
        if not oracle:
            return
        try:
            from tasdiq.runners.trace import PREMISE_RULES, measured_classes
        except Exception:
            PREMISE_RULES = frozenset({"none", "invalidate", "narrowing"})

            def measured_classes(arm: str, rule: str | None = None) -> frozenset[str]:
                return frozenset()
        for arm in (CLASSICAL_ARM, PRIMARY_ARM):
            g = [r for r in oracle if r.get("arm") == arm and r["rule"] in PREMISE_RULES]
            if not g:
                continue
            excluded = measured_classes(arm)
            k = 0
            measured: dict[str, int] = {}
            for r in g:
                for c in _classes(r.get("intent_violations")):
                    if c in measured_classes(arm, str(r["rule"])):
                        measured[c] = measured.get(c, 0) + 1
                    else:
                        k += 1
            executions = sum(_count(r.get("executions")) for r in g)
            self.count_claim(f"h2_premise_intent_violations_{arm}", k, f"INTENT-VIOLATION executions of arm {arm} on oracle sessions under the premise rules, excluding its measured classes ({', '.join(sorted(excluded)) or 'none'}).", source, "intent_violations")
            self.binomial_rate(f"h2_premise_intent_violation_rate_{arm}", k, executions, f"The same count per rail execution of arm {arm}; the exact upper bound is the reported bound of the premise.", source, "intent_violations", threshold=0.0)
            for c, v in sorted(measured.items()):
                self.count_claim(f"h2_premise_measured_{arm}_{c}", v, f"Measured class of arm {arm} under the premise rules: executions of class {c}, reported rather than required to be zero.", source, "intent_violations")
        other_rules = sorted({str(r["rule"]) for r in oracle if r.get("arm") == PRIMARY_ARM and r["rule"] not in PREMISE_RULES})
        for rule in other_rules:
            g = [r for r in oracle if r.get("arm") == PRIMARY_ARM and str(r["rule"]) == rule]
            excluded = measured_classes(PRIMARY_ARM, rule)
            k = 0
            measured = {}
            for r in g:
                for c in _classes(r.get("intent_violations")):
                    if c in excluded:
                        measured[c] = measured.get(c, 0) + 1
                    else:
                        k += 1
            self.count_claim(f"h2_measured_rule_intent_violations_{PRIMARY_ARM}_{rule}", k, f"INTENT-VIOLATION executions of arm {PRIMARY_ARM} under the {rule} rule on oracle sessions outside its measured classes ({', '.join(sorted(excluded)) or 'none'}); the rule takes no part in the premise.", source, "intent_violations")
            for c, v in sorted(measured.items()):
                self.count_claim(f"h2_measured_rule_{PRIMARY_ARM}_{rule}_{c}", v, f"Measured class of arm {PRIMARY_ARM} under the {rule} rule: executions of class {c}, reported rather than required to be zero.", source, "intent_violations")

    def _add_revision_curves(self, rows: list[dict[str, Any]], w: Any, source: Mapping[str, Any]) -> None:
        """Operating points of the revision rules at the primary window, with H3.

        The ``rev_curve_`` claims cover the revision family and the ``rev_all_`` claims every
        family. An arm that enforces a window of its own (B3, which takes no swept window) is
        read at that window, as the enforcement claims read it.
        """
        at_window = [r for r in rows if r.get("window_s") == w or r.get("arm") == ELEVATION_ARM]
        fam = [r for r in at_window if r.get("family") == REVISION_FAMILY]
        if not fam:
            return
        unn_field = "unnecessary_confirmations_narrowing" if any("unnecessary_confirmations_narrowing" in r for r in fam) else "unnecessary_confirmations"
        points: dict[str, list[dict[str, Any]]] = {}
        for scope, prefix, subset in (("the revision family", "rev_curve", fam), ("every family", "rev_all", at_window)):
            scoped: dict[str, list[dict[str, Any]]] = {}
            for r in subset:
                tok = curve_token(str(r.get("arm")), r["rule"])
                if tok is not None:
                    scoped.setdefault(tok, []).append(r)
            if prefix == "rev_curve":
                points = scoped
                self.groups["revision_curve"] = {"tokens": sorted(points), "window": w}
            for tok, g in sorted(scoped.items()):
                cl = [r.get("seed_id") for r in g]
                executions = sum(_count(r.get("executions")) for r in g)
                where = f"operating point {tok} on {scope} at window {w} s (an arm with a window of its own at that window)"
                self.count_claim(f"{prefix}_{tok}_sessions", len(g), f"Sessions at {where}.", source, "variant_id")
                self.cluster_rate(f"{prefix}_{tok}_unnecessary_reconf", [_count(r.get(unn_field)) > 0 for r in g], cl, f"Share of sessions with at least one unnecessary re-confirmation under the narrowing reading of the label ({unn_field}) at {where}.", source, unn_field)
                self.cluster_mean(f"{prefix}_{tok}_confirmations_per_session", [float(_count(r.get("confirmations"))) for r in g], cl, "count", f"Mean number of confirmation requests per session at {where}.", source, "confirmations")
                self.count_claim(f"{prefix}_{tok}_intent_violations", sum(r["n_intent"] for r in g), f"Executed INTENT-VIOLATION labels at {where}.", source, "intent_violations")
                self.binomial_rate(f"{prefix}_{tok}_intent_violation_rate", sum(r["n_intent"] for r in g), executions, f"INTENT-VIOLATION executions per rail execution at {where}.", source, "intent_violations")
                self.binomial_rate(f"{prefix}_{tok}_unconfirmed_intended_share", sum(r["n_unconfirmed_intended"] for r in g), executions, f"Executions that were unconfirmed but intended per rail execution at {where} (the relaxation of the Art. 5(1)(d) reading that the rule permits).", source, "unconfirmed_intended")
                self.cluster_rate(f"{prefix}_{tok}_completion", [bool(r.get("completed")) for r in g], cl, f"Completion rate at {where}.", source, "completed")
        narrowing = {r.get("variant_id"): r for r in points.get(NARROWING_RULE, [])}
        for base_tok in (PRIMARY_RULE, CLASSICAL_ARM):
            base = {r.get("variant_id"): r for r in points.get(base_tok, [])}
            keys = [k for k in base if k in narrowing]
            cid = f"h3_unnecessary_reconf_diff_{base_tok}_minus_{NARROWING_RULE}"
            if not keys:
                self.ledger.add(cid, None, "fraction", n=0, computed_by="tasdiq.analysis.stats.paired_cluster_diff", description=f"Paired difference in the unnecessary re-confirmation session rate between {base_tok} and the narrowing rule on the revision family; no aligned sessions.", threshold=THRESHOLDS["h3_unnecessary_reconf_diff"], **self._src(source))
                continue
            tested = base_tok == PRIMARY_RULE
            note = " Secondary family of the protocol: two-sided paired cluster permutation p-value (rules swapped within whole seeds) with its Holm adjustment across the family." if tested else ""
            self.paired_diff(cid, [1.0 if _count(base[k].get(unn_field)) > 0 else 0.0 for k in keys], [1.0 if _count(narrowing[k].get(unn_field)) > 0 else 0.0 for k in keys], [base[k].get("seed_id") for k in keys], "fraction", f"Paired difference (fraction of sessions) in the unnecessary re-confirmation session rate between {base_tok} and the narrowing rule on the revision family at window {w} s, aligned on the variant and clustered on the seed.{note}", source, unn_field, threshold=THRESHOLDS["h3_unnecessary_reconf_diff"], permutation=tested, family="secondary" if tested else None)
        for tok in (NARROWING_RULE, PRIMARY_RULE, CLASSICAL_ARM):
            g = points.get(tok)
            if g:
                k = sum(r["n_intent"] for r in g)
                self.binomial_rate(f"h3_{tok}_intent_violation_rate", k, sum(_count(r.get("executions")) for r in g), f"INTENT-VIOLATION executions per rail execution of {tok} on the revision family; the exact upper bound is reported when the count is zero.", source, "intent_violations", threshold=0.0)
        b3 = [r for r in at_window if r.get("arm") == ELEVATION_ARM]
        if b3:
            executions = sum(_count(r.get("executions")) for r in b3)
            classes: dict[str, int] = {}
            for r in b3:
                for c in _classes(r.get("intent_violations")):
                    classes[c] = classes.get(c, 0) + 1
            for c in ("substitution", "superseded", "amount_change", "currency_change"):
                if c in classes:
                    self.binomial_rate(f"h3_{ELEVATION_ARM}_intent_class_{c}_rate", classes[c], executions, f"INTENT-VIOLATION executions of class {c} per rail execution of {ELEVATION_ARM} over every family at the window the arm enforces.", source, "intent_violations", threshold=0.0)

    def _add_window_curves(self, rows: list[dict[str, Any]], windows: Sequence[Any], source: Mapping[str, Any]) -> None:
        """Stale-execution and forced re-confirmation rates per window on the delayed-execution family, with the three denominators."""
        fam = [r for r in rows if r.get("family") == DELAY_FAMILY]
        if not fam:
            return
        armtoks = sorted({r["armtok"] for r in fam})
        primary = arm_token(PRIMARY_ARM, PRIMARY_RULE)
        self.groups["window_curve"] = {"arms": armtoks, "windows": [w for w in windows if w is not None], "primary": primary}
        for w in windows:
            if w is None:
                continue
            wt = int(w)
            for a in armtoks:
                g = [r for r in fam if r["armtok"] == a and r.get("window_s") == w]
                if not g:
                    continue
                valid = [r for r in g if r["n_capture_class"] == 0]
                cl = [r.get("seed_id") for r in valid]
                suffixes = [f"_{a}"] + ([""] if a == primary else [])
                for suffix in suffixes:
                    where = f"arm {a} at window {w} s on the delayed-execution family"
                    self.cluster_rate(f"win_curve_{wt}_stale_rate{suffix}", [_count(r.get("stale_executions")) > 0 for r in valid], cl, f"Share of sessions with a stale execution among sessions whose consent was accepted and valid at capture, {where}.", source, "stale_executions")
                    self.cluster_rate(f"win_curve_{wt}_forced_reconf_rate{suffix}", [_count(r.get("forced_reconfirmations")) > 0 for r in valid], cl, f"Share of sessions with a forced re-confirmation among sessions whose consent was accepted and valid at capture, {where}.", source, "forced_reconfirmations")
                    self.count_claim(f"win_curve_{wt}_sessions{suffix}", len(valid), f"Sessions with a valid captured consent, {where}.", source, "variant_id")
                    self.cluster_rate(f"den_stale_valid_consents_{wt}{suffix}", [_count(r.get("stale_executions")) > 0 for r in valid], cl, f"Second denominator of the window outcome: stale executions among consents accepted and valid at capture, {where}.", source, "stale_executions")
                    self.cluster_rate(f"den_end_to_end_invalid_{wt}{suffix}", [(r["n_intent"] > 0 or r["n_capture_class"] > 0) for r in g], [r.get("seed_id") for r in g], f"Third denominator of the window outcome: sessions with an end-to-end invalid outcome (an executed INTENT-VIOLATION or a capture-stage error) among all sessions, {where}.", source, "intent_violations")
            reference = [r for r in fam if r.get("arm") == NO_MONITOR_ARM and r.get("window_s") == w]
            if reference:
                self.cluster_rate(f"den_context_change_{wt}", [_count(r.get("stale_executions")) > 0 for r in reference], [r.get("seed_id") for r in reference], f"First denominator of the window outcome: sessions whose execution falls after the reference window of {w} s regardless of the monitor, read from the no-monitor arm {NO_MONITOR_ARM} on the delayed-execution family.", source, "stale_executions")

    # -- live track --------------------------------------------------------------------

    def add_live(self, episodes: Sequence[Mapping[str, Any]], source: Mapping[str, Any], steps: Sequence[Mapping[str, Any]] = ()) -> None:
        """Live-track claims per model, language and arm with the two denominators, H5 and the secondary family.

        Episodes of the dev split (the tasks used to tune the harness) and their steps are
        left out; a record without a split field belongs to the test split.
        """
        rows = [dict(r) for r in ordered_records(episodes, "live") if str(r.get("split") or "test") != "dev"]
        if not rows:
            return
        test_ids = {r.get("episode_id") for r in rows}
        steps = [s for s in steps if s.get("episode_id") in test_ids]
        for r in rows:
            v = r.get("valid")
            r["is_valid"] = (_count(r.get("parse_failures")) == 0) if v is None else bool(v)
        models = sorted({str(r.get("model")) for r in rows})
        langs = sorted({str(r.get("lang")) for r in rows})
        arms = sorted({str(r.get("arm")) for r in rows})
        families = sorted({str(r.get("family")) for r in rows})
        self.groups["live"] = {"models": models, "model_slugs": {m: model_slug(m) for m in models}, "langs": langs, "arms": arms, "families": families}
        for m in models:
            ms = model_slug(m)
            for lang in langs + ["all"]:
                for a in arms:
                    g = [r for r in rows if r.get("model") == m and r.get("arm") == a and (lang == "all" or r.get("lang") == lang)]
                    if not g:
                        continue
                    tok = f"{ms}_{lang}_{a.lower()}"
                    self._add_live_group(g, tok, f"model {m}, language {lang}, arm {a}", source)
                    for fam in families:
                        fg = [r for r in g if r.get("family") == fam]
                        if fg:
                            self._add_live_group(fg, f"{tok}_{fam}", f"model {m}, language {lang}, arm {a}, family {fam}", source, brief=True)
            self._add_h5(rows, m, source)
            self._add_secondary(rows, m, arms, source)
        self._add_live_latency(rows, steps, source)

    def _add_live_group(self, g: list[dict[str, Any]], tok: str, where: str, source: Mapping[str, Any], brief: bool = False) -> None:
        valid = [r for r in g if r["is_valid"]]
        cl_all = [r.get("task_id") for r in g]
        cl_valid = [r.get("task_id") for r in valid]
        self.count_claim(f"live_n_{tok}", len(g), f"Episodes under {where}.", source, "episode_id")
        self.count_claim(f"live_n_valid_{tok}", len(valid), f"Valid episodes (no parse failure) under {where}.", source, "valid")
        self.cluster_rate(f"live_reproposal_{tok}", [r.get("reproposed_superseded") for r in valid], cl_valid, f"Share of valid episodes with a superseded record whose agent re-proposed the superseded transaction under {where}.", source, "reproposed_superseded")
        self.cluster_rate(f"live_duplicate_{tok}", [bool(r.get("duplicated_current")) for r in valid], cl_valid, f"Share of valid episodes with a duplicate execution of the current record under {where}.", source, "duplicated_current")
        self.cluster_rate(f"live_unsafe_{tok}", [_count(r.get("unsafe_executions")) > 0 for r in valid], cl_valid, f"Share of valid episodes with at least one unsafe execution (superseded, duplicate or forbidden) under {where}.", source, "unsafe_executions")
        self.cluster_rate(f"live_completion_{tok}", [bool(r.get("completed_intended")) for r in valid], cl_valid, f"Share of valid episodes in which the intended payment after the last user change executed exactly once under {where}.", source, "completed_intended")
        if brief:
            return
        self.cluster_rate(f"live_reproposal_alleps_{tok}", [r.get("reproposed_superseded") for r in g], cl_all, f"Re-proposal share over all episodes with a superseded record under {where}.", source, "reproposed_superseded")
        self.cluster_rate(f"live_duplicate_alleps_{tok}", [bool(r.get("duplicated_current")) for r in g], cl_all, f"Duplicate share over all episodes under {where}.", source, "duplicated_current")
        self.cluster_rate(f"live_unsafe_alleps_{tok}", [_count(r.get("unsafe_executions")) > 0 for r in g], cl_all, f"Unsafe-execution share over all episodes under {where}.", source, "unsafe_executions")
        self.cluster_rate(f"live_completion_alleps_{tok}", [bool(r.get("completed_intended")) for r in g], cl_all, f"Completion share over all episodes under {where}.", source, "completed_intended")
        self.cluster_rate(f"live_parse_failure_{tok}", [_count(r.get("parse_failures")) > 0 for r in g], cl_all, f"Share of episodes with at least one parse failure under {where}.", source, "parse_failures")
        self.binomial_rate(f"live_parse_failure_steps_{tok}", sum(_count(r.get("parse_failures")) for r in g), sum(_count(r.get("steps")) for r in g), f"Parse failures per agent step under {where}.", source, "parse_failures")
        self.cluster_rate(f"live_unconfirmed_{tok}", [_count(r.get("unconfirmed_executions")) > 0 for r in valid], cl_valid, f"Share of valid episodes with at least one UNCONFIRMED execution under {where}.", source, "unconfirmed_executions")
        self.count_claim(f"live_unsafe_executions_{tok}", sum(_count(r.get("unsafe_executions")) for r in g), f"Unsafe executions summed over episodes under {where}.", source, "unsafe_executions")
        self.count_claim(f"live_executions_{tok}", sum(_count(r.get("executions")) for r in g), f"Rail executions summed over episodes under {where}.", source, "executions")
        self.count_claim(f"live_confirmations_{tok}", sum(_count(r.get("confirmations")) for r in g), f"Confirmation requests summed over episodes under {where}.", source, "confirmations")

    def _add_h5(self, rows: list[dict[str, Any]], m: str, source: Mapping[str, Any]) -> None:
        ms = model_slug(m)
        behaviour = [r for r in rows if r.get("model") == m and r.get("arm") == LIVE_BEHAVIOUR_ARM and r["is_valid"]]
        if behaviour:
            self.cluster_rate(f"h5_reproposal_{ms}", [r.get("reproposed_superseded") for r in behaviour], [r.get("task_id") for r in behaviour], f"Re-proposal rate of the superseded transaction over valid episodes of model {m} without a monitor ({LIVE_BEHAVIOUR_ARM}), clustered on the task.", source, "reproposed_superseded", threshold=THRESHOLDS["h5_reproposal"])
        monitored = [r for r in rows if r.get("model") == m and r.get("arm") == LIVE_MONITOR_ARM and r["is_valid"]]
        if monitored:
            k = sum(_count(r.get("unsafe_executions")) for r in monitored)
            self.count_claim(f"h5_unsafe_executions_behind_monitor_{ms}", k, f"Superseded, duplicate or forbidden executions behind the monitor ({LIVE_MONITOR_ARM}) over valid episodes of model {m}.", source, "unsafe_executions")
            self.binomial_rate(f"h5_unsafe_rate_behind_monitor_{ms}", sum(1 for r in monitored if _count(r.get("unsafe_executions")) > 0), len(monitored), f"Share of valid episodes with an unsafe execution behind the monitor for model {m}; the exact upper bound is the reported bound.", source, "unsafe_executions", threshold=0.0)
            change = [r for r in monitored if r.get("family") in CHANGE_FAMILIES]
            control = [r for r in monitored if r.get("family") == CONTROL_FAMILY]
            cid = f"h5_completion_diff_change_minus_control_{ms}"
            if change and control:
                mat = np.asarray([[1.0 if r.get("completed_intended") else 0.0, 1.0 if r.get("family") in CHANGE_FAMILIES else 0.0] for r in change + control], dtype=float)
                self.cluster_stat(cid, mat, [r.get("task_id") for r in change + control], _lang_diff_stat, "fraction", f"Completion rate of the revised legitimate transaction behind the monitor on the revision and cancellation families minus the completion rate on the no-change control, model {m}, clustered on the task. Secondary family of the protocol: two-sided cluster permutation p-value (family labels permuted across whole tasks) with its Holm adjustment across the family.", source, "completed_intended", threshold=-THRESHOLDS["h5_completion_loss"], permutation="label", family="secondary")
            else:
                self.ledger.add(cid, None, "fraction", n=len(change) + len(control), computed_by="tasdiq.analysis.stats.cluster_bootstrap", description=f"Completion difference between the change families and the control behind the monitor for model {m}; one side has no episodes.", threshold=-THRESHOLDS["h5_completion_loss"], **self._src(source))

    def _add_secondary(self, rows: list[dict[str, Any]], m: str, arms: Sequence[str], source: Mapping[str, Any]) -> None:
        """Arabic-minus-English differences per arm, paired on the task and seed: the exploratory family, Holm-adjusted on its own."""
        ms = model_slug(m)
        for a in arms:
            g = [r for r in rows if r.get("model") == m and r.get("arm") == a and r["is_valid"]]
            ar = {(r.get("task_id"), r.get("seed")): r for r in g if r.get("lang") == "ar"}
            en = {(r.get("task_id"), r.get("seed")): r for r in g if r.get("lang") == "en"}
            for name, field in (("reproposal", "reproposed_superseded"), ("duplicate", "duplicated_current")):
                keys = [k for k in ar if k in en and ar[k].get(field) is not None and en[k].get(field) is not None]
                cid = f"sec_{name}_lang_diff_ar_minus_en_{ms}_{a.lower()}"
                if not keys:
                    continue
                self.paired_diff(cid, [1.0 if ar[k].get(field) else 0.0 for k in keys], [1.0 if en[k].get(field) else 0.0 for k in keys], [k[0] for k in keys], "fraction", f"Arabic-minus-English {name} rate difference over valid episodes of model {m} under arm {a}, paired on the task and seed and clustered on the task. Exploratory family (outside the pre-stated secondary family): two-sided paired cluster permutation p-value with its Holm adjustment across the live language differences only.", source, field, permutation=True, family="exploratory")

    def _add_live_latency(self, rows: list[dict[str, Any]], steps: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> None:
        by_model: dict[str, list[float]] = {}
        monitor: list[float] = []
        for s in steps:
            v = s.get("model_seconds")
            if v is not None:
                by_model.setdefault(str(s.get("model")), []).append(float(v))
            for d in s.get("decisions") or []:
                t = d.get("monitor_seconds") if isinstance(d, Mapping) else None
                if t is not None:
                    monitor.append(float(t))
        if not by_model:
            for r in rows:
                v = r.get("model_seconds")
                steps_n = _count(r.get("steps"))
                if v is not None and steps_n:
                    by_model.setdefault(str(r.get("model")), []).append(float(v) / steps_n)
        latency = self.groups.setdefault("latency", [])
        for m, vals in sorted(by_model.items()):
            tok = f"model_live_{model_slug(m)}"
            latency.append({"stage": "model", "pipeline": "live", "model": m, "token": tok})
            for stat, v in _quantiles(vals).items():
                self.value_claim(f"lat_{tok}_{stat}", v, "count" if stat == "n" else "seconds", f"Latency of one agent step of model {m} on the live track: {stat} over steps.", source, "model_seconds", n=len(vals))
        if monitor:
            latency.append({"stage": "monitor", "pipeline": "live", "model": "all", "token": "monitor_live"})
            for stat, v in _quantiles(monitor).items():
                self.value_claim(f"lat_monitor_live_{stat}", v, "count" if stat == "n" else "seconds", f"Latency of one monitor hook call (consent request, execution or user event) on the live track: {stat} over decisions.", source, "monitor_seconds", n=len(monitor))
        peak = max((int(r.get("peak_rss_bytes") or 0) for r in rows), default=0)
        if peak:
            self.value_claim("mem_peak_rss_bytes_live", peak, "bytes", "Peak resident set size of a live-track worker.", source, "peak_rss_bytes")

    # -- verify track ----------------------------------------------------------------

    def add_verify(self, results: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> None:
        """State counts and invariant violations of the bounded exploration per arm, rule and depth."""
        tokens = []
        for res in results:
            tok = f"{res.get('arm')}_{res.get('rule')}_d{res.get('depth')}"
            tokens.append(tok)
            where = f"arm {res.get('arm')} under rule {res.get('rule')} to depth {res.get('depth')}"
            self.count_claim(f"ver_states_{tok}", int(res.get("states", 0)), f"Reachable states of the bounded exploration, {where}.", source, "states")
            self.count_claim(f"ver_transitions_{tok}", int(res.get("transitions", 0)), f"Transitions explored, {where}.", source, "transitions")
            self.count_claim(f"ver_violations_{tok}", int(res.get("n_violations", 0)), f"Invariant violations found, {where}.", source, "n_violations")
            self.count_claim(f"ver_intent_departures_{tok}", int(res.get("n_intent_departures", 0)), f"Intent departures found, {where}.", source, "n_intent_departures")
            self.count_claim(f"ver_executions_max_{tok}", int(res.get("executions_max", 0)), f"Maximum executions of one consent on any path, {where}.", source, "executions_max")
            self.text_claim(f"ver_complete_{tok}", "yes" if res.get("complete") else "no", f"Whether the exploration exhausted the state space within the depth bound, {where}.", source, "complete")
        self.groups["verify"] = {"tokens": tokens}

    # -- hypothesis verdicts ------------------------------------------------------------

    def _claim(self, cid: str) -> Claim | None:
        return self.ledger.get(cid) if cid in self.ledger else None

    def add_verdicts(self, source: Mapping[str, Any] | None = None) -> None:
        """Verdicts of H1 to H5 from the decision quantities and their pre-stated thresholds."""
        self.verdict_claim("h1_verdict", *self._h1_verdict(), source)
        self.verdict_claim("h2_verdict", *self._h2_verdict(), source)
        self.verdict_claim("h3_verdict", *self._h3_verdict(), source)
        self.verdict_claim("h4_verdict", *self._h4_verdict(), source)
        self.verdict_claim("h5_verdict", *self._h5_verdict(), source)

    def _primary_capture(self) -> tuple[str, str] | None:
        cap = self.groups.get("capture")
        if not cap or PROPOSED_PIPELINE not in cap["pipelines"]:
            return None
        counts = {m: (self._claim(f"cap_n_all_{PROPOSED_PIPELINE}_{model_slug(m)}") or Claim("", 0, "count")).value or 0 for m in cap["models"]}
        if not counts:
            return None
        m = max(counts, key=lambda k: (counts[k], k))
        return m, model_slug(m)

    def _h1_verdict(self) -> tuple[str, str]:
        primary = self._primary_capture()
        if primary is None:
            return INCONCLUSIVE, "H1: no capture track under the direct pipeline."
        m, ms = primary
        diffs = {}
        for lang in ("ar", "mix"):
            c = self._claim(f"h1_invariance_diff_en_minus_{lang}_{PROPOSED_PIPELINE}_{ms}")
            if c is not None and c.value is not None and c.ci is not None:
                diffs[lang] = c
        if not diffs:
            return INCONCLUSIVE, f"H1: no English-minus-Arabic or English-minus-mixed invariance difference is computable for model {m}."
        thr = THRESHOLDS["h1_invariance_diff"]
        strong = all(c.value >= thr and c.ci["lo"] > 0 for c in diffs.values())
        weak = all(c.value < thr or c.ci["lo"] <= 0 for c in diffs.values())
        conc = [self._claim(f"h1_concentration_{lang}_{PROPOSED_PIPELINE}_{ms}") for lang in diffs]
        conc_vals = [c.value for c in conc if c is not None and c.value is not None]
        concentrated = any(v > THRESHOLDS["h1_concentration"] for v in conc_vals)
        detail = "; ".join(f"en minus {lang} {c.value:.4f} [{c.ci['lo']:.4f}, {c.ci['hi']:.4f}]" for lang, c in diffs.items())
        if len(diffs) < 2:
            return INCONCLUSIVE, f"H1: only one comparison language present ({detail}); both differences are required."
        if strong and concentrated:
            return SUPPORTS, f"H1: both differences at least {thr} with intervals above zero and failures concentrated in two named phenomena ({detail})."
        if weak or (conc_vals and not concentrated):
            return REFUTES, f"H1: {detail}; concentration {conc_vals}."
        return INCONCLUSIVE, f"H1: mixed evidence ({detail}; concentration {conc_vals})."

    def _h2_verdict(self) -> tuple[str, str]:
        premise = [self._claim(f"h2_premise_intent_violations_{arm}") for arm in (CLASSICAL_ARM, PRIMARY_ARM)]
        if any(c is None for c in premise):
            return INCONCLUSIVE, "H2: the oracle trace track with B4 and the proposed system is missing, so the premise is unchecked."
        if any(c.value for c in premise):
            return REFUTES, "H2: the premise fails; an arm shows intent violations outside its measured classes on the oracle track."
        primary = self._primary_capture()
        if primary is None:
            return INCONCLUSIVE, "H2: premise holds; no capture track to measure the reduction."
        m, ms = primary
        red = self._claim(f"h2_capture_error_reduction_{PROPOSED_PIPELINE}_vs_{SAME_PARSER_PIPELINE}_{ms}")
        cost = self._claim(f"h2_ask_unambiguous_added_{PROPOSED_PIPELINE}_minus_{SAME_PARSER_PIPELINE}_{ms}")
        translations = [(comparator, self._claim(f"h2_reduction_diff_{PROPOSED_PIPELINE}_minus_{comparator}_{ms}")) for comparator in TRANSLATION_COMPARATORS]
        translations = [(comparator, tr) for comparator, tr in translations if tr is not None and tr.value is not None and tr.ci is not None]
        if red is None or red.value is None or cost is None or cost.value is None:
            return INCONCLUSIVE, f"H2: premise holds; the same-parser pipeline {SAME_PARSER_PIPELINE} is absent for model {m}, so no reduction is measured."
        if red.value <= 0 or (red.ci and red.ci["hi"] <= 0) or cost.value > THRESHOLDS["h2_added_confirmations"]:
            return REFUTES, f"H2: reduction {red.value:.4f}, added confirmations {cost.value:.4f}."
        if not translations:
            return INCONCLUSIVE, f"H2: reduction {red.value:.4f}, added confirmations {cost.value:.4f}; translate-then-bind not measured."
        detail = ", ".join(f"direct minus {comparator} {tr.value:.4f} [{tr.ci['lo']:.4f}, {tr.ci['hi']:.4f}]" for comparator, tr in translations)
        if red.value >= THRESHOLDS["h2_relative_reduction"] and red.ci["lo"] > 0 and all(tr.ci["lo"] > 0 for _, tr in translations):
            return SUPPORTS, f"H2: reduction {red.value:.4f} [{red.ci['lo']:.4f}, {red.ci['hi']:.4f}], added confirmations {cost.value:.4f}, {detail}."
        matched = [f"{comparator} {tr.value:.4f} [{tr.ci['lo']:.4f}, {tr.ci['hi']:.4f}]" for comparator, tr in translations if tr.ci["hi"] <= 0]
        if matched:
            return REFUTES, f"H2: a translate-then-bind comparator matches or exceeds the reduction (direct minus {'; '.join(matched)})."
        return INCONCLUSIVE, f"H2: reduction {red.value:.4f} [{red.ci['lo']:.4f}, {red.ci['hi']:.4f}], {detail}."

    def _h3_verdict(self) -> tuple[str, str]:
        diff = self._claim(f"h3_unnecessary_reconf_diff_{PRIMARY_RULE}_minus_{NARROWING_RULE}")
        narrowing = self._claim(f"h3_{NARROWING_RULE}_intent_violation_rate")
        invalidate = self._claim(f"h3_{PRIMARY_RULE}_intent_violation_rate")
        if diff is None or diff.value is None or narrowing is None:
            return INCONCLUSIVE, "H3: the revision family under the invalidate and narrowing rules is missing."
        if narrowing.count:
            return REFUTES, f"H3: the narrowing rule admitted {narrowing.count} executed intent violations and is withdrawn."
        thr = THRESHOLDS["h3_unnecessary_reconf_diff"]
        b3 = [self._claim(f"h3_{ELEVATION_ARM}_intent_class_{c}_rate") for c in ("substitution", "superseded")]
        b3_lower = any(c is not None and c.ci is not None and c.ci["lo"] > 0 for c in b3)
        detail = f"invalidate minus narrowing {diff.value:.4f} [{diff.ci['lo']:.4f}, {diff.ci['hi']:.4f}]"
        if diff.value < thr or diff.ci["lo"] <= 0:
            return REFUTES, f"H3: {detail}; below {thr} or interval includes zero."
        if invalidate is not None and invalidate.count:
            return INCONCLUSIVE, f"H3: {detail}, but the invalidate rule admitted executed violations."
        if not b3_lower:
            return INCONCLUSIVE, f"H3: {detail}; B3 shows no violation class with a lower bound above zero."
        return SUPPORTS, f"H3: {detail} at zero executed violations; B3 admits a violation class with a lower bound above zero."

    def _h4_verdict(self) -> tuple[str, str]:
        wc = self.groups.get("window_curve")
        if not wc:
            return INCONCLUSIVE, "H4: no delayed-execution family on the trace track."
        points = []
        for w in wc["windows"]:
            s = self._claim(f"win_curve_{int(w)}_stale_rate")
            f = self._claim(f"win_curve_{int(w)}_forced_reconf_rate")
            if s is not None and s.value is not None and f is not None and f.value is not None:
                points.append((int(w), s, f))
        if len(points) < THRESHOLDS["h4_min_windows"]:
            return INCONCLUSIVE, f"H4: {len(points)} window values present; at least {int(THRESHOLDS['h4_min_windows'])} are required."
        stale = [p[1].value for p in points]
        forced = [p[2].value for p in points]
        monotone = all(a <= b for a, b in zip(stale, stale[1:])) and all(a >= b for a, b in zip(forced, forced[1:]))
        flat = len(set(stale)) == 1 or len(set(forced)) == 1
        feasible = [w for w, s, f in points if s.value <= THRESHOLDS["h4_stale_budget"] and f.value <= THRESHOLDS["h4_forced_budget"]]
        detail = "; ".join(f"{w} s: stale {s.value:.4f}, forced {f.value:.4f}" for w, s, f in points)
        if monotone and not flat and feasible:
            return SUPPORTS, f"H4: both curves monotone over {len(points)} windows and windows {feasible} meet both budgets ({detail})."
        if flat or not feasible:
            return REFUTES, f"H4: flat curve or no window within both budgets ({detail})."
        return INCONCLUSIVE, f"H4: curves not monotone ({detail})."

    def _h5_verdict(self) -> tuple[str, str]:
        live = self.groups.get("live")
        if not live:
            return INCONCLUSIVE, "H5: no live track."
        rep = {m: self._claim(f"h5_reproposal_{model_slug(m)}") for m in live["models"]}
        rep = {m: c for m, c in rep.items() if c is not None and c.value is not None}
        if not rep:
            return INCONCLUSIVE, "H5: no valid episodes with a superseded record without a monitor."
        thr, floor = THRESHOLDS["h5_reproposal"], THRESHOLDS["h5_reproposal_floor"]
        strong = [m for m, c in rep.items() if c.value >= thr and c.ci is not None and c.ci["lo"] > floor]
        all_low = all(c.value < floor and c.ci is not None and c.ci["hi"] < thr for c in rep.values())
        all_low_point = all(c.value < floor for c in rep.values())
        unsafe = [self._claim(f"h5_unsafe_executions_behind_monitor_{model_slug(m)}") for m in live["models"]]
        unsafe_vals = [c.count for c in unsafe if c is not None]
        comp = [self._claim(f"h5_completion_diff_change_minus_control_{model_slug(m)}") for m in live["models"]]
        comp_vals = [c.value for c in comp if c is not None and c.value is not None]
        detail = "; ".join(f"{m}: {c.value:.4f} [{c.ci['lo']:.4f}, {c.ci['hi']:.4f}]" for m, c in rep.items() if c.ci)
        if comp_vals and any(v < -THRESHOLDS["h5_completion_loss"] for v in comp_vals):
            return REFUTES, f"H5: the monitor lowers completion of the revised transaction by more than {THRESHOLDS['h5_completion_loss']} against the control (differences {comp_vals})."
        if all_low:
            return REFUTES, f"H5: re-proposal below {floor} for every model with upper bounds below {thr} ({detail})."
        if all_low_point:
            return INCONCLUSIVE, f"H5: re-proposal below {floor} for every model but an upper bound reaches {thr} ({detail})."
        if not strong or not unsafe_vals or not comp_vals:
            return INCONCLUSIVE, f"H5: re-proposal {detail}; unsafe executions behind the monitor {unsafe_vals}; completion differences {comp_vals}."
        if any(unsafe_vals):
            return REFUTES, f"H5: executions behind the monitor were unsafe ({unsafe_vals})."
        return SUPPORTS, f"H5: re-proposal {detail}; zero unsafe executions behind the monitor; completion differences {comp_vals}."


def _lang_diff_stat(mat: np.ndarray) -> float:
    """Mean of column 0 where column 1 is 1 minus the mean where column 1 is 0."""
    a = mat[mat[:, 1] == 1.0, 0]
    b = mat[mat[:, 1] == 0.0, 0]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(a.mean() - b.mean())


def _relative_reduction_stat(mat: np.ndarray) -> float:
    """``(mean(col 0) - mean(col 1)) / mean(col 0)``; not finite when column 0 has no errors."""
    base = float(mat[:, 0].mean())
    if base == 0.0:
        return float("nan")
    return (base - float(mat[:, 1].mean())) / base


def _abs_reduction_stat(mat: np.ndarray) -> float:
    return float(mat[:, 0].mean() - mat[:, 1].mean())


def load_track(run_dirs: Sequence[Path], track: str, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Records of a track over its run directories, deduplicated on the track's record key.

    Returns the records and one manifest entry per run (run id, path, record count and the
    hashes of its manifest.json).
    """
    key_fields = RECORD_KEYS.get(track)
    seen: set[tuple[Any, ...]] = set()
    records: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        path = run_dir / RECORD_FILES[track]
        rows = read_jsonl(path) if track != "verify" else list((read_json(path) or {}).get("results") or [])
        kept = 0
        for r in rows:
            if r is None:
                continue
            if key_fields is not None:
                key = tuple(r.get(k) for k in key_fields)
                if key in seen:
                    continue
                seen.add(key)
            r = dict(r)
            r["_source"] = {"run": run_dir.name, "file": _rel_path(path, root)}
            records.append(r)
            kept += 1
        manifest = read_json(run_dir / "manifest.json") or {}
        entries.append({
            "track": track,
            "run_id": run_dir.name,
            "path": _rel_path(run_dir, root),
            "records": kept,
            "config_hash": (manifest.get("config") or {}).get("config_hash"),
            "code_hash": (manifest.get("config") or {}).get("code_hash"),
            "peak_rss_bytes": manifest.get("peak_rss_bytes"),
            "elapsed_seconds": manifest.get("elapsed_seconds"),
        })
    return records, entries


def _records_source(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The run and file names carried by the records' ``_source`` entries, or ``None`` when they carry none."""
    runs = sorted({str(r["_source"]["run"]) for r in records if r.get("_source")})
    files = sorted({str(r["_source"]["file"]) for r in records if r.get("_source")})
    if not runs:
        return None
    return {"run": ",".join(runs), "file": ",".join(files)}


def _source_of(records: Sequence[Mapping[str, Any]], run_dirs: Sequence[Path], track: str, root: Path) -> dict[str, Any]:
    runs = sorted({str(r["_source"]["run"]) for r in records if r.get("_source")})
    files = sorted({str(r["_source"]["file"]) for r in records if r.get("_source")})
    if not runs and run_dirs:
        runs = [p.name for p in run_dirs]
        files = [_rel_path(p / RECORD_FILES[track], root) for p in run_dirs]
    return {"run": ",".join(runs), "file": ",".join(files)}


def build_ledger(
    results_dir: Path | str,
    data_dir: Path | str | None = None,
    runs: Mapping[str, Sequence[str]] | None = None,
    n_boot: int = 2000,
    seed: int = 20261006,
    root: Path | str | None = None,
    log: Callable[[str], None] | None = None,
    n_perm: int = 10000,
) -> tuple[Ledger, dict[str, Any]]:
    """Build the claim ledger and its manifest from the run directories under ``results_dir``.

    ``runs`` restricts each track to the named run directories; by default every run
    directory holding the track's record file is read.  ``data_dir`` adds the benchmark
    composition claims.  ``n_boot`` is the number of cluster bootstrap resamples and
    ``n_perm`` the number of permutations of each permutation test.  Returns the ledger
    (path ``results_dir/ledger.json``) and the manifest mapping, which lists the runs that
    fed the ledger, the group tokens found (arms, windows, families, languages, pipelines,
    models), the bootstrap and permutation settings and the claim ids of the two Holm
    families.
    """
    results = Path(results_dir)
    root_path = Path(root) if root is not None else (results.parent if results.name == "results" else CODE_ROOT)
    ledger = Ledger(results / "ledger.json")
    builder = LedgerBuilder(ledger, n_boot=n_boot, seed=seed, root=root_path, n_perm=n_perm)
    selected = select_runs(results, runs)
    entries: list[dict[str, Any]] = []
    if data_dir is not None and Path(data_dir).is_dir():
        builder.add_benchmark(Path(data_dir))
    for track in TRACKS:
        run_dirs = selected.get(track, [])
        if not run_dirs:
            continue
        records, track_entries = load_track(run_dirs, track, root_path)
        entries.extend(track_entries)
        source = _source_of(records, run_dirs, track, root_path)
        if log:
            log(f"ledger: {track}: {len(records)} records from {[p.name for p in run_dirs]}")
        if track == "capture":
            builder.add_capture(records, source)
        elif track == "trace":
            builder.add_trace(records, source)
        elif track == "live":
            steps: list[dict[str, Any]] = []
            for run_dir in run_dirs:
                steps.extend(read_jsonl(run_dir / "steps.jsonl"))
            builder.add_live(records, source, steps)
        elif track == "verify":
            builder.add_verify(records, source)
        for e in track_entries:
            if e.get("peak_rss_bytes") and f"mem_peak_rss_bytes_{track}" not in ledger:
                builder.value_claim(f"mem_peak_rss_bytes_{track}", int(e["peak_rss_bytes"]), "bytes", f"Peak resident set size recorded in the {track} run manifest.", {"run": e["run_id"], "file": e["path"] + "/manifest.json"}, "peak_rss_bytes")
    builder.finish_secondary()
    builder.add_verdicts()
    manifest = {
        "ledger": _rel_path(ledger.path, root_path),
        "claims": len(ledger),
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "runs": entries,
        "groups": builder.groups,
        "thresholds": dict(THRESHOLDS),
        "secondary_family": [cid for cid, _ in builder.secondary],
        "exploratory_family": [cid for cid, _ in builder.exploratory],
    }
    return ledger, _json_safe(manifest)


def write_ledger(results_dir: Path | str, ledger: Ledger, manifest: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write ``ledger.json`` and ``manifest.json`` under ``results_dir`` and return both paths."""
    results = Path(results_dir)
    ledger_path = ledger.write(results / "ledger.json")
    manifest_path = results / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return ledger_path, manifest_path
