"""Figure generation from the claim ledger and its manifest.

Six figures are drawn from ``results/ledger.json`` and the group tokens recorded in
``results/manifest.json``; nothing is recomputed from the run records.

F1  capture invariance and sensitivity by language and pipeline (grouped bars with intervals)
F2  stage attribution: capture-class and enforcement-class scenario-run shares per arm and language
F3  revision-rule operating points: unnecessary re-confirmation against executed intent violations
F4  validity-window curves: stale executions and forced re-confirmations against the window
F5  live outcomes: re-proposal, duplicate, unsafe and completion rates by model, language and arm
F6  latency from the stored quantiles: (a) model calls per pipeline and model, (b) non-model stages

Every figure is a vector PDF and a PNG, 3.45 in wide (7.16 in for F2 when it is drawn
per language), serif text at 6 to 8 pt, thin lines and one fixed colour set.  A track
that is absent from the ledger leaves its panel empty with a label saying so.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

FIGURE_IDS: tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5", "F6")
SINGLE_COLUMN_IN = 3.45
DOUBLE_COLUMN_IN = 7.16
DPI_PNG = 300

PALETTE: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999", "#000000")
"""Fixed colour set (Okabe and Ito); hues are assigned to entities in fixed order, never cycled."""

LANG_ORDER: tuple[str, ...] = ("en", "ar", "mix", "all")
LANG_LABELS: dict[str, str] = {"en": "English", "ar": "Arabic", "mix": "Code-switched", "all": "All"}
PIPELINE_ORDER: tuple[str, ...] = ("direct", "noncanon", "nogate", "translate", "translate_mt", "english_reference")
PIPELINE_LABELS: dict[str, str] = {"direct": "direct", "noncanon": "no canonicaliser", "nogate": "no gate", "translate": "translate (same model)", "translate_mt": "translate (MT)", "english_reference": "English reference"}
ARM_ORDER: tuple[str, ...] = ("PromptOnly", "B0", "B1", "B2", "B3", "B4", "B4Surface", "B5", "B6", "B7", "Tasdiq_invalidate", "Tasdiq_narrowing", "Tasdiq_cap_settle", "TasdiqNonAtomic_invalidate", "TasdiqNonAtomic_narrowing", "TasdiqNonAtomic_cap_settle", "TrustAgent", "AskAlways", "RefuseAll", "Tasdiq")
CURVE_ORDER: tuple[str, ...] = ("B4", "invalidate", "narrowing", "cap_settle", "B3", "B5", "B7")
"""Revision-rule operating points drawn in F3; arms without a revision rule are left out."""
CURVE_LABELS: dict[str, str] = {"B4": "B4 (invalidate, no channel)", "invalidate": "invalidate on change", "narrowing": "narrowing tolerant", "cap_settle": "cap and settle", "B3": "B3 elevation", "B5": "B5 lease ledger", "B7": "B7 policy narrowing"}
CURVE_MARKERS: dict[str, str] = {"B4": "s", "invalidate": "o", "narrowing": "^", "cap_settle": "D", "B3": "v", "B5": "P", "B7": "X"}
STAGE_ORDER: tuple[str, ...] = ("model", "translate", "canonicalise", "gate", "monitor")
STAGE_COLOURS: dict[str, str] = {"translate": PALETTE[2], "canonicalise": PALETTE[3], "gate": PALETTE[6], "monitor": PALETTE[7]}
"""Colours of the non-model stages in panel (b) of F6, disjoint from the model colours of panel (a) (PALETTE[0] and PALETTE[1])."""

RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 0.8,
    "lines.markersize": 3.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "patch.linewidth": 0.5,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
    "savefig.dpi": DPI_PNG,
    "figure.dpi": 100,
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def require_inputs(results_dir: Path | str) -> Path:
    """Return the ledger path when the results needed for the figures exist."""
    results = Path(results_dir)
    ledger = results / "ledger.json"
    if not ledger.is_file():
        raise FileNotFoundError(f"no claim ledger at {ledger}; run the tracks and build the ledger before drawing figures")
    return ledger


def load_inputs(results_dir: Path | str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The claims of ``ledger.json`` and the ``groups`` mapping of ``manifest.json`` (empty when absent)."""
    ledger = require_inputs(results_dir)
    claims = json.loads(ledger.read_text(encoding="utf-8"))
    manifest_path = Path(results_dir) / "manifest.json"
    groups: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = dict(manifest.get("groups") or {})
    return claims, groups


class Point:
    """A claim's value with its interval bounds (``None`` bounds when the claim has no interval)."""

    __slots__ = ("value", "lo", "hi", "n")

    def __init__(self, value: float, lo: float | None, hi: float | None, n: int | None) -> None:
        self.value = value
        self.lo = lo
        self.hi = hi
        self.n = n

    def err(self) -> tuple[float, float]:
        lo = 0.0 if self.lo is None else max(0.0, self.value - self.lo)
        hi = 0.0 if self.hi is None else max(0.0, self.hi - self.value)
        return lo, hi


def point(claims: Mapping[str, Mapping[str, Any]], cid: str) -> Point | None:
    """The point of a claim, or ``None`` when the claim is absent or has no value."""
    c = claims.get(cid)
    if c is None or c.get("value") is None or isinstance(c.get("value"), str):
        return None
    ci = c.get("ci") or {}
    return Point(float(c["value"]), ci.get("lo"), ci.get("hi"), c.get("n"))


def _ordered(items: Sequence[str], order: Sequence[str]) -> list[str]:
    rank = {k: i for i, k in enumerate(order)}
    return sorted(items, key=lambda k: (rank.get(k, len(order)), k))


def colour_for(key: str, order: Sequence[str]) -> str:
    """Palette colour of an entity by its fixed position in ``order`` (unknown keys follow alphabetically)."""
    keys = _ordered(list(order) + ([key] if key not in order else []), order)
    return PALETTE[keys.index(key) % len(PALETTE)]


def _empty(ax: Any, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=7, color="#555555", wrap=True)
    ax.set_xticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _tidy(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _percent_axis(ax: Any, label: str | None) -> None:
    if label is not None:
        ax.set_ylabel(label)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))


def _pct(p: Point) -> tuple[float, tuple[float, float]]:
    lo, hi = p.err()
    return 100.0 * p.value, (100.0 * lo, 100.0 * hi)


ARM_SHORT_LABELS: dict[str, str] = {
    "B4Surface": "B4S",
    "Tasdiq_invalidate": "T-inv",
    "Tasdiq_narrowing": "T-nar",
    "Tasdiq_cap_settle": "T-cap",
    "TasdiqNonAtomic_invalidate": "NA-inv",
    "TasdiqNonAtomic_narrowing": "NA-nar",
    "TasdiqNonAtomic_cap_settle": "NA-cap",
    "TrustAgent": "Trust",
    "AskAlways": "Ask",
    "RefuseAll": "Refuse",
}
"""Tick labels of the arms in F2, the same abbreviations as the enforcement tables."""


def _arm_label(tok: str) -> str:
    return ARM_SHORT_LABELS.get(tok, tok)


# ---------------------------------------------------------------------------
# F1: capture invariance and sensitivity
# ---------------------------------------------------------------------------


PIPELINE_SHORT_LABELS: dict[str, str] = {"direct": "direct", "noncanon": "no canon.", "nogate": "no gate", "translate": "transl. (same)", "translate_mt": "transl. (MT)", "english_reference": "Eng. ref."}
"""Short tick labels of the capture pipelines in F1."""


def _model_groups(columns: Sequence[tuple[Any, ...]], models: Sequence[str], index: int) -> tuple[list[str], list[float], list[float]]:
    """The models present among the columns, the x centre of each model's group and the boundaries between groups.

    columns lists the (pipeline, model) or (model, language) tuples drawn at x = 0, 1, ...;
    index names the tuple position holding the model.
    """
    present = [m for m in models if any(col[index] == m for col in columns)]
    centres, boundaries = [], []
    for k, m in enumerate(present):
        xs = [i for i, col in enumerate(columns) if col[index] == m]
        centres.append((min(xs) + max(xs)) / 2.0)
        if k > 0:
            boundaries.append(min(xs) - 0.5)
    return present, centres, boundaries


def _model_dividers(ax: Any, columns: Sequence[tuple[Any, ...]], models: Sequence[str], index: int) -> None:
    """Draw a dotted separator between consecutive model groups when more than one model is present."""
    if len(models) < 2:
        return
    for b in _model_groups(columns, models, index)[2]:
        ax.axvline(b, color="#999999", linewidth=0.5, linestyle=":", zorder=0)


def _model_group_axis(ax: Any, columns: Sequence[tuple[Any, ...]], models: Sequence[str], index: int, offset_pt: float) -> None:
    """Label groups of columns with their model name on a second x axis below the tick labels.

    The dotted separators of :func:`_model_dividers` are drawn as well; nothing is drawn
    for a single model.
    """
    if len(models) < 2:
        return
    present, centres, _ = _model_groups(columns, models, index)
    _model_dividers(ax, columns, models, index)
    sec = ax.secondary_xaxis("bottom")
    sec.spines["bottom"].set_visible(False)
    sec.spines["bottom"].set_position(("outward", offset_pt))
    sec.set_xticks(centres)
    sec.set_xticklabels(present, fontsize=7)
    sec.tick_params(length=0)


def fig_capture(claims: Mapping[str, Mapping[str, Any]], groups: Mapping[str, Any]) -> Figure:
    """Grouped bars of invariance and sensitivity per pipeline and model (x) and language (hue), one panel each.

    Only pipeline and model pairs with a claim are drawn; the model of each group of columns
    is named below the pipeline labels when more than one model is present, and the dotted
    divider between models is drawn in both panels. The two panels share one y label.
    """
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COLUMN_IN, 2.5), sharex=True)
    cap = groups.get("capture")
    if not cap:
        for ax, title in zip(axes, ("Invariance", "Sensitivity")):
            _empty(ax, f"{title}: no capture track")
        fig.tight_layout()
        return fig
    models = list(cap["models"])
    slugs = dict(cap.get("model_slugs") or {})
    pipelines = _ordered(cap["pipelines"], PIPELINE_ORDER)
    langs = [lang for lang in _ordered(cap["langs"], LANG_ORDER) if lang != "all"]
    columns = [(p, m) for m in models for p in pipelines if any(f"cap_{metric}_{lang}_{p}_{slugs.get(m, m)}" in claims for metric in ("invariance", "sensitivity") for lang in langs)]
    width = 0.8 / max(1, len(langs))
    for ax, metric, title in zip(axes, ("invariance", "sensitivity"), ("Invariance", "Sensitivity")):
        drawn = 0
        for j, lang in enumerate(langs):
            xs, ys, errs = [], [], []
            for i, (p, m) in enumerate(columns):
                pt = point(claims, f"cap_{metric}_{lang}_{p}_{slugs.get(m, m)}")
                if pt is None:
                    continue
                y, e = _pct(pt)
                xs.append(i + (j - (len(langs) - 1) / 2) * width)
                ys.append(y)
                errs.append(e)
            if xs:
                drawn += len(xs)
                ax.bar(xs, ys, width=width * 0.92, color=colour_for(lang, LANG_ORDER), label=LANG_LABELS.get(lang, lang), yerr=list(zip(*errs)), error_kw={"elinewidth": 0.5, "capsize": 1.5, "capthick": 0.5, "ecolor": "#333333"})
        if drawn == 0:
            _empty(ax, f"{title}: no members with a reference")
            continue
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([PIPELINE_SHORT_LABELS.get(p, p) for p, _ in columns], rotation=40, ha="right", rotation_mode="anchor", fontsize=7)
        ax.set_xlim(-0.6, len(columns) - 0.4)
        ax.set_title(title, loc="left")
        _percent_axis(ax, None)
        _tidy(ax)
    _model_dividers(axes[0], columns, models, 1)
    _model_group_axis(axes[1], columns, models, 1, 36)
    fig.supylabel("Rate (% of members)", fontsize=RC["axes.labelsize"], x=0.01)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.05, 0.06, 1, 1), h_pad=0.6)
    return fig


# ---------------------------------------------------------------------------
# F2: stage attribution
# ---------------------------------------------------------------------------


def _own_window_token(claims: Mapping[str, Mapping[str, Any]], arm: str, lang: str, wt: str) -> str:
    """The window token of an arm's attribution claims: the primary window, else the single window the arm ran at (B3)."""
    if f"attr_capture_class_{arm}_{lang}_{wt}" in claims:
        return wt
    prefix = f"attr_capture_class_{arm}_{lang}_w"
    found = sorted(cid[len(prefix) - 1:] for cid in claims if cid.startswith(prefix))
    return found[0] if len(found) == 1 else wt


def fig_attribution(claims: Mapping[str, Mapping[str, Any]], groups: Mapping[str, Any]) -> Figure:
    """Stacked bars of capture-class and enforcement-class scenario-run shares per arm.

    One panel over all languages is drawn when the ledger holds the ``all`` claims; otherwise one
    panel per language of the trace track.
    """
    trace = groups.get("trace")
    langs = [lang for lang in _ordered(trace["langs"], LANG_ORDER)] if trace else ["en"]
    if trace and any(cid.startswith("attr_capture_class_") and cid.split("_")[-2] == "all" for cid in claims):
        langs = ["all"]
    width_in = SINGLE_COLUMN_IN if len(langs) == 1 else DOUBLE_COLUMN_IN
    fig, axes = plt.subplots(1, len(langs), figsize=(width_in, 1.9), sharey=True, squeeze=False)
    axes = axes[0]
    if not trace:
        _empty(axes[0], "Stage attribution: no trace track")
        fig.tight_layout()
        return fig
    arms = _ordered(trace["arms"], ARM_ORDER)
    w = trace["primary_window"]
    wt = "wnone" if w is None else f"w{int(w)}"
    for ax, lang in zip(axes, langs):
        xs, cap_vals, enf_vals = [], [], []
        for i, a in enumerate(arms):
            own = _own_window_token(claims, a, lang, wt)
            c = point(claims, f"attr_capture_class_{a}_{lang}_{own}")
            e = point(claims, f"attr_enforcement_class_{a}_{lang}_{own}")
            if c is None and e is None:
                continue
            xs.append(i)
            cap_vals.append(100.0 * (c.value if c else 0.0))
            enf_vals.append(100.0 * (e.value if e else 0.0))
        if not xs:
            _empty(ax, f"{LANG_LABELS.get(lang, lang)}: no scenario runs")
            continue
        ax.bar(xs, cap_vals, width=0.7, color=PALETTE[0], label="capture class")
        ax.bar(xs, enf_vals, width=0.7, bottom=cap_vals, color=PALETTE[1], label="enforcement class")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([_arm_label(a) for a in arms], rotation=60, ha="right", rotation_mode="anchor", fontsize=7)
        own = [(a, _own_window_token(claims, a, lang, wt)) for a in arms]
        suffix = "".join(f"; {_arm_label(a)} at {tok[1:]} s" for a, tok in own if tok != wt and tok.startswith("w"))
        prefix = "All languages" if lang == "all" else LANG_LABELS.get(lang, lang)
        ax.set_title(f"{prefix} (window {w} s{suffix})", loc="left")
        _tidy(ax)
    _percent_axis(axes[0], "Scenario runs with an error (%)")
    top = max((max(ax.get_ylim()[1], 1.0) for ax in axes), default=100.0)
    for ax in axes:
        ax.set_ylim(0, min(100.0, max(5.0, top)))
    axes[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        if not handles:
            handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# F3: revision-rule operating points
# ---------------------------------------------------------------------------


def fig_revision(claims: Mapping[str, Mapping[str, Any]], groups: Mapping[str, Any]) -> Figure:
    """Unnecessary re-confirmation rate against executed intent-violation rate per revision rule.

    The legend names the rules only; the share of executions unconfirmed but intended under
    each rule is a ledger claim (``rev_curve_<rule>_unconfirmed_intended_share``) quoted in
    the caption rather than in the legend.
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.3))
    rc = groups.get("revision_curve")
    if not rc:
        _empty(ax, "Revision rules: no revision family on the trace track")
        fig.tight_layout()
        return fig
    tokens = [tok for tok in CURVE_ORDER if tok in rc["tokens"]]
    drawn = 0
    for tok in tokens:
        x = point(claims, f"rev_curve_{tok}_intent_violation_rate")
        y = point(claims, f"rev_curve_{tok}_unnecessary_reconf")
        if x is None or y is None:
            continue
        drawn += 1
        xv, xe = _pct(x)
        yv, ye = _pct(y)
        colour = colour_for(tok, CURVE_ORDER)
        own = point(claims, f"cal_window_s_{tok}")
        window_note = f" ({own.value:g} s window)" if own is not None and isinstance(own.value, (int, float)) and rc.get("window") is not None and own.value != rc["window"] else ""
        ax.errorbar([xv], [yv], xerr=[[xe[0]], [xe[1]]], yerr=[[ye[0]], [ye[1]]], fmt=CURVE_MARKERS.get(tok, "o"), color=colour, ecolor=colour, elinewidth=0.5, capsize=1.5, capthick=0.5, markerfacecolor="white", markeredgewidth=0.8, label=f"{CURVE_LABELS.get(tok, tok)}{window_note}")
    if drawn == 0:
        _empty(ax, "Revision rules: no operating points")
        fig.tight_layout()
        return fig
    ax.set_xlabel("Executed intent violations (% of executions)")
    ax.set_ylabel("Unnecessary re-confirmations,\nnarrowing label (% of scenario runs)")
    ax.set_xlim(left=-1)
    ax.set_ylim(bottom=-1)
    _tidy(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F4: validity-window curves
# ---------------------------------------------------------------------------


def fig_window(claims: Mapping[str, Mapping[str, Any]], groups: Mapping[str, Any]) -> Figure:
    """Stale-execution and forced re-confirmation rates of the proposed system against the window (log x)."""
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 1.9))
    wc = groups.get("window_curve")
    if not wc:
        _empty(ax, "Validity window: no delayed-execution family on the trace track")
        fig.tight_layout()
        return fig
    windows = sorted(int(w) for w in wc["windows"])
    drawn = 0
    for metric, label, colour in (("stale_rate", "stale executions", PALETTE[1]), ("forced_reconf_rate", "forced re-confirmations", PALETTE[0])):
        xs, ys, los, his = [], [], [], []
        for w in windows:
            pt = point(claims, f"win_curve_{w}_{metric}")
            if pt is None:
                continue
            y, e = _pct(pt)
            xs.append(w)
            ys.append(y)
            los.append(e[0])
            his.append(e[1])
        if not xs:
            continue
        drawn += 1
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o-", color=colour, ecolor=colour, elinewidth=0.5, capsize=1.5, capthick=0.5, label=label)
    if drawn == 0:
        _empty(ax, "Validity window: no window points")
        fig.tight_layout()
        return fig
    ax.axhline(1.0, color=PALETTE[1], linestyle=":", linewidth=0.6)
    ax.axhline(10.0, color=PALETTE[0], linestyle=":", linewidth=0.6)
    ax.text(windows[0], 1.0, " budget 1%", fontsize=6, va="bottom", color=PALETTE[1])
    ax.text(windows[0], 10.0, " budget 10%", fontsize=6, va="bottom", color=PALETTE[0])
    ax.set_xscale("log")
    ax.set_xticks(windows)
    ax.set_xticklabels([str(w) for w in windows])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Validity window (s)")
    ax.set_ylabel("Scenario runs (%)")
    ax.set_ylim(bottom=0)
    _tidy(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F5: live outcomes
# ---------------------------------------------------------------------------


def fig_live(claims: Mapping[str, Mapping[str, Any]], groups: Mapping[str, Any]) -> Figure:
    """Re-proposal, duplicate, unsafe and completion rates per model and language (x) and arm (hue)."""
    metrics = (("reproposal", "Re-proposal"), ("duplicate", "Duplicate"), ("unsafe", "Unsafe execution"), ("completion", "Completion"))
    fig, axes2 = plt.subplots(2, 2, figsize=(SINGLE_COLUMN_IN, 2.65), sharey=True)
    axes = axes2.ravel()
    live = groups.get("live")
    if not live:
        for ax, (_, title) in zip(axes, metrics):
            _empty(ax, f"{title}: no live track")
        fig.tight_layout()
        return fig
    models = list(live["models"])
    slugs = dict(live.get("model_slugs") or {})
    langs = [lang for lang in _ordered(live["langs"], LANG_ORDER) if lang != "all"]
    arms = _ordered(live["arms"], ARM_ORDER)
    columns = [(m, lang) for m in models for lang in langs]
    width = 0.8 / max(1, len(arms))
    for ax, (metric, title) in zip(axes, metrics):
        drawn = 0
        for j, a in enumerate(arms):
            xs, ys, errs = [], [], []
            for i, (m, lang) in enumerate(columns):
                pt = point(claims, f"live_{metric}_{slugs.get(m, m)}_{lang}_{a.lower()}")
                if pt is None:
                    continue
                y, e = _pct(pt)
                xs.append(i + (j - (len(arms) - 1) / 2) * width)
                ys.append(y)
                errs.append(e)
            if xs:
                drawn += len(xs)
                ax.bar(xs, ys, width=width * 0.92, color=colour_for(a, ARM_ORDER), label=a, yerr=list(zip(*errs)), error_kw={"elinewidth": 0.5, "capsize": 1.5, "capthick": 0.5, "ecolor": "#333333"})
        if drawn == 0:
            _empty(ax, f"{title}: no valid episodes")
            continue
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([LANG_LABELS.get(lang, lang) for _, lang in columns], fontsize=6)
        ax.set_xlim(-0.6, len(columns) - 0.4)
        ax.set_title(title, loc="left")
        _tidy(ax)
        _model_group_axis(ax, columns, models, 0, 10)
    for ax in axes2[:, 0]:
        _percent_axis(ax, "Valid episodes (%)")
    handles, labels = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.0, w_pad=0.6)
    return fig


# ---------------------------------------------------------------------------
# F6: latency per stage
# ---------------------------------------------------------------------------


MODEL_CALL_ORDER: tuple[str, ...] = PIPELINE_ORDER + ("live",)
"""Columns of the model-call panel of F6: the capture pipelines, then the live agent step."""
MODEL_CALL_LABELS: dict[str, str] = {"direct": "direct", "noncanon": "no\ncanon.", "nogate": "no\ngate", "translate": "translate\n(same)", "translate_mt": "translate\n(MT)", "english_reference": "English\nref.", "live": "live\nstep"}
NON_MODEL_STAGES: tuple[tuple[str, str, str, str], ...] = (
    ("translate", "translate_mt", "translate\n(MT)", "translate"),
    ("translate", "translate", "translate\n(same)", "translate"),
    ("canonicalise", "direct", "canonical-\niser", "canonicalise"),
    ("gate", "direct", "gate", "gate"),
    ("monitor", "trace", "monitor\n(trace)", "monitor"),
    ("monitor", "live", "monitor\n(live)", "monitor"),
)
"""Non-model boxes of F6 as (stage, pipeline, label, colour key); canonicalise and gate are shown for the direct pipeline."""


def _log_axis(ax: Any, decades_only: bool) -> None:
    """Logarithmic y axis with plain tick labels: 1, 2, 5 per decade, or the decades alone."""
    ax.set_yscale("log")
    subs = (1.0,) if decades_only else (1.0, 2.0, 5.0)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=subs, numticks=30))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if v >= 1 else f"{v:.10f}".rstrip("0")))


def _box_stats(claims: Mapping[str, Mapping[str, Any]], tok: str) -> tuple[dict[str, Any], int] | None:
    """The bxp statistics of a latency token (box p25 to p75, whiskers p05 to p95, line median) and its n, or None."""
    q = {k: point(claims, f"lat_{tok}_{k}") for k in ("p05", "p25", "median", "p75", "p95", "n")}
    if any(q[k] is None for k in ("p05", "p25", "median", "p75", "p95")):
        return None
    stats = {"med": q["median"].value, "q1": q["p25"].value, "q3": q["p75"].value, "whislo": q["p05"].value, "whishi": q["p95"].value, "fliers": []}
    n = int(q["n"].value) if q["n"] is not None else 0
    return stats, n


def _draw_boxes(ax: Any, stats: Sequence[dict[str, Any]], positions: Sequence[float], colours: Sequence[str], width: float) -> None:
    boxes = ax.bxp(list(stats), positions=list(positions), widths=width, showfliers=False, patch_artist=True, medianprops={"color": "#000000", "linewidth": 0.8}, boxprops={"linewidth": 0.5}, whiskerprops={"linewidth": 0.5}, capprops={"linewidth": 0.5})
    for patch, colour in zip(boxes["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)


def _latency_models(entries: Sequence[Mapping[str, Any]], groups: Mapping[str, Any]) -> list[str]:
    """The models with timed model calls, in the order of the capture group, then the live group, then the rest."""
    order: list[str] = []
    for group in ("capture", "live"):
        for m in list((groups.get(group) or {}).get("models") or []):
            if m not in order:
                order.append(m)
    for e in entries:
        if e.get("stage") == "model" and e.get("model") not in order:
            order.append(str(e.get("model")))
    present = {str(e.get("model")) for e in entries if e.get("stage") == "model"}
    return [m for m in order if m in present]


def fig_latency(claims: Mapping[str, Mapping[str, Any]], groups: Mapping[str, Any]) -> Figure:
    """Latency per stage in two panels of one column, both on a logarithmic y axis.

    Panel (a) shows the model call seconds per capture pipeline and the live agent step,
    one box per pipeline and model (models as colours when there are several); panel (b)
    shows the non-model stages: the opus-mt and same-model translation calls, the
    canonicaliser and the gate of the direct pipeline (primary model), and one monitor
    hook call on the trace and on the live track, coloured by stage from a colour set
    disjoint from the model colours of panel (a). Boxes span p25 to p75, whiskers p05 to
    p95, the line is the median; n is the number of timed calls.
    """
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(SINGLE_COLUMN_IN, 2.6))
    entries = list(groups.get("latency") or [])
    models = _latency_models(entries, groups)
    columns = _ordered(sorted({str(e.get("pipeline")) for e in entries if e.get("stage") == "model"}), MODEL_CALL_ORDER)
    drawn_a = 0
    handles: list[Any] = []
    if columns and models:
        width = 0.8 / len(models)
        for j, m in enumerate(models):
            stats, positions, colours = [], [], []
            for i, pipeline in enumerate(columns):
                tok = next((e["token"] for e in entries if e.get("stage") == "model" and e.get("pipeline") == pipeline and e.get("model") == m), None)
                box = _box_stats(claims, tok) if tok else None
                if box is None:
                    continue
                stats.append(box[0])
                positions.append(i + (j - (len(models) - 1) / 2) * width)
                colours.append(PALETTE[j % len(PALETTE)])
            if stats:
                drawn_a += len(stats)
                _draw_boxes(ax_a, stats, positions, colours, width * 0.85)
                handles.append(matplotlib.patches.Patch(facecolor=PALETTE[j % len(PALETTE)], alpha=0.75, linewidth=0.5, edgecolor="#000000", label=m))
    if drawn_a == 0:
        _empty(ax_a, "(a) Model calls: no timed calls in the ledger")
    else:
        labels = []
        for pipeline in columns:
            ns = [_box_stats(claims, e["token"])[1] for e in entries if e.get("stage") == "model" and e.get("pipeline") == pipeline and e.get("model") in models and _box_stats(claims, e["token"]) is not None]
            labels.append(MODEL_CALL_LABELS.get(pipeline, pipeline) + "".join(f"\nn={n}" for n in ns))
        ax_a.set_xticks(range(len(columns)))
        ax_a.set_xticklabels(labels, fontsize=6)
        ax_a.set_xlim(-0.6, len(columns) - 0.4)
        _log_axis(ax_a, decades_only=False)
        ax_a.set_ylabel("Seconds per model call")
        ax_a.set_title("(a) Model calls", loc="left", fontsize=8)
        if len(handles) > 1:
            ax_a.legend(handles=handles, loc="upper left", ncol=len(handles), fontsize=7, handlelength=1.0, borderaxespad=0.2)
        _tidy(ax_a)
    primary = models[0] if models else None
    stats_b, labels_b, colours_b = [], [], []
    for stage, pipeline, label, colour_key in NON_MODEL_STAGES:
        matches = [e for e in entries if e.get("stage") == stage and e.get("pipeline") == pipeline]
        matches.sort(key=lambda e: (e.get("model") != primary, str(e.get("model"))))
        box = None
        for e in matches:
            box = _box_stats(claims, e["token"])
            if box is not None:
                break
        if box is None:
            continue
        stats_b.append(box[0])
        labels_b.append(f"{label}\nn={box[1]}")
        colours_b.append(STAGE_COLOURS.get(colour_key, colour_for(colour_key, STAGE_ORDER)))
    if not stats_b:
        _empty(ax_b, "(b) Other stages: no timed calls in the ledger")
    else:
        _draw_boxes(ax_b, stats_b, list(range(len(stats_b))), colours_b, 0.6)
        ax_b.set_xticks(range(len(labels_b)))
        ax_b.set_xticklabels(labels_b, fontsize=6)
        ax_b.set_xlim(-0.6, len(labels_b) - 0.4)
        _log_axis(ax_b, decades_only=True)
        ax_b.set_ylabel("Seconds per call")
        ax_b.set_title("(b) Other stages", loc="left", fontsize=8)
        _tidy(ax_b)
    fig.tight_layout(h_pad=1.2)
    return fig


FIGURES: tuple[tuple[str, Any], ...] = (
    ("tasdiq_fig1", fig_capture),
    ("tasdiq_fig2", fig_attribution),
    ("tasdiq_fig3", fig_revision),
    ("tasdiq_fig4", fig_window),
    ("tasdiq_fig5", fig_live),
    ("tasdiq_fig6", fig_latency),
)


def make_figures(results_dir: Path | str = "results", out_dir: Path | str = "figures", formats: Sequence[str] = ("pdf", "png")) -> list[Path]:
    """Draw every figure from the ledger into ``out_dir`` and return the written paths."""
    claims, groups = load_inputs(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with plt.rc_context(RC):
        for name, draw in FIGURES:
            fig = draw(claims, groups)
            for fmt in formats:
                path = out / f"{name}.{fmt}"
                fig.savefig(path, format=fmt, bbox_inches="tight", pad_inches=0.02, metadata={"Creator": "Matplotlib"} if fmt == "pdf" else None)
                written.append(path)
            plt.close(fig)
    return written
