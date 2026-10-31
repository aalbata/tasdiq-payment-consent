"""Table filling from the claim ledger.

Templates are LaTeX table bodies with placeholders ``<<claim_id>>`` (the value) and
``<<claim_id:ci>>`` (the interval as ``[lo, hi]``); ``<<claim_id:n>>`` and
``<<claim_id:count>>`` give the sample size and the numerator.  :func:`fill_tables`
replaces every placeholder from ``results/ledger.json``, asserts that each placeholder
occurs exactly once in its template and that none remains in the output, and writes the
filled bodies next to each other in the output directory.  Rounding happens only here:
fractions are printed as percentages with one decimal, counts as integers, seconds with
three significant figures and bytes as gigabytes with two decimals.

:func:`write_templates` generates the template set from the ledger and the group tokens
of ``results/manifest.json`` (benchmark composition, baseline ladder and calibration,
capture results, enforcement rates, enforcement counts by family, live results and cost),
so the row and column sets
follow the languages, pipelines, arms and models present in the results.  Existing
templates are kept unless ``overwrite`` is set, so hand edits survive a rebuild.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PLACEHOLDER = re.compile(r"<<([A-Za-z0-9_.-]+)(?::(ci|n|count))?>>")
MISSING = "n/a"

TABLE_NAMES: tuple[str, ...] = (
    "tasdiq_tab_benchmark",
    "tasdiq_tab_ladder",
    "tasdiq_tab_capture",
    "tasdiq_tab_enforcement",
    "tasdiq_tab_enforcement_counts",
    "tasdiq_tab_live",
    "tasdiq_tab_cost",
)

LANG_LABELS: dict[str, str] = {"en": "English", "ar": "Arabic", "mix": "Code-switched", "all": "All"}
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
"""Column headers of the enforcement tables; an arm absent from the mapping keeps its token."""
PIPELINE_LABELS: dict[str, str] = {"direct": "direct", "noncanon": "no canonicaliser", "nogate": "no gate", "translate": "translate (same model)", "translate_mt": "translate (MT)", "english_reference": "English reference"}
ARM_ORDER: tuple[str, ...] = ("PromptOnly", "B0", "B1", "B2", "B3", "B4", "B4Surface", "B5", "B6", "B7", "Tasdiq", "Tasdiq_invalidate", "Tasdiq_narrowing", "Tasdiq_cap_settle", "TasdiqNonAtomic", "TasdiqNonAtomic_invalidate", "TasdiqNonAtomic_narrowing", "TasdiqNonAtomic_cap_settle", "TrustAgent", "AskAlways", "RefuseAll")

LADDER: dict[str, dict[str, str]] = {
    "B0": {"binding": "none", "rule": "none", "revocation": "none", "replay": "none", "key": "none", "source": "no-monitor control"},
    "B1": {"binding": "schema check only", "rule": "none", "revocation": "none", "replay": "none", "key": "none", "source": "schema-only control"},
    "B2": {"binding": "allowlist and caps", "rule": "none", "revocation": "none", "replay": "key presence", "key": "client", "source": "static restrictions"},
    "B3": {"binding": "effect class per interaction", "rule": "none (interaction elevated)", "revocation": "none", "replay": "none", "key": "none", "source": "time-boxed elevation form"},
    "B4": {"binding": "record hash (payee, amount, currency)", "rule": "invalidate on change", "revocation": "not applied (no channel)", "replay": "idempotency ledger", "key": "client (agent)", "source": "transaction-bound form"},
    "B4Surface": {"binding": "rendered confirmation text", "rule": "invalidate on change", "revocation": "not applied (no channel)", "replay": "idempotency ledger", "key": "client (agent)", "source": "surface-bound variant of B4"},
    "B5": {"binding": "record hash and confirmation event", "rule": "new confirmation", "revocation": "issued slots revoked", "replay": "slot key replay", "key": "server (record and slot)", "source": "lease ledger form"},
    "B6": {"binding": "tool and argument fingerprint", "rule": "none", "revocation": "none", "replay": "none", "key": "none", "source": "per-call approval form"},
    "B7": {"binding": "policy set (payees, amount cap)", "rule": "narrowing applied, expansion asks", "revocation": "policy narrowed to empty", "replay": "none", "key": "none", "source": "policy narrowing form"},
    "Tasdiq": {"binding": "record hash and consent channel", "rule": "invalidate, narrowing or cap and settle", "revocation": "record revoked on cancellation", "replay": "ledger keyed by record and slot", "key": "server (record and slot)", "source": "this work"},
    "TasdiqNonAtomic": {"binding": "as Tasdiq, check and consume separated", "rule": "as Tasdiq", "revocation": "as Tasdiq", "replay": "as Tasdiq", "key": "server (record and slot)", "source": "ablation of this work"},
    "TrustAgent": {"binding": "agent claim of confirmation", "rule": "as B4 otherwise", "revocation": "not applied", "replay": "idempotency ledger", "key": "client (agent)", "source": "trust-the-agent control"},
    "AskAlways": {"binding": "fresh presentation per execution", "rule": "as B4", "revocation": "not applied", "replay": "idempotency ledger", "key": "client (agent)", "source": "ask-always control"},
    "RefuseAll": {"binding": "none (every request denied)", "rule": "none", "revocation": "none", "replay": "none", "key": "none", "source": "refuse-all control"},
}
"""Descriptive columns of the baseline ladder; the window column is filled from the ledger."""


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def escape_latex(text: str) -> str:
    """Escape the characters of ``text`` that LaTeX reads as commands."""
    out = str(text)
    for ch, rep in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        out = out.replace(ch, rep)
    return out


def _format_number(value: Any, unit: str) -> str:
    if value is None:
        return MISSING
    if isinstance(value, str):
        return escape_latex(value)
    if unit == "fraction":
        return f"{100.0 * float(value):.1f}"
    if unit == "count":
        return f"{int(round(float(value)))}"
    if unit == "seconds":
        return f"{float(value):.3g}"
    if unit == "bytes":
        return f"{float(value) / 1e9:.2f}"
    if unit in ("text", "verdict"):
        return escape_latex(str(value))
    return f"{float(value):.3g}"


def format_value(claim: Mapping[str, Any]) -> str:
    """The displayed value of a claim under the rounding rule of its unit."""
    return _format_number(claim.get("value"), str(claim.get("unit")))


def format_ci(claim: Mapping[str, Any]) -> str:
    """The displayed interval of a claim as ``[lo, hi]`` in the unit of its value."""
    ci = claim.get("ci")
    if not ci or ci.get("lo") is None or ci.get("hi") is None:
        return MISSING
    unit = str(claim.get("unit"))
    return f"[{_format_number(ci['lo'], unit)}, {_format_number(ci['hi'], unit)}]"


def format_field(claim: Mapping[str, Any], field: str | None) -> str:
    """The displayed text of a placeholder: the value, or its ``ci``, ``n`` or ``count``."""
    if field is None:
        return format_value(claim)
    if field == "ci":
        return format_ci(claim)
    v = claim.get(field)
    return MISSING if v is None else f"{int(v)}"


# ---------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------


def require_inputs(results_dir: Path | str, templates_dir: Path | str) -> tuple[Path, Path]:
    """Return the ledger path and the template directory when both exist."""
    ledger = Path(results_dir) / "ledger.json"
    templates = Path(templates_dir)
    if not ledger.is_file():
        raise FileNotFoundError(f"no claim ledger at {ledger}; run the tracks and build the ledger before filling tables")
    if not templates.is_dir():
        raise FileNotFoundError(f"no table templates at {templates}")
    return ledger, templates


def load_claims(results_dir: Path | str) -> dict[str, dict[str, Any]]:
    """The claims of ``results_dir/ledger.json``."""
    return json.loads((Path(results_dir) / "ledger.json").read_text(encoding="utf-8"))


def load_groups(results_dir: Path | str) -> dict[str, Any]:
    """The ``groups`` mapping of ``results_dir/manifest.json`` (empty when absent)."""
    path = Path(results_dir) / "manifest.json"
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")).get("groups") or {})


def fill_text(text: str, claims: Mapping[str, Mapping[str, Any]], name: str = "template") -> str:
    """Replace every placeholder of ``text`` exactly once from ``claims``.

    Raises ``AssertionError`` when a placeholder occurs more than once in the template,
    ``KeyError`` when a placeholder names a claim that is not in the ledger, and
    ``AssertionError`` when a placeholder survives in the output.
    """
    occurrences = Counter(m.group(0) for m in PLACEHOLDER.finditer(text))
    repeated = sorted(k for k, v in occurrences.items() if v != 1)
    assert not repeated, f"{name}: placeholders occurring more than once: {repeated}"
    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(text)} - set(claims))
    if missing:
        raise KeyError(f"{name}: placeholders without a ledger claim: {missing}")

    def replace(match: re.Match[str]) -> str:
        return format_field(claims[match.group(1)], match.group(2))

    filled = PLACEHOLDER.sub(replace, text)
    assert "<<" not in filled and ">>" not in filled, f"{name}: a placeholder remains after filling"
    assert len(occurrences) == 0 or filled != text, f"{name}: nothing was replaced"
    return filled


def fill_tables(results_dir: Path | str = "results", templates_dir: Path | str = "tables", out_dir: Path | str = "tables_filled", write_missing: bool = True) -> list[Path]:
    """Fill every ``*.tex`` template from the ledger into ``out_dir`` and return the written paths.

    With ``write_missing`` the generated template set is written first for any template
    name that does not yet exist (existing templates are kept).
    """
    if write_missing:
        write_templates(results_dir, templates_dir, overwrite=False)
    ledger, templates = require_inputs(results_dir, templates_dir)
    claims = load_claims(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    files = sorted(templates.glob("*.tex"))
    if not files:
        raise FileNotFoundError(f"no .tex templates under {templates}")
    for path in files:
        filled = fill_text(path.read_text(encoding="utf-8"), claims, path.name)
        target = out / path.name
        target.write_text(filled, encoding="utf-8", newline="\n")
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def _ph(claims: Mapping[str, Any], cid: str, field: str | None = None) -> str:
    """A placeholder for ``cid`` when the claim exists, else the missing marker.

    An interval placeholder is left empty when the claim carries no interval.
    """
    if cid not in claims:
        return MISSING
    if field == "ci" and not (claims[cid].get("ci") or {}).get("hi") is not None:
        return ""
    return f"<<{cid}:{field}>>" if field else f"<<{cid}>>"


def _ordered(items: Sequence[str], order: Sequence[str]) -> list[str]:
    rank = {k: i for i, k in enumerate(order)}
    return sorted(items, key=lambda k: (rank.get(k, len(order)), k))


def _tabular(spec: str, header: str, rows: Sequence[str], note: str) -> str:
    body = "\n".join(rows)
    return f"% {note}\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"


def template_benchmark(claims: Mapping[str, Any], groups: Mapping[str, Any]) -> str:
    bench = groups.get("benchmark") or {}
    families = list(bench.get("families") or [])
    langs = _ordered(bench.get("langs") or [], ("en", "ar", "mix"))
    kinds = list(bench.get("kinds") or [])
    rows: list[str] = []
    rows.append("\\multicolumn{5}{l}{\\emph{Seeds and variants per family}} \\\\")
    for fam in families:
        rows.append(f"{escape_latex(fam)} & {_ph(claims, f'bench_seeds_{fam}')} & {_ph(claims, f'bench_variants_{fam}')} & & \\\\")
    rows.append(f"total & {_ph(claims, 'bench_seeds_total')} & {_ph(claims, 'bench_variants_total')} & & \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{Variants per language and kind}} \\\\")
    head = " & ".join(escape_latex(k) for k in kinds)
    rows.append(f"language & {head}" + " & " * max(0, 3 - len(kinds)) + " & total \\\\")
    for lang in langs:
        cells = " & ".join(_ph(claims, f"bench_variants_{lang}_{k}") for k in kinds)
        rows.append(f"{LANG_LABELS.get(lang, lang)} & {cells}" + " & " * max(0, 3 - len(kinds)) + f" & {_ph(claims, f'bench_variants_{lang}')} \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{Splits and review}} \\\\")
    rows.append(f"dev (seeds, variants) & {_ph(claims, 'bench_seeds_dev')} & {_ph(claims, 'bench_variants_dev')} & & \\\\")
    rows.append(f"test (seeds, variants) & {_ph(claims, 'bench_seeds_test')} & {_ph(claims, 'bench_variants_test')} & & \\\\")
    rows.append(f"excluded variants (near duplicates) & {_ph(claims, 'bench_variants_excluded')} & & & \\\\")
    for cid in sorted(c for c in claims if c.startswith("bench_review_status_")):
        rows.append(f"review status {escape_latex(cid[len('bench_review_status_'):])} (variants) & {_ph(claims, cid)} & & & \\\\")
    rows.append(f"author review set (rows) & {_ph(claims, 'bench_review_rows')} & & & \\\\")
    return _tabular("lrrrr", "family & seeds & variants & & ", rows, "Benchmark composition: counts are integers; n/a marks a count absent from the ledger.")


LADDER_SPEC = r"l>{\raggedright\arraybackslash}p{1.0in}>{\raggedright\arraybackslash}p{0.95in}c>{\raggedright\arraybackslash}p{0.8in}>{\raggedright\arraybackslash}p{0.75in}>{\raggedright\arraybackslash}p{0.65in}>{\raggedright\arraybackslash}p{0.85in}"
"""Column specification of the ladder table; wrapped columns keep it within the text width."""


def template_ladder(claims: Mapping[str, Any], groups: Mapping[str, Any]) -> str:
    arms = sorted((c[len("cal_window_s_"):] for c in claims if c.startswith("cal_window_s_")), key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else len(ARM_ORDER), a))
    rows: list[str] = []
    for arm in arms:
        d = LADDER.get(arm, {"binding": MISSING, "rule": MISSING, "revocation": MISSING, "replay": MISSING, "key": MISSING, "source": MISSING})
        rows.append(f"{escape_latex(arm)} & {escape_latex(d['binding'])} & {escape_latex(d['rule'])} & {_ph(claims, f'cal_window_s_{arm}')} & {escape_latex(d['revocation'])} & {escape_latex(d['replay'])} & {escape_latex(d['key'])} & {escape_latex(d['source'])} \\\\")
    rows.append(f"\\midrule\n\\multicolumn{{8}}{{l}}{{windows swept (s): {_ph(claims, 'cal_windows_swept')}; the window column gives the default window of an arm whose window is swept}} \\\\")
    return _tabular(LADDER_SPEC, "arm & binding & revision rule & window (s) & revocation & replay & key provenance & source", rows, "Baseline ladder and calibration: the window column is the value each arm enforces in seconds (none for an arm without a window).")


CAPTURE_PIPELINE_LABELS: dict[str, str] = dict(PIPELINE_LABELS, translate="translate (same)")
"""Pipeline labels of the capture table; the same-model translation label is shortened so that the table fits the text width at footnotesize."""


def template_capture(claims: Mapping[str, Any], groups: Mapping[str, Any]) -> str:
    cap = groups.get("capture") or {}
    models = list(cap.get("models") or [])
    slugs = dict(cap.get("model_slugs") or {})
    pipelines = _ordered(cap.get("pipelines") or [], tuple(PIPELINE_LABELS))
    langs = _ordered(cap.get("langs") or [], ("en", "ar", "mix")) + (["all"] if cap else [])
    rows: list[str] = []
    for m in models:
        ms = slugs.get(m, m)
        model_rows: list[str] = []
        for p in pipelines:
            for lang in langs:
                tok = f"{lang}_{p}_{ms}"
                if f"cap_n_{tok}" not in claims:
                    continue
                model_rows.append(" & ".join([
                    CAPTURE_PIPELINE_LABELS.get(p, p), LANG_LABELS.get(lang, lang),
                    _ph(claims, f"cap_n_{tok}"),
                    f"{_ph(claims, f'cap_invariance_{tok}')} {_ph(claims, f'cap_invariance_{tok}', 'ci')}",
                    f"{_ph(claims, f'cap_sensitivity_{tok}')} {_ph(claims, f'cap_sensitivity_{tok}', 'ci')}",
                    _ph(claims, f"cap_exact_match_{tok}"),
                    _ph(claims, f"cap_ask_unambiguous_{tok}"),
                    _ph(claims, f"cap_ask_ambiguous_{tok}"),
                    _ph(claims, f"cap_parse_failure_{tok}"),
                ]) + " \\\\")
        if model_rows:
            if rows:
                rows.append("\\midrule")
            rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{escape_latex(m)}}}}} \\\\")
            rows.extend(model_rows)
    if not rows:
        rows.append("\\multicolumn{9}{l}{no capture track in the ledger} \\\\")
    header = " & ".join(["pipeline", "language", "members", "invariance (\\%) [CI]", "sensitivity (\\%) [CI]", _stack("exact", "match (\\%)"), _stack("asks,", "unambiguous (\\%)"), _stack("asks,", "ambiguous (\\%)"), _stack("parse", "failure (\\%)")])
    return _tabular("llrllrrrr", header, rows, "Capture results by language and pipeline, one block per model (the model is named in an italic group row): percentages with one decimal; intervals [CI] are 95 percent cluster bootstrap on the seed (exact Clopper-Pearson for zero or full counts); translate (same) is the same-model translation pipeline.")


def _own_window_token(claims: Mapping[str, Any], arm: str, wt: str) -> str:
    """The window token of an arm's enforcement claims: the primary window, else the single window the arm ran at (B3)."""
    if f"enf_sessions_{arm}_{wt}" in claims:
        return wt
    pattern = re.compile(rf"^enf_sessions_{re.escape(arm)}_(w\d+)$")
    found = sorted(m.group(1) for cid in claims for m in [pattern.match(cid)] if m)
    return found[0] if len(found) == 1 else wt


def _arm_header(tok: str) -> str:
    return escape_latex(ARM_SHORT_LABELS.get(tok, tok))


def _enforcement_frame(claims: Mapping[str, Any], groups: Mapping[str, Any]) -> tuple[list[str], dict[str, str], list[tuple[str, str]], Any, str] | None:
    """The arms, their window tokens, the arms read at a window of their own, the primary window and the column header."""
    trace = groups.get("trace") or {}
    arms = _ordered(trace.get("arms") or [], ARM_ORDER)
    if not arms:
        return None
    w = trace.get("primary_window")
    wt = "wnone" if w is None else f"w{int(w)}"
    tokens = {a: _own_window_token(claims, a, wt) for a in arms}
    own = [(a, tok[1:]) for a, tok in tokens.items() if tok != wt and tok.startswith("w")]
    header = " & ".join(_arm_header(a) + ("$^{\\dagger}$" if tokens[a] != wt else "") for a in arms)
    return arms, tokens, own, w, header


def _enforcement_footnote(rows: list[str], arms: Sequence[str], own: Sequence[tuple[str, str]], note: str) -> str:
    if own:
        footnote = "; ".join(f"{_arm_header(a)} enforces a window of its own and is read at {seconds} s" for a, seconds in own)
        rows.append("\\midrule")
        rows.append(f"\\multicolumn{{{len(arms) + 1}}}{{l}}{{$^{{\\dagger}}$ {footnote}.}} \\\\")
        note += " A dagger marks an arm that enforces a window of its own and is read at that window (" + ", ".join(f"{a} at {seconds} s" for a, seconds in own) + ")."
    return note


ENFORCEMENT_LABEL_COLUMN = ">{\\raggedright\\arraybackslash}p{1.8in}"
"""Column specification of the row labels of the enforcement rate table; the long labels wrap (needs the array package)."""


def _stack(*lines: str) -> str:
    """A table header cell of several lines, set as a nested one-column tabular aligned on its bottom line."""
    return "\\begin{tabular}[b]{@{}c@{}}" + "\\\\".join(lines) + "\\end{tabular}"


def template_enforcement(claims: Mapping[str, Any], groups: Mapping[str, Any]) -> str:
    """Rates over all families per arm at the primary window; the arm headers are the short labels."""
    frame = _enforcement_frame(claims, groups)
    rows: list[str] = []
    if frame is None:
        rows.append("\\multicolumn{2}{l}{no trace track in the ledger} \\\\")
        return _tabular("ll", "rate & arm", rows, "Enforcement rates by arm.")
    arms, tokens, own, w, header = frame
    for label, metric in (("intent violations per execution", "enf_intent_violation_rate"), ("runs with an intent violation", "enf_intent_violation_sessions"), ("unconfirmed executions per execution", "enf_unconfirmed_rate"), ("completion", "enf_completion"), ("runs with an unnecessary confirmation (invalidation-on-change label)", "enf_unnecessary_confirmation_sessions"), ("runs with an unnecessary re-confirmation (narrowing label)", "enf_unnecessary_reconf_narrowing_sessions"), ("false blocks per authorised request", "enf_false_block_rate"), ("runs with a capture-class error", "enf_capture_class_sessions")):
        rows.append(f"{label} & " + " & ".join(_ph(claims, f"{metric}_{a}_{tokens[a]}") for a in arms) + " \\\\")
    rows.append("scenario runs (count) & " + " & ".join(_ph(claims, f"enf_sessions_{a}_{tokens[a]}") for a in arms) + " \\\\")
    note = f"Enforcement rates over all families at window {w} s by arm: percentages with one decimal over all scenario runs, the last row the count of scenario runs; n/a marks a rate without executions. The two unnecessary-confirmation rows carry the two labels of the protocol, the invalidation-on-change label and the narrowing label. Arm headers are the short labels of ARM_SHORT_LABELS; the label column wraps."
    note = _enforcement_footnote(rows, arms, own, note)
    return _tabular(ENFORCEMENT_LABEL_COLUMN + "r" * len(arms), f"rate (\\%) & {header}", rows, note)


def template_enforcement_counts(claims: Mapping[str, Any], groups: Mapping[str, Any]) -> str:
    """Executed intent violations as counts by scenario family and arm at the primary window."""
    frame = _enforcement_frame(claims, groups)
    rows: list[str] = []
    if frame is None:
        rows.append("\\multicolumn{2}{l}{no trace track in the ledger} \\\\")
        return _tabular("ll", "family & arm", rows, "Executed intent violations by family and arm.")
    arms, tokens, own, w, header = frame
    families = list((groups.get("trace") or {}).get("families") or [])
    for fam in families:
        rows.append(f"{escape_latex(fam)} & " + " & ".join(_ph(claims, f"enf_intent_violations_{a}_{fam}_{tokens[a]}") for a in arms) + " \\\\")
    rows.append("all families & " + " & ".join(_ph(claims, f"enf_intent_violations_{a}_{tokens[a]}") for a in arms) + " \\\\")
    note = f"Executed intent violations (count) by scenario family and arm at window {w} s: counts are integers; n/a marks a cell without scenario runs. Arm headers are the short labels of ARM_SHORT_LABELS."
    note = _enforcement_footnote(rows, arms, own, note)
    return _tabular("l" + "r" * len(arms), f"family & {header}", rows, note)


def template_live(claims: Mapping[str, Any], groups: Mapping[str, Any]) -> str:
    live = groups.get("live") or {}
    models = list(live.get("models") or [])
    slugs = dict(live.get("model_slugs") or {})
    langs = _ordered(live.get("langs") or [], ("en", "ar")) + (["all"] if live else [])
    arms = _ordered(live.get("arms") or [], ARM_ORDER)
    rows: list[str] = []
    for m in models:
        ms = slugs.get(m, m)
        for lang in langs:
            for a in arms:
                tok = f"{ms}_{lang}_{a.lower()}"
                if f"live_n_{tok}" not in claims:
                    continue
                rows.append(" & ".join([
                    escape_latex(m), LANG_LABELS.get(lang, lang), escape_latex(a),
                    _ph(claims, f"live_n_{tok}"), _ph(claims, f"live_n_valid_{tok}"),
                    f"{_ph(claims, f'live_reproposal_{tok}')} {_ph(claims, f'live_reproposal_{tok}', 'ci')}",
                    f"{_ph(claims, f'live_duplicate_{tok}')} {_ph(claims, f'live_duplicate_{tok}', 'ci')}",
                    f"{_ph(claims, f'live_unsafe_{tok}')} {_ph(claims, f'live_unsafe_{tok}', 'ci')}",
                    f"{_ph(claims, f'live_completion_{tok}')} {_ph(claims, f'live_completion_{tok}', 'ci')}",
                    _ph(claims, f"live_parse_failure_{tok}"),
                ]) + " \\\\")
    if not rows:
        rows.append("\\multicolumn{10}{l}{no live track in the ledger} \\\\")
    return _tabular("lllrrllllr", "model & language & arm & episodes & valid & re-proposal (\\%) [95\\% CI] & duplicate (\\%) [95\\% CI] & unsafe (\\%) [95\\% CI] & completion (\\%) [95\\% CI] & parse failure (\\%)", rows, "Live results: rates over valid episodes (parse-failure rate over all episodes); intervals are cluster bootstrap on the task.")


def template_cost(claims: Mapping[str, Any], groups: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for e in groups.get("latency") or []:
        tok = e["token"]
        rows.append(" & ".join([escape_latex(e.get("stage", "")), escape_latex(PIPELINE_LABELS.get(e.get("pipeline", ""), e.get("pipeline", ""))), escape_latex(e.get("model", "")), _ph(claims, f"lat_{tok}_n"), _ph(claims, f"lat_{tok}_median"), _ph(claims, f"lat_{tok}_p95"), _ph(claims, f"lat_{tok}_p99"), _ph(claims, f"lat_{tok}_max")]) + " \\\\")
    if not rows:
        rows.append("\\multicolumn{8}{l}{no timed calls in the ledger} \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{8}{l}{\\emph{Peak resident set size (GB)}} \\\\")
    for track in ("trace", "capture", "live"):
        cid = f"mem_peak_rss_bytes_{track}"
        rows.append(f"{track} & & & & {_ph(claims, cid)} & & & \\\\")
    return _tabular("lllrrrrr", "stage & pipeline & model & calls & median (s) & p95 (s) & p99 (s) & max (s)", rows, "Cost: latency per stage in seconds (three significant figures) and peak memory in gigabytes (two decimals).")


TEMPLATES: dict[str, Any] = {
    "tasdiq_tab_benchmark": template_benchmark,
    "tasdiq_tab_ladder": template_ladder,
    "tasdiq_tab_capture": template_capture,
    "tasdiq_tab_enforcement": template_enforcement,
    "tasdiq_tab_enforcement_counts": template_enforcement_counts,
    "tasdiq_tab_live": template_live,
    "tasdiq_tab_cost": template_cost,
}


def write_templates(results_dir: Path | str, templates_dir: Path | str, overwrite: bool = False) -> list[Path]:
    """Write the generated templates under ``templates_dir`` and return the paths written.

    An existing template is left untouched unless ``overwrite`` is set.
    """
    claims = load_claims(results_dir)
    groups = load_groups(results_dir)
    out = Path(templates_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, make in TEMPLATES.items():
        path = out / f"{name}.tex"
        if path.exists() and not overwrite:
            continue
        path.write_text(make(claims, groups), encoding="utf-8", newline="\n")
        written.append(path)
    return written
