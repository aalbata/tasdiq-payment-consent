"""Draw the system and trust-boundary diagram (manuscript Figure 1) with matplotlib.

The figure places the user, the agent, the untrusted content, the confirmation surface, the
monitor (canonicaliser, ambiguity gate and atomic check-and-consume), the consent store, the
idempotency ledger, the payment rail, the payee directory and the clock, draws the trust boundary
as a dashed rectangle and labels every arrow with the message it carries.  Output: a vector PDF
3.45 in wide and a PNG render, both named tasdiq_fig0 in the chosen output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

WIDTH_IN = 3.45
HEIGHT_IN = 2.90
DPI_PNG = 300

TRUSTED = "#0072B2"
UNTRUSTED = "#D55E00"
NEUTRAL = "#555555"
LABEL_PT = 5.0

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
}


def _box(ax, x, y, w, h, text, colour, dashed=False, fontsize=6.3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.03",
        linewidth=0.7,
        edgecolor=colour,
        facecolor="white",
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="black", linespacing=1.1)


def _arrow(ax, p0, p1, colour=NEUTRAL, shrink=1.0):
    arrow = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, linewidth=0.6, color=colour, shrinkA=shrink, shrinkB=shrink)
    ax.add_patch(arrow)


def _label(ax, x, y, text, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=LABEL_PT, color="black", linespacing=1.05)


def draw(out_dir: Path) -> list[Path]:
    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_IN)
    ax.set_ylim(0, HEIGHT_IN)
    ax.axis("off")

    # Trust boundary around the surface, the monitor, the stores, the rail, the directory and the clock.
    ax.add_patch(Rectangle((1.40, 0.08), 2.00, 2.78, linewidth=0.8, edgecolor=TRUSTED, facecolor="#F3F7FB", linestyle="--"))
    ax.text(3.36, 0.14, "trusted computing base", fontsize=5.8, color=TRUSTED, ha="right", va="center", style="italic")

    # Left column (outside the boundary).
    _box(ax, 0.06, 2.40, 0.86, 0.40, "User\n(principal)", NEUTRAL)
    _box(ax, 0.06, 1.30, 0.86, 0.75, "Agent\n(language-model\nplanner, untrusted)", UNTRUSTED)
    _box(ax, 0.06, 0.25, 0.86, 0.50, "Untrusted content\n(pages, invoices,\nmessages)", UNTRUSTED, dashed=True)

    # Column A inside the boundary.
    _box(ax, 1.50, 2.30, 0.80, 0.42, "Confirmation\nsurface\n(non-agentic)", TRUSTED)
    _box(ax, 1.50, 1.20, 0.80, 0.78, "Monitor:\ncanonicaliser,\nambiguity gate,\natomic check-\nand-consume", TRUSTED)
    _box(ax, 1.50, 0.25, 0.80, 0.47, "Payment rail\n(simulated bank)", TRUSTED)

    # Column B inside the boundary.
    _box(ax, 2.62, 1.86, 0.70, 0.46, "Consent store\n(records, state\nmachine, window)", TRUSTED, fontsize=5.8)
    _box(ax, 2.62, 1.32, 0.70, 0.42, "Idempotency\nledger (keys,\nreceipts)", TRUSTED, fontsize=5.8)
    _box(ax, 2.62, 0.76, 0.70, 0.38, "Payee directory\n(stable payee_id)", TRUSTED, fontsize=5.8)
    _box(ax, 2.62, 0.25, 0.70, 0.28, "Clock", TRUSTED, fontsize=5.8)

    # Arrows with labels placed in free space beside them.
    _arrow(ax, (0.49, 2.40), (0.49, 2.05))
    _label(ax, 0.53, 2.22, "utterance\n(English, Arabic,\nmixed)", ha="left")
    _arrow(ax, (0.49, 0.75), (0.49, 1.30))
    _label(ax, 0.53, 1.02, "retrieved text", ha="left")
    _arrow(ax, (0.92, 2.60), (1.50, 2.60))
    _label(ax, 1.21, 2.73, "approve, decline,\nrevise, cancel")
    _arrow(ax, (0.92, 1.80), (1.50, 1.80))
    _label(ax, 1.21, 1.92, "raw extraction,\nexecution request")
    _arrow(ax, (1.50, 1.48), (0.92, 1.48))
    _label(ax, 1.21, 1.37, "ALLOW, DENY,\nASK")
    _arrow(ax, (1.80, 1.98), (1.80, 2.30), colour=TRUSTED)
    _label(ax, 1.76, 2.14, "draft\nrecord", ha="right")
    _arrow(ax, (2.10, 2.30), (2.10, 1.98), colour=TRUSTED)
    _label(ax, 2.14, 2.14, "consent\nevent", ha="left")
    _arrow(ax, (2.30, 1.92), (2.62, 1.92), colour=TRUSTED)
    _label(ax, 2.46, 2.00, "consume", va="bottom")
    _arrow(ax, (2.30, 1.52), (2.62, 1.52), colour=TRUSTED)
    _label(ax, 2.46, 1.60, "key", va="bottom")
    _arrow(ax, (1.72, 1.20), (1.72, 0.72), colour=TRUSTED)
    _label(ax, 1.68, 0.96, "debit\nonce", ha="right")
    _arrow(ax, (2.08, 0.72), (2.08, 1.20), colour=TRUSTED)
    _label(ax, 2.12, 0.96, "receipt\nor fault", ha="left")
    _arrow(ax, (2.62, 0.95), (2.30, 1.26), colour=TRUSTED)
    _label(ax, 2.50, 1.16, "payee_id", va="bottom")
    _arrow(ax, (2.62, 0.39), (2.30, 0.48), colour=TRUSTED)
    _label(ax, 2.46, 0.50, "time", va="bottom")

    # Legend.
    ax.plot([0.08, 0.26], [0.12, 0.12], color=TRUSTED, linewidth=0.7)
    ax.text(0.29, 0.12, "trusted", fontsize=5.5, va="center")
    ax.plot([0.60, 0.78], [0.12, 0.12], color=UNTRUSTED, linewidth=0.7)
    ax.text(0.81, 0.12, "untrusted", fontsize=5.5, va="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "tasdiq_fig0.pdf"
    png = out_dir / "tasdiq_fig0.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=DPI_PNG)
    plt.close(fig)
    return [pdf, png]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the system and trust-boundary diagram to tasdiq_fig0.pdf and tasdiq_fig0.png.")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "figures"), help="output directory (default: figures/ under the code directory)")
    args = parser.parse_args(argv)
    for path in draw(Path(args.out)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
