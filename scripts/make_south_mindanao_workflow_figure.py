#!/usr/bin/env python
"""Render a publication-quality, horizontal workflow figure of the South
Mindanao prediction method.

Summarizes how the PRODUCTION PINN v3 model (trained on the full Cotabato set)
is transferred to the South Mindanao slope-unit dataset with no retraining,
mirroring the pipeline in `scripts/predict_south_mindanao_v2_8.py`.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = PROJECT_ROOT / "figures" / "south_mindanao_prediction_workflow.png"

plt.rcParams.update({"font.family": "DejaVu Sans"})

INK = "#1f2933"
MUTED = "#55606b"
PAGE_BG = "#ffffff"

# Palette: (fill, border, header-text) per semantic role.
PAL = {
    "model":  ("#d5dae1", "#6b7783", "#39424c"),
    "data":   ("#c9e6d4", "#4e9e78", "#2f6f52"),
    "proc":   ("#d7d4ef", "#7a72c4", "#463f8c"),
    "phys":   ("#efd9a6", "#c39a3f", "#856512"),
}

# (role, title, subtitle, [pill sub-steps])
STAGES = [
    ("model", "Production model", "Frozen PINN v3",
     ["Trained on full Cotabato set",
      "Physics-informed architecture",
      "Weights frozen (no retraining)",
      "Paired transform manifest"]),
    ("data", "Input data", "South Mindanao",
     ["Slope-unit GeoPackage",
      "layer = joined_layer",
      "Geospatial predictors",
      "New, unseen region"]),
    ("proc", "Align to training", "Match the training distribution",
     ["Rename columns → training schema",
      "Convert physics units",
      "Impute from training medians",
      "Slope filter ≥ 10°",
      "Collapse soil types (25 → 3)",
      "Replay manifest + unit audit"]),
    ("phys", "Inference", "Frozen forward pass",
     ["Mohr–Coulomb Factor of Safety",
      "Critical acceleration",
      "Newmark displacement",
      "Sigmoid → susceptibility"]),
    ("data", "Outputs", "Maps + tables",
     ["Susceptibility p(s) ∈ [0, 1]",
      "FoS + displacement maps",
      "Cohesion c′ + friction φ′",
      "GPKG + CSV + choropleths"]),
]

EQUATIONS = [
    r"$\mathrm{FoS} = \dfrac{c'}{\gamma\,h\,\sin\alpha} + \dfrac{\tan\varphi'}{\tan\alpha}"
    r" - \dfrac{m\,\gamma_w\,\tan\varphi'}{\gamma\,\tan\alpha}$",
    r"$a_c = (\mathrm{FoS} - 1)\,g\,\sin\alpha$",
    r"$D = f_{\mathrm{Newmark}}(a_c,\ \mathrm{PGA})$",
    r"$p(s) = \dfrac{1}{1 + e^{\,5 - D}}$",
]

CARD_W = 3.0
GAP = 0.55
MARGIN = 0.55
Y_TOP = 11.0
HEADER_H = 1.05
N_STAGES = 5
TOTAL_W = 2 * MARGIN + N_STAGES * CARD_W + (N_STAGES - 1) * GAP  # 18.3
TOTAL_H = 12.0
PILL_LINE_H = 0.30
PILL_PAD = 0.30
PILL_GAP = 0.16
WRAP = 24


def pill_lines(text):
    return textwrap.fill(text, WRAP).split("\n")


def pill_height(text):
    return len(pill_lines(text)) * PILL_LINE_H + PILL_PAD


def draw_card(ax, x0, role, title, subtitle, pills):
    fill, border, htext = PAL[role]

    # Measure total card height so tops align while bodies size to content.
    body_h = sum(pill_height(p) + PILL_GAP for p in pills) + 0.30
    card_h = HEADER_H + body_h + 0.25
    y0 = Y_TOP - card_h

    # Soft shadow.
    ax.add_patch(FancyBboxPatch(
        (x0 + 0.05, y0 - 0.07), CARD_W, card_h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=0, facecolor="#c4ccd4", alpha=0.4, zorder=1))
    # Card body (light tint).
    ax.add_patch(FancyBboxPatch(
        (x0, y0), CARD_W, card_h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.6, edgecolor=border, facecolor=fill + "55", zorder=2))

    cx = x0 + CARD_W / 2
    ax.text(cx, Y_TOP - 0.42, title, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color=htext, zorder=4)
    ax.text(cx, Y_TOP - 0.78, subtitle, ha="center", va="center",
            fontsize=9.5, color=MUTED, style="italic", zorder=4)
    ax.plot([x0 + 0.3, x0 + CARD_W - 0.3], [Y_TOP - HEADER_H] * 2,
            color=border, alpha=0.5, linewidth=1.1, zorder=3)

    # Stacked pill sub-steps.
    ty = Y_TOP - HEADER_H - 0.22
    for p in pills:
        lines = pill_lines(p)
        ph = len(lines) * PILL_LINE_H + PILL_PAD
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.22, ty - ph), CARD_W - 0.44, ph,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            linewidth=0, facecolor=fill, alpha=0.95, zorder=3))
        ax.text(cx, ty - ph / 2, "\n".join(lines), ha="center", va="center",
                fontsize=9.3, color=INK, linespacing=1.25, zorder=4)
        ty -= ph + PILL_GAP

    return x0, x0 + CARD_W, card_h


def main():
    fig, ax = plt.subplots(figsize=(TOTAL_W, TOTAL_H))
    fig.patch.set_facecolor(PAGE_BG)
    ax.set_facecolor(PAGE_BG)
    ax.set_xlim(0, TOTAL_W)
    ax.set_ylim(0, TOTAL_H)
    ax.axis("off")

    cx_page = TOTAL_W / 2
    ax.text(cx_page, 11.55, "South Mindanao landslide susceptibility mapping",
            ha="center", va="center", fontsize=20, fontweight="bold", color=INK)

    # ---- Horizontal flow of stage cards (tops aligned) ---- #
    edges = []
    for i, (role, title, subtitle, pills) in enumerate(STAGES):
        x0 = MARGIN + i * (CARD_W + GAP)
        left, right, _ = draw_card(ax, x0, role, title, subtitle, pills)
        edges.append((left, right))

    arrow_y = Y_TOP - HEADER_H / 2
    for (_, right), (left, _) in zip(edges[:-1], edges[1:]):
        ax.add_patch(FancyArrowPatch(
            (right + 0.06, arrow_y), (left - 0.06, arrow_y),
            arrowstyle="-|>", mutation_scale=22, linewidth=2.4,
            color="#7b8794", zorder=6))

    # ---- Embedded-physics banner (the core of step 4) ---- #
    bx0, bx1, by0, by1 = MARGIN, TOTAL_W - MARGIN, 1.95, 3.35
    fill, border, htext = PAL["phys"]
    ax.add_patch(FancyBboxPatch(
        (bx0, by0), bx1 - bx0, by1 - by0,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6, edgecolor=border, facecolor=fill + "40",
        linestyle="--", zorder=2))
    ax.text((bx0 + bx1) / 2, by1 - 0.32,
            "Embedded physics layer  ·  Mohr–Coulomb → Newmark → susceptibility  "
            "(inside step 4, Inference)",
            ha="center", va="center", fontsize=11, fontweight="bold", color=htext)
    span = bx1 - bx0
    # Non-uniform placement: the 3-term FoS needs more horizontal room.
    eq_fracs = [0.19, 0.47, 0.68, 0.88]
    for eq, frac in zip(EQUATIONS, eq_fracs):
        ax.text(bx0 + span * frac, by0 + 0.52, eq, ha="center", va="center",
                fontsize=14.5, color=INK)

    # ---- Legend ---- #
    legend = [("model", "Model"), ("data", "Data / outputs"),
              ("proc", "Alignment"), ("phys", "Physics inference")]
    lx = (TOTAL_W - 3 * 2.9 - 1.0) / 2
    ly = 1.0
    for role, label in legend:
        fill, border, _ = PAL[role]
        ax.add_patch(FancyBboxPatch(
            (lx, ly - 0.16), 0.42, 0.32,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            linewidth=1.2, edgecolor=border, facecolor=fill, zorder=3))
        ax.text(lx + 0.58, ly, label, ha="left", va="center",
                fontsize=10, color=INK)
        lx += 2.9

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, facecolor=PAGE_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved workflow figure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
