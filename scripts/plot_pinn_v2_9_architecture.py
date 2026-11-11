"""Architecture figure for the v2-9 grid PINN and its three wetness variants.

Draws one shared pipeline (inputs -> geotechnical MLP -> c, phi -> FS -> a_c -> Newmark D ->
learnable threshold -> P(landslide)) with the wetness module shown as three swappable lanes:
A pure earthquake-induced (m = 0), B estimated recharge (WetnessRatioLayer), C fixed recharge
(WetnessLayerFixedRecharge). A notation panel defines every symbol with its unit.
See docs/pinn_v2-9_architecture.md for the full specification.

    venv/bin/python scripts/plot_pinn_v2_9_architecture.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT_STEM = "pinn_v2-9_architecture"

# (fill, edge) per role; the same mapping is drawn as the legend.
STYLES = {
    "input": ("#EEEEEE", "#6B6B6B"),
    "nn": ("#DCE9F7", "#2F6DB5"),
    "physics": ("#FCE8D5", "#D9822B"),
    "learnable": ("#DFF2E1", "#3A8F4B"),
    "user": ("#EADFF5", "#7B4FB0"),
    "loss": ("#F9DEDE", "#C0392B"),
}
LEGEND = [
    ("input", "data input"),
    ("nn", "neural network"),
    ("physics", "fixed physics equation"),
    ("learnable", "learnable physical scalar / table"),
    ("user", "user-supplied constant"),
    ("loss", "training loss"),
]
ARROW_COLOR = "#333333"
STUB_COLOR = "#6B6B6B"
BACKPROP_COLOR = "#C0392B"
TITLE_SIZE = 12.5
BODY_SIZE = 11.5
EQ_SIZE = 14.5

# Every symbol drawn in the figure, with meaning and unit. Order = reading order in the panel
# (filled column by column).
NOTATION = [
    (r"$\theta$", "slope angle  [°]"),
    (r"$\gamma$", "soil unit weight = BD · g  [kN/m³]"),
    (r"$\gamma_w$", "unit weight of water = 9.81 kN/m³"),
    (r"$H$", "soil thickness (failure depth)  [m]"),
    (r"$A$", "upslope contributing area  [m²]"),
    (r"$b$", "contour width = DEM cell size  [m]"),
    (r"$s$", "soil texture class (12 USDA)"),
    (r"$P$", "precipitation  [mm/month]"),
    (r"$z_1, z_2$", "raw outputs of the MLP"),
    (r"$\sigma(\cdot)$", "logistic sigmoid  1 / (1 + e$^{-x}$)"),
    (r"$c$", "soil cohesion  [kPa]"),
    (r"$\phi$", "internal friction angle  [rad]"),
    (r"$m$", "saturated fraction  $h_w / H$  [–]"),
    (r"$R$", "steady recharge rate  [m/h]"),
    (r"$K$", "saturated hydraulic conductivity  [m/h]"),
    (r"$u_k$", "learnable per-class offset of $K$"),
    (r"$\mathrm{FS}$", "factor of safety  [–]"),
    (r"$g$", "gravity = 9.81 m/s²"),
    (r"$a_c$", "critical acceleration  [m/s²]"),
    (r"$\mathrm{PGA}$", "peak ground acceleration  [m/s²]"),
    (r"$k$", r"critical-acceleration ratio  $a_c / \mathrm{PGA}$"),
    (r"$D$", "Newmark displacement  [cm]"),
    (r"$D_0$", "learnable critical displacement  [cm]"),
    (r"$a$", "learnable threshold steepness  [–]"),
    (r"$p$", "landslide probability  [0, 1]"),
]
NOTATION_COLUMNS = 4


def box(ax, x, y, w, h, role, title, body="", *, title_size=TITLE_SIZE,
        body_size=BODY_SIZE, body_y=None, dashed=False):
    """Rounded box with a bold title at the top and a body centred below it (or at body_y)."""
    fill, edge = STYLES[role]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor="none" if dashed else fill, edgecolor=edge,
        linewidth=1.4, linestyle="--" if dashed else "-", zorder=1 if dashed else 2,
    ))
    if title:
        ax.text(x + w / 2, y + h - 0.9, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", zorder=3)
    if body:
        centre = body_y if body_y is not None else y + (h - 2.4) / 2
        ax.text(x + w / 2, centre, body, ha="center", va="center",
                fontsize=body_size, linespacing=1.6, zorder=3)


def tag(ax, x, y, text, role, size=10.5):
    """Small coloured label marking where a lane's recharge R comes from."""
    fill, edge = STYLES[role]
    ax.text(x, y, text, ha="center", va="center", fontsize=size, zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=fill, edgecolor=edge, linewidth=1.1))


def arrow(ax, points, *, color=ARROW_COLOR, dashed=False, label=None, label_xy=None,
          label_kw=None):
    """Polyline through `points`; the last segment carries the arrow head."""
    ls = (0, (4, 3)) if dashed else "-"
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.4, linestyle=ls, zorder=1.5)
    ax.annotate("", xy=points[-1], xytext=points[-2], zorder=1.5,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.4,
                                linestyle=ls, mutation_scale=13, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_xy, label, **{"fontsize": 9, "color": color, **(label_kw or {})})


def draw_inputs(ax):
    box(ax, 1, 46, 14, 11, "input", "Feature vector x",
        "z-scored features\n$\\mathit{type}$ → embedding")
    box(ax, 1, 2, 14, 41, "input", "Physics inputs",
        "(raw units)\n\n"
        "$\\theta$   slope\n"
        "$\\gamma$   unit weight\n"
        "$H$   soil thickness\n"
        "$A$   upslope area\n"
        "$s$   soil texture class\n"
        "$P$   precipitation\n"
        "$\\mathrm{PGA}$",
        body_size=12, body_y=30)


def draw_geotech(ax):
    box(ax, 20, 46, 25, 11, "nn", "Geotechnical MLP",
        "input: x \\ {Prc, Slope, Elev, BUK}\n"
        "Dense 64 → 8 × [Dense–BN–LeakyReLU]\n"
        "→ Dense 2 → $(z_1,\\ z_2)$")
    box(ax, 49, 46, 13, 11, "nn", "Bounded strength",
        "$c = 30\\,\\sigma(z_1)$  kPa\n$\\phi = \\sigma(z_2)$  rad", body_size=EQ_SIZE)
    arrow(ax, [(15.6, 51.5), (19.4, 51.5)])
    arrow(ax, [(45.6, 51.5), (48.4, 51.5)])


def draw_wetness(ax):
    box(ax, 20, 2, 42, 38, "physics", "", dashed=True)
    ax.text(41, 39.2, "Wetness module  →  $m = h_w / H$   (one variant per model)",
            ha="center", va="top", fontsize=TITLE_SIZE + 0.5, fontweight="bold")

    box(ax, 22, 30, 38, 6.5, "physics", "A   Pure earthquake-induced",
        "$m = 0$    (dry slope; rainfall disconnected)", body_size=EQ_SIZE - 1.5)

    box(ax, 22, 3.5, 9, 24.5, "learnable", "$K(s)$",
        "Ksat table\nper soil class\n+ learnable\noffset $u_k$", title_size=13)

    lane_x, lane_w = 33.5, 26.5
    box(ax, lane_x, 16.5, lane_w, 11.5, "physics", "B   Estimated recharge",
        "$m = \\min\\!\\left(1,\\ \\dfrac{R\\,\\ln(1+A)}{K\\;H\\;\\sin\\theta}\\right)$",
        body_size=EQ_SIZE, body_y=22.6)
    tag(ax, lane_x + lane_w / 2, 18.4,
        "$R$ = NN(x, $P$) per cell,   $10^{-5} \\leq R \\leq 10^{-1}$ m/h", "nn")

    box(ax, lane_x, 3.5, lane_w, 11.5, "physics", "C   Fixed recharge",
        "$m = \\min\\!\\left(1,\\ \\dfrac{R\\;(A/b)}{K\\;H\\;\\sin\\theta}\\right)$",
        body_size=EQ_SIZE, body_y=9.6)
    tag(ax, lane_x + lane_w / 2, 5.4, "$R$ = user input (e.g. 3 mm/day)", "user")

    arrow(ax, [(31.6, 22.5), (33, 22.5)])
    arrow(ax, [(31.6, 9), (33, 9)])
    # Physics inputs feed the wetness module.
    arrow(ax, [(15.6, 20), (19.4, 20)])


def draw_physics_chain(ax):
    x, w = 67, 25
    cx = x + w / 2
    box(ax, x, 45, w, 12, "physics", "Infinite-slope factor of safety",
        "$\\mathrm{FS} = \\dfrac{c}{\\gamma\\,H\\,\\sin\\theta}"
        " + \\dfrac{\\tan\\phi}{\\tan\\theta}\\left(1 - m\\,\\dfrac{\\gamma_w}{\\gamma}\\right)$",
        body_size=EQ_SIZE, body_y=50)
    box(ax, x, 32, w, 10, "physics", "Critical acceleration",
        "$a_c = (\\mathrm{FS} - 1)\\; g\\,\\sin\\theta$\n"
        "$k = \\mathrm{softclip}_{[0.05,\\ 0.75]}\\!\\left(\\dfrac{a_c}{\\mathrm{PGA}}\\right)$",
        body_size=EQ_SIZE - 0.5, body_y=35.6)
    box(ax, x, 19, w, 10, "physics", "Newmark displacement",
        "$\\ln D = 0.215 + \\ln\\!\\left[(1-k)^{2.341}\\; k^{-1.438}\\right] + 0.51$\n"
        "(Jibson 2007 regression;  $D$ in cm)",
        body_size=EQ_SIZE - 2, body_y=22.6)
    box(ax, x, 6, w, 10, "learnable", "Learnable threshold",
        "$p = \\sigma\\!\\left(a\\,\\left[\\ln D - \\ln D_0\\right]\\right)$\n"
        "$D_0$ and $a$ learned",
        body_size=EQ_SIZE - 0.5, body_y=9.7)
    for y_top, y_bottom in ((45, 42), (32, 29), (19, 16)):
        arrow(ax, [(cx, y_top - 0.4), (cx, y_bottom + 0.4)])

    # c, phi into FS; m out of the wetness module up into FS.
    arrow(ax, [(62.6, 53.5), (66.4, 53.5)])
    arrow(ax, [(62.6, 21), (64.5, 21), (64.5, 48.5), (66.4, 48.5)],
          label="$m$", label_xy=(63.2, 33), label_kw=dict(fontsize=14))

    # Raw physics inputs reused by the chain, drawn as stubs to keep the routing clean.
    for y, text in ((51, "$\\theta,\\ \\gamma,\\ H$"), (37, "$\\theta,\\ \\mathrm{PGA}$")):
        arrow(ax, [(96, y), (92.6, y)], color=STUB_COLOR)
        ax.text(96.3, y, text, va="center", fontsize=12, color=STUB_COLOR)


def draw_output(ax):
    box(ax, 94, 6, 9, 10, "nn", "P(landslide)", "$p$\n(final_head)", title_size=12, body_size=11)
    box(ax, 94, 19, 9, 10, "loss", "Loss",
        "Dice + w-BCE\nvs. inventory\nweights 1 : 5", title_size=12, body_size=10.5)
    arrow(ax, [(92.6, 11), (93.6, 11)])
    arrow(ax, [(98.5, 16.4), (98.5, 18.6)])
    # Gradients flow back through every physics block into the MLP (and K / recharge head).
    arrow(ax, [(101.5, 29.4), (101.5, 60), (32.5, 60), (32.5, 57.6)],
          color=BACKPROP_COLOR, dashed=True,
          label="∂L back-propagated through the physics chain into the MLP, $K(s)$ and the recharge head",
          label_xy=(67, 60.6), label_kw=dict(ha="center", va="bottom", style="italic", fontsize=10))


def draw_legend(ax, y):
    x = 1
    for role, label in LEGEND:
        fill, edge = STYLES[role]
        ax.add_patch(FancyBboxPatch((x, y - 0.9), 2.2, 1.8, boxstyle="round,pad=0.1,rounding_size=0.3",
                                    facecolor=fill, edgecolor=edge, linewidth=1.2))
        ax.text(x + 3, y, label, va="center", fontsize=11.5)
        x += 3.8 + 0.66 * len(label)


def draw_notation(ax, top, bottom):
    """Symbol table: column-major, symbol right-aligned against its description."""
    ax.add_patch(FancyBboxPatch((1, bottom), 102.5, top - bottom,
                                boxstyle="round,pad=0.25,rounding_size=0.8",
                                facecolor="#FAFAFA", edgecolor="#BBBBBB", linewidth=1.0))
    ax.text(2, top - 0.6, "Notation", fontsize=TITLE_SIZE, fontweight="bold", va="top")
    rows = -(-len(NOTATION) // NOTATION_COLUMNS)
    col_w = 102 / NOTATION_COLUMNS
    row_h = (top - bottom - 3.2) / rows
    for i, (symbol, meaning) in enumerate(NOTATION):
        col, row = divmod(i, rows)
        x0 = 2 + col * col_w
        y = top - 3.4 - row * row_h
        ax.text(x0 + 4.5, y, symbol, ha="right", va="center", fontsize=13)
        ax.text(x0 + 5.5, y, meaning, ha="left", va="center", fontsize=11.5)


def main():
    # One typeface for text and maths, so inline symbols match the surrounding words.
    plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(0, 104)
    ax.set_ylim(-24, 63)
    ax.set_axis_off()

    draw_inputs(ax)
    draw_geotech(ax)
    draw_wetness(ax)
    draw_physics_chain(ax)
    draw_output(ax)
    draw_legend(ax, y=-2.5)
    draw_notation(ax, top=-5.5, bottom=-23.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_DIR / OUT_STEM}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
