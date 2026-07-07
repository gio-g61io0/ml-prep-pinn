#!/usr/bin/env python
"""Verify WHY precipitation misleads the transferred model on the South
Mindanao earthquake-induced landslide (EIL) inventory.

Hypothesis (from the presentation): the v3 inventory is purely EIL from the
08 Jun 2026 Mw 7.8 Offshore Sarangani earthquake, so landslides occurred where
SHAKING + steep slopes were — not where it rains. The Cotabato-trained PINN
carries a "wet soil -> landslide" prior (pore-pressure term in FoS), so it paints
the high-rainfall NW as susceptible even though the inventory has no slides there.

This script quantifies that by binning the 1M-cell prediction table:
  - model susceptibility vs precipitation   (expected: rises with rain)
  - ACTUAL landslide rate vs precipitation   (expected: flat / falling)
  - both vs PGA                              (expected: landslide rate rises hard)
  - point-biserial correlation of each feature + susceptibility with the label.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = PROJECT_ROOT / "outputs" / "sm_v3_prod_analysis.csv"
OUT = PROJECT_ROOT / "figures" / "sm_v3_precip_verification.png"

INK = "#1f2933"
RAIN = "#1c9bd6"     # precipitation / susceptibility (model's belief)
TRUTH = "#d7191c"    # actual landslides (reality)
PGA_C = "#e08529"

PRETTY = {
    "Slope_mean": "Slope", "BUK_mean": "Bulk unit wt.", "Prc_mean": "Precipitation",
    "ContributingFactor_mean": "Catchment area", "SoilThc_mean": "Soil thickness",
    "Elev_mean": "Elevation", "PGA2_max": "PGA (shaking)", "susceptibility": "Model susceptibility",
}
FEATS = ["PGA2_max", "Slope_mean", "Elev_mean", "SoilThc_mean",
         "ContributingFactor_mean", "BUK_mean", "Prc_mean", "susceptibility"]


def binned(df, xcol, nbins=10):
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = df["landslide"].to_numpy()
    s = pd.to_numeric(df["susceptibility"], errors="coerce")
    m = np.isfinite(x) & np.isfinite(s)
    x, y, s = x[m].to_numpy(), y[m], s[m].to_numpy()
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    centers, ls_rate, susc = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() > 100:
            centers.append(np.median(x[sel]))
            ls_rate.append(y[sel].mean() * 100)
            susc.append(s[sel].mean())
    return np.array(centers), np.array(ls_rate), np.array(susc)


def dual_panel(ax, df, xcol, xlabel):
    c, ls, su = binned(df, xcol)
    ax.plot(c, ls, "-o", color=TRUTH, lw=2.2, ms=5, label="actual landslide rate")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("actual landslide rate (%)", color=TRUTH)
    ax.tick_params(axis="y", labelcolor=TRUTH)
    ax2 = ax.twinx()
    ax2.plot(c, su, "-s", color=RAIN, lw=2.2, ms=5, label="model susceptibility")
    ax2.set_ylabel("mean model susceptibility", color=RAIN)
    ax2.tick_params(axis="y", labelcolor=RAIN)
    ax.grid(alpha=0.25)
    return ax2


def main():
    df = pd.read_csv(CSV)
    y = df["landslide"].to_numpy()
    print(f"Loaded {len(df):,} cells | {int(y.sum()):,} landslides ({y.mean():.3%})")

    # ---- Point-biserial correlations with the EIL label ---- #
    corrs = {}
    for c in FEATS:
        v = pd.to_numeric(df[c], errors="coerce")
        m = np.isfinite(v)
        r, _ = pointbiserialr(y[m], v[m])
        corrs[c] = r
    print("\nCorrelation with EIL landslide label (point-biserial):")
    for c in sorted(corrs, key=corrs.get, reverse=True):
        print(f"  {PRETTY.get(c,c):22s} r = {corrs[c]:+.4f}")

    # ---- Median comparison: landslide vs non-landslide cells ---- #
    print("\nMedian in landslide vs non-landslide cells:")
    for c in ["PGA2_max", "Slope_mean", "Prc_mean"]:
        a = df.loc[df.landslide == 1, c].median()
        b = df.loc[df.landslide == 0, c].median()
        print(f"  {PRETTY.get(c,c):22s} LS={a:.4g}  non-LS={b:.4g}  ratio={a/b:.2f}")

    # ---- Figure ---- #
    fig = plt.figure(figsize=(16, 6.6))
    gs = fig.add_gridspec(1, 3, wspace=0.55, width_ratios=[1, 1, 1.05])

    ax = fig.add_subplot(gs[0, 0])
    dual_panel(ax, df, "Prc_mean", "Precipitation (mm)")
    ax.set_title("(a) Rain: model susceptibility ↑, real slides ↓\n"
                 "→ model chases rainfall where there are NO slides",
                 fontsize=11, fontweight="bold", loc="left", color=INK)

    ax = fig.add_subplot(gs[0, 1])
    dual_panel(ax, df, "Slope_mean", "Slope (degrees)")
    ax.set_title("(b) Slope: model AND reality ↑ together\n"
                 "→ the one physical signal that transfers",
                 fontsize=11, fontweight="bold", loc="left", color=INK)

    # (c) correlation bars.
    ax = fig.add_subplot(gs[0, 2])
    order = sorted(corrs, key=corrs.get)
    names = [PRETTY.get(c, c) for c in order]
    vals = [corrs[c] for c in order]
    colors = []
    for c in order:
        if c == "Prc_mean":
            colors.append(RAIN)
        elif c == "Slope_mean":
            colors.append("#2f9e6f")
        elif c == "susceptibility":
            colors.append("#8e44ad")
        else:
            colors.append("#94a3b0")
    ax.barh(names, vals, color=colors, edgecolor="white")
    ax.set_xlim(min(vals) * 1.7, max(vals) * 1.6)
    ax.axvline(0, color="#55606b", lw=1)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.3f} ", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
    ax.set_xlabel("correlation with actual EIL landslides")
    ax.set_title("(c) What actually marks the landslides?\n"
                 "slope positive · rain negative · PGA ≈ 0 (region uniformly shaken)",
                 fontsize=11, fontweight="bold", loc="left", color=INK)

    fig.suptitle("Why precipitation misleads the transferred model — "
                 "the inventory is earthquake-triggered, not rainfall-triggered",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
