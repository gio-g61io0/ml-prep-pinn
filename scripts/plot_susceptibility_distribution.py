#!/usr/bin/env python
"""Plot the distribution of predicted landslide susceptibility.

Reads only the `susceptibility` column from a prediction CSV (so it stays light
on the multi-hundred-MB South Mindanao output) and renders:
  1. Histogram on a linear y-axis (shows how dominant the low-susceptibility
     mass is).
  2. Histogram on a log y-axis (reveals the shape of the sparse high-
     susceptibility tail).
  3. A susceptibility-class bar chart using the standard 5-class scheme.

Usage:
    python scripts/plot_susceptibility_distribution.py [csv_path] [out_png]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "outputs" / "south_mindanao_susceptibility_v2_8.csv"
DEFAULT_PNG = PROJECT_ROOT / "outputs" / "south_mindanao_susceptibility_distribution_v2_8.png"

# Standard 5-class susceptibility scheme (equal-interval on [0, 1]).
CLASS_BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
CLASS_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PNG

    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["susceptibility"])
    s = pd.to_numeric(df["susceptibility"], errors="coerce")
    s = s[np.isfinite(s)].to_numpy()
    n = len(s)
    if n == 0:
        raise ValueError("No finite susceptibility values found.")

    # ---- Summary stats --------------------------------------------------- #
    qs = np.percentile(s, [1, 5, 25, 50, 75, 95, 99])
    print(f"Source: {csv_path.name}")
    print(f"N finite slope units: {n:,}")
    print(f"min={s.min():.4f}  mean={s.mean():.4f}  median={np.median(s):.4f}  max={s.max():.4f}")
    print("percentiles  p1={:.4f} p5={:.4f} p25={:.4f} p50={:.4f} p75={:.4f} p95={:.4f} p99={:.4f}"
          .format(*qs))

    counts, _ = np.histogram(s, bins=CLASS_BOUNDS)
    print("\nSusceptibility classes:")
    for label, c in zip(CLASS_LABELS, counts):
        print(f"  {label:10s} {c:>10,}  ({100 * c / n:5.2f}%)")

    # ---- Figure ---------------------------------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bins = np.linspace(0, 1, 51)

    axes[0].hist(s, bins=bins, color="#4575b4", edgecolor="white", linewidth=0.3)
    axes[0].set_title("Susceptibility distribution (linear)")
    axes[0].set_xlabel("Susceptibility")
    axes[0].set_ylabel("Slope-unit count")
    axes[0].axvline(np.median(s), color="k", ls="--", lw=1, label=f"median={np.median(s):.3f}")
    axes[0].legend()

    axes[1].hist(s, bins=bins, color="#d73027", edgecolor="white", linewidth=0.3)
    axes[1].set_yscale("log")
    axes[1].set_title("Susceptibility distribution (log count)")
    axes[1].set_xlabel("Susceptibility")
    axes[1].set_ylabel("Slope-unit count (log)")

    axes[2].bar(CLASS_LABELS, counts, color=CLASS_COLORS, edgecolor="black", linewidth=0.5)
    axes[2].set_title("Susceptibility classes")
    axes[2].set_ylabel("Slope-unit count")
    axes[2].tick_params(axis="x", rotation=30)
    for i, c in enumerate(counts):
        axes[2].text(i, c, f"{100 * c / n:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"South Mindanao — PINN v2-8 production model  (N={n:,})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved distribution -> {out_png}")


if __name__ == "__main__":
    main()
