#!/usr/bin/env python
"""Presentation figures for the South Mindanao zero-shot transfer, validated
against the v3 landslide inventory.

Reads `outputs/sm_v3_prod_analysis.csv` (produced by
`scripts/predict_south_mindanao_v3_labeled.py`) and emits four figures:

  1. figures/sm_v3_transferability.png   domain-shift audit (SM vs training)
  2. figures/sm_v3_physics_consistency.png physics fields + label coincidence
  3. figures/sm_v3_susceptibility_validation.png class stats + frequency ratio
  4. figures/sm_v3_drivers.png            permutation importance (+ SHAP)
"""

import os
import sys
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

CSV = PROJECT_ROOT / "outputs" / "sm_v3_prod_analysis.csv"
AUDIT = PROJECT_ROOT / "outputs" / "sm_v3_domainshift.json"
MANIFEST = PROJECT_ROOT / "feature_manifests" / "v1_cotabato_transforms_production.json"
FIG = PROJECT_ROOT / "figures"

INK = "#1f2933"
ACCENT = "#2b6cb0"
GOOD = "#2f9e6f"
BAD = "#d7191c"

CLASS_BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
CLASS_COLORS = ["#2c7bb6", "#abd9e9", "#ffd97d", "#fdae61", "#d7191c"]

# Pretty labels for model-input features.
PRETTY = {
    "Slope_mean": "Slope", "BUK_mean": "Bulk unit wt.", "Prc_mean": "Precip.",
    "ContributingFactor_mean": "Catchment area", "SoilThc_mean": "Soil thickness",
    "Elev_mean": "Elevation", "PGA2_max": "PGA", "soil_texture_idx": "Soil texture",
    "type": "Soil type",
}


def load():
    df = pd.read_csv(CSV)
    audit = json.load(open(AUDIT))
    return df, audit


# --------------------------------------------------------------------------- #
# Figure 1 — Transferability / domain shift
# --------------------------------------------------------------------------- #
def fig_transferability(df, audit):
    feats = [c for c in audit if audit[c].get("ratio") is not None]
    feats = sorted(feats, key=lambda c: audit[c]["ratio"])
    ratios = [audit[c]["ratio"] for c in feats]
    names = [PRETTY.get(c, c) for c in feats]

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.55, wspace=0.35)

    # (a) ratio bars with in-distribution band.
    ax = fig.add_subplot(gs[0, :2])
    colors = [GOOD if 0.5 <= r <= 2.0 else BAD for r in ratios]
    ax.axvspan(0.5, 2.0, color=GOOD, alpha=0.10, zorder=0)
    ax.axvline(1.0, color="#7b8794", lw=1, ls="--")
    ax.barh(names, ratios, color=colors, edgecolor="white")
    for i, r in enumerate(ratios):
        ax.text(r * 1.08, i, f"{r:.2f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.2, max(5, max(ratios) * 1.3))
    ax.set_xlabel("South Mindanao median ÷ Cotabato training median  (log scale)")
    ax.set_title("(a) Feature distribution shift vs training",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.text(0.5, -0.28, "green band = within 0.5–2× of training (in-distribution)",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#55606b")

    # (b) summary text.
    ax = fig.add_subplot(gs[0, 2:])
    ax.axis("off")
    n_in = sum(1 for r in ratios if 0.5 <= r <= 2.0)
    off = [(PRETTY.get(c, c), audit[c]["ratio"]) for c in feats
           if not (0.5 <= audit[c]["ratio"] <= 2.0)]
    elev = audit.get("Elev_mean", {})
    lines = [
        ("Transferability summary", "header"),
        (f"{n_in} / {len(ratios)} features within the training envelope (0.5–2×)", "ok"),
    ]
    for nm, r in off:
        lines.append((f"{nm}: {r:.2f}× training median — off-distribution", "bad"))
    if "pct_above_clip" in elev:
        lines.append((f"Elevation: {elev['pct_above_clip']:.1f}% of cells above the "
                      f"training clip ceiling", "note"))
    lines.append(("Off-distribution features flag where the transfer is\n"
                  "extrapolating and predictions carry more uncertainty.", "note"))
    y = 0.95
    for text, kind in lines:
        if kind == "header":
            ax.text(0, y, text, fontsize=12.5, fontweight="bold", color=INK); y -= 0.16
        elif kind == "ok":
            ax.text(0, y, "✓ " + text, fontsize=10.5, color=GOOD); y -= 0.14
        elif kind == "bad":
            ax.text(0, y, "▲ " + text, fontsize=10.5, color=BAD); y -= 0.14
        else:
            ax.text(0, y, text, fontsize=9.5, color="#55606b", style="italic"); y -= 0.20

    # (c) per-feature SM histograms with training-median line.
    show = feats[-4:] if len(feats) >= 4 else feats  # most-shifted first
    show = feats[:4]
    for j, c in enumerate(feats[:4]):
        ax = fig.add_subplot(gs[1, j])
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        lo, hi = np.percentile(vals, [1, 99])
        ax.hist(vals.clip(lo, hi), bins=40, color=ACCENT, alpha=0.75)
        tm = audit[c]["train_median"]
        if tm is not None:
            ax.axvline(tm, color=BAD, lw=1.8, label="train median")
        ax.axvline(vals.median(), color=INK, lw=1.4, ls="--", label="SM median")
        ax.set_title(PRETTY.get(c, c), fontsize=9.5)
        ax.set_yticks([])
        if j == 0:
            ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Transferability — is South Mindanao in the model's training domain?",
                 fontsize=15, fontweight="bold", y=0.99)
    out = FIG / "sm_v3_transferability.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# --------------------------------------------------------------------------- #
# Figure 2 — Physics consistency
# --------------------------------------------------------------------------- #
def fig_physics(df):
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30)
    y = df["landslide"].to_numpy()

    # (a) cohesion by soil type.
    ax = fig.add_subplot(gs[0, 0])
    order = df["type"].value_counts().index.tolist()
    data = [df.loc[df["type"] == t, "cohesion"].dropna() for t in order]
    ax.boxplot(data, labels=order, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#cfe3f5"), medianprops=dict(color=ACCENT))
    ax.set_title("(a) Cohesion by soil type", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("cohesion (model units)")
    ax.tick_params(axis="x", rotation=20, labelsize=8)

    # (b) internal friction by soil type.
    ax = fig.add_subplot(gs[0, 1])
    data = [df.loc[df["type"] == t, "internal_friction"].dropna() for t in order]
    ax.boxplot(data, labels=order, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#d7ecd9"), medianprops=dict(color=GOOD))
    ax.set_title("(b) Internal friction by soil type", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("φ′ (model units)")
    ax.tick_params(axis="x", rotation=20, labelsize=8)

    # (c) FoS histogram with FoS=1 marker.
    ax = fig.add_subplot(gs[0, 2])
    fos = pd.to_numeric(df["fos"], errors="coerce").dropna()
    lo, hi = np.percentile(fos, [1, 99])
    ax.hist(fos.clip(lo, hi), bins=50, color="#f0a24b", alpha=0.85)
    ax.axvline(1.0, color=BAD, lw=2, label="FoS = 1 (limit)")
    pct_unstable = float((fos < 1.0).mean() * 100)
    ax.set_title(f"(c) Factor of Safety\n{pct_unstable:.1f}% of cells below FoS = 1",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("FoS"); ax.set_yticks([]); ax.legend(fontsize=8)

    # (d)-(f) monotonic behavior: mean susceptibility vs slope / PGA / FoS.
    def binned(ax, xcol, xlabel, expect):
        x = pd.to_numeric(df[xcol], errors="coerce")
        s = pd.to_numeric(df["susceptibility"], errors="coerce")
        m = np.isfinite(x) & np.isfinite(s)
        x, s = x[m], s[m]
        q = np.quantile(x, np.linspace(0, 1, 13))
        q = np.unique(q)
        idx = np.digitize(x, q[1:-1])
        mids, means = [], []
        for b in range(len(q) - 1):
            sel = idx == b
            if sel.sum() > 50:
                mids.append(x[sel].median()); means.append(s[sel].mean())
        ax.plot(mids, means, "-o", color=ACCENT, ms=4)
        ax.set_xlabel(xlabel); ax.set_ylabel("mean susceptibility")
        ax.set_title(f"{xlabel}  (expect {expect})", fontsize=10, fontweight="bold", loc="left")
        ax.grid(alpha=0.25)

    binned(fig.add_subplot(gs[1, 0]), "Slope_mean", "(d) Slope", "training: ↑")
    binned(fig.add_subplot(gs[1, 1]), "PGA2_max", "(e) PGA", "training: ↑")
    binned(fig.add_subplot(gs[1, 2]), "fos", "(f) FoS", "training: ↓")

    fig.suptitle("Physics diagnostics — how the embedded Newmark physics responds under transfer",
                 fontsize=15, fontweight="bold", y=0.98)
    out = FIG / "sm_v3_physics_consistency.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# --------------------------------------------------------------------------- #
# Figure 3 — Susceptibility characterization + frequency-ratio validation
# --------------------------------------------------------------------------- #
def fig_validation(df):
    s = pd.to_numeric(df["susceptibility"], errors="coerce").to_numpy()
    y = df["landslide"].to_numpy()
    m = np.isfinite(s)
    s, y = s[m], y[m]

    cls = np.digitize(s, CLASS_BOUNDS[1:-1])  # 0..4
    counts = np.array([(cls == k).sum() for k in range(5)])
    ls_counts = np.array([y[cls == k].sum() for k in range(5)])
    area_frac = counts / counts.sum()
    ls_frac = ls_counts / max(ls_counts.sum(), 1)
    overall = y.mean()
    dens = np.divide(ls_counts, counts, out=np.zeros(5), where=counts > 0)
    fr = dens / overall  # frequency ratio per class

    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    # (a) area share per class.
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(CLASS_LABELS, area_frac * 100, color=CLASS_COLORS, edgecolor="white")
    for i, v in enumerate(area_frac * 100):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of slope-unit area")
    ax.set_title("(a) Susceptibility class distribution", fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(axis="x", rotation=15, labelsize=9)

    # (b) frequency ratio per class (validation).
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(CLASS_LABELS, fr, color=CLASS_COLORS, edgecolor="white")
    ax.axhline(1.0, color="#55606b", ls="--", lw=1, label="expected if no skill")
    for i, v in enumerate(fr):
        ax.text(i, v, f"{v:.1f}×", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("landslide frequency ratio")
    ax.set_title("(b) Frequency ratio — validated on inventory\n"
                 "monotonic increase = correct ranking",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=15, labelsize=9)

    # (c) success-rate curve.
    ax = fig.add_subplot(gs[0, 2])
    order = np.argsort(-s)
    cum_area = np.arange(1, len(s) + 1) / len(s)
    cum_ls = np.cumsum(y[order]) / max(y.sum(), 1)
    ax.plot(cum_area * 100, cum_ls * 100, color=ACCENT, lw=2)
    ax.plot([0, 100], [0, 100], "--", color="#7b8794", lw=1, label="random")
    ax.set_xlabel("% of area (ranked most→least susceptible)")
    ax.set_ylabel("% of actual landslides captured")
    ax.set_title(f"(c) Success-rate curve\nROC AUC = {auc:.3f} · PR AUC = {ap:.3f}",
                 fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right")
    # annotate capture at top 20% area
    k20 = int(0.20 * len(s))
    cap20 = cum_ls[k20] * 100
    ax.annotate(f"top 20% area\ncaptures {cap20:.0f}% of slides",
                xy=(20, cap20), xytext=(35, max(cap20 - 30, 15)),
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color="#55606b"))

    fig.suptitle("Susceptibility validation against the South Mindanao inventory (zero-shot)",
                 fontsize=15, fontweight="bold", y=1.0)
    out = FIG / "sm_v3_susceptibility_validation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")
    return {"auc": auc, "ap": ap, "fr": fr.tolist(), "capture_top20": float(cap20)}


# --------------------------------------------------------------------------- #
# Figure 4 — Driver analysis (permutation importance + SHAP)
# --------------------------------------------------------------------------- #
def fig_drivers(df):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from py_files.GallenModel_v1 import NewmarkActivation
    from py_files.data import apply_log_transform, apply_clip_thresholds, dataframe_to_dataset
    from py_files.helpers import set_seed
    import predict_south_mindanao_v2_8 as base

    set_seed(42)
    tm = json.load(open(MANIFEST))
    model = load_model(base.MODEL_DIR / "production-model-v3.keras",
                       custom_objects={"NewmarkActivation": NewmarkActivation})
    input_cols = [t.name.split(":")[0] for t in model.inputs]
    numeric_inputs = [c for c in input_cols if c != "type"]

    # Balanced subsample: all positives + capped negatives, for a fast, stable AUC.
    pos = df[df["landslide"] == 1]
    neg = df[df["landslide"] == 0].sample(n=min(120_000, (df["landslide"] == 0).sum()),
                                          random_state=42)
    sub = pd.concat([pos, neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    def prep(frame):
        fx = frame[input_cols + ["landslide"]].copy()
        fx = apply_log_transform(fx, tm["log_transformed_cols"])
        fx = apply_clip_thresholds(fx, tm["clip_thresholds"])
        return fx

    def predict(frame):
        ds = dataframe_to_dataset(prep(frame), shuffle=False, batch_size=512)
        out = model.predict(ds, verbose=0)
        return out["final_head"].flatten() if isinstance(out, dict) else out.flatten()

    y = sub["landslide"].to_numpy()
    base_auc = roc_auc_score(y, predict(sub))

    rng = np.random.default_rng(42)
    imp = {}
    for c in input_cols:
        perm = sub.copy()
        perm[c] = rng.permutation(perm[c].to_numpy())
        imp[c] = base_auc - roc_auc_score(y, predict(perm))
    order = sorted(imp, key=imp.get)
    names = [PRETTY.get(c, c) for c in order]
    vals = [imp[c] for c in order]

    # ---- SHAP (optional) on continuous features ---- #
    shap_ok = False
    try:
        import shap
        cont = [c for c in numeric_inputs if c != "soil_texture_idx"]
        modal_type = sub["type"].mode().iloc[0]
        modal_idx = float(sub["soil_texture_idx"].mode().iloc[0])
        fixed = {c: sub[c].median() for c in numeric_inputs}

        def f(X):
            frame = pd.DataFrame(X, columns=cont)
            for c in numeric_inputs:
                if c not in cont:
                    frame[c] = modal_idx if c == "soil_texture_idx" else fixed[c]
            frame["type"] = modal_type
            frame["landslide"] = 0
            return predict(frame)

        bg = shap.kmeans(sub[cont].sample(300, random_state=1).to_numpy(), 12)
        expl = shap.KernelExplainer(f, bg)
        Xs = sub[cont].sample(200, random_state=2).to_numpy()
        sv = expl.shap_values(Xs, nsamples=100, silent=True)
        shap_ok = True
    except Exception as exc:
        print(f"  [warn] SHAP skipped: {exc}")

    ncol = 2 if shap_ok else 1
    fig, axes = plt.subplots(1, ncol, figsize=(8 * ncol, 6))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    colors = [GOOD if v > 0 else BAD for v in vals]
    ax.barh(names, np.array(vals), color=colors, edgecolor="white")
    ax.axvline(0, color="#55606b", lw=1)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.3f} ", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.set_xlabel("AUC drop when feature is permuted")
    ax.set_title(f"(a) Permutation importance   (baseline AUC = {base_auc:.3f})",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.text(0.5, -0.16,
            "positive = feature helps ranking   ·   negative = feature MISLEADS "
            "the model here (scrambling it improves AUC)",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#55606b")

    if shap_ok:
        import shap
        plt.sca(axes[1])
        shap.summary_plot(sv, features=Xs, feature_names=[PRETTY.get(c, c) for c in cont],
                          show=False, plot_size=None)
        axes[1].set_title("(b) SHAP — direction & magnitude of drivers",
                          fontsize=11.5, fontweight="bold", loc="left")

    fig.suptitle("What drives the South Mindanao susceptibility predictions?",
                 fontsize=15, fontweight="bold", y=1.02)
    out = FIG / "sm_v3_drivers.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    df, audit = load()
    print(f"Loaded {len(df):,} rows | positives {int(df['landslide'].sum()):,}")
    fig_transferability(df, audit)
    fig_physics(df)
    stats = fig_validation(df)
    print(f"Validation: AUC={stats['auc']:.3f} AP={stats['ap']:.3f} "
          f"top20%-capture={stats['capture_top20']:.0f}%")
    fig_drivers(df)


if __name__ == "__main__":
    main()
