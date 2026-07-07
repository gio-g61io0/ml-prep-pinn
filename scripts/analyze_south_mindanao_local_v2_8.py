#!/usr/bin/env python
"""Compare the two locally-trained South Mindanao PINNs (rainfall ON vs pure-EIL)
against the earthquake-induced landslide inventory, using honest out-of-fold
predictions. Also overlays the failed zero-shot transfer (AUC 0.45) as a baseline.

Reads outputs/sm_local_{rain,eil}_scored.csv (from
scripts/train_south_mindanao_local_v2_8.py) and emits:
  figures/sm_local_{rain,eil}_validation.png  per-model ROC/PR + frequency ratio + success
  figures/sm_local_{rain,eil}_drivers.png     per-model permutation importance (+SHAP)
  figures/sm_local_compare.png                head-to-head ROC / freq ratio / precip driver
"""

import os
import sys
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

INK, ACCENT, GOOD, BAD = "#1f2933", "#2b6cb0", "#2f9e6f", "#d7191c"
RAIN_C, EIL_C, XFER_C = "#1c9bd6", "#2f9e6f", "#9aa5b1"

CLASS_BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
CLASS_COLORS = ["#2c7bb6", "#abd9e9", "#ffd97d", "#fdae61", "#d7191c"]
PRETTY = {
    "Slope_mean": "Slope", "BUK_mean": "Bulk unit wt.", "Prc_mean": "Precipitation",
    "ContributingFactor_mean": "Catchment area", "SoilThc_mean": "Soil thickness",
    "Elev_mean": "Elevation", "PGA2_max": "PGA", "soil_texture_idx": "Soil texture",
    "type": "Soil type",
}
SCORE = "susceptibility_oof"


def scored_path(tag):
    return PROJECT_ROOT / "outputs" / f"sm_local_{tag}_scored.csv"


def class_freq_ratio(s, y):
    cls = np.digitize(s, CLASS_BOUNDS[1:-1])
    counts = np.array([(cls == k).sum() for k in range(5)])
    ls = np.array([y[cls == k].sum() for k in range(5)])
    dens = np.divide(ls, counts, out=np.zeros(5), where=counts > 0)
    fr = dens / max(y.mean(), 1e-9)
    return counts / counts.sum(), fr


def success_curve(s, y):
    order = np.argsort(-s)
    cum_area = np.arange(1, len(s) + 1) / len(s)
    cum_ls = np.cumsum(y[order]) / max(y.sum(), 1)
    return cum_area * 100, cum_ls * 100


def fig_validation(tag, df):
    s = pd.to_numeric(df[SCORE], errors="coerce").to_numpy()
    y = df["landslide"].to_numpy()
    m = np.isfinite(s)
    s, y = s[m], y[m]
    auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
    area_frac, fr = class_freq_ratio(s, y)
    ca, cl = success_curve(s, y)

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(CLASS_LABELS, area_frac * 100, color=CLASS_COLORS, edgecolor="white")
    ax.set_ylabel("% of slope units"); ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.set_title("(a) Susceptibility class distribution", fontsize=11, fontweight="bold", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(CLASS_LABELS, fr, color=CLASS_COLORS, edgecolor="white")
    ax.axhline(1.0, color="#55606b", ls="--", lw=1)
    for i, v in enumerate(fr):
        ax.text(i, v, f"{v:.1f}×", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("landslide frequency ratio"); ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.set_title("(b) Frequency ratio (held-out)\nmonotonic ↑ = correct ranking",
                 fontsize=11, fontweight="bold", loc="left")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ca, cl, color=ACCENT, lw=2)
    ax.plot([0, 100], [0, 100], "--", color="#7b8794", lw=1, label="random")
    k20 = int(0.2 * len(s))
    ax.set_title(f"(c) Success-rate curve\nROC AUC = {auc:.3f} · PR AUC = {ap:.3f}",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("% area (ranked)"); ax.set_ylabel("% landslides captured")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right")

    label = "rainfall physics ON" if tag == "rain" else "pure EIL (no rainfall)"
    fig.suptitle(f"Local SM model — {label} — validated on the EIL inventory",
                 fontsize=15, fontweight="bold", y=1.02)
    out = PROJECT_ROOT / "figures" / f"sm_local_{tag}_validation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")
    return {"auc": auc, "ap": ap, "fr": fr.tolist()}


def _load_model_and_predict_fn(tag):
    """Load the best fold model + its manifest; return (predict_fn, input_cols)."""
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from py_files.GallenModel_v1 import NewmarkActivation
    from py_files.data import apply_log_transform, apply_clip_thresholds, dataframe_to_dataset

    summary = json.load(open(PROJECT_ROOT / "outputs" / "sm_local_summary.json"))
    rec = next(r for r in summary if r["tag"] == tag)
    best_fold = int(np.argmax(rec["fold_aucs"])) + 1
    model_dir = PROJECT_ROOT / "trained_models" / f"south_mindanao_{tag}_v2_8"
    tdir = PROJECT_ROOT / "feature_manifests" / f"south_mindanao_{tag}_v2_8"
    tm = json.load(open(tdir / f"v1_cotabato_transforms_fold{best_fold}.json"))
    model = load_model(model_dir / f"fold-{best_fold}-model-v3.keras",
                       custom_objects={"NewmarkActivation": NewmarkActivation})
    input_cols = [t.name.split(":")[0] for t in model.inputs]

    def predict(frame):
        fx = frame[input_cols + ["landslide"]].copy()
        fx = apply_log_transform(fx, tm["log_transformed_cols"])
        fx = apply_clip_thresholds(fx, tm["clip_thresholds"])
        ds = dataframe_to_dataset(fx, shuffle=False, batch_size=512)
        out = model.predict(ds, verbose=0)
        return out["final_head"].flatten() if isinstance(out, dict) else out.flatten()

    return predict, input_cols


def permutation_importance(tag, df):
    """AUC drop when each input is permuted (best-fold model, apparent). The point
    for pure-EIL is a structural check: precipitation importance must be ~0."""
    predict, input_cols = _load_model_and_predict_fn(tag)
    y = df["landslide"].to_numpy()
    base_auc = roc_auc_score(y, predict(df))
    rng = np.random.default_rng(42)
    imp = {}
    for c in input_cols:
        perm = df.copy()
        perm[c] = rng.permutation(perm[c].to_numpy())
        imp[c] = base_auc - roc_auc_score(y, predict(perm))
    return base_auc, imp


def fig_drivers(tag, df, base_auc, imp):
    order = sorted(imp, key=imp.get)
    names = [PRETTY.get(c, c) for c in order]
    vals = [imp[c] for c in order]
    colors = [RAIN_C if c == "Prc_mean" else (GOOD if imp[c] > 0 else BAD) for c in order]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(names, vals, color=colors, edgecolor="white")
    ax.axvline(0, color="#55606b", lw=1)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.3f} ", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.set_xlabel("AUC drop when feature is permuted")
    label = "rainfall ON" if tag == "rain" else "pure EIL"
    ax.set_title(f"Permutation importance — {label}   (baseline AUC = {base_auc:.3f})",
                 fontsize=12, fontweight="bold", loc="left")
    out = PROJECT_ROOT / "figures" / f"sm_local_{tag}_drivers.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def fig_compare(rain, eil, imp_rain, imp_eil):
    """Head-to-head: ROC overlay (+ transfer baseline), frequency ratio, precip driver."""
    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.30)

    # (a) ROC overlay
    ax = fig.add_subplot(gs[0, 0])
    for df, c, lab in [(rain, RAIN_C, "rainfall ON"), (eil, EIL_C, "pure EIL")]:
        s = pd.to_numeric(df[SCORE], errors="coerce").to_numpy()
        y = df["landslide"].to_numpy()
        mm = np.isfinite(s)
        fpr, tpr, _ = roc_curve(y[mm], s[mm])
        auc = roc_auc_score(y[mm], s[mm])
        ax.plot(fpr, tpr, color=c, lw=2.2, label=f"{lab} (AUC {auc:.3f})")
    # transfer baseline
    xfer = PROJECT_ROOT / "outputs" / "sm_v3_prod_analysis.csv"
    if xfer.exists():
        t = pd.read_csv(xfer)
        s = pd.to_numeric(t["susceptibility"], errors="coerce").to_numpy()
        y = t["landslide"].to_numpy()
        mm = np.isfinite(s)
        fpr, tpr, _ = roc_curve(y[mm], s[mm])
        auc = roc_auc_score(y[mm], s[mm])
        ax.plot(fpr, tpr, color=XFER_C, lw=1.8, ls=":", label=f"zero-shot transfer (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#c0c7ce", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("(a) ROC — region-trained vs transfer", fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.25)

    # (b) frequency ratio side by side
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(5)
    for i, (df, c, lab) in enumerate([(rain, RAIN_C, "rainfall ON"), (eil, EIL_C, "pure EIL")]):
        s = pd.to_numeric(df[SCORE], errors="coerce").to_numpy()
        y = df["landslide"].to_numpy()
        mm = np.isfinite(s)
        _, fr = class_freq_ratio(s[mm], y[mm])
        ax.bar(x + (i - 0.5) * 0.4, fr, width=0.4, color=c, label=lab, edgecolor="white")
    ax.axhline(1.0, color="#55606b", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, fontsize=8.5)
    ax.set_ylabel("frequency ratio")
    ax.set_title("(b) Frequency ratio by class", fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(fontsize=9)

    # (c) precipitation permutation importance
    ax = fig.add_subplot(gs[0, 2])
    pv = [imp_rain.get("Prc_mean", 0.0), imp_eil.get("Prc_mean", 0.0)]
    bars = ax.bar(["rainfall ON", "pure EIL"], pv, color=[RAIN_C, EIL_C], edgecolor="white")
    ax.axhline(0, color="#55606b", lw=1)
    for b, v in zip(bars, pv):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10)
    ax.set_ylabel("precip permutation importance (AUC drop)")
    ax.set_title("(c) Does precipitation still matter?\npure-EIL should be ≈ 0",
                 fontsize=11.5, fontweight="bold", loc="left")

    fig.suptitle("Region-trained South Mindanao PINN: rainfall physics ON vs pure-EIL",
                 fontsize=15, fontweight="bold", y=1.02)
    out = PROJECT_ROOT / "figures" / "sm_local_compare.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    results = {}
    imps = {}
    for tag in ("rain", "eil"):
        df = pd.read_csv(scored_path(tag))
        print(f"\n[{tag}] {len(df):,} rows | positives {int(df['landslide'].sum()):,}")
        results[tag] = fig_validation(tag, df)
        base_auc, imp = permutation_importance(tag, df)
        imps[tag] = imp
        fig_drivers(tag, df, base_auc, imp)
        results[tag]["base_auc"] = base_auc
        results[tag]["precip_importance"] = imp.get("Prc_mean")
        print(f"[{tag}] AUC={results[tag]['auc']:.3f} AP={results[tag]['ap']:.3f} "
              f"precip_perm_imp={imp.get('Prc_mean'):+.4f}")

    rain = pd.read_csv(scored_path("rain"))
    eil = pd.read_csv(scored_path("eil"))
    fig_compare(rain, eil, imps["rain"], imps["eil"])

    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD (held-out OOF on undersampled frame)")
    for tag in ("rain", "eil"):
        r = results[tag]
        print(f"  {tag:5s}  AUC={r['auc']:.3f}  PR-AUC={r['ap']:.3f}  "
              f"precip-importance={r['precip_importance']:+.4f}")


if __name__ == "__main__":
    main()
