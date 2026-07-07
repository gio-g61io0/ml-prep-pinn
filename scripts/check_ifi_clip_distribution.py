"""Check the internal friction angle distribution after clipping.

Loads each fold from `v4-1-1dce-repro-ifi-clip-0-40/`, runs inference on both
training (SU_15_Training1) and validation (SU_15_Validation1) sets, and pulls
two intermediate signals:

  * pre-clip IFI  — output of `internal_friction` (sigmoid in [0, 1] rad)
  * post-clip IFI — output of `ifi_clip_0_40` (IFIClipLayer)

Reports per-fold stats, the fraction of pre-clip values that were saturated by
the clip, and saves histograms + a CSV summary to figures/ifi_clip_check/.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Triggers registration of all custom layers (CohesionLayer, InternalFrictionLayer,
# IFIClipLayer, DisplacementLayer, NewmarkActivation, DiceCrossEntropyLoss).
from py_files import GallenModel_v1, Landslidev2_Old  # noqa: F401
from py_files.data import dataframe_to_dataset


CHECKPOINT_DIR = Path(
    "/Users/giogonzales/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/trainedCotabatoPhase7/historical/v4-1-1dce-repro-ifi-clip-0-40"
)
TRAIN_PATH = Path("/Users/giogonzales/Documents/ml-prep/mlprep/datasets/SU_15_Training1.gpkg")
VAL_PATH = Path("/Users/giogonzales/Documents/ml-prep/mlprep/datasets/SU_15_Validation1.gpkg")
OUT_DIR = Path("/Users/giogonzales/Documents/ml-prep/mlprep/figures/ifi_clip_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_DEG = (0.0, 40.0)
PRE_CLIP_LAYER = "internal_friction"
POST_CLIP_LAYER = "ifi_clip_0_40"


def load_split(path: Path) -> gpd.GeoDataFrame:
    df = gpd.read_file(str(path))
    drop_cols = [
        "landslide_probability",
        "landslide_preds",
        "confusion",
        "sus_pinn_landslide",
        "sus_pinn_ground truth",
        "ds",
        "cohesion",
        "internal_friction",
        "descriptio",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df = df[df["Slope_mean"] >= 10]
    df.dropna(subset=list(df.columns), inplace=True)
    return df


def feature_columns(df) -> list[str]:
    exclude = {
        "DN",
        "BD_mean",
        "geometry",
        "PGA2_max",
        "Soil Type",
        "description",
        "descriptio",
        "predicted_susceptibility",
    }
    return [c for c in df.columns if c not in exclude]


def heads_for(model: tf.keras.Model) -> tf.keras.Model:
    return tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(PRE_CLIP_LAYER).output,
            model.get_layer(POST_CLIP_LAYER).output,
        ],
    )


def fold_stats(name: str, fold: int, pre_rad: np.ndarray, post_rad: np.ndarray) -> dict:
    pre_deg = np.degrees(pre_rad)
    post_deg = np.degrees(post_rad)

    inside = (pre_deg >= CLIP_DEG[0]) & (pre_deg <= CLIP_DEG[1])
    saturated_high = pre_deg > CLIP_DEG[1]
    saturated_low = pre_deg < CLIP_DEG[0]
    within_bounds = ((post_deg >= CLIP_DEG[0]) & (post_deg <= CLIP_DEG[1])).all()

    return {
        "split": name,
        "fold": fold,
        "n": int(pre_deg.size),
        "pre_min": float(pre_deg.min()),
        "pre_max": float(pre_deg.max()),
        "pre_mean": float(pre_deg.mean()),
        "pre_median": float(np.median(pre_deg)),
        "post_min": float(post_deg.min()),
        "post_max": float(post_deg.max()),
        "post_mean": float(post_deg.mean()),
        "post_median": float(np.median(post_deg)),
        "frac_clipped_high": float(saturated_high.mean()),
        "frac_clipped_low": float(saturated_low.mean()),
        "frac_unchanged": float(inside.mean()),
        "post_within_bounds": bool(within_bounds),
    }


def plot_split_histograms(split: str, pre_list: list[np.ndarray], post_list: list[np.ndarray]) -> None:
    pre_all_deg = np.degrees(np.concatenate(pre_list))
    post_all_deg = np.degrees(np.concatenate(post_list))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].hist(pre_all_deg, bins=60, alpha=0.85, color="steelblue", edgecolor="black", linewidth=0.2)
    axes[0].axvline(CLIP_DEG[0], color="k", linestyle="--", alpha=0.6)
    axes[0].axvline(CLIP_DEG[1], color="k", linestyle="--", alpha=0.6, label="Clip bounds")
    axes[0].set_title(f"{split} — pre-clip IFI (from `internal_friction`)")
    axes[0].set_xlabel("Internal friction angle (°)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(post_all_deg, bins=60, alpha=0.85, color="darkorange", edgecolor="black", linewidth=0.2)
    axes[1].axvline(CLIP_DEG[0], color="k", linestyle="--", alpha=0.6)
    axes[1].axvline(CLIP_DEG[1], color="k", linestyle="--", alpha=0.6)
    axes[1].set_title(f"{split} — post-clip IFI (from `ifi_clip_0_40`)")
    axes[1].set_xlabel("Internal friction angle (°)")

    fig.suptitle(f"IFI distribution across all 10 folds — {split} ({pre_all_deg.size:,} samples)")
    plt.tight_layout()
    out = OUT_DIR / f"ifi_distribution_{split.lower()}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def plot_per_fold_post(split: str, post_list: list[np.ndarray]) -> None:
    """Faceted post-clip histograms, one per fold, to spot fold-level differences."""
    n_folds = len(post_list)
    cols = 5
    rows = math.ceil(n_folds / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    for i, post in enumerate(post_list):
        ax = axes[i]
        ax.hist(np.degrees(post), bins=40, color="darkorange", alpha=0.85, edgecolor="black", linewidth=0.2)
        ax.axvline(CLIP_DEG[0], color="k", linestyle="--", alpha=0.5)
        ax.axvline(CLIP_DEG[1], color="k", linestyle="--", alpha=0.5)
        ax.set_title(f"Fold {i + 1}")
    for j in range(len(post_list), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Post-clip IFI per fold — {split}")
    fig.supxlabel("Internal friction angle (°)")
    plt.tight_layout()
    out = OUT_DIR / f"ifi_post_per_fold_{split.lower()}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def run_split(name: str, df, feature_cols: list[str]) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    ds = dataframe_to_dataset(df[feature_cols], shuffle=False)
    folds = sorted(int(p.stem.split("-")[1]) for p in CHECKPOINT_DIR.glob("fold-*-model-0.keras"))
    rows: list[dict] = []
    pre_per_fold: list[np.ndarray] = []
    post_per_fold: list[np.ndarray] = []

    print(f"\n=== {name} ({len(df):,} samples) ===")
    skipped: list[int] = []
    for fold in folds:
        ckpt = CHECKPOINT_DIR / f"fold-{fold}-model-0.keras"
        try:
            model = load_model(str(ckpt))
        except (ValueError, TypeError) as exc:
            # Pre-fix checkpoints saved a `Lambda(tf.clip_by_value)` whose closure
            # can't survive serialization. Skip with a warning so the report still
            # covers the folds that were saved after IFIClipLayer landed.
            print(f"  fold {fold:>2}: SKIP (stale Lambda checkpoint — retrain to include)  [{exc.__class__.__name__}]")
            skipped.append(fold)
            tf.keras.backend.clear_session()
            continue
        head = heads_for(model)
        pre, post = head.predict(ds, verbose=0)
        pre = np.asarray(pre).flatten()
        post = np.asarray(post).flatten()

        stats = fold_stats(name, fold, pre, post)
        rows.append(stats)
        pre_per_fold.append(pre)
        post_per_fold.append(post)

        print(
            f"  fold {fold:>2}: pre [{stats['pre_min']:5.2f}°, {stats['pre_max']:5.2f}°] mean={stats['pre_mean']:5.2f}°"
            f"  | post [{stats['post_min']:5.2f}°, {stats['post_max']:5.2f}°] mean={stats['post_mean']:5.2f}°"
            f"  | clipped >40°: {stats['frac_clipped_high']:6.2%}  <0°: {stats['frac_clipped_low']:6.2%}"
            f"  | bounds OK: {stats['post_within_bounds']}"
        )

        tf.keras.backend.clear_session()

    if skipped:
        print(f"  → skipped folds: {skipped}  (retrain via scripts/train_v411dce_clip.py to populate)")
    return rows, pre_per_fold, post_per_fold


def main() -> None:
    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)
    train_cols = feature_columns(train_df)
    val_cols = feature_columns(val_df)

    rows_train, pre_train, post_train = run_split("Training", train_df, train_cols)
    rows_val, pre_val, post_val = run_split("Validation", val_df, val_cols)

    all_rows = rows_train + rows_val
    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUT_DIR / "ifi_clip_distribution_per_fold.csv", index=False)

    plot_split_histograms("Training", pre_train, post_train)
    plot_split_histograms("Validation", pre_val, post_val)
    plot_per_fold_post("Training", post_train)
    plot_per_fold_post("Validation", post_val)

    print("\n=== Aggregate over 10 folds ===")
    for split_name, rows in [("Training", rows_train), ("Validation", rows_val)]:
        d = pd.DataFrame(rows)
        print(
            f"  {split_name:>10}: mean pre={d.pre_mean.mean():5.2f}°  "
            f"mean post={d.post_mean.mean():5.2f}°  "
            f"mean clipped(high)={d.frac_clipped_high.mean():6.2%}  "
            f"mean clipped(low)={d.frac_clipped_low.mean():6.2%}  "
            f"all-bounds-ok={d.post_within_bounds.all()}"
        )
    print(f"\nWrote summary CSV + 4 plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
