"""Compare v4-1-1dce-repro (no clip) vs v4-1-1dce-repro-ifi-clip-25-45.

For each fold:
  * Predict susceptibility on SU_15_Validation1.
  * Extract intermediate cohesion and friction predictions.
  * Compute AUC / balanced accuracy.

Then aggregate and plot:
  * Per-fold AUC bar chart.
  * IFI distribution before/after clip (in degrees).
  * Cohesion distribution before/after clip — checks for optimizer compensation.
  * Susceptibility distribution shift.
  * Side-by-side susceptibility maps for a chosen fold.
  * Susceptibility delta map (clipped - original) for the same fold.

Writes plots to figures/v411dce_ifi_clip/ and a markdown summary to figures/v411dce_ifi_clip/SUMMARY.md.
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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as cx
import sklearn.metrics
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model

keras.config.enable_unsafe_deserialization()  # clipped model uses a Lambda for tf.clip_by_value

from py_files import GallenModel_v1, Landslidev2_Old  # noqa: F401 — registers custom objects
from py_files.data import dataframe_to_dataset

# Import the trained-with-clip builder so we can reconstruct the architecture and
# load weights (the saved Lambda can't be auto-deserialized because its output
# shape isn't inferable from a (None,) input).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v411dce_clip import build_model as build_clipped_model, load_training_df  # noqa: E402


ORIG_DIR = Path(
    "/Users/giogonzales/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/trainedCotabatoPhase7/historical/v4-1-1dce-repro"
)
CLIPPED_DIR = Path(
    "/Users/giogonzales/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/trainedCotabatoPhase7/historical/v4-1-1dce-repro-ifi-clip-25-45"
)
VAL_PATH = Path("/Users/giogonzales/Documents/ml-prep/mlprep/datasets/SU_15_Validation1.gpkg")
FIG_DIR = Path("/Users/giogonzales/Documents/ml-prep/mlprep/figures/v411dce_ifi_clip")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLIP_DEG = (25.0, 45.0)
MAP_FOLD = 4  # which fold to use for the susceptibility maps


def load_validation_df() -> gpd.GeoDataFrame:
    df = gpd.read_file(str(VAL_PATH))
    df.drop(
        columns=[
            "landslide_probability",
            "landslide_preds",
            "confusion",
            "sus_pinn_landslide",
            "sus_pinn_ground truth",
            "ds",
            "cohesion",
            "internal_friction",
            "descriptio",
        ],
        inplace=True,
    )
    df = df[df["Slope_mean"] >= 10]
    df.dropna(subset=list(df.columns), inplace=True)
    return df


def predict_with_heads(model, val_ds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (susceptibility, cohesion, friction_radians)."""
    sus = model.predict(val_ds, verbose=0).flatten()

    coh_layer = model.get_layer("cohesion_layer")
    ifi_layer = model.get_layer("internal_friction")
    head_model = tf.keras.Model(
        inputs=model.input, outputs=[coh_layer.output, ifi_layer.output]
    )
    coh, ifi = head_model.predict(val_ds, verbose=0)
    return sus, np.asarray(coh).flatten(), np.asarray(ifi).flatten()


def per_fold_metrics(y_true: np.ndarray, sus: np.ndarray) -> dict:
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, sus)
    auc = sklearn.metrics.auc(fpr, tpr)
    acc = sklearn.metrics.balanced_accuracy_score(y_true, sus > 0.5)
    return {"auc": auc, "bal_acc": acc, "mean_sus": float(np.mean(sus))}


def main() -> None:
    val_df = load_validation_df()
    feature_cols = [
        c
        for c in val_df.columns
        if c
        not in (
            "DN",
            "BD_mean",
            "geometry",
            "PGA2_max",
            "Soil Type",
            "description",
            "descriptio",
            "predicted_susceptibility",
        )
    ]
    val_ds = dataframe_to_dataset(val_df[feature_cols], shuffle=False)
    y_true = val_df["landslide"].to_numpy()

    # The clipped builder adapts the StringLookup vocab from whatever dataset it sees;
    # we need that vocab to match training (4 tokens) or the Sus_0 kernel shape won't
    # line up with the saved weights. Build the adapt-source from the training gpkg.
    numeric_cols = [c for c in feature_cols if c not in ("landslide", "type")]
    categorical_cols = ["type"]
    pga_col = "PGA1_max"

    train_df = load_training_df()
    train_feature_cols = [c for c in train_df.columns if c in feature_cols]
    train_ds_for_build = dataframe_to_dataset(train_df[train_feature_cols], shuffle=False)

    folds = sorted(int(p.stem.split("-")[1]) for p in ORIG_DIR.glob("fold-*-model-0.keras"))
    rows: list[dict] = []
    map_payload: dict[str, np.ndarray] = {}

    for fold in folds:
        orig_path = ORIG_DIR / f"fold-{fold}-model-0.keras"
        clip_path = CLIPPED_DIR / f"fold-{fold}-model-0.keras"
        if not clip_path.exists():
            print(f"Fold {fold}: clipped checkpoint missing, skipping.")
            continue

        orig_model = load_model(str(orig_path))
        sus_o, coh_o, ifi_o = predict_with_heads(orig_model, val_ds)

        tf.keras.backend.clear_session()  # avoid layer-name collisions across folds
        clipped_model = build_clipped_model(train_ds_for_build, numeric_cols, categorical_cols, pga_col)
        clipped_model.load_weights(str(clip_path))
        sus_c, coh_c, ifi_c = predict_with_heads(clipped_model, val_ds)

        m_o = per_fold_metrics(y_true, sus_o)
        m_c = per_fold_metrics(y_true, sus_c)

        ifi_o_deg = np.degrees(ifi_o)
        ifi_c_deg = np.degrees(ifi_c)
        below = float(np.mean(ifi_o_deg < CLIP_DEG[0]))
        above = float(np.mean(ifi_o_deg > CLIP_DEG[1]))

        rows.append(
            {
                "fold": fold,
                "auc_orig": m_o["auc"],
                "auc_clip": m_c["auc"],
                "auc_delta": m_c["auc"] - m_o["auc"],
                "balacc_orig": m_o["bal_acc"],
                "balacc_clip": m_c["bal_acc"],
                "mean_sus_orig": m_o["mean_sus"],
                "mean_sus_clip": m_c["mean_sus"],
                "ifi_mean_deg_orig": float(np.mean(ifi_o_deg)),
                "ifi_mean_deg_clip": float(np.mean(ifi_c_deg)),
                "ifi_min_deg_orig": float(np.min(ifi_o_deg)),
                "ifi_max_deg_orig": float(np.max(ifi_o_deg)),
                "frac_below_25_orig": below,
                "frac_above_45_orig": above,
                "coh_mean_orig": float(np.mean(coh_o)),
                "coh_mean_clip": float(np.mean(coh_c)),
            }
        )

        if fold == MAP_FOLD:
            map_payload = {
                "sus_o": sus_o,
                "sus_c": sus_c,
                "ifi_o": ifi_o,
                "ifi_c": ifi_c,
                "coh_o": coh_o,
                "coh_c": coh_c,
            }
        print(
            f"Fold {fold}: AUC orig={m_o['auc']:.4f} clip={m_c['auc']:.4f}"
            f"  | IFI orig mean={np.mean(ifi_o_deg):.2f}° (range {np.min(ifi_o_deg):.1f}–{np.max(ifi_o_deg):.1f})"
            f"  | clipped fraction={below + above:.1%}"
        )

    if not rows:
        print("No folds compared. Did the clipped training finish?")
        return

    summary = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    summary.to_csv(FIG_DIR / "per_fold_summary.csv", index=False)

    # 1. Per-fold AUC bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    width = 0.4
    x = np.arange(len(summary))
    ax.bar(x - width / 2, summary["auc_orig"], width, label="Original (no clip)")
    ax.bar(x + width / 2, summary["auc_clip"], width, label="Clipped [25°, 45°]")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["fold"].astype(int))
    ax.set_xlabel("Fold")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Per-fold AUC: original vs IFI-clipped")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "auc_per_fold.png", dpi=140)
    plt.close()

    # 2. IFI distribution for the map fold
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.hist(np.degrees(map_payload["ifi_o"]), bins=40, alpha=0.6, label="Original", density=True)
    ax.hist(np.degrees(map_payload["ifi_c"]), bins=40, alpha=0.6, label="Clipped [25°, 45°]", density=True)
    ax.axvline(CLIP_DEG[0], color="k", linestyle="--", alpha=0.6, label="Clip bounds")
    ax.axvline(CLIP_DEG[1], color="k", linestyle="--", alpha=0.6)
    ax.set_xlabel("Internal friction angle (°)")
    ax.set_ylabel("Density")
    ax.set_title(f"IFI distribution — fold {MAP_FOLD} (validation)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"ifi_hist_fold{MAP_FOLD}.png", dpi=140)
    plt.close()

    # 3. Cohesion distribution shift (compensation check)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.hist(map_payload["coh_o"], bins=40, alpha=0.6, label="Original", density=True)
    ax.hist(map_payload["coh_c"], bins=40, alpha=0.6, label="Clipped", density=True)
    ax.set_xlabel("Cohesion (kPa)")
    ax.set_ylabel("Density")
    ax.set_title(f"Cohesion distribution — fold {MAP_FOLD} (validation)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"cohesion_hist_fold{MAP_FOLD}.png", dpi=140)
    plt.close()

    # 4. Susceptibility distribution
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.hist(map_payload["sus_o"], bins=40, alpha=0.6, label="Original", density=True)
    ax.hist(map_payload["sus_c"], bins=40, alpha=0.6, label="Clipped", density=True)
    ax.set_xlabel("Predicted susceptibility")
    ax.set_ylabel("Density")
    ax.set_title(f"Susceptibility distribution — fold {MAP_FOLD} (validation)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"susceptibility_hist_fold{MAP_FOLD}.png", dpi=140)
    plt.close()

    # 5. Side-by-side susceptibility maps
    gdf = val_df.to_crs(epsg=3857).copy()
    gdf["sus_orig"] = map_payload["sus_o"]
    gdf["sus_clip"] = map_payload["sus_c"]
    gdf["sus_delta"] = gdf["sus_clip"] - gdf["sus_orig"]

    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, col, title in zip(axes, ["sus_orig", "sus_clip"], [
        f"Original — fold {MAP_FOLD}",
        f"IFI clipped [25°, 45°] — fold {MAP_FOLD}",
    ]):
        gdf.plot(column=col, cmap="plasma_r", ax=ax, norm=norm)
        sm = plt.cm.ScalarMappable(cmap="plasma_r", norm=norm)
        fig.colorbar(sm, ax=ax)
        ax.set_title(title)
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"susceptibility_maps_fold{MAP_FOLD}.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 6. Susceptibility delta map (signed)
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    delta_max = float(np.percentile(np.abs(gdf["sus_delta"]), 99))
    delta_norm = mcolors.TwoSlopeNorm(vmin=-delta_max, vcenter=0, vmax=delta_max)
    gdf.plot(column="sus_delta", cmap="RdBu_r", ax=ax, norm=delta_norm)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=delta_norm)
    fig.colorbar(sm, ax=ax, label="Susceptibility delta (clipped − original)")
    ax.set_title(f"Susceptibility delta — fold {MAP_FOLD} (red = clip raises risk)")
    cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"susceptibility_delta_fold{MAP_FOLD}.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 7. Markdown summary
    summary_md = FIG_DIR / "SUMMARY.md"
    with summary_md.open("w") as f:
        f.write("# v4-1-1dce vs IFI-clipped [25°, 45°] — comparison\n\n")
        f.write(f"Map fold: {MAP_FOLD}.  Validation rows: {len(val_df)}.\n\n")
        f.write("## Per-fold metrics\n\n")
        f.write(summary.to_csv(index=False, float_format="%.4f"))
        f.write("\n\n## Aggregates\n\n")
        mean_auc_o = summary["auc_orig"].mean()
        mean_auc_c = summary["auc_clip"].mean()
        f.write(f"- Mean AUC original: **{mean_auc_o:.4f}** ± {summary['auc_orig'].std():.4f}\n")
        f.write(f"- Mean AUC clipped:  **{mean_auc_c:.4f}** ± {summary['auc_clip'].std():.4f}\n")
        f.write(f"- Mean ΔAUC (clipped − original): **{(mean_auc_c - mean_auc_o):+.4f}**\n")
        f.write(
            f"- Fraction of validation pixels with original IFI < 25°: mean **{summary['frac_below_25_orig'].mean():.1%}**\n"
        )
        f.write(
            f"- Fraction of validation pixels with original IFI > 45°: mean **{summary['frac_above_45_orig'].mean():.1%}**\n"
        )
        f.write(
            f"- Mean cohesion shift (clipped − original): **{(summary['coh_mean_clip'] - summary['coh_mean_orig']).mean():+.2f} kPa**\n"
        )

    print(f"\nWrote summary CSV + markdown + plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
