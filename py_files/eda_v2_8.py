"""Exploratory Data Analysis for the v2-8 Cotabato training + validation pair.

Produces schema/missingness, univariate distributions (with skew/outlier flags
that mirror per-fold transform behaviour), target balance, correlations, and
train/validation drift reports.

Run from the project root:

    source venv/bin/activate
    python py_files/eda_v2_8.py

Outputs are written to ``eda_outputs/`` (created if missing).

The training file goes through the same ``preprocessing_v2`` filter the v2-8
notebook uses. The validation file goes through the same rename + BUK unit
conversion + median-imputation + slope filter + soil-texture indexing the
notebook applies, but **without** the per-fold manifest log/clip transforms
so that train vs validation distributions are compared in the same raw space.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import json
import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_files.data import (
    apply_clip_thresholds, apply_imputation_medians, apply_log_transform,
    apply_missingness_indicators, dataframe_to_dataset, preprocessing_v2,
)
from py_files.helpers import add_soil_texture_index

# ---------------------------------------------------------------------------
# Constants (single source of truth for the v2-8 pipeline)
# ---------------------------------------------------------------------------

TRAIN_FILE = Path(
    "~/Documents/ml-prep/ML-PREP-2025/learn/data/SU_17_training_v3_contri.gpkg"
).expanduser()
VAL_FILE = Path(
    "~/Documents/ml-prep/ML-PREP-2025/learn/data/Merged_PINN_Features_2.gpkg"
).expanduser()

DEFAULT_OUT_DIR = PROJECT_ROOT / "eda_outputs"
DEFAULT_MODEL_SAVE_PATH = Path(
    "/Users/giogonzales/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/"
    "trainedCotabatoPhase7/historical/v8"
)
DEFAULT_TRANSFORMS_DIR = PROJECT_ROOT / "feature_manifests"
FOLD_FOR_VAL_PREDICTION = 1  # matches notebook cell 21
OOF_FILENAME = "oof_preds.npy"

COLUMNS_DROP = [
    "Landslide1", "descriptio", "sus_pinn_ground truth", "ds",
    "cohesion", "internal_friction", "sus_pinn_landslide",
    "confusion", "landslide_preds", "landslide_probability",
    "Lithology", "LITHO", "Geomorphology", "LITHODESC",
    "LITHO_2", "LITHODESC_2", "value",
]

VALIDATION_RENAME = {
    "slope": "Slope_mean",
    "bulkdensity": "BUK_mean",
    "pga": "PGA2_max",
    "prc": "Prc_mean",
    "contributingfactor": "ContributingFactor_mean",
    "soilthickness": "SoilThc_mean",
    "elevation": "Elev_mean",
    "clay": "Clay_mean",
    "silt": "Silt_mean",
    "sand": "Sand_mean",
}

# Mirrors PHYSICS_FEATURES in the v2-8 notebook (cell 8) and
# train_rainfall_v3.py per-fold transform exclusions.
PHYSICS_FEATURES = {
    "Slope_mean", "BUK_mean", "PGA2_max",
    "Prc_mean", "ContributingFactor_mean",
    "SoilThc_mean", "LULC_majority",
}

# From feature_manifests/v1_cotabato.json (final_features section).
FINAL_NUMERIC_FEATURES = [
    "Slope_mean", "BUK_mean", "Prc_mean", "ContributingFactor_mean",
    "SoilThc_mean", "soil_texture_idx", "PGA2_max", "Elev_mean",
]
FINAL_CATEGORICAL_FEATURES = ["type"]

SKEW_THRESHOLD = 1.0
LOWER_PCT, UPPER_PCT = 1, 99
DRIFT_KS_THRESHOLD = 0.1
SLOPE_FILTER_DEG = 10.0
TSNE_SAMPLE_DEFAULT = 3000
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_training() -> pd.DataFrame:
    """Load training data with the same filtering the v2-8 notebook uses.

    The index is reset so that ``df.iloc[i]`` aligns with ``oof_preds[i]``
    produced by ``train_model_rainfall_v3`` (which uses StratifiedKFold over
    the post-preprocessing dataframe).
    """
    print(f"[load] training: {TRAIN_FILE}")
    df = gpd.read_file(TRAIN_FILE)
    print(f"  raw rows: {len(df):,}")
    df, _columns, _numeric_cols = preprocessing_v2(df, columns_drop=COLUMNS_DROP)
    df = add_soil_texture_index(df)
    return df.reset_index(drop=True)


def load_validation() -> pd.DataFrame:
    """Load validation data through the v2-8 cell-20 pipeline (sans manifest transforms)."""
    print(f"[load] validation: {VAL_FILE}")
    df = gpd.read_file(VAL_FILE)
    print(f"  raw rows: {len(df):,}")
    df = df.rename(columns=VALIDATION_RENAME)

    # Bulk density unit fix: g/cm^3 * 100 -> kN/m^3.
    if "BUK_mean" in df.columns:
        df["BUK_mean"] = df["BUK_mean"] * 9.81 / 100

    impute_cols = [c for c in VALIDATION_RENAME.values() if c in df.columns]
    if impute_cols:
        df[impute_cols] = df[impute_cols].fillna(df[impute_cols].median(numeric_only=True))

    if "Slope_mean" in df.columns:
        n_before = len(df)
        df = df[df["Slope_mean"] >= SLOPE_FILTER_DEG].reset_index(drop=True)
        print(f"  slope filter (>= {SLOPE_FILTER_DEG} deg): {n_before:,} -> {len(df):,}")

    if "soiltype" in df.columns:
        df["type"] = df["soiltype"].fillna(-1).astype(int).astype(str)

    df = add_soil_texture_index(df)
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])
    return df


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_schema_missingness(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    rows = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isnull().sum())
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n_missing": n_missing,
            "pct_missing": round(100 * n_missing / max(len(s), 1), 4),
            "n_unique": int(s.nunique(dropna=True)),
        })
    schema = pd.DataFrame(rows).sort_values("pct_missing", ascending=False)
    out = out_dir / f"{name}_schema.csv"
    schema.to_csv(out, index=False)
    total_cells = len(df) * len(df.columns)
    total_nan = int(df.isnull().sum().sum())
    print(f"[schema:{name}] rows={len(df):,} cols={len(df.columns)} "
          f"nan_cells={total_nan:,}/{total_cells:,} -> {out.name}")


def _percentile(s: pd.Series, q: float) -> float:
    s = s.dropna()
    if len(s) == 0:
        return float("nan")
    return float(np.percentile(s, q))


def report_distributions(
    df: pd.DataFrame, numeric_cols: list[str], name: str, out_dir: Path,
) -> None:
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        skew = float(s.skew()) if s.std() > 0 else 0.0
        col_min = float(s.min())
        in_physics = col in PHYSICS_FEATURES
        would_log1p = (abs(skew) > SKEW_THRESHOLD) and (col_min >= 0) and (not in_physics)
        would_clip = not in_physics
        rows.append({
            "column": col,
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": col_min,
            "p1": _percentile(s, LOWER_PCT),
            "p25": _percentile(s, 25),
            "p50": _percentile(s, 50),
            "p75": _percentile(s, 75),
            "p99": _percentile(s, UPPER_PCT),
            "max": float(s.max()),
            "skew": skew,
            "would_log1p": would_log1p,
            "would_clip": would_clip,
            "in_physics_set": in_physics,
        })
    summary = pd.DataFrame(rows)
    out_csv = out_dir / f"{name}_distributions.csv"
    summary.to_csv(out_csv, index=False)

    # Histogram grid
    cols_present = [r["column"] for r in rows]
    n = len(cols_present)
    if n > 0:
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, cols_present):
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            ax.hist(s, bins=40, color="steelblue", edgecolor="white")
            ax.set_title(col, fontsize=10)
            ax.tick_params(labelsize=8)
        for ax in axes[len(cols_present):]:
            ax.set_visible(False)
        fig.suptitle(f"{name} — feature distributions", y=1.02, fontsize=12)
        fig.tight_layout()
        out_png = out_dir / f"{name}_distributions.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[distributions:{name}] {n} features -> {out_csv.name}, {out_png.name}")
    else:
        print(f"[distributions:{name}] no numeric features available")


def report_target_balance(df: pd.DataFrame, out_dir: Path) -> None:
    if "landslide" not in df.columns:
        print("[target] no `landslide` column; skipping")
        return

    overall_rate = float(df["landslide"].mean())
    overall = pd.DataFrame([{
        "group": "__overall__",
        "level": "all",
        "n": int(len(df)),
        "n_positive": int(df["landslide"].sum()),
        "positive_rate": overall_rate,
    }])

    parts = [overall]
    for group_col in ("type", "soil_texture_idx"):
        if group_col not in df.columns:
            continue
        agg = (
            df.groupby(group_col)["landslide"]
              .agg(n="count", n_positive="sum", positive_rate="mean")
              .reset_index()
              .rename(columns={group_col: "level"})
        )
        agg.insert(0, "group", group_col)
        parts.append(agg)

    balance = pd.concat(parts, ignore_index=True)
    out_csv = out_dir / "target_balance.csv"
    balance.to_csv(out_csv, index=False)
    print(f"[target] overall positive_rate={overall_rate:.4f}; n={len(df):,} -> {out_csv.name}")

    # Bar plots
    for group_col, fname in (("type", "target_balance_by_type.png"),
                             ("soil_texture_idx", "target_balance_by_soil_texture.png")):
        if group_col not in df.columns:
            continue
        agg = (
            df.groupby(group_col)["landslide"]
              .agg(n="count", positive_rate="mean")
              .reset_index()
              .sort_values(group_col)
        )
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(agg)), 4))
        bars = ax.bar(agg[group_col].astype(str), agg["positive_rate"], color="indianred")
        ax.axhline(overall_rate, color="black", linestyle="--", linewidth=1,
                   label=f"overall={overall_rate:.3f}")
        for bar, n in zip(bars, agg["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"n={int(n)}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("positive rate")
        ax.set_xlabel(group_col)
        ax.set_title(f"landslide positive rate by {group_col}")
        ax.legend(loc="upper right", fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def report_correlations(
    df: pd.DataFrame, numeric_cols: list[str], name: str, out_dir: Path,
) -> None:
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        print(f"[corr:{name}] fewer than 2 numeric cols; skipping")
        return

    corr = df[cols].corr()
    abs_corr = corr.abs()
    upper = abs_corr.where(np.triu(np.ones(abs_corr.shape, dtype=bool), k=1))
    pair_rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = upper.iat[i, j]
            if pd.notna(v) and v > 0.9:
                pair_rows.append({"feature_1": cols[i], "feature_2": cols[j], "correlation": float(v)})
    pairs = pd.DataFrame(pair_rows, columns=["feature_1", "feature_2", "correlation"])
    if not pairs.empty:
        pairs = pairs.sort_values("correlation", ascending=False)
    out_csv = out_dir / f"{name}_correlations.csv"
    pairs.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 2, 0.7 * len(cols) + 2))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(f"{name} — correlation matrix")
    fig.tight_layout()
    out_png = out_dir / f"{name}_correlation_heatmap.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[corr:{name}] {len(pairs)} pairs above |r|>0.9 -> {out_csv.name}, {out_png.name}")


def report_drift(
    df_train: pd.DataFrame, df_val: pd.DataFrame,
    numeric_cols: list[str], out_dir: Path,
) -> None:
    cols = [c for c in numeric_cols if c in df_train.columns and c in df_val.columns]
    if not cols:
        print("[drift] no shared numeric columns; skipping")
        return

    rows = []
    for col in cols:
        train_s = pd.to_numeric(df_train[col], errors="coerce").dropna().to_numpy()
        val_s = pd.to_numeric(df_val[col], errors="coerce").dropna().to_numpy()
        if len(train_s) == 0 or len(val_s) == 0:
            continue
        ks = stats.ks_2samp(train_s, val_s)
        train_std = float(train_s.std()) if train_s.std() > 0 else float("nan")
        rows.append({
            "column": col,
            "n_train": int(len(train_s)),
            "n_val": int(len(val_s)),
            "mean_train": float(train_s.mean()),
            "mean_val": float(val_s.mean()),
            "std_train": float(train_s.std()),
            "std_val": float(val_s.std()),
            "mean_shift_z": (float(val_s.mean()) - float(train_s.mean())) / train_std
                            if train_std and not math.isnan(train_std) else float("nan"),
            "std_ratio_val_over_train": float(val_s.std()) / train_std
                                        if train_std and not math.isnan(train_std) else float("nan"),
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "drift_flag": float(ks.statistic) > DRIFT_KS_THRESHOLD,
        })
    drift = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
    out_csv = out_dir / "drift.csv"
    drift.to_csv(out_csv, index=False)

    n = len(cols)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        train_s = pd.to_numeric(df_train[col], errors="coerce").dropna()
        val_s = pd.to_numeric(df_val[col], errors="coerce").dropna()
        ax.hist(train_s, bins=40, density=True, alpha=0.5, color="steelblue", label="train")
        ax.hist(val_s, bins=40, density=True, alpha=0.5, color="darkorange", label="val")
        ks_row = drift.loc[drift["column"] == col].iloc[0]
        ax.set_title(f"{col}\nKS={ks_row['ks_stat']:.3f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("train vs validation — distribution overlay", y=1.02, fontsize=12)
    fig.tight_layout()
    out_png = out_dir / "drift_overlay.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_drifted = int(drift["drift_flag"].sum())
    print(f"[drift] {n_drifted}/{len(drift)} features flagged (KS>{DRIFT_KS_THRESHOLD}) -> "
          f"{out_csv.name}, {out_png.name}")


def _load_oof_predictions(model_save_path: Path, expected_len: int) -> np.ndarray | None:
    """Return OOF preds aligned to ``df_train.iloc[i]`` or None if unavailable."""
    oof_path = Path(model_save_path) / OOF_FILENAME
    if not oof_path.exists():
        print(f"[tsne:susc] no OOF preds at {oof_path} -- skipping training susceptibility plot")
        return None
    preds = np.load(oof_path)
    if len(preds) != expected_len:
        print(f"[tsne:susc] OOF length {len(preds)} != training rows {expected_len}; "
              f"skipping training susceptibility plot")
        return None
    return preds


def _predict_validation(
    val_sample: pd.DataFrame, model_save_path: Path,
    transforms_dir: Path, fold: int = FOLD_FOR_VAL_PREDICTION,
) -> np.ndarray | None:
    """Predict susceptibility on the val sample using the saved fold checkpoint.

    Mirrors notebook cell 20 + 21: replay manifest transforms (indicators,
    medians, log, clip), build the dataset, run the saved model. Returns
    None and skips gracefully if any artifact is missing.
    """
    model_path = Path(model_save_path) / f"fold-{fold}-model-v3.keras"
    manifest_path = Path(transforms_dir) / f"v1_cotabato_transforms_fold{fold}.json"
    if not model_path.exists():
        print(f"[tsne:susc] no fold-{fold} checkpoint at {model_path} -- "
              f"skipping validation susceptibility plot")
        return None
    if not manifest_path.exists():
        print(f"[tsne:susc] no fold-{fold} manifest at {manifest_path} -- "
              f"skipping validation susceptibility plot")
        return None

    # Local import so EDA still loads without tensorflow being importable.
    from tensorflow.keras.models import load_model
    from py_files.GallenModel_v1 import NewmarkActivation
    from py_files.LandslideRainfall_v3 import LandslideRainFallV3

    with open(manifest_path) as f:
        meta = json.load(f)

    val = val_sample.copy()
    val = apply_missingness_indicators(val, meta.get("imputed_indicator_cols", []))
    val = apply_imputation_medians(val, meta.get("imputation_medians", {}))
    val = apply_log_transform(val, meta.get("log_transformed_cols", []))
    val = apply_clip_thresholds(val, meta.get("clip_thresholds", {}))
    val = add_soil_texture_index(val)
    val["landslide"] = 0  # dummy so dataframe_to_dataset can pop it

    model = load_model(
        model_path, custom_objects={"NewmarkActivation": NewmarkActivation},
    )
    input_cols = [t.name.split(":")[0] for t in model.inputs]
    missing = [c for c in input_cols if c not in val.columns]
    if missing:
        print(f"[tsne:susc] val sample missing required model inputs {missing}; skipping")
        return None

    val_ds = dataframe_to_dataset(
        val[input_cols + ["landslide"]], shuffle=False, batch_size=128,
    )
    val_ds_mo = LandslideRainFallV3.to_multi_output_ds(val_ds)
    preds = model.predict(val_ds_mo, verbose=0)["final_head"].flatten()
    return preds


def _stratified_sample(df: pd.DataFrame, n: int, by: str, rng: np.random.Generator) -> pd.DataFrame:
    """Sample ``n`` rows from ``df`` proportionally across ``by`` groups."""
    if len(df) <= n:
        return df.copy()
    groups = df.groupby(by)
    per_group = []
    for level, group in groups:
        take = max(1, int(round(n * len(group) / len(df))))
        take = min(take, len(group))
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        per_group.append(df.loc[idx])
    sampled = pd.concat(per_group, axis=0)
    if len(sampled) > n:
        keep = rng.choice(sampled.index.to_numpy(), size=n, replace=False)
        sampled = sampled.loc[keep]
    return sampled


def report_tsne(
    df_train: pd.DataFrame, df_val: pd.DataFrame,
    numeric_cols: list[str], out_dir: Path,
    sample_size: int = TSNE_SAMPLE_DEFAULT,
    *,
    model_save_path: Path | None = None,
    transforms_dir: Path | None = None,
    with_susceptibility: bool = True,
) -> None:
    """Project training and validation rows to 2D via t-SNE.

    Subsamples each dataset (stratified by ``landslide`` for training),
    standardizes features using training mean/std (no validation leakage),
    runs t-SNE on the combined matrix so points land in a shared 2D space,
    and renders three colorings:

    - training points by landslide label
    - training points by ``type``
    - training vs validation overlay

    Saves ``tsne_coords.csv`` with all points + metadata for further use.
    """
    cols = [c for c in numeric_cols if c in df_train.columns and c in df_val.columns]
    if len(cols) < 2:
        print("[tsne] fewer than 2 shared numeric cols; skipping")
        return

    rng = np.random.default_rng(TSNE_RANDOM_STATE)

    train_sample = _stratified_sample(df_train, sample_size, by='landslide', rng=rng)
    val_sample = (df_val.sample(min(sample_size, len(df_val)), random_state=TSNE_RANDOM_STATE)
                  if len(df_val) > 0 else df_val)

    # Standardize using TRAINING stats only, then apply to both. Prevents val
    # data from influencing the scaling.
    train_mean = train_sample[cols].mean()
    train_std = train_sample[cols].std().replace(0, 1)
    train_X = ((train_sample[cols] - train_mean) / train_std).to_numpy(dtype=np.float32)
    val_X = ((val_sample[cols] - train_mean) / train_std).to_numpy(dtype=np.float32)

    combined_X = np.vstack([train_X, val_X])
    perplexity = min(TSNE_PERPLEXITY, max(5, combined_X.shape[0] // 4))

    print(f"[tsne] running on {combined_X.shape[0]} points "
          f"(train={len(train_sample)}, val={len(val_sample)}) "
          f"x {combined_X.shape[1]} features, perplexity={perplexity}")
    tsne = TSNE(
        n_components=2, perplexity=perplexity,
        init='pca', learning_rate='auto',
        random_state=TSNE_RANDOM_STATE, n_iter=1000,
    )
    coords = tsne.fit_transform(combined_X)
    train_coords = coords[: len(train_sample)]
    val_coords = coords[len(train_sample):]

    # Save coords + metadata
    train_meta = pd.DataFrame({
        'dataset': 'training',
        'landslide': train_sample['landslide'].to_numpy() if 'landslide' in train_sample else np.nan,
        'type': train_sample['type'].astype(str).to_numpy() if 'type' in train_sample else '',
        'soil_texture_idx': (train_sample['soil_texture_idx'].to_numpy()
                              if 'soil_texture_idx' in train_sample else np.nan),
        'PGA2_max': (train_sample['PGA2_max'].to_numpy()
                     if 'PGA2_max' in train_sample else np.nan),
        'x': train_coords[:, 0], 'y': train_coords[:, 1],
    })
    val_meta = pd.DataFrame({
        'dataset': 'validation',
        'landslide': np.nan,
        'type': val_sample['type'].astype(str).to_numpy() if 'type' in val_sample else '',
        'soil_texture_idx': (val_sample['soil_texture_idx'].to_numpy()
                              if 'soil_texture_idx' in val_sample else np.nan),
        'PGA2_max': (val_sample['PGA2_max'].to_numpy()
                     if 'PGA2_max' in val_sample else np.nan),
        'x': val_coords[:, 0], 'y': val_coords[:, 1],
    })
    coords_df = pd.concat([train_meta, val_meta], ignore_index=True)
    coords_df.to_csv(out_dir / 'tsne_coords.csv', index=False)

    # Plot 1: training by landslide
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color in [(0, 'steelblue'), (1, 'crimson')]:
        mask = train_meta['landslide'] == label
        ax.scatter(train_coords[mask, 0], train_coords[mask, 1],
                   s=6, alpha=0.5, color=color, label=f'landslide={label}')
    ax.set_title('t-SNE — training points by landslide')
    ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    fig.tight_layout()
    fig.savefig(out_dir / 'tsne_train_by_landslide.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: training by type
    if 'type' in train_sample:
        fig, ax = plt.subplots(figsize=(7, 6))
        levels = sorted(train_meta['type'].dropna().unique())
        palette = plt.cm.tab10.colors
        for i, level in enumerate(levels):
            mask = train_meta['type'] == level
            ax.scatter(train_coords[mask, 0], train_coords[mask, 1],
                       s=6, alpha=0.5, color=palette[i % len(palette)], label=str(level))
        ax.set_title('t-SNE — training points by type')
        ax.legend(loc='best', fontsize=9)
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
        fig.tight_layout()
        fig.savefig(out_dir / 'tsne_train_by_type.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Plot 3: train vs validation overlay
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(train_coords[:, 0], train_coords[:, 1],
               s=6, alpha=0.4, color='steelblue', label=f'training (n={len(train_sample)})')
    ax.scatter(val_coords[:, 0], val_coords[:, 1],
               s=6, alpha=0.5, color='darkorange', label=f'validation (n={len(val_sample)})')
    ax.set_title('t-SNE — training vs validation overlay')
    ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    fig.tight_layout()
    fig.savefig(out_dir / 'tsne_train_vs_val.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 4: training and validation side-by-side, colored by PGA2_max.
    # Shared color scale (train+val combined min/max) so the two panels are
    # directly comparable. If t-SNE preserves PGA structure, you'll see a
    # smooth gradient across the cloud; if PGA is scrambled, it means other
    # features dominate the local-neighbor structure.
    if 'PGA2_max' in train_sample.columns and 'PGA2_max' in val_sample.columns:
        train_pga = train_sample['PGA2_max'].to_numpy()
        val_pga = val_sample['PGA2_max'].to_numpy()
        pga_min = float(np.nanmin(np.concatenate([train_pga, val_pga])))
        pga_max = float(np.nanmax(np.concatenate([train_pga, val_pga])))

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
        sc0 = axes[0].scatter(train_coords[:, 0], train_coords[:, 1],
                              c=train_pga, cmap='viridis', vmin=pga_min, vmax=pga_max,
                              s=8, alpha=0.7)
        axes[0].set_title(f'training (n={len(train_sample)})')
        axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')

        sc1 = axes[1].scatter(val_coords[:, 0], val_coords[:, 1],
                              c=val_pga, cmap='viridis', vmin=pga_min, vmax=pga_max,
                              s=8, alpha=0.7)
        axes[1].set_title(f'validation (n={len(val_sample)})')
        axes[1].set_xlabel('t-SNE 1')

        cbar = fig.colorbar(sc1, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label('PGA2_max (g)')
        fig.suptitle('t-SNE — colored by PGA2_max (shared scale)', y=1.02, fontsize=12)
        fig.savefig(out_dir / 'tsne_by_pga.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    n_pngs = 4

    # Plot 5: training and validation colored by predicted susceptibility.
    # Training uses out-of-fold predictions from oof_preds.npy aligned by
    # positional index. Validation runs the fold-1 checkpoint with manifest
    # transforms replayed. Either source is skipped gracefully if missing.
    if with_susceptibility and model_save_path is not None and transforms_dir is not None:
        train_susc = None
        val_susc = None

        oof = _load_oof_predictions(model_save_path, expected_len=len(df_train))
        if oof is not None:
            train_susc = oof[train_sample.index.to_numpy()]

        val_susc = _predict_validation(
            val_sample, model_save_path, transforms_dir, fold=FOLD_FOR_VAL_PREDICTION,
        )

        if train_susc is not None or val_susc is not None:
            fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

            if train_susc is not None:
                sc0 = axes[0].scatter(train_coords[:, 0], train_coords[:, 1],
                                      c=train_susc, cmap='magma', vmin=0, vmax=1,
                                      s=8, alpha=0.7)
                axes[0].set_title(f'training OOF (n={len(train_sample)})')
            else:
                axes[0].scatter(train_coords[:, 0], train_coords[:, 1],
                                color='lightgrey', s=8, alpha=0.5)
                axes[0].set_title('training (no OOF available)')
                sc0 = None
            axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')

            if val_susc is not None:
                sc1 = axes[1].scatter(val_coords[:, 0], val_coords[:, 1],
                                      c=val_susc, cmap='magma', vmin=0, vmax=1,
                                      s=8, alpha=0.7)
                axes[1].set_title(f'validation fold-1 (n={len(val_sample)})')
            else:
                axes[1].scatter(val_coords[:, 0], val_coords[:, 1],
                                color='lightgrey', s=8, alpha=0.5)
                axes[1].set_title('validation (no checkpoint available)')
                sc1 = None
            axes[1].set_xlabel('t-SNE 1')

            # Single colorbar on whichever side has a mappable.
            mappable = sc1 if sc1 is not None else sc0
            if mappable is not None:
                cbar = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.02)
                cbar.set_label('predicted susceptibility')
            fig.suptitle('t-SNE — colored by predicted susceptibility (shared scale 0-1)',
                         y=1.02, fontsize=12)
            fig.savefig(out_dir / 'tsne_by_susceptibility.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            n_pngs += 1
            print(f"[tsne:susc] wrote tsne_by_susceptibility.png "
                  f"(train_susc={train_susc is not None}, val_susc={val_susc is not None})")

            # Persist the susceptibility values alongside coords for downstream use.
            sus_train = (train_susc.tolist() if train_susc is not None
                         else [None] * len(train_sample))
            sus_val = (val_susc.tolist() if val_susc is not None
                       else [None] * len(val_sample))
            sus_series = pd.Series(sus_train + sus_val, name='predicted_susceptibility')
            coords_df['predicted_susceptibility'] = sus_series.values
            coords_df.to_csv(out_dir / 'tsne_coords.csv', index=False)

    print(f"[tsne] wrote tsne_coords.csv + {n_pngs} PNGs to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--tsne-sample", type=int, default=TSNE_SAMPLE_DEFAULT,
                        help=f"Rows per dataset for t-SNE (default: {TSNE_SAMPLE_DEFAULT})")
    parser.add_argument("--no-tsne", action="store_true",
                        help="Skip the t-SNE projection (it can take ~30s)")
    parser.add_argument("--model-save-path", type=Path, default=DEFAULT_MODEL_SAVE_PATH,
                        help=f"Path to fold checkpoints + oof_preds.npy "
                             f"(default: {DEFAULT_MODEL_SAVE_PATH})")
    parser.add_argument("--transforms-dir", type=Path, default=DEFAULT_TRANSFORMS_DIR,
                        help=f"Path to per-fold manifests (default: {DEFAULT_TRANSFORMS_DIR})")
    parser.add_argument("--no-susceptibility", action="store_true",
                        help="Skip the t-SNE-by-predicted-susceptibility plot even if "
                             "checkpoints / OOF predictions are available")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] output dir: {out_dir}")

    df_train = load_training()
    df_val = load_validation()

    train_numeric = [c for c in FINAL_NUMERIC_FEATURES if c in df_train.columns]
    val_numeric = [c for c in FINAL_NUMERIC_FEATURES if c in df_val.columns]

    report_schema_missingness(df_train, "training", out_dir)
    report_schema_missingness(df_val, "validation", out_dir)

    report_distributions(df_train, train_numeric, "training", out_dir)
    report_distributions(df_val, val_numeric, "validation", out_dir)

    report_target_balance(df_train, out_dir)

    report_correlations(df_train, train_numeric, "training", out_dir)

    report_drift(df_train, df_val, FINAL_NUMERIC_FEATURES, out_dir)

    if not args.no_tsne:
        report_tsne(df_train, df_val, FINAL_NUMERIC_FEATURES, out_dir,
                    sample_size=args.tsne_sample,
                    model_save_path=args.model_save_path,
                    transforms_dir=args.transforms_dir,
                    with_susceptibility=not args.no_susceptibility)

    # Top-line summary
    drift_path = out_dir / "drift.csv"
    corr_path = out_dir / "training_correlations.csv"
    summary = {
        "out_dir": str(out_dir),
        "train_rows": int(len(df_train)),
        "val_rows": int(len(df_val)),
        "final_numeric_features": FINAL_NUMERIC_FEATURES,
        "final_categorical_features": FINAL_CATEGORICAL_FEATURES,
    }
    if drift_path.exists():
        drift = pd.read_csv(drift_path)
        summary["top3_drifted"] = drift.head(3)[["column", "ks_stat"]].to_dict(orient="records")
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        summary["top3_correlated"] = corr.head(3).to_dict(orient="records")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[main] done -> {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
