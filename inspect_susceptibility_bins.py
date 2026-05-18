"""Inspect susceptibility distribution across the 5 bins used in Table 5A.4.

Bins (Landslide Susceptibility ranges):
    0.000 - 0.125   (very low)
    0.126 - 0.375   (low)
    0.376 - 0.625   (moderate)
    0.626 - 0.875   (high)
    0.876 - 1.000   (very high)

Predicts on the held-out validation GeoPackage (`Merged_PINN_Features_2.gpkg`)
with both:
  - the data-driven baseline from `train_data_driven`
  - the v2-8 PINN (`train_rainfall_v3`)

Usage
-----
Default (no slope filter -- matches the row count in Table 5A.4):

    python inspect_susceptibility_bins.py

With v8's slope >= 10 filter (matches the v2-8 notebook validation cell):

    python inspect_susceptibility_bins.py --slope-filter 10

Custom fold / paths:

    python inspect_susceptibility_bins.py \\
        --dd-model .../v8_data_driven/fold-2-model-data-driven.keras \\
        --pinn-model .../v8/fold-2-model-v3.keras
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_files.data import (
    apply_clip_thresholds,
    apply_log_transform,
    dataframe_to_dataset,
)
from py_files.helpers import add_soil_texture_index

# Register custom layers used by both models (needed for `load_model`).
# Importing these modules has the side effect of executing the
# @tf.keras.utils.register_keras_serializable() decorators inside them.
from py_files import train_data_driven  # noqa: F401  (CastToFloat32)
from py_files.GallenModel import (  # noqa: F401
    CriticalAcceleration,
    DisplacementIntermediate,
    FosLayer,
)
from py_files.GallenModel_v1 import (  # noqa: F401
    ClipLayer,
    CohesionLayer,
    DisplacementLayerRainFall,
    InternalFrictionLayer,
    NewmarkActivation,
    WetnessLayer,
)
from py_files.GallenModel_v3 import HydraulicConductivityLayerV3  # noqa: F401
from py_files.Landslidev2_Old import DiceCrossEntropyLoss  # noqa: F401

# ---------------------------------------------------------------------------
# Constants matching the v2-8 notebook validation pipeline
# ---------------------------------------------------------------------------

DEFAULT_VALIDATION = (
    "~/Documents/ml-prep/ML-PREP-2025/learn/data/Merged_PINN_Features_2.gpkg"
)
DEFAULT_DD_MODEL = (
    "/Users/giogonzales/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/"
    "trainedCotabatoPhase7/historical/v8_data_driven/fold-1-model-data-driven.keras"
)
DEFAULT_PINN_MODEL = (
    "/Users/giogonzales/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/"
    "trainedCotabatoPhase7/historical/v8/fold-1-model-v3.keras"
)
DEFAULT_TRANSFORMS = (
    PROJECT_ROOT / "feature_manifests" / "v1_cotabato_transforms.json"
)

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

# Bin edges aligned with Table 5A.4. Using bins=[0, 0.125, 0.375, 0.625, 0.875, 1.0001]
# with `right=False` would lump 0.125 into the low bin; the table writes the bins as
# closed-open on the upper side (0-0.125, 0.126-0.375 ...) so we replicate that by
# nudging the inner edges by +eps and using right=True (pd.cut default).
BIN_EDGES = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0 + 1e-9]
BIN_LABELS = [
    "0 - 0.125",
    "0.126 - 0.375",
    "0.376 - 0.625",
    "0.626 - 0.875",
    "0.876 - 1.0",
]


# ---------------------------------------------------------------------------
# Validation-set preprocessing (mirrors v2-8 cell 20)
# ---------------------------------------------------------------------------

def load_and_prep_validation(
    path: str,
    transform_meta: dict,
    slope_filter_deg: float | None = None,
) -> gpd.GeoDataFrame:
    """Reproduce the v2-8 validation cell on `Merged_PINN_Features_2.gpkg`."""
    expanded = os.path.expanduser(path)
    df = gpd.read_file(expanded)
    df = df.rename(columns=VALIDATION_RENAME)

    # Bulk density (g/cm^3 * 100) -> unit weight kN/m^3.
    df["BUK_mean"] = df["BUK_mean"] * 9.81 / 100

    impute_cols = list(VALIDATION_RENAME.values())
    df[impute_cols] = df[impute_cols].fillna(
        df[impute_cols].median(numeric_only=True)
    )

    n_before_filter = len(df)
    if slope_filter_deg is not None:
        df = df[df["Slope_mean"] >= slope_filter_deg].reset_index(drop=True)
        print(
            f"Slope filter (>= {slope_filter_deg} deg): "
            f"{n_before_filter:,} -> {len(df):,} rows"
        )
    else:
        df = df.reset_index(drop=True)
        print(f"No slope filter applied: {len(df):,} rows")

    df["type"] = df["soiltype"].fillna(-1).astype(int).astype(str)

    df = apply_log_transform(df, transform_meta["log_transformed_cols"])
    df = apply_clip_thresholds(df, transform_meta["clip_thresholds"])
    df = add_soil_texture_index(df)

    if "label" in df.columns:
        has_labels = df["label"].astype(str).str.strip().ne("").any()
    else:
        has_labels = False
    if has_labels:
        df["landslide"] = (
            pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        )
    else:
        df["landslide"] = 0
    return df


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _input_cols(model: tf.keras.Model) -> list[str]:
    return [t.name.split(":")[0] for t in model.inputs]


def predict_data_driven(model: tf.keras.Model, df: pd.DataFrame, batch_size: int) -> np.ndarray:
    cols = _input_cols(model)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Data-driven model missing inputs: {missing}")
    ds = dataframe_to_dataset(
        df[cols + ["landslide"]].copy(), shuffle=False, batch_size=batch_size,
    )
    return model.predict(ds, verbose=0).flatten()


def predict_pinn(model: tf.keras.Model, df: pd.DataFrame, batch_size: int) -> np.ndarray:
    cols = _input_cols(model)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"PINN model missing inputs: {missing}")
    ds = dataframe_to_dataset(
        df[cols + ["landslide"]].copy(), shuffle=False, batch_size=batch_size,
    )
    out = model.predict(ds, verbose=0)
    if isinstance(out, dict):
        # Multi-output PINN -- `final_head` holds the susceptibility logit/prob.
        if "final_head" not in out:
            raise KeyError(
                f"PINN output dict missing 'final_head'; got keys {list(out.keys())}"
            )
        return np.asarray(out["final_head"]).flatten()
    return np.asarray(out).flatten()


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_counts(preds: np.ndarray) -> pd.Series:
    """Histogram counts using the Table 5A.4 bin edges.

    NaN predictions (e.g. PINN Newmark chain returning 0/0 on flat slopes)
    are surfaced as an explicit "NaN" bucket so the column totals reconcile
    with the input row count.
    """
    arr = np.asarray(preds, dtype=float)
    nan_count = int(np.isnan(arr).sum())
    finite = arr[~np.isnan(arr)]
    clipped = np.clip(finite, 0.0, 1.0)
    cats = pd.cut(clipped, bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True)
    counts = cats.value_counts().reindex(BIN_LABELS, fill_value=0).astype(int)
    counts["NaN"] = nan_count
    counts.name = "count"
    return counts


def bin_summary_table(named_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per model with counts across the 5 bins (+ NaN)."""
    rows = {name: bin_counts(preds) for name, preds in named_preds.items()}
    df = pd.DataFrame(rows).T
    df["total"] = df.sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", default=DEFAULT_VALIDATION,
                   help="Path to Merged_PINN_Features_2.gpkg (default: %(default)s)")
    p.add_argument("--dd-model", default=DEFAULT_DD_MODEL,
                   help="Data-driven fold checkpoint (.keras)")
    p.add_argument("--pinn-model", default=DEFAULT_PINN_MODEL,
                   help="PINN v3 fold checkpoint (.keras)")
    p.add_argument("--transforms", default=str(DEFAULT_TRANSFORMS),
                   help="v1_cotabato_transforms.json path")
    p.add_argument("--slope-filter", type=float, default=None,
                   help=("Min slope (deg) to keep. Omit for no filter "
                         "(matches the 481,133-row total in Table 5A.4)."))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output-csv", default=None,
                   help="Optional CSV path to save the bin-count table.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Validation file : {args.validation}")
    print(f"Data-driven ckpt: {args.dd_model}")
    print(f"PINN v3 ckpt    : {args.pinn_model}")
    print(f"Transforms      : {args.transforms}")
    print()

    with open(args.transforms) as f:
        transform_meta = json.load(f)

    df_val = load_and_prep_validation(
        args.validation, transform_meta, slope_filter_deg=args.slope_filter,
    )

    print("\nLoading models...")
    dd_model = load_model(args.dd_model)
    pinn_model = load_model(args.pinn_model)
    print(f"  Data-driven inputs: {_input_cols(dd_model)}")
    print(f"  PINN inputs       : {_input_cols(pinn_model)}")

    print("\nRunning inference...")
    dd_preds = predict_data_driven(dd_model, df_val, args.batch_size)
    pinn_preds = predict_pinn(pinn_model, df_val, args.batch_size)

    summary = bin_summary_table({
        "Data Driven": dd_preds,
        "PINN v8 (v2-8)": pinn_preds,
    })

    print("\nSusceptibility distribution across Table 5A.4 bins")
    print("-" * 70)
    print(summary.to_string())
    print()
    print("Per-bin percentage of total rows:")
    pct = summary.drop(columns=["total"]).div(summary["total"], axis=0) * 100
    print(pct.round(2).to_string())
    print()
    print("Prediction summary statistics (finite values only):")
    stats = pd.DataFrame({
        "Data Driven": pd.Series(dd_preds).dropna().describe(),
        "PINN v8":      pd.Series(pinn_preds).dropna().describe(),
    })
    print(stats.round(4).to_string())
    dd_nan = int(np.isnan(dd_preds).sum())
    pinn_nan = int(np.isnan(pinn_preds).sum())
    if dd_nan or pinn_nan:
        print(f"\nNaN predictions: Data Driven={dd_nan}, PINN v8={pinn_nan} "
              "(see NaN column above; PINN NaNs usually come from the "
              "Newmark physics chain on degenerate slopes).")

    if args.output_csv:
        summary.to_csv(args.output_csv)
        print(f"\nWrote: {args.output_csv}")


if __name__ == "__main__":
    main()
