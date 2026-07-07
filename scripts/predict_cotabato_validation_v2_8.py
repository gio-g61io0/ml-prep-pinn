#!/usr/bin/env python
"""Map landslide susceptibility on the North Cotabato VALIDATION set
(`Merged_PINN_Features_2.gpkg`) using the PINN v2-8 PRODUCTION model.

The production model was trained on the Cotabato training set
(`SU_17_training_v3_contri.gpkg`) by `scripts/train_production_v2_8.py`. That
training data is left untouched here — this script only runs inference on the
held-out validation file and writes:
  - susceptibility PNG choropleth
  - a GPKG with susceptibility + geotechnical params (cohesion, internal
    friction) + FOS + Newmark displacement per slope unit.

Preprocessing replays notebook cell 23 EXACTLY, except the manifest is the
production manifest (paired with the production model), not fold-1's:
  rename -> BUK unit convert -> missingness indicators -> impute from TRAINING
  medians -> slope filter (>=10 deg) -> soiltype->type -> log -> clip ->
  soil_texture_idx. This keeps every value on the distribution the production
  NormalizationLayer was adapted on (no training-serving skew).

If the validation file carries a non-empty `label` column, a validation AUC is
printed as a bonus; otherwise the run is inference-only.
"""

import os
import sys
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script import

import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
from tensorflow.keras.models import load_model

from py_files.helpers import set_seed, add_soil_texture_index
from py_files.data import (
    apply_log_transform,
    apply_clip_thresholds,
    apply_missingness_indicators,
    apply_imputation_medians,
    dataframe_to_dataset,
)
from py_files.GallenModel_v1 import NewmarkActivation
from predict_cotabato_v2_8 import save_susceptibility_map  # DRY: same choropleth

set_seed(42)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
VALIDATION_PATH = Path(
    "~/Documents/ml-prep/ML-PREP-2025/learn/data/Merged_PINN_Features_2.gpkg"
).expanduser()
MODEL_DIR = Path(
    "~/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/"
    "trainedCotabatoPhase7/historical/v8"
).expanduser()
MODEL_PATH = MODEL_DIR / "production-model-v3.keras"
TRANSFORMS_PATH = PROJECT_ROOT / "feature_manifests" / "v1_cotabato_transforms_production.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "cotabato_validation_susceptibility_v2_8.gpkg"
MAP_PATH = PROJECT_ROOT / "outputs" / "cotabato_validation_susceptibility_map_v2_8.png"

SLOPE_FILTER_DEG = 10.0

# Validation column name -> training schema name (notebook cell 23).
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


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found: {MODEL_PATH}\n"
            "Run scripts/train_production_v2_8.py first."
        )
    if not TRANSFORMS_PATH.exists():
        raise FileNotFoundError(f"Production manifest not found: {TRANSFORMS_PATH}")
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Validation file not found: {VALIDATION_PATH}")

    print(f"Model:      {MODEL_PATH.name}")
    print(f"Manifest:   {TRANSFORMS_PATH.name}")
    model = load_model(MODEL_PATH, custom_objects={"NewmarkActivation": NewmarkActivation})
    with open(TRANSFORMS_PATH) as f:
        transform_meta = json.load(f)

    input_cols = [t.name.split(":")[0] for t in model.inputs]
    print(f"Model inputs: {input_cols}")

    # ---- Load & rename (cell 23) ----------------------------------------- #
    vdf = gpd.read_file(VALIDATION_PATH)
    print(f"\nLoaded {len(vdf):,} rows from {VALIDATION_PATH.name}")
    vdf = vdf.rename(columns=VALIDATION_RENAME)

    # BUK: g/cm^3 * 100 -> kN/m^3 unit weight (matches training median ~11.3).
    vdf["BUK_mean"] = vdf["BUK_mean"] * 9.81 / 100

    # ---- Replay manifest: indicators -> impute (TRAINING medians) -------- #
    vdf = apply_missingness_indicators(vdf, transform_meta.get("imputed_indicator_cols", []))
    vdf = apply_imputation_medians(vdf, transform_meta.get("imputation_medians", {}))

    # Safety net: any residual NaN among mapped features -> validation median.
    impute_cols = [c for c in VALIDATION_RENAME.values() if c in vdf.columns]
    residual_na = vdf[impute_cols].isnull().sum()
    if int(residual_na.sum()) > 0:
        print(f"  [warn] residual NaNs after manifest fill: "
              f"{residual_na[residual_na > 0].to_dict()}; using validation median")
        vdf[impute_cols] = vdf[impute_cols].fillna(vdf[impute_cols].median(numeric_only=True))

    # ---- Slope filter (match training domain) ---------------------------- #
    n_before = len(vdf)
    vdf = vdf[vdf["Slope_mean"] >= SLOPE_FILTER_DEG].reset_index(drop=True)
    print(f"Slope filter (>= {SLOPE_FILTER_DEG} deg): {n_before:,} -> {len(vdf):,} rows "
          f"(dropped {n_before - len(vdf):,})")

    # ---- soiltype -> `type` (string; OOV -> learned embedding) ----------- #
    vdf["type"] = vdf["soiltype"].fillna(-1).astype(int).astype(str)

    # ---- Replay manifest transforms: log -> clip -> soil texture idx ----- #
    vdf = apply_log_transform(vdf, transform_meta["log_transformed_cols"])
    vdf = apply_clip_thresholds(vdf, transform_meta["clip_thresholds"])
    vdf = add_soil_texture_index(vdf)

    # ---- Labels (inference-only if `label` empty) ------------------------ #
    has_labels = (
        "label" in vdf.columns and vdf["label"].astype(str).str.strip().ne("").any()
    )
    if has_labels:
        vdf["landslide"] = pd.to_numeric(vdf["label"], errors="coerce").fillna(0).astype(int)
        print(f"Labels: {int(vdf['landslide'].sum()):,} positives / {len(vdf):,} rows")
    else:
        vdf["landslide"] = 0
        print("Labels: none (inference-only run)")

    missing = [c for c in input_cols if c not in vdf.columns]
    if missing:
        raise ValueError(f"Validation file is missing required model inputs: {missing}")

    ds = dataframe_to_dataset(
        vdf[input_cols + ["landslide"]].copy(), shuffle=False, batch_size=128,
    )

    # ---- Predict susceptibility ------------------------------------------ #
    out = model.predict(ds)
    susceptibility = out["final_head"].flatten() if isinstance(out, dict) else out.flatten()
    vdf["susceptibility"] = susceptibility

    # ---- Geotech + FOS + displacement ------------------------------------ #
    physics_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs={
            "fos": model.get_layer("fos_layer").output,
            "displacement": model.get_layer("displacement_layer").output,
            "cohesion": model.get_layer("cohesion_layer").output,
            "internal_friction": model.get_layer("internal_friction").output,
        },
    )
    phys = physics_extractor.predict(ds)
    for name in ("fos", "displacement", "cohesion", "internal_friction"):
        vdf[name] = np.asarray(phys[name]).reshape(len(vdf), -1)[:, 0]

    for name in ("fos", "displacement", "cohesion", "internal_friction"):
        v = pd.to_numeric(vdf[name], errors="coerce")
        print(f"{name:18s} med={np.nanmedian(v):.4f} min={np.nanmin(v):.4f} max={np.nanmax(v):.4f}")

    finite = np.isfinite(susceptibility)
    n_nan = int((~finite).sum())
    if n_nan:
        print(f"\n[!] {n_nan:,}/{len(susceptibility):,} predictions are NaN/inf.")
    s = susceptibility[finite]
    print(f"Susceptibility (finite): min={s.min():.4f} med={np.median(s):.4f} "
          f"mean={s.mean():.4f} max={s.max():.4f}")

    if has_labels and finite.any():
        from sklearn.metrics import roc_auc_score
        y = vdf.loc[finite, "landslide"].values
        if len(np.unique(y)) > 1:
            print(f"Validation AUC (production model): {roc_auc_score(y, s):.4f}")

    # ---- Save GPKG + PNG -------------------------------------------------- #
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    vdf.to_file(OUTPUT_PATH, driver="GPKG")
    save_susceptibility_map(
        vdf, "susceptibility", MAP_PATH,
        f"PINN v2-8 production — North Cotabato validation — Susceptibility (N={len(vdf):,})",
    )
    print(f"\nSaved predictions -> {OUTPUT_PATH}")
    print(f"Saved map         -> {MAP_PATH}")


if __name__ == "__main__":
    main()
