#!/usr/bin/env python
"""Train the PINN v2-8 PRODUCTION model on the full Cotabato training set.

Background
----------
`cotabato_new_slope_unit_v2-8.ipynb` trains five models via 5-fold
StratifiedKFold. Cross-validation is an *evaluation* procedure: it estimates
how well the modelling recipe generalises (mean±std OOF AUC). Each fold model
sees only ~80% of the data and is tied to an arbitrary validation split, so
deploying a single fold (e.g. `fold-1-model-v3.keras`) for South Mindanao /
Cianjur inference is defensible only as a quick sanity check — not as the
final model.

This script produces the deployable artifact: ONE model trained on the full
training set with the identical architecture, class weights `{0:1, 1:5}`,
optimizer, Dice-CE loss, and early-stopping recipe. A small stratified holdout
(`--val-frac`, default 0.15) drives `val_final_head_auc` for early stopping;
everything else is training data.

Reproduction of the notebook pipeline (cells 10, 13, 18, 20, 21):
  1. `preprocessing_v2(track_imputation=True)` — slope filter, median impute,
     missingness indicators, imputation medians.
  2. `add_soil_texture_index` — USDA texture -> soil_texture_idx.
  3. Load the frozen GA-EN feature manifest `v1_cotabato.json`.
  4. `train_production_rainfall_v3` — derives per-run log/clip transforms from
     the training portion, writes `v1_cotabato_transforms_production.json`, and
     saves `production-model-v3.keras`.

The resulting model + manifest pair is what `predict_south_mindanao_v2_8.py`
loads for inference. Keep the two in sync: re-run this script whenever the
feature manifest or preprocessing changes, then re-run the prediction script.
"""

import os
import sys
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import geopandas as gpd

from py_files.helpers import set_seed, add_soil_texture_index
from py_files.data import preprocessing_v2
from py_files.train_rainfall_v3 import train_production_rainfall_v3

set_seed(42)

# --------------------------------------------------------------------------- #
# Config — mirrors cell 10 of cotabato_new_slope_unit_v2-8.ipynb
# --------------------------------------------------------------------------- #
FILE_PATH = Path(
    "~/Documents/ml-prep/ML-PREP-2025/learn/data/SU_17_training_v3_contri.gpkg"
).expanduser()
MODEL_SAVE_PATH = Path(
    "~/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/"
    "trainedCotabatoPhase7/historical/v8"
).expanduser()
FEATURE_MANIFEST_PATH = PROJECT_ROOT / "feature_manifests" / "v1_cotabato.json"
TRANSFORMS_DIR = PROJECT_ROOT / "feature_manifests"

# Same drop list and physics-feature set as the notebook (cells 13, 15).
COLUMNS_DROP = [
    "Landslide1", "descriptio", "sus_pinn_ground truth", "ds",
    "cohesion", "internal_friction", "sus_pinn_landslide",
    "confusion", "landslide_preds", "landslide_probability",
    "Lithology", "LITHO", "Geomorphology", "LITHODESC",
    "LITHO_2", "LITHODESC_2", "value",
]
PHYSICS_FEATURES = {
    "Slope_mean", "BUK_mean", "PGA2_max",          # Newmark inputs
    "Prc_mean", "ContributingFactor_mean",         # wetness layer inputs
    "SoilThc_mean",                                # soil thickness -> FOS
    "LULC_majority",                               # categorical-like
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-frac", type=float, default=0.15,
        help="Stratified holdout fraction used only for the early-stopping "
             "signal (default: 0.15). The rest is training data.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {FILE_PATH}")
    if not FEATURE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {FEATURE_MANIFEST_PATH}\n"
            "Run notebooks/feature_selection.ipynb first to generate it."
        )
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load & preprocess (notebook cell 13) ------------------------- #
    df = gpd.read_file(FILE_PATH)
    print(f"Raw dataset: {len(df):,} rows")
    df, columns, numeric_cols, imputed_indicator_cols, imputation_medians = preprocessing_v2(
        df, columns_drop=COLUMNS_DROP, track_imputation=True,
    )
    print(f"Imputation indicators: {imputed_indicator_cols}")

    # ---- 2. Soil texture index (notebook cell 18) ------------------------ #
    df = add_soil_texture_index(df[columns].copy())

    # ---- 3. Load frozen GA-EN feature manifest (notebook cell 20) -------- #
    with open(FEATURE_MANIFEST_PATH) as f:
        manifest = json.load(f)
    pga_col = manifest["pga_col"]
    selected_numerical = manifest["final_features"]["numerical"]
    selected_categorical = manifest["final_features"]["categorical"]

    # Imputation indicators are first-class features; append any not already
    # selected (survives only if their source column is still selected).
    indicator_extras = [c for c in imputed_indicator_cols if c not in selected_numerical]
    selected_numerical = selected_numerical + indicator_extras
    selected_feature_cols = selected_numerical + selected_categorical + ["landslide"]

    print(f"PGA column:         {pga_col}")
    print(f"Numerical features: {selected_numerical}")
    print(f"Categorical:        {selected_categorical}")

    # ---- 4. Train production model (notebook cell 21, no CV) ------------- #
    model_path, holdout_auc = train_production_rainfall_v3(
        df,
        selected_numerical,
        selected_categorical,
        selected_feature_cols,
        pga_col,
        str(MODEL_SAVE_PATH),
        epochs=args.epochs,
        batch_size=args.batch_size,
        physics_features=PHYSICS_FEATURES,
        skew_threshold=1.0,
        clip_lower_pct=1,
        clip_upper_pct=99,
        transforms_dir=TRANSFORMS_DIR,
        categorical_encoder="embedding",
        imputed_indicator_cols=imputed_indicator_cols,
        imputation_medians=imputation_medians,
        val_frac=args.val_frac,
    )

    print("\n" + "=" * 70)
    print("PRODUCTION MODEL READY")
    print("=" * 70)
    print(f"Model:    {model_path}")
    print(f"Manifest: {TRANSFORMS_DIR / 'v1_cotabato_transforms_production.json'}")
    print(f"Holdout AUC: {holdout_auc:.4f}")
    print("\nNext: run scripts/predict_south_mindanao_v2_8.py (now points at the")
    print("production model + manifest) to map South Mindanao susceptibility.")


if __name__ == "__main__":
    main()
