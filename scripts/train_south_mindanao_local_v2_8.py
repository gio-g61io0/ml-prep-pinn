#!/usr/bin/env python
"""Train two region-specific South Mindanao PINNs on the labeled v3 inventory and
score them out-of-fold, isolating rainfall as the only difference:

  - BASELINE  (use_rainfall=True) : rainfall pore-pressure physics ON  (original v3)
  - PURE-EIL  (use_rainfall=False): rainfall fully disconnected (dry static FoS +
                                    Prc kept out of the residual DNN)

Both are trained on the SAME undersampled frame, SAME 5 folds, SAME seed, so any
difference in held-out skill is attributable to rainfall. Reproduces the vetted
preprocessing from notebooks/south_mindanao_train_v2_8.ipynb.

Outputs per model (row-aligned on the undersampled training frame):
  outputs/sm_local_{tag}_scored.csv   features + type + soil_texture + landslide
                                      + susceptibility_oof (honest held-out) + physics
  trained_models/south_mindanao_{tag}_v2_8/   fold checkpoints + oof_preds.npy
  feature_manifests/south_mindanao_{tag}_v2_8/ per-fold transform manifests
"""

import os
import sys
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, average_precision_score

from py_files.helpers import set_seed, add_soil_texture_index
from py_files.data import apply_log_transform, apply_clip_thresholds, dataframe_to_dataset
from py_files.train_rainfall_v3 import train_model_rainfall_v3
from py_files.GallenModel_v1 import NewmarkActivation
import predict_south_mindanao_v2_8 as base

INPUT_PATH = PROJECT_ROOT / "datasets" / "SlopeUnit_SouthMindanao_v3.gpkg"
LAYER = "joined_layer"
SLOPE_FILTER_DEG = 10.0
NEG_PER_POS = 20

SELECTED_NUMERICAL = [
    "Slope_mean", "BUK_mean", "Prc_mean", "ContributingFactor_mean",
    "SoilThc_mean", "soil_texture_idx", "PGA2_max", "Elev_mean",
]
SELECTED_CATEGORICAL = ["type"]
PGA_COL = "PGA2_max"
PHYSICS_FEATURES = {
    "Slope_mean", "BUK_mean", "PGA2_max",
    "Prc_mean", "ContributingFactor_mean", "SoilThc_mean",
}
PHYSICS_LAYERS = {  # output name -> candidate layer names
    "fos": ["fos_layer"],
    "cohesion": ["cohesion_layer"],
    "internal_friction": ["internal_friction_layer", "internal_friction"],
    "displacement": ["displacement_intermediate", "displacement_layer"],
}


def build_frame():
    """Load v3, preprocess exactly as the notebook, undersample. Returns
    (gdf_train, df, feature_cols, imputation_medians)."""
    src_cols = list(base.SM_RENAME.keys()) + ["landslide"]
    gdf = gpd.read_file(INPUT_PATH, layer=LAYER, columns=src_cols)
    print(f"Raw v3: {len(gdf):,} rows | positives {int(gdf['landslide'].sum()):,}")
    gdf = gdf.rename(columns=base.SM_RENAME)

    for col, factor in base.UNIT_CONVERSIONS.items():
        if col in gdf.columns and factor != 1.0:
            gdf[col] = gdf[col] * factor
    for col in base.ZERO_IS_MISSING:
        if col in gdf.columns:
            gdf.loc[gdf[col] == 0, col] = np.nan
    gdf["type"] = gdf["type"].map(base.map_soil_type)

    n_before = len(gdf)
    gdf = gdf[gdf["Slope_mean"] >= SLOPE_FILTER_DEG].reset_index(drop=True)
    print(f"Slope filter >= {SLOPE_FILTER_DEG}: {n_before:,} -> {len(gdf):,} "
          f"(positives {int(gdf['landslide'].sum()):,})")

    impute_cols = [c for c in SELECTED_NUMERICAL if c != "soil_texture_idx"] + \
                  ["Clay_mean", "Sand_mean", "Silt_mean"]
    impute_cols = [c for c in dict.fromkeys(impute_cols) if c in gdf.columns]
    medians = gdf[impute_cols].median(numeric_only=True)
    imputation_medians = {c: float(medians[c]) for c in impute_cols if pd.notna(medians[c])}
    gdf[impute_cols] = gdf[impute_cols].fillna(medians)

    gdf = add_soil_texture_index(gdf)
    assert gdf["soil_texture_idx"].notna().all()

    pos = gdf[gdf["landslide"] == 1]
    neg = gdf[gdf["landslide"] == 0].sample(
        n=min(int((gdf["landslide"] == 0).sum()), NEG_PER_POS * len(pos)), random_state=42)
    gdf_train = pd.concat([pos, neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Undersampled frame: {len(gdf_train):,} rows | positives "
          f"{int(gdf_train['landslide'].sum()):,} ({gdf_train['landslide'].mean():.2%})")

    feature_cols = SELECTED_NUMERICAL + SELECTED_CATEGORICAL + ["landslide"]
    df = pd.DataFrame(gdf_train[feature_cols]).copy()
    assert df[SELECTED_NUMERICAL].isna().sum().sum() == 0, "NaNs remain in numeric inputs"
    return gdf_train, df, feature_cols, imputation_medians


def extract_physics(model_dir, transforms_dir, gdf_train, feature_cols, best_fold):
    """Extract intermediate physics fields from the best fold, on the training frame."""
    with open(Path(transforms_dir) / f"v1_cotabato_transforms_fold{best_fold}.json") as f:
        tm = json.load(f)
    m = load_model(f"{model_dir}/fold-{best_fold}-model-v3.keras",
                   custom_objects={"NewmarkActivation": NewmarkActivation})
    input_cols = [t.name.split(":")[0] for t in m.inputs]

    fx = gdf_train[feature_cols].copy()
    fx = apply_log_transform(fx, tm["log_transformed_cols"])
    fx = apply_clip_thresholds(fx, tm["clip_thresholds"])
    ds = dataframe_to_dataset(fx[input_cols + ["landslide"]], shuffle=False, batch_size=512)

    outputs, resolved = {}, {}
    layer_names = {l.name for l in m.layers}
    for out_name, candidates in PHYSICS_LAYERS.items():
        for cand in candidates:
            if cand in layer_names:
                outputs[out_name] = m.get_layer(cand).output
                resolved[out_name] = cand
                break
    print(f"  physics layers resolved: {resolved}")
    extractor = tf.keras.Model(inputs=m.inputs, outputs=outputs)
    phys = extractor.predict(ds, verbose=0)
    return {k: np.asarray(v).reshape(len(gdf_train), -1)[:, 0] for k, v in phys.items()}


def train_and_score(tag, use_rainfall, gdf_train, df, feature_cols, imputation_medians):
    print("\n" + "=" * 70)
    print(f"TRAINING [{tag}]  use_rainfall={use_rainfall}")
    print("=" * 70)
    model_dir = PROJECT_ROOT / "trained_models" / f"south_mindanao_{tag}_v2_8"
    transforms_dir = PROJECT_ROOT / "feature_manifests" / f"south_mindanao_{tag}_v2_8"
    model_dir.mkdir(parents=True, exist_ok=True)
    transforms_dir.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    oof_preds, fold_aucs = train_model_rainfall_v3(
        df, SELECTED_NUMERICAL, SELECTED_CATEGORICAL, feature_cols, PGA_COL,
        str(model_dir), physics_features=PHYSICS_FEATURES,
        skew_threshold=1.0, clip_lower_pct=1, clip_upper_pct=99,
        transforms_dir=str(transforms_dir), categorical_encoder="embedding",
        imputed_indicator_cols=[], imputation_medians=imputation_medians,
        use_rainfall=use_rainfall,
    )
    oof = np.asarray(oof_preds, dtype=float)
    y = df["landslide"].to_numpy()
    oof_auc = roc_auc_score(y, oof)
    oof_ap = average_precision_score(y, oof)
    print(f"[{tag}] per-fold AUC: {[round(a, 4) for a in fold_aucs]}")
    print(f"[{tag}] mean fold AUC: {np.mean(fold_aucs):.4f} | "
          f"OOF AUC: {oof_auc:.4f} | OOF PR-AUC: {oof_ap:.4f} "
          f"(prevalence {y.mean():.3%})")

    best_fold = int(np.argmax(fold_aucs)) + 1
    tf.keras.backend.clear_session()
    physics = extract_physics(model_dir, transforms_dir, gdf_train, feature_cols, best_fold)

    scored = gdf_train.drop(columns=[c for c in ["geometry"] if c in gdf_train.columns]).copy()
    scored["susceptibility_oof"] = oof
    for k, v in physics.items():
        scored[k] = v
    out_csv = PROJECT_ROOT / "outputs" / f"sm_local_{tag}_scored.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_csv, index=False)
    print(f"[{tag}] saved -> {out_csv}  ({scored.shape})")
    tf.keras.backend.clear_session()
    return {"tag": tag, "fold_aucs": [float(a) for a in fold_aucs],
            "oof_auc": float(oof_auc), "oof_ap": float(oof_ap),
            "prevalence": float(y.mean())}


def main():
    set_seed(42)
    gdf_train, df, feature_cols, imputation_medians = build_frame()
    summary = []
    summary.append(train_and_score("rain", True, gdf_train, df, feature_cols, imputation_medians))
    summary.append(train_and_score("eil", False, gdf_train, df, feature_cols, imputation_medians))

    out_json = PROJECT_ROOT / "outputs" / "sm_local_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 70)
    print("SUMMARY (held-out OOF on undersampled frame)")
    for s in summary:
        print(f"  {s['tag']:5s}  OOF AUC={s['oof_auc']:.4f}  PR-AUC={s['oof_ap']:.4f}")
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
