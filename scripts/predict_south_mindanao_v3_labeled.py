#!/usr/bin/env python
"""Run the PRODUCTION Cotabato PINN on the LABELED South Mindanao v3 dataset.

Because the production model was trained only on Cotabato, every South Mindanao
slope unit is a genuine held-out test — so joining the v3 `landslide` inventory
to the predictions gives an honest external validation of the zero-shot transfer.

Replays the exact production inference pipeline used in
`scripts/predict_south_mindanao_v2_8.py` (rename -> unit convert -> zeros->NaN ->
impute training medians -> slope filter -> soil-type collapse -> log -> clip ->
soil-texture index), but reads v3 (which carries the label) and keeps it aligned.

Outputs (row-aligned on the slope-filtered v3 set):
  outputs/sm_v3_prod_analysis.csv    features (raw, pre log/clip) + type +
                                     soil_texture + susceptibility + physics
                                     (fos, displacement, cohesion,
                                     internal_friction) + landslide
  outputs/sm_v3_domainshift.json     per-feature SM vs training median audit
  figures/sm_v3_susceptibility_classes.png   5-class susceptibility choropleth
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

from py_files.helpers import set_seed, add_soil_texture_index
from py_files.data import (
    apply_log_transform,
    apply_clip_thresholds,
    apply_missingness_indicators,
    apply_imputation_medians,
    dataframe_to_dataset,
)
from py_files.GallenModel_v1 import NewmarkActivation

# Reuse the vetted transfer config so the pipeline stays identical.
import predict_south_mindanao_v2_8 as base

set_seed(42)

INPUT_PATH = PROJECT_ROOT / "datasets" / "SlopeUnit_SouthMindanao_v3.gpkg"
LAYER = "joined_layer"
MODEL_PATH = base.MODEL_DIR / "production-model-v3.keras"
TRANSFORMS_PATH = PROJECT_ROOT / "feature_manifests" / "v1_cotabato_transforms_production.json"
OUT_CSV = PROJECT_ROOT / "outputs" / "sm_v3_prod_analysis.csv"
OUT_JSON = PROJECT_ROOT / "outputs" / "sm_v3_domainshift.json"
OUT_MAP = PROJECT_ROOT / "figures" / "sm_v3_susceptibility_classes.png"

PHYSICS_LAYERS = {
    "fos": "fos_layer",
    "displacement": "displacement_layer",
    "cohesion": "cohesion_layer",
    "internal_friction": "internal_friction",
}
CLASS_BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
CLASS_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]


def plot_class_map(gdf, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(CLASS_BOUNDS, cmap.N)
    fig, ax = plt.subplots(figsize=(9, 8))
    gdf.plot(column="susceptibility", cmap=cmap, norm=norm, ax=ax)
    ax.set_axis_off()
    ax.set_title("South Mindanao — susceptibility class (production PINN, zero-shot)",
                 fontsize=13, fontweight="bold")
    handles = [Patch(facecolor=c, edgecolor="none", label=l)
               for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
    ax.legend(handles=handles, title="Susceptibility", loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Model:    {MODEL_PATH}")
    model = load_model(MODEL_PATH, custom_objects={"NewmarkActivation": NewmarkActivation})
    input_cols = [t.name.split(":")[0] for t in model.inputs]
    print(f"Model inputs ({len(input_cols)}): {input_cols}")

    with open(TRANSFORMS_PATH) as f:
        tm = json.load(f)
    training_medians = tm.get("imputation_medians", {})

    src_cols = list(base.SM_RENAME.keys()) + ["landslide"]
    gdf = gpd.read_file(INPUT_PATH, layer=LAYER, columns=src_cols)
    print(f"\nLoaded {len(gdf):,} rows | positives: {int(gdf['landslide'].sum()):,}")
    gdf = gdf.rename(columns=base.SM_RENAME)

    for col, factor in base.UNIT_CONVERSIONS.items():
        if col in gdf.columns and factor != 1.0:
            gdf[col] = gdf[col] * factor
    for col in base.ZERO_IS_MISSING:
        if col in gdf.columns:
            gdf.loc[gdf[col] == 0, col] = np.nan

    gdf = apply_missingness_indicators(gdf, tm.get("imputed_indicator_cols", []))
    gdf = apply_imputation_medians(gdf, training_medians)

    # Slope filter to match the training domain.
    n_before = len(gdf)
    gdf = gdf[gdf["Slope_mean"] >= base.SLOPE_FILTER_DEG].reset_index(drop=True)
    print(f"Slope filter (>= {base.SLOPE_FILTER_DEG} deg): {n_before:,} -> {len(gdf):,} "
          f"(positives: {int(gdf['landslide'].sum()):,})")

    gdf["type"] = gdf["type"].map(base.map_soil_type)
    gdf["Elev_raw"] = gdf["Elev_mean"]

    # soil_texture_idx is a model input; derive it (from Clay/Sand/Silt) first.
    gdf = add_soil_texture_index(gdf)

    # Safety-net impute for any residual NaN in model-input numerics.
    numeric_inputs = [c for c in input_cols if c != "type"]
    resid = gdf[numeric_inputs].isnull().sum()
    if int(resid.sum()) > 0:
        gdf[numeric_inputs] = gdf[numeric_inputs].fillna(gdf[numeric_inputs].median(numeric_only=True))

    # ---- Snapshot RAW features (post-impute, PRE log/clip) for domain-shift ---- #
    raw_snapshot = gdf[numeric_inputs + ["type", "soil_texture", "landslide"]].copy()

    # ---- Domain-shift audit (SM vs training medians) ---- #
    audit = {}
    for col in numeric_inputs:
        tmed = training_medians.get(col)
        smed = float(np.nanmedian(gdf[col]))
        entry = {"sm_median": smed, "train_median": tmed}
        if tmed not in (None, 0):
            entry["ratio"] = smed / tmed
        cb = tm["clip_thresholds"].get(col)
        if cb is not None:
            lo, hi = cb
            entry["pct_below_clip"] = float((gdf[col] < lo).mean() * 100)
            entry["pct_above_clip"] = float((gdf[col] > hi).mean() * 100)
        audit[col] = entry
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Saved domain-shift audit -> {OUT_JSON}")

    # ---- Replay manifest transforms then predict ---- #
    gdf = apply_log_transform(gdf, tm["log_transformed_cols"])
    gdf = apply_clip_thresholds(gdf, tm["clip_thresholds"])

    ds = dataframe_to_dataset(gdf[input_cols + ["landslide"]].copy(),
                              shuffle=False, batch_size=256)
    out = model.predict(ds, verbose=0)
    susceptibility = out["final_head"].flatten() if isinstance(out, dict) else out.flatten()
    gdf["susceptibility"] = susceptibility

    physics_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs={k: model.get_layer(v).output for k, v in PHYSICS_LAYERS.items()},
    )
    phys = physics_extractor.predict(ds, verbose=0)
    for name in PHYSICS_LAYERS:
        gdf[name] = np.asarray(phys[name]).reshape(len(gdf), -1)[:, 0]

    y = gdf["landslide"].to_numpy()
    finite = np.isfinite(susceptibility)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(y[finite], susceptibility[finite])
        ap = average_precision_score(y[finite], susceptibility[finite])
        print(f"\nEXTERNAL VALIDATION (zero-shot transfer):")
        print(f"  prevalence: {y.mean():.4%}  positives={int(y.sum()):,}/{len(y):,}")
        print(f"  ROC AUC   = {auc:.4f}")
        print(f"  PR  AUC   = {ap:.4f}  (baseline {y.mean():.4f})")
    except Exception as exc:
        print(f"  [warn] metric computation failed: {exc}")

    # ---- Save analysis table (raw features + predictions + label) ---- #
    analysis = raw_snapshot.copy()
    for c in ["susceptibility"] + list(PHYSICS_LAYERS):
        analysis[c] = gdf[c].to_numpy()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(OUT_CSV, index=False)
    print(f"Saved analysis table -> {OUT_CSV}  ({analysis.shape})")

    # ---- Classified susceptibility map ---- #
    try:
        plot_class_map(gdf, OUT_MAP)
        print(f"Saved class map -> {OUT_MAP}")
    except Exception as exc:
        print(f"  [warn] class map skipped: {exc}")


if __name__ == "__main__":
    main()
