#!/usr/bin/env python
"""Map landslide susceptibility over the North Cotabato TRAINING region using
the PINN v2-8 PRODUCTION model (`production-model-v3.keras`).

This is the training-domain companion to `predict_south_mindanao_v2_8.py`. It
runs the production model over the full Cotabato slope-unit dataset that the
model was trained on, so you get a susceptibility map for the study area
itself.

Pipeline (identical prep to scripts/train_production_v2_8.py):
  preprocessing_v2 (slope filter, median impute, missingness indicators)
    -> add_soil_texture_index
    -> replay production manifest transforms (log -> clip)
    -> predict with production model
    -> extract intermediate physics (FOS, displacement, cohesion, friction)
    -> save GPKG + susceptibility PNG.

CAVEAT (in-sample optimism): the production model was trained on ~85% of these
rows (a 15% stratified holdout drove early stopping). Predictions here are
therefore partly in-sample and look better than true held-out performance. For
an unbiased number use the CV out-of-fold AUC from the notebook. This map is a
study-area deliverable, not a performance estimate.
"""

import os
import sys
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import geopandas as gpd
import tensorflow as tf
from tensorflow.keras.models import load_model

from py_files.helpers import set_seed, add_soil_texture_index
from py_files.data import (
    preprocessing_v2,
    apply_log_transform,
    apply_clip_thresholds,
    dataframe_to_dataset,
)
from py_files.GallenModel_v1 import NewmarkActivation

set_seed(42)

# --------------------------------------------------------------------------- #
# Config — mirrors scripts/train_production_v2_8.py
# --------------------------------------------------------------------------- #
FILE_PATH = Path(
    "~/Documents/ml-prep/ML-PREP-2025/learn/data/SU_17_training_v3_contri.gpkg"
).expanduser()
MODEL_DIR = Path(
    "~/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/"
    "trainedCotabatoPhase7/historical/v8"
).expanduser()
MODEL_PATH = MODEL_DIR / "production-model-v3.keras"
TRANSFORMS_PATH = PROJECT_ROOT / "feature_manifests" / "v1_cotabato_transforms_production.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "cotabato_susceptibility_v2_8.gpkg"
MAP_PATH = PROJECT_ROOT / "outputs" / "cotabato_susceptibility_map_v2_8.png"

COLUMNS_DROP = [
    "Landslide1", "descriptio", "sus_pinn_ground truth", "ds",
    "cohesion", "internal_friction", "sus_pinn_landslide",
    "confusion", "landslide_preds", "landslide_probability",
    "Lithology", "LITHO", "Geomorphology", "LITHODESC",
    "LITHO_2", "LITHODESC_2", "value",
]


def save_susceptibility_map(gdf, col, out_png, title, vmin=0.0, vmax=1.0):
    """Render a per-slope-unit susceptibility choropleth on a fixed [0,1] scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(8, 7))
    gdf.plot(column=col, cmap="plasma_r", ax=ax, norm=norm)

    sm = plt.cm.ScalarMappable(cmap="plasma_r", norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Susceptibility")
    ax.set_title(title)
    ax.set_axis_off()

    try:  # basemap is best-effort (needs network + projected CRS)
        import contextily as cx
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
    except Exception as exc:
        print(f"  [map] basemap skipped: {exc}")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found: {MODEL_PATH}\n"
            "Run scripts/train_production_v2_8.py first."
        )
    if not TRANSFORMS_PATH.exists():
        raise FileNotFoundError(f"Production manifest not found: {TRANSFORMS_PATH}")

    print(f"Model:    {MODEL_PATH.name}")
    print(f"Manifest: {TRANSFORMS_PATH.name}")
    model = load_model(MODEL_PATH, custom_objects={"NewmarkActivation": NewmarkActivation})
    with open(TRANSFORMS_PATH) as f:
        transform_meta = json.load(f)

    input_cols = [t.name.split(":")[0] for t in model.inputs]
    print(f"Model inputs: {input_cols}")

    # ---- Load & preprocess exactly as production training did ------------- #
    gdf_raw = gpd.read_file(FILE_PATH)
    print(f"\nLoaded {len(gdf_raw):,} rows from {FILE_PATH.name}")
    df, columns, numeric_cols, _, _ = preprocessing_v2(
        gdf_raw, columns_drop=COLUMNS_DROP, track_imputation=True,
    )

    # preprocessing_v2 keeps geometry on the returned GeoDataFrame; capture it
    # (aligned to the filtered index) before selecting model-input columns.
    geometry = df.geometry.copy()
    crs = df.crs

    df = add_soil_texture_index(df[columns].copy())

    # ---- Replay production manifest transforms: log -> clip --------------- #
    df = apply_log_transform(df, transform_meta["log_transformed_cols"])
    df = apply_clip_thresholds(df, transform_meta["clip_thresholds"])

    missing = [c for c in input_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required model inputs: {missing}")

    # ---- Predict ---------------------------------------------------------- #
    ds = dataframe_to_dataset(
        df[input_cols + ["landslide"]].copy(), shuffle=False, batch_size=128,
    )
    out = model.predict(ds)
    susceptibility = out["final_head"].flatten() if isinstance(out, dict) else out.flatten()

    # ---- Intermediate physics -------------------------------------------- #
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

    # ---- Assemble output GeoDataFrame (reattach geometry) ----------------- #
    out_gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)
    out_gdf["susceptibility"] = susceptibility
    for name in ("fos", "displacement", "cohesion", "internal_friction"):
        out_gdf[name] = np.asarray(phys[name]).reshape(len(out_gdf), -1)[:, 0]

    finite = np.isfinite(susceptibility)
    n_nan = int((~finite).sum())
    if n_nan:
        print(f"\n[!] {n_nan:,}/{len(susceptibility):,} predictions are NaN/inf.")
    s = susceptibility[finite]
    print(f"\nSusceptibility (finite): min={s.min():.4f} med={np.median(s):.4f} "
          f"mean={s.mean():.4f} max={s.max():.4f}")
    if "landslide" in out_gdf.columns:
        pos = out_gdf["landslide"].sum()
        print(f"Rows: {len(out_gdf):,}  (landslide inventory positives={int(pos):,})")

    # ---- Save GPKG + PNG -------------------------------------------------- #
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(OUTPUT_PATH, driver="GPKG")
    save_susceptibility_map(
        out_gdf, "susceptibility", MAP_PATH,
        f"PINN v2-8 production — North Cotabato — Susceptibility (N={len(out_gdf):,})",
    )
    print(f"\nSaved predictions -> {OUTPUT_PATH}")
    print(f"Saved map         -> {MAP_PATH}")


if __name__ == "__main__":
    main()
