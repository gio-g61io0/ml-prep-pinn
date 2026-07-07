#!/usr/bin/env python
"""Predict landslide susceptibility on the South Mindanao slope-unit dataset
using the PINN v3 model trained in `cotabato_new_slope_unit_v2-8.ipynb`.

By default this loads the PRODUCTION model (`production-model-v3.keras`,
trained on the full Cotabato set by `scripts/train_production_v2_8.py`) and its
paired manifest (`v1_cotabato_transforms_production.json`). It replays the same
inference pipeline the notebook uses (median fill -> slope filter -> log ->
clip -> soil texture index) so the trained NormalizationLayer sees values on the
distribution it was adapted on. Set USE_PRODUCTION = False to fall back to a
single CV fold for a quick sanity check.

UNIT / SCALE RECONCILIATION (South Mindanao v2)
-----------------------------------------------
The South Mindanao file uses different column NAMES and, for some physics
features, different UNITS than the Cotabato training data. Names are remapped via
SM_RENAME; units via UNIT_CONVERSIONS. Scale check vs. the fold-1 training
medians (see the plan / `feature_manifests/v1_cotabato_transforms_fold1.json`):

    feature (SM col)              training med   conversion   post-conv med
    Slope_mean  (Slope_mean)      21.83 deg      none         8.9   (flatter terrain, not a unit issue)
    BUK_mean    (buk_mean)        11.29          x9.81/100    ~11.7 (g/cm^3*100 -> kN/m^3)
    PGA2_max    (PGA_mean)        0.225          NONE (x1.0)  ~0.36 (already in model's unit; NOT /100 like Cianjur)
    Elev_mean   (Elev_mean)       592.8 m        none         254   (lowland; clipped to [74,1936] by manifest)
    Prc_mean    (Precip_mea)      223.5 mm       x1000        ~134  (m -> mm)
    SoilThc_mean(SoilThickn)      13.09 m        x0.001       ~2.5  (mm -> m; ~5x gap REMAINS, accepted)
    ContributingFactor_mean(CatchmentA) 935     NONE (x1.0)  ~486  (median ok; extreme tail left as-is, no clip)
    Clay/Sand/Silt               ~370/343/287   none         near-identical -> soil_texture_idx

The `type` categorical arrives as `SoilType` (25 Philippine soil-series names)
and is collapsed to the model's 3-class vocabulary via `map_soil_type`.

The script prints a UNIT AUDIT comparing post-conversion medians against training
medians so you can verify each feature lands in-distribution before trusting
predictions. The SoilThc_mean ~5x gap is expected and documented.
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

set_seed(42)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
INPUT_PATH = PROJECT_ROOT / "datasets" / "SlopeUnit_SouthMindanao_v2.gpkg"
LAYER = "joined_layer"
MODEL_DIR = Path(
    "~/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/"
    "trainedCotabatoPhase7/historical/v8"
).expanduser()
# Production model: ONE model trained on the full Cotabato set (see
# scripts/train_production_v2_8.py), not an arbitrary CV fold. Model + manifest
# must always be replayed as a pair. To fall back to a single CV fold for a
# quick sanity check, set USE_PRODUCTION = False and pick a FOLD.
USE_PRODUCTION = True
FOLD = 1  # only used when USE_PRODUCTION is False
if USE_PRODUCTION:
    MODEL_PATH = MODEL_DIR / "production-model-v3.keras"
    TRANSFORMS_PATH = PROJECT_ROOT / "feature_manifests" / "v1_cotabato_transforms_production.json"
else:
    MODEL_PATH = MODEL_DIR / f"fold-{FOLD}-model-v3.keras"
    TRANSFORMS_PATH = PROJECT_ROOT / "feature_manifests" / f"v1_cotabato_transforms_fold{FOLD}.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "south_mindanao_susceptibility_v2_8.gpkg"

SLOPE_FILTER_DEG = 10.0  # training set was filtered to slopes >= ~10.5 deg

# South Mindanao column name -> training schema name.
SM_RENAME = {
    "Slope_mean": "Slope_mean",
    "buk_mean": "BUK_mean",
    "PGA_mean": "PGA2_max",
    "Precip_mea": "Prc_mean",
    "CatchmentA": "ContributingFactor_mean",
    "SoilThickn": "SoilThc_mean",
    "Elev_mean": "Elev_mean",
    "Clay_mean": "Clay_mean",
    "Silt_mean": "Silt_mean",
    "Sand_mean": "Sand_mean",
    "SoilType": "type",  # collapsed to the 3-class training vocabulary by map_soil_type()
}

# Per-feature multiplier applied AFTER rename, BEFORE the manifest transforms.
UNIT_CONVERSIONS = {
    "BUK_mean": 9.81 / 100.0,  # g/cm^3 * 100  ->  kN/m^3 unit weight (matches v2-8 / Cianjur)
    "Prc_mean": 1000.0,        # m -> mm (SM ~0.13 m -> ~134 mm; training median ~223)
    "SoilThc_mean": 0.001,     # mm -> m (SM 2487 -> ~2.5 m; training median ~13, ~5x gap accepted)
    # PGA2_max: NOT converted — SM PGA_mean is already in the model's unit (~0.36 vs 0.225).
    #           (Cianjur needed /100 because it was in cm/s^2; South Mindanao does not.)
    # ContributingFactor_mean: left as-is per user; median matches (~486 vs ~935), extreme tail unclipped.
}

# Columns where a literal 0 means "missing" rather than a true measurement.
# Zeros are set to NaN before the training-median fill so they get imputed
# instead of poisoning the physics: a 0 bulk density or soil thickness sits in
# the FOS denominator (bulk_density * thickness * sin(slope)) -> divide-by-zero.
# Clay/Sand/Silt are added here so all-zero granulometry rows don't corrupt the
# USDA soil_texture_idx derivation.
ZERO_IS_MISSING = {"SoilThc_mean", "BUK_mean", "Clay_mean", "Sand_mean", "Silt_mean"}


def map_soil_type(value):
    """Collapse a South Mindanao soil-series name into the model's 3-class `type`
    vocabulary {Undifferentiated, Sandy Clay Loam, Loam}.

    The trained StringLookup was adapted only on those three classes, so any
    other label falls to the OOV embedding. Rule (case-insensitive):
      - undifferentiated / Mountain soil / Hydrosol / peat / null -> Undifferentiated
      - contains "clay" (clay, clay loam, silty clay loam)         -> Sandy Clay Loam
      - everything else (loam, sandy loam, loamy sand, silt loam)  -> Loam
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Undifferentiated"
    s = str(value).lower()
    if ("undifferentiated" in s or "undefferentiated" in s
            or "mountain soil" in s or "hydrosol" in s or "peat" in s):
        return "Undifferentiated"
    if "clay" in s:
        return "Sandy Clay Loam"
    return "Loam"


def print_unit_audit(df, training_medians):
    """Compare each model-input feature's median against the training median."""
    print("\n" + "=" * 78)
    print("UNIT AUDIT — post-conversion South Mindanao median vs training median")
    print("=" * 78)
    print(f"{'feature':28s} {'train_med':>12s} {'sm_med':>12s} {'ratio':>8s}  flag")
    print("-" * 78)
    for col in [
        "Slope_mean", "Elev_mean", "BUK_mean", "PGA2_max",
        "Prc_mean", "ContributingFactor_mean", "SoilThc_mean",
        "Clay_mean", "Silt_mean", "Sand_mean",
    ]:
        if col not in df.columns:
            continue
        cmed = float(np.nanmedian(df[col]))
        tmed = training_medians.get(col)
        if tmed is None or tmed == 0:
            ratio_str, flag = "  n/a", ""
        else:
            ratio = cmed / tmed
            ratio_str = f"{ratio:8.3f}"
            flag = "  <-- OFF" if (ratio < 0.5 or ratio > 2.0) else "  ok"
        tmed_str = f"{tmed:12.4f}" if tmed is not None else f"{'n/a':>12s}"
        print(f"{col:28s} {tmed_str} {cmed:12.4f} {ratio_str}{flag}")
    print("=" * 78 + "\n")


def save_choropleth(gdf, col, out_png, title, cbar_label,
                    cmap="plasma_r", vmin=None, vmax=None):
    """Render and save a per-slope-unit choropleth for any continuous column.

    vmin/vmax default to the robust 2nd/98th percentiles of the finite values
    so outliers don't wash out the color scale. Pass explicit bounds for
    fixed-range fields like [0, 1].
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: save to file, never block on plt.show()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    vals = pd.to_numeric(gdf[col], errors="coerce")
    finite = vals[np.isfinite(vals)]
    lo = vmin if vmin is not None else (float(np.percentile(finite, 2)) if len(finite) else 0.0)
    hi = vmax if vmax is not None else (float(np.percentile(finite, 98)) if len(finite) else 1.0)
    if hi <= lo:
        hi = lo + 1e-6
    norm = mcolors.Normalize(vmin=lo, vmax=hi)

    fig, ax = plt.subplots(figsize=(8, 7))
    gdf.plot(column=col, cmap=cmap, ax=ax, norm=norm)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_axis_off()

    try:  # basemap is best-effort (needs network + projected CRS)
        import contextily as cx
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
    except Exception as exc:
        print(f"  [map] basemap skipped for {col}: {exc}")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    # ---- Load model & manifest -------------------------------------------- #
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not TRANSFORMS_PATH.exists():
        raise FileNotFoundError(f"Transform manifest not found: {TRANSFORMS_PATH}")

    print(f"Model:    {MODEL_PATH}")
    print(f"Manifest: {TRANSFORMS_PATH.name}")
    model = load_model(MODEL_PATH, custom_objects={"NewmarkActivation": NewmarkActivation})

    with open(TRANSFORMS_PATH) as f:
        transform_meta = json.load(f)
    training_medians = transform_meta.get("imputation_medians", {})

    input_cols = [t.name.split(":")[0] for t in model.inputs]
    print(f"Model inputs: {input_cols}")

    # ---- Load South Mindanao data ----------------------------------------- #
    gdf = gpd.read_file(INPUT_PATH, layer=LAYER)
    print(f"\nLoaded {len(gdf):,} rows from {INPUT_PATH.name} (layer={LAYER})")

    missing_src = [c for c in SM_RENAME if c not in gdf.columns]
    if missing_src:
        raise ValueError(f"South Mindanao file is missing expected source columns: {missing_src}")

    gdf = gdf.rename(columns=SM_RENAME)

    # ---- Unit conversions ------------------------------------------------- #
    for col, factor in UNIT_CONVERSIONS.items():
        if col in gdf.columns and factor != 1.0:
            gdf[col] = gdf[col] * factor
            print(f"  unit-converted {col} x {factor:g}")

    # ---- Treat configured zeros as missing -------------------------------- #
    for col in ZERO_IS_MISSING:
        if col in gdf.columns:
            n_zero = int((gdf[col] == 0).sum())
            if n_zero:
                gdf.loc[gdf[col] == 0, col] = np.nan
                print(f"  {col}: {n_zero:,} zero values -> NaN (will be median-filled)")

    # ---- Missingness indicators (no-op for fold-1; empty list) ------------ #
    gdf = apply_missingness_indicators(gdf, transform_meta.get("imputed_indicator_cols", []))

    # ---- Fill NaNs from TRAINING medians (Mechanism A) -------------------- #
    gdf = apply_imputation_medians(gdf, training_medians)

    # Safety net: any residual NaN among mapped numeric features -> SM median.
    impute_cols = [c for c in SM_RENAME.values() if c in gdf.columns and c != "type"]
    residual_na = gdf[impute_cols].isnull().sum()
    if int(residual_na.sum()) > 0:
        print(f"  [warn] residual NaNs after manifest fill: "
              f"{residual_na[residual_na > 0].to_dict()}; using South Mindanao median")
        gdf[impute_cols] = gdf[impute_cols].fillna(gdf[impute_cols].median(numeric_only=True))

    # ---- Slope filter (match training domain) ----------------------------- #
    n_before = len(gdf)
    gdf = gdf[gdf["Slope_mean"] >= SLOPE_FILTER_DEG].reset_index(drop=True)
    print(f"Slope filter (>= {SLOPE_FILTER_DEG} deg): {n_before:,} -> {len(gdf):,} rows "
          f"(dropped {n_before - len(gdf):,})")

    # ---- SoilType -> `type` (collapse 25 series to the 3-class vocabulary) - #
    gdf["type"] = gdf["type"].map(map_soil_type)
    print(f"  type distribution: {gdf['type'].value_counts(dropna=False).to_dict()}")

    # Keep raw (post-impute, pre-clip) elevation for cross-checking the maps.
    # The model's Elev_mean gets clipped to [74, 1936], which would flatten peaks.
    gdf["Elev_raw"] = gdf["Elev_mean"]

    # ---- Replay manifest transforms: log -> clip -> soil texture idx ------ #
    gdf = apply_log_transform(gdf, transform_meta["log_transformed_cols"])
    gdf = apply_clip_thresholds(gdf, transform_meta["clip_thresholds"])
    gdf = add_soil_texture_index(gdf)

    print_unit_audit(gdf, training_medians)

    # ---- No landslide inventory for South Mindanao: dummy label column ----- #
    gdf["landslide"] = 0

    # ---- Build dataset from the model's actual input schema --------------- #
    missing = [c for c in input_cols if c not in gdf.columns]
    if missing:
        raise ValueError(f"South Mindanao data is missing required model inputs: {missing}")

    ds = dataframe_to_dataset(
        gdf[input_cols + ["landslide"]].copy(), shuffle=False, batch_size=128
    )

    # ---- Predict ---------------------------------------------------------- #
    out = model.predict(ds)
    susceptibility = out["final_head"].flatten() if isinstance(out, dict) else out.flatten()
    gdf["susceptibility"] = susceptibility

    # ---- Intermediate physics: FOS, displacement, cohesion, friction ------ #
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
        gdf[name] = np.asarray(phys[name]).reshape(len(gdf), -1)[:, 0]

    for name, unit in [("FOS", ""), ("Displacement", " cm"), ("Cohesion", ""),
                       ("Internal_friction", "")]:
        col = name.lower()
        print(f"{name:18s} med={np.nanmedian(gdf[col]):.4f}{unit} "
              f"min={np.nanmin(gdf[col]):.4f} max={np.nanmax(gdf[col]):.4f}")

    finite = np.isfinite(susceptibility)
    n_nan = int((~finite).sum())
    if n_nan:
        print(f"\n[!] {n_nan:,}/{len(susceptibility):,} predictions are NaN/inf.")
        print("    Likely a 0 / out-of-range physics input feeding a divide in the")
        print("    Newmark FOS (bulk density, soil thickness, or slope). Check the")
        print("    UNIT_CONVERSIONS and ZERO_IS_MISSING config and rerun.")
    if finite.any():
        s = susceptibility[finite]
        print(f"Susceptibility (finite only): min={s.min():.4f} "
              f"med={np.median(s):.4f} max={s.max():.4f}")

    # ---- Save ------------------------------------------------------------- #
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_gdf = gdf.drop(columns=[c for c in ["landslide"] if c in gdf.columns])
    out_gdf.to_file(OUTPUT_PATH, driver="GPKG")
    csv_path = OUTPUT_PATH.with_suffix(".csv")
    out_gdf.drop(columns="geometry").to_csv(csv_path, index=False)

    # ---- Maps: susceptibility (fixed 0-1) + physics fields (auto-scaled) -- #
    figs_dir = OUTPUT_PATH.parent
    stem = "south_mindanao"
    maps = [
        ("susceptibility",     "Susceptibility",            "plasma_r", 0.0, 1.0),
        ("fos",                "Factor of Safety",          "RdYlGn",   None, None),
        ("PGA2_max",           "PGA",                       "inferno",  None, None),
        ("cohesion",           "Cohesion (model units)",    "viridis",  None, None),
        ("internal_friction",  "Internal friction (model units)", "cividis", None, None),
        ("Elev_raw",           "Elevation (m)",             "terrain",  None, None),
        ("Slope_mean",         "Slope (deg)",               "YlOrRd",   None, None),
    ]
    map_paths = []
    for col, label, cmap, vmn, vmx in maps:
        png = figs_dir / f"{stem}_{col}_map_v2_8.png"
        save_choropleth(out_gdf, col, png, f"PINN v8 — South Mindanao — {label}",
                        label, cmap=cmap, vmin=vmn, vmax=vmx)
        map_paths.append(png)

    print(f"\nSaved predictions -> {OUTPUT_PATH}")
    print(f"Saved table       -> {csv_path}")
    for p in map_paths:
        print(f"Saved map         -> {p}")


if __name__ == "__main__":
    main()
