#!/usr/bin/env python
"""Predict landslide susceptibility on the Cianjur EQ 2022 slope-unit dataset
using the PINN v3 model trained in `cotabato_new_slope_unit_v2-8.ipynb`.

This replays the exact inference pipeline from cell 20 of that notebook
(median fill -> slope filter -> log -> clip -> soil texture index) so the
trained NormalizationLayer sees values on the distribution it was adapted on.

IMPORTANT — UNIT / SCALE MISMATCH
---------------------------------
The Cianjur file uses different column NAMES *and* different UNITS than the
Cotabato training data. Names are remapped via CIANJUR_RENAME. Units are
reconciled via UNIT_CONVERSIONS below. Four physics features arrive on scales
that differ from training by 1-3 orders of magnitude:

    feature                  training median    Cianjur raw median
    PGA2_max                 0.225 (g)          14.67   (~65x -> assume %g, /100)
    Prc_mean                 223.5              0.092   (~2400x -> UNKNOWN unit)
    ContributingFactor_mean  935               10.0     (~90x  -> UNKNOWN def)
    SoilThc_mean             13.1              0.0 med  (UNKNOWN unit; half are 0)

These feed the Newmark / wetness physics layers directly. If the conversion
factors below are wrong, the physics outputs are meaningless. The script prints
a UNIT AUDIT comparing post-conversion medians against training medians so you
can verify each feature lands in-distribution before trusting predictions.
Edit UNIT_CONVERSIONS once you confirm the true provenance of each column.
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
INPUT_PATH = Path("~/Downloads/SlopeUnit_CianjurEQ2022.gpkg").expanduser()
MODEL_DIR = Path(
    "~/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/"
    "trainedCotabatoPhase7/historical/v8"
).expanduser()
FOLD = 1  # cell-20 replays fold-1's transform manifest; keep model + manifest in sync
MODEL_PATH = MODEL_DIR / f"fold-{FOLD}-model-v3.keras"
TRANSFORMS_PATH = PROJECT_ROOT / "feature_manifests" / f"v1_cotabato_transforms_fold{FOLD}.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "cianjur_eq2022_susceptibility_v2_8.gpkg"

SLOPE_FILTER_DEG = 10.0  # training set was filtered to slopes >= ~10.5 deg

# Cianjur column name -> training schema name.
CIANJUR_RENAME = {
    "slopemean": "Slope_mean",
    "bulkdensitymean": "BUK_mean",
    "PGAmean": "PGA2_max",
    "prcmean": "Prc_mean",
    "contributingfactor": "ContributingFactor_mean",
    "soilthic": "SoilThc_mean",
    "elevation": "Elev_mean",
    "claymean": "Clay_mean",
    "siltmean": "Silt_mean",
    "sandmean": "Sand_mean",
    # `soiltype` is handled separately -> `type` string
    # `Landslide` is the (optional) label
}

# Per-feature multiplier applied AFTER rename, BEFORE the manifest transforms.
# Defaults bring each feature toward the training median where the conversion
# is known; ambiguous features default to 1.0 and are flagged in the audit.
UNIT_CONVERSIONS = {
    "BUK_mean": 9.81 / 100.0,  # g/cm^3 * 100  ->  kN/m^3 unit weight (confirmed, matches v2-8)
    "PGA2_max": 1.0 / 100.0,   # raw cm/s^2 -> m/s^2 to match the training PGA2_max scale.
                               #   Verified: training PGA2_max is 0.093-0.530 (median 0.225);
                               #   /100 puts Cianjur at 0.020-0.624 (median 0.147), in-distribution.
                               #   (/980.665 -> g would give 0.002-0.064, far below training.)
    "Prc_mean": 1000.0,        # user: m -> mm (Cianjur ~0.09 m -> ~92 mm; training median ~223).
    "ContributingFactor_mean": 1.0,  # left as-is per user (Cianjur ~10 vs training ~935; OOD physics input).
    "SoilThc_mean": 0.001,     # user: mm -> m (training median ~13).
}

# Columns where a literal 0 means "missing" rather than a true measurement.
# Zeros are set to NaN before the training-median fill so they get imputed
# instead of poisoning the physics: a 0 bulk density or soil thickness sits in
# the FOS denominator (bulk_density * thickness * sin(slope)) -> divide-by-zero.
ZERO_IS_MISSING = {"SoilThc_mean", "BUK_mean"}

# Training medians (from the fold-1 transform manifest) used only for the audit.
TRAINING_MEDIANS_FOR_AUDIT = None  # filled from manifest at runtime


def print_unit_audit(df, training_medians):
    """Compare each model-input feature's median against the training median."""
    print("\n" + "=" * 78)
    print("UNIT AUDIT — post-conversion Cianjur median vs training median")
    print("=" * 78)
    print(f"{'feature':28s} {'train_med':>12s} {'cianjur_med':>12s} {'ratio':>8s}  flag")
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
    so outliers (e.g. negative FOS, 136 cm displacement) don't wash out the
    color scale. Pass explicit bounds for fixed-range fields like [0, 1].
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


def save_roc_curve(y_true, y_score, out_png, title):
    """Compute AUC and save the ROC curve. Returns the AUC (or None)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    mask = np.isfinite(y_score)
    y_true, y_score = np.asarray(y_true)[mask], np.asarray(y_score)[mask]
    if len(np.unique(y_true)) < 2:
        print("  [roc] only one class present — skipping ROC plot")
        return None

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"PINN v8 (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return auc


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

    # ---- Load Cianjur data ------------------------------------------------ #
    gdf = gpd.read_file(INPUT_PATH)
    print(f"\nLoaded {len(gdf):,} rows from {INPUT_PATH.name}")

    missing_src = [c for c in CIANJUR_RENAME if c not in gdf.columns]
    if missing_src:
        raise ValueError(f"Cianjur file is missing expected source columns: {missing_src}")

    gdf = gdf.rename(columns=CIANJUR_RENAME)

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

    # Safety net: any residual NaN among mapped features -> Cianjur median.
    impute_cols = [c for c in CIANJUR_RENAME.values() if c in gdf.columns]
    residual_na = gdf[impute_cols].isnull().sum()
    if int(residual_na.sum()) > 0:
        print(f"  [warn] residual NaNs after manifest fill: "
              f"{residual_na[residual_na > 0].to_dict()}; using Cianjur median")
        gdf[impute_cols] = gdf[impute_cols].fillna(gdf[impute_cols].median(numeric_only=True))

    # ---- Slope filter (match training domain) ----------------------------- #
    n_before = len(gdf)
    gdf = gdf[gdf["Slope_mean"] >= SLOPE_FILTER_DEG].reset_index(drop=True)
    print(f"Slope filter (>= {SLOPE_FILTER_DEG} deg): {n_before:,} -> {len(gdf):,} rows "
          f"(dropped {n_before - len(gdf):,})")

    # ---- soiltype -> `type` string (OOV -> learned embedding) ------------- #
    gdf["type"] = gdf["soiltype"].fillna(-1).astype(int).astype(str)

    # Keep raw (post-impute, pre-clip) elevation for cross-checking the maps.
    # The model's Elev_mean gets clipped to [74, 1936], which would flatten peaks.
    gdf["Elev_raw"] = gdf["Elev_mean"]

    # ---- Replay manifest transforms: log -> clip -> soil texture idx ------ #
    gdf = apply_log_transform(gdf, transform_meta["log_transformed_cols"])
    gdf = apply_clip_thresholds(gdf, transform_meta["clip_thresholds"])
    gdf = add_soil_texture_index(gdf)

    print_unit_audit(gdf, training_medians)

    # ---- Labels (optional) ------------------------------------------------ #
    has_labels = "Landslide" in gdf.columns and gdf["Landslide"].notna().any()
    gdf["landslide"] = (
        pd.to_numeric(gdf["Landslide"], errors="coerce").fillna(0).astype(int)
        if has_labels else 0
    )

    # ---- Build dataset from the model's actual input schema --------------- #
    missing = [c for c in input_cols if c not in gdf.columns]
    if missing:
        raise ValueError(f"Cianjur data is missing required model inputs: {missing}")

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

    if has_labels and finite.any():
        from sklearn.metrics import roc_auc_score
        y = gdf["landslide"].values[finite]
        if len(np.unique(y)) == 2:
            print(f"AUC (Cianjur labels, finite rows): "
                  f"{roc_auc_score(y, susceptibility[finite]):.4f}")
        print(f"Label balance: {pd.Series(gdf['landslide'].values).value_counts().to_dict()}")

    # ---- Save ------------------------------------------------------------- #
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Drop the internal `landslide` helper (case-insensitive collision with the
    # original `Landslide` column breaks the GPKG writer).
    out_gdf = gdf.drop(columns=[c for c in ["landslide"] if c in gdf.columns])
    out_gdf.to_file(OUTPUT_PATH, driver="GPKG")
    csv_path = OUTPUT_PATH.with_suffix(".csv")
    out_gdf.drop(columns="geometry").to_csv(csv_path, index=False)

    # ---- Maps: susceptibility (fixed 0-1) + physics fields (auto-scaled) -- #
    figs_dir = OUTPUT_PATH.parent
    stem = "cianjur_eq2022"
    maps = [
        ("susceptibility",     "Susceptibility",            "plasma_r", 0.0, 1.0),
        ("fos",                "Factor of Safety",          "RdYlGn",   None, None),
        ("PGA2_max",           "PGA (g)",                   "inferno",  None, None),
        ("cohesion",           "Cohesion (model units)",    "viridis",  None, None),
        ("internal_friction",  "Internal friction (model units)", "cividis", None, None),
        ("Elev_raw",           "Elevation (m)",             "terrain",  None, None),
        ("Slope_mean",         "Slope (deg)",               "YlOrRd",   None, None),
    ]
    map_paths = []
    for col, label, cmap, vmn, vmx in maps:
        png = figs_dir / f"{stem}_{col}_map_v2_8.png"
        save_choropleth(out_gdf, col, png, f"PINN v8 — Cianjur EQ 2022 — {label}",
                        label, cmap=cmap, vmin=vmn, vmax=vmx)
        map_paths.append(png)

    # ---- ROC / AUC (only when the Cianjur inventory labels are present) ---- #
    roc_path = None
    if has_labels:
        roc_path = figs_dir / f"{stem}_roc_v2_8.png"
        save_roc_curve(gdf["landslide"].values, susceptibility, roc_path,
                       "ROC — PINN v8 on Cianjur EQ 2022")

    print(f"\nSaved predictions -> {OUTPUT_PATH}")
    print(f"Saved table       -> {csv_path}")
    for p in map_paths:
        print(f"Saved map         -> {p}")
    if roc_path:
        print(f"Saved ROC         -> {roc_path}")


if __name__ == "__main__":
    main()
