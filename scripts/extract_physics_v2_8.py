"""Extract intermediate physics outputs from a v2-8 (v3 PINN) fold model.

Mirrors notebook cells 1–11 + cell 43 of `notebooks/cotabato_new_slope_unit_v2-8.ipynb`:
loads the training dataframe through `preprocessing_v2`, replays the per-fold
log/clip transform manifest, then runs a multi-output Keras model that exposes
the physics chain (cohesion, internal friction, critical acceleration, factor
of safety, Newmark displacement) and writes one row per pixel to a GeoPackage
(geometry re-attached positionally from the raw training GPKG using the
preserved row index).

Defaults target fold 5 (`fold-5-model-v3.keras`) and the v8 MODEL_SAVE_PATH used
by the notebook. Override via CLI flags.

Usage:
    source venv/bin/activate
    python scripts/extract_physics_v2_8.py
    python scripts/extract_physics_v2_8.py --fold 3 --out /tmp/physics_fold3.gpkg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402  (sys.path tweak above)
from tensorflow.keras import Model  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402

from py_files.GallenModel_v1 import NewmarkActivation  # noqa: E402
from py_files.data import (  # noqa: E402
    apply_clip_thresholds,
    apply_log_transform,
    dataframe_to_dataset,
    preprocessing_v2,
)
from py_files.helpers import add_soil_texture_index, set_seed  # noqa: E402


DEFAULT_MODEL_SAVE_PATH = (
    "/Users/giogonzales/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/"
    "trainedCotabatoPhase7/historical/v8"
)
DEFAULT_TRAINING_FILE = (
    "~/Documents/ml-prep/ML-PREP-2025/learn/data/SU_17_training_v3_contri.gpkg"
)
DEFAULT_MANIFESTS_DIR = PROJECT_ROOT / "feature_manifests"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "image_outputs"

COLUMNS_DROP = [
    "Landslide1", "descriptio", "sus_pinn_ground truth", "ds",
    "cohesion", "internal_friction", "sus_pinn_landslide",
    "confusion", "landslide_preds", "landslide_probability",
    "Lithology", "LITHO", "Geomorphology", "LITHODESC",
    "LITHO_2", "LITHODESC_2", "value",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=5,
                        help="Fold number to extract (default: 5)")
    parser.add_argument("--model-save-path", default=DEFAULT_MODEL_SAVE_PATH,
                        help="Directory containing fold-{N}-model-v3.keras files")
    parser.add_argument("--training-file", default=DEFAULT_TRAINING_FILE,
                        help="Training GeoPackage (same one the notebook uses)")
    parser.add_argument("--manifests-dir", default=str(DEFAULT_MANIFESTS_DIR),
                        help="Directory holding v1_cotabato_transforms_fold{N}.json")
    parser.add_argument("--out", default=None,
                        help="Output GeoPackage path. Defaults to "
                             "image_outputs/physics_chain_fold{N}.gpkg")
    parser.add_argument("--layer", default="physics_chain",
                        help="Layer name inside the GeoPackage (default: physics_chain)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--include-residual", action="store_true",
                        help="Also include physics_prob and final_head columns")
    return parser.parse_args()


def load_fold_manifest(manifests_dir: Path, fold: int) -> dict:
    manifest_path = manifests_dir / f"v1_cotabato_transforms_fold{fold}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Per-fold manifest not found: {manifest_path}. "
            "Re-run notebook cell 17 (train_model_rainfall_v3) to write it."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"[manifest] {manifest_path.name}")
    print(f"  log_transformed_cols: {manifest.get('log_transformed_cols', [])}")
    print(f"  clip_thresholds:      {len(manifest.get('clip_thresholds', {}))} columns")
    return manifest


def build_extractor(model: Model, include_residual: bool) -> Model:
    """Wrap the trained model to expose physics intermediates per sample.

    `critical_acceleration` is a tuple layer (ac, ac/pga); we keep only `ac`.
    """
    crit_layer_output = model.get_layer("critical_acceleration").output
    if isinstance(crit_layer_output, (list, tuple)):
        critical_acceleration_out = crit_layer_output[0]
    else:
        critical_acceleration_out = crit_layer_output

    outputs = {
        "cohesion":             model.get_layer("cohesion_layer").output,
        "internal_friction":    model.get_layer("internal_friction").output,
        "fos":                  model.get_layer("fos_layer").output,
        "critical_acceleration": critical_acceleration_out,
        "displacement":         model.get_layer("displacement_layer").output,
    }
    if include_residual:
        outputs["physics_prob"] = model.get_layer("physics_prob").output
        outputs["final_head"] = model.get_layer("final_head").output
    return Model(inputs=model.inputs, outputs=outputs)


def summarize(name: str, values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "feature": name,
        "min":     float(np.min(finite)) if finite.size else None,
        "p05":     float(np.percentile(finite, 5)) if finite.size else None,
        "median":  float(np.median(finite)) if finite.size else None,
        "mean":    float(np.mean(finite)) if finite.size else None,
        "p95":     float(np.percentile(finite, 95)) if finite.size else None,
        "max":     float(np.max(finite)) if finite.size else None,
    }


def main() -> int:
    args = parse_args()
    set_seed(42)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

    fold = args.fold
    model_path = Path(args.model_save_path) / f"fold-{fold}-model-v3.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    output_path = Path(args.out) if args.out else (
        DEFAULT_OUTPUT_DIR / f"physics_chain_fold{fold}.gpkg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load + preprocess the training dataframe exactly like the notebook does.
    #    Keep the raw GeoDataFrame around so we can re-attach geometry by index
    #    (preprocessing_v2 filters rows but does not reset the index — same
    #     strategy used by `_attach_geometry` in notebook cell 45).
    training_file = os.path.expanduser(args.training_file)
    print(f"[data] reading {training_file}")
    raw_gdf = gpd.read_file(training_file)
    df = raw_gdf.copy()
    print(f"  raw rows: {len(df):,}")

    df, columns, numeric_cols, _imputed_cols, _imputation_medians = preprocessing_v2(
        df, columns_drop=COLUMNS_DROP, track_imputation=True,
    )
    df = add_soil_texture_index(df[columns].copy())
    print(f"  after preprocessing: {len(df):,} rows, {len(df.columns)} columns")

    # 2) Replay the per-fold log/clip manifest
    manifest = load_fold_manifest(Path(args.manifests_dir), fold)
    log_cols = manifest.get("log_transformed_cols", [])
    clip_thresholds = manifest.get("clip_thresholds", {})
    eval_df = apply_log_transform(df.copy(), log_cols)
    eval_df = apply_clip_thresholds(eval_df, clip_thresholds)

    # 3) Load the model (only NewmarkActivation needs explicit registration —
    #    every other custom layer is decorated with register_keras_serializable)
    print(f"[model] loading {model_path}")
    model = load_model(
        str(model_path),
        custom_objects={"NewmarkActivation": NewmarkActivation},
    )

    # 4) Build the dataset using the input columns the saved model expects
    input_cols = [t.name.split(":")[0] for t in model.inputs]
    missing = [c for c in input_cols if c not in eval_df.columns]
    if missing:
        raise KeyError(
            f"Columns required by the model are missing from the dataframe: "
            f"{missing}"
        )

    has_label = "landslide" in eval_df.columns
    cols_for_ds = input_cols + (["landslide"] if has_label else [])
    ds = dataframe_to_dataset(
        eval_df[cols_for_ds].assign(
            landslide=eval_df["landslide"] if has_label else 0
        ),
        shuffle=False,
        batch_size=args.batch_size,
    )

    # 5) Predict each intermediate physics value
    print("[predict] running extractor")
    extractor = build_extractor(model, include_residual=args.include_residual)
    preds = extractor.predict(ds, verbose=0)

    # Carry forward the *cleaned* features that were actually fed to the model
    # (post-preprocessing_v2 + post log/clip manifest replay). This makes the
    # output GeoPackage the cleaned prediction-input dataset enriched with the
    # physics-chain values the model produced.
    out = eval_df.copy()
    if isinstance(out, gpd.GeoDataFrame) and "geometry" in out.columns:
        out = pd.DataFrame(out.drop(columns="geometry"))
    out.insert(0, "row_index", eval_df.index.to_numpy())

    out["cohesion_kpa"] = np.asarray(preds["cohesion"]).squeeze()
    out["internal_friction_rad"] = np.asarray(preds["internal_friction"]).squeeze()
    out["internal_friction_deg"] = np.degrees(out["internal_friction_rad"])
    out["critical_acceleration"] = np.asarray(preds["critical_acceleration"]).squeeze()
    out["fos"] = np.asarray(preds["fos"]).squeeze()
    out["displacement_cm"] = np.asarray(preds["displacement"]).squeeze()

    if args.include_residual:
        out["physics_prob"] = np.asarray(preds["physics_prob"]).squeeze()
        out["final_head"] = np.asarray(preds["final_head"]).squeeze()

    # Re-attach geometry by index. preprocessing_v2 filters rows without
    # resetting the index, so eval_df.index points at raw_gdf row positions.
    idx = eval_df.index
    if not idx.is_unique:
        raise RuntimeError("Preprocessed index is not unique — cannot align geometry.")
    if idx.max() >= len(raw_gdf) or idx.min() < 0:
        raise RuntimeError(
            f"Preprocessed index out of range for raw GDF "
            f"(idx [{idx.min()}, {idx.max()}] vs raw len {len(raw_gdf)})."
        )
    geom = raw_gdf.geometry.iloc[idx.to_numpy()].to_numpy()
    out_gdf = gpd.GeoDataFrame(out, geometry=geom, crs=raw_gdf.crs)

    if output_path.exists():
        output_path.unlink()  # avoid GPKG layer-append surprises
    out_gdf.to_file(output_path, layer=args.layer, driver="GPKG")
    print(
        f"[write] {output_path} layer={args.layer!r} "
        f"({len(out_gdf):,} rows, {len(out_gdf.columns)} cols, crs={out_gdf.crs})"
    )

    summary_cols = [
        "cohesion_kpa",
        "internal_friction_rad",
        "internal_friction_deg",
        "critical_acceleration",
        "fos",
        "displacement_cm",
    ]
    if args.include_residual:
        summary_cols += ["physics_prob", "final_head"]
    summary_df = pd.DataFrame([summarize(c, out[c].values) for c in summary_cols])
    print("\n=== Physics chain summary (fold {} on training data) ===".format(fold))
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
