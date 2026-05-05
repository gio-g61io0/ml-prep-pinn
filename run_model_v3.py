
import argparse
import os
import numpy as np
import geopandas as gpd
import tensorflow as tf
from tensorflow.keras.models import load_model

from py_files.data import preprocessing, dataframe_to_dataset
from py_files.helpers import add_soil_texture_index

# Custom layers (needed for deserialization even though they are
# registered via @register_keras_serializable)
from py_files.GallenModel import CriticalAcceleration, DisplacementIntermediate, FosLayer
from py_files.GallenModel_v1 import (
    NewmarkActivation,
    DisplacementLayerRainFall,
    WetnessLayer,
    ClipLayer,
    CohesionLayer,
    InternalFrictionLayer,
)
from py_files.GallenModel_v3 import HydraulicConductivityLayerV3
from py_files.Landslidev2_Old import DiceCrossEntropyLoss
from py_files.metrics import (
    export_to_geopackage,
    create_slope_unit_template, rasterize_from_template,
)

DEFAULT_DATA = "~/Documents/ml-prep/ML-PREP-2025/learn/data/SU_17_training_v3_contri.gpkg"
MODEL_SAVE_PATH = (
    "/Users/giogonzales/Documents/ml-prep/ML-PREP-2025/learn/"
    "trainedWeights/trainedCotabatoPhase7/historical/v7"
)
DEFAULT_MODEL = f"{MODEL_SAVE_PATH}/fold-1-model-v3.keras"

COLUMNS_DROP = [
    "Landslide1", "descriptio", "sus_pinn_ground truth", "ds",
    "cohesion", "internal_friction", "sus_pinn_landslide",
    "confusion", "landslide_preds", "landslide_probability",
    "Lithology", "LITHO", "Geomorphology", "LITHODESC",
    "LITHO_2", "LITHODESC_2", "value",
]

BATCH_SIZE = 128


def load_and_preprocess(data_path: str):
    """Load GeoPackage, run preprocessing, add soil texture index.

    Returns the preprocessed GeoDataFrame, feature columns, and the
    *full* (unfiltered) GeoDataFrame for gap-free rasterization.
    """
    df_full = gpd.read_file(data_path)

    # Drop columns that exist in the dataframe (some may not be present)
    cols_to_drop = [c for c in COLUMNS_DROP if c in df_full.columns]
    df, columns, numeric_cols = preprocessing(df_full.copy(), columns_drop=cols_to_drop)
    df = add_soil_texture_index(df)

    feature_cols = columns + ["soil_texture_idx"]
    return df, feature_cols, df_full


def load_trained_model(model_path: str):
    """Load a saved .keras model with all custom objects."""
    custom_objects = {
        "NewmarkActivation": NewmarkActivation,
        "DisplacementLayerRainFall": DisplacementLayerRainFall,
        "WetnessLayer": WetnessLayer,
        "ClipLayer": ClipLayer,
        "CohesionLayer": CohesionLayer,
        "InternalFrictionLayer": InternalFrictionLayer,
        "CriticalAcceleration": CriticalAcceleration,
        "DisplacementIntermediate": DisplacementIntermediate,
        "FosLayer": FosLayer,
        "HydraulicConductivityLayerV3": HydraulicConductivityLayerV3,
        "DiceCrossEntropyLoss": DiceCrossEntropyLoss,
    }
    return load_model(model_path, custom_objects=custom_objects)


def extract_intermediate(model, dataset, layer_name):
    """Build a sub-model and predict intermediate outputs."""
    sub = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
    return sub.predict(dataset)


def run(data_path: str, model_path: str, output_dir: str = None, pixel_size: float = 30.0):
    print(f"Loading data from {data_path}")
    df, feature_cols, df_full = load_and_preprocess(data_path)
    
    print(f"  Samples after preprocessing: {len(df)}")
    print(f"  COLS : {feature_cols}")
    print(df['soil_texture_idx'].head())

    ds = dataframe_to_dataset(df[feature_cols].copy(), shuffle=False, batch_size=BATCH_SIZE)

    print(f"Loading model from {model_path}")
    model = load_trained_model(model_path)
    model.summary()

    susceptibility = model.predict(ds).flatten()


    fos = extract_intermediate(model, ds, "fos_layer").flatten()
    print(f"  Factor of Safety     — mean: {fos.mean():.4f}, std: {fos.std():.4f}")

    ac_outputs = extract_intermediate(model, ds, "critical_acceleration")
    ac = ac_outputs[0].flatten()
    acpg = ac_outputs[1].flatten()
    print(f"  Critical Accel (ac)  — mean: {ac.mean():.4f}, std: {ac.std():.4f}")
    print(f"  ac / PGA (acpg)      — mean: {acpg.mean():.4f}, std: {acpg.std():.4f}")

    displacement = extract_intermediate(model, ds, "displacement_layer").flatten()
    print(f"  Displacement (cm)    — mean: {displacement.mean():.4f}, std: {displacement.std():.4f}")

    cohesion = extract_intermediate(model, ds, "cohesion_layer").flatten()
    print(f"  Cohesion (kPa)       — mean: {cohesion.mean():.4f}, std: {cohesion.std():.4f}")

    ifi = extract_intermediate(model, ds, "internal_friction").flatten()
    print(f"  Int. Friction (rad)  — mean: {ifi.mean():.4f}, std: {ifi.std():.4f}")

    wetness = extract_intermediate(model, ds, "m_clip").flatten()
    print(f"  Wetness (m)          — mean: {wetness.mean():.4f}, std: {wetness.std():.4f}")

    k_layer = model.get_layer("hydraulic_conductivity_v3")
    u_k = k_layer.u_k.numpy()
    k_min = k_layer.k_min.numpy()
    k_max = k_layer.k_max.numpy()
    k_cmh = k_min + (k_max - k_min) * tf.nn.sigmoid(u_k).numpy()
    print("\n  Learned K (cm/h) per USDA soil texture:")
    soil_names = [
        "Sand", "Loamy Sand", "Sandy Loam", "Silt Loam", "Loam", "Silt",
        "Sandy Clay Loam", "Clay Loam", "Silty Clay Loam", "Sandy Clay",
        "Silty Clay", "Clay",
    ]
    for i, name in enumerate(soil_names):
        print(f"    {name:20s}: {k_cmh[i]:.4f}")

    print("\nPlotting boxplots and susceptibility map …")
    # plot_boxplot(cohesion, 'Cohesion (kPa)')
    # plot_boxplot(susceptibility, 'Susceptibility')
    # plot_boxplot(ifi, 'Internal Friction Angle (rad)')
    # plot_boxplot(wetness, 'Wetness (m)')
    # plot_susceptibility_map(df, susceptibility, "PINN v3 (fold-1)")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # --- Vector export (GeoPackage) ---
        gpkg_path = os.path.join(output_dir, "geotechnical_values.gpkg")
        print(f"\nExporting GeoPackage to {gpkg_path} …")
        export_to_geopackage(df, {
            "cohesion_kpa": cohesion,
            "internal_friction_rad": ifi,
            "wetness_m": wetness,
            "fos": fos,
            "displacement_cm": displacement,
            "susceptibility": susceptibility,
        }, gpkg_path)

        # --- Slope-unit raster export (GeoTIFFs) ---
        # Use the FULL (unfiltered) GeoDataFrame for the template so that
        # all slope-unit polygons are rasterized — no gaps from dropped rows.
        template_path = os.path.join(output_dir, "slope_unit_template.tif")
        print(f"\nCreating slope-unit template at {template_path} (pixel_size={pixel_size}) …")
        create_slope_unit_template(df_full, template_path, pixel_size=pixel_size)

        # Build a full-length value array: predicted values for kept rows,
        # nodata for rows dropped during preprocessing.
        nodata_val = -9999.0
        kept_indices = df.index.values  # original row indices that survived preprocessing

        print(f"Exporting slope-unit GeoTIFFs to {output_dir} …")
        layers = {
            "cohesion": (cohesion, "Cohesion (kPa)"),
            "internal_friction": (ifi, "Internal Friction Angle (rad)"),
            "wetness": (wetness, "Wetness (m)"),
            "fos": (fos, "Factor of Safety"),
            "displacement": (displacement, "Displacement (cm)"),
            "susceptibility": (susceptibility, "Susceptibility"),
        }
        for fname, (vals, desc) in layers.items():
            full_vals = np.full(len(df_full), nodata_val, dtype=np.float64)
            full_vals[kept_indices] = vals
            rasterize_from_template(
                template_path, full_vals,
                os.path.join(output_dir, f"{fname}.tif"),
                layer_name=desc,
            )

    return {
        "susceptibility": susceptibility,
        "fos": fos,
        "critical_acceleration": ac,
        "acpg": acpg,
        "displacement": displacement,
        "cohesion": cohesion,
        "internal_friction": ifi,
        "wetness": wetness,
        "k_cmh": k_cmh,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINN Landslide Model v3 inference")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to GeoPackage file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .keras model checkpoint")
    parser.add_argument("--output-dir", default=None, help="Directory for GeoTIFF exports (skip if omitted)")
    parser.add_argument("--pixel-size", type=float, default=30.0, help="GeoTIFF pixel size in CRS units (default: 30)")
    args = parser.parse_args()

    results = run(args.data, args.model, output_dir=args.output_dir, pixel_size=args.pixel_size)
