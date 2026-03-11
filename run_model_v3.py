
import argparse
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
from py_files.metrics import plot_susceptibility_map

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
    """Load GeoPackage, run preprocessing, add soil texture index."""
    df = gpd.read_file(data_path)

    # Drop columns that exist in the dataframe (some may not be present)
    cols_to_drop = [c for c in COLUMNS_DROP if c in df.columns]
    df, columns, numeric_cols = preprocessing(df, columns_drop=cols_to_drop)
    df = add_soil_texture_index(df)

    feature_cols = columns + ["soil_texture_idx"]
    return df, feature_cols


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


def run(data_path: str, model_path: str):
    print(f"Loading data from {data_path}")
    df, feature_cols = load_and_preprocess(data_path)
    print(f"  Samples after preprocessing: {len(df)}")

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

    print("\nPlotting susceptibility map …")
    plot_susceptibility_map(df, susceptibility, "PINN v3 (fold-1)")

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
    args = parser.parse_args()

    results = run(args.data, args.model)
