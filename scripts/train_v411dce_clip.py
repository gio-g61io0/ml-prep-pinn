"""Retrain the v4-1-1dce architecture with a hard clip on the internal friction angle.

Mirrors notebooks/cotabato_new_slope_unit_v2-3-train.ipynb so the only difference
between the two model families is the IFI constraint. Writes 10 fold checkpoints
to OUTPUT_DIR.

Clip: tf.clip_by_value(ifi_rad, deg_to_rad(25), deg_to_rad(45))
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import geopandas as gpd
import tensorflow as tf
from tensorflow.keras import layers, Model
import sklearn
from sklearn.model_selection import StratifiedKFold

from py_files.data import dataframe_to_dataset, NormalizationLayer, CategoricalEncoderLayer
from py_files.GallenModel_v1 import (
    CohesionLayer,
    InternalFrictionLayer,
    IFIClipLayer,
    DisplacementLayer,
    NewmarkActivation,
)
from py_files.Landslidev2_Old import DiceCrossEntropyLoss


MLP_UNITS = [32, 64, 8, 64, 32, 8, 32, 8]
LEAKY_ALPHA = 0.2
NEWMARK_THRESHOLD = 2.0
LEARNING_RATE = 1e-5
EPOCHS = 200
BATCH_SIZE = 128
CLASS_WEIGHT = {0: 1, 1: 5}
N_SPLITS = 10

IFI_CLIP_DEG = (0.0, 40.0)
IFI_CLIP_RAD = tuple(math.radians(d) for d in IFI_CLIP_DEG)

DATA_PATH = Path("~/Documents/ml-prep/ML-PREP-2025/learn/data/SU_15_Training1.gpkg").expanduser()
OUTPUT_DIR = Path(
    "/Users/giogonzales/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/trainedCotabatoPhase7/historical/v4-1-1dce-repro-ifi-clip-0-40"
)


def build_model(train_ds, numeric_cols, categorical_cols, pga_col):
    all_inputs: list[tf.Tensor] = []
    encoded: list[tf.Tensor] = []
    pga_input = None

    for header in numeric_cols:
        x_in = tf.keras.Input((1,), name=header)
        if header == pga_col:
            pga_input = x_in
            continue
        x_norm = NormalizationLayer(header, train_ds)(x_in)
        all_inputs.append(x_in)
        encoded.append(x_norm)

    for header in categorical_cols:
        x_in = tf.keras.Input((1,), name=header, dtype="string")
        x_enc = CategoricalEncoderLayer(header, train_ds, dtype="string")(x_in)
        all_inputs.append(x_in)
        encoded.append(x_enc)

    if pga_input is None:
        raise ValueError(f"PGA column {pga_col!r} not in numeric_cols")

    by_name = {t.name.split(":")[0]: t for t in all_inputs}
    slope = by_name["Slope_mean"]
    bulk_density = by_name["BUK_mean"]

    x = layers.concatenate(encoded)
    x = layers.Dense(
        64, name="Sus_0", kernel_initializer="random_normal", bias_initializer="random_normal"
    )(x)
    for i, units in enumerate(MLP_UNITS, start=1):
        x = layers.Dense(
            units,
            name=f"Sus_{i}",
            kernel_initializer="random_normal",
            bias_initializer="random_normal",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA)(x)

    x = layers.Dense(2, name="geotechnical_param")(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA)(x)

    coh = CohesionLayer()(x)
    ifi = InternalFrictionLayer()(x)
    ifi = IFIClipLayer(IFI_CLIP_RAD[0], IFI_CLIP_RAD[1], name="ifi_clip_0_40")(ifi)

    ds = DisplacementLayer()([coh, ifi, slope, pga_input, bulk_density])
    ds = layers.LeakyReLU(negative_slope=LEAKY_ALPHA)(ds)
    sus = NewmarkActivation(threshold=NEWMARK_THRESHOLD)(ds)

    model = Model(inputs=all_inputs + [pga_input], outputs=sus)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=DiceCrossEntropyLoss(),
        metrics=[
            tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5),
            tf.keras.metrics.AUC(curve="ROC"),
            "accuracy",
        ],
    )
    return model


def load_training_df():
    df = gpd.read_file(str(DATA_PATH))
    df.drop(
        columns=[
            "landslide_probability",
            "landslide_preds",
            "confusion",
            "sus_pinn_landslide",
            "sus_pinn_ground truth",
            "ds",
            "cohesion",
            "internal_friction",
        ],
        inplace=True,
    )
    df = df[df["Slope_mean"] >= 10]
    df.dropna(subset=list(df.columns), inplace=True)
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_df()
    cols_remove = ["DN", "BD_mean", "geometry", "PGA2_max", "Soil Type", "description", "descriptio"]
    feature_cols = [c for c in df.columns if c not in cols_remove]
    numeric_cols = [c for c in feature_cols if c not in ("landslide", "type")]
    categorical_cols = ["type"]
    pga_col = "PGA1_max"

    print(f"Training set rows: {len(df)}  features: {feature_cols}")
    print(f"IFI clip (deg): {IFI_CLIP_DEG}  (rad): {IFI_CLIP_RAD}")
    print(f"Output dir: {OUTPUT_DIR}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    aucs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["landslide"]), start=1):
        print(f"\n=== Fold {fold}/{N_SPLITS} ===")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        train_ds = dataframe_to_dataset(train_df[feature_cols])
        val_ds = dataframe_to_dataset(val_df[feature_cols], shuffle=False)

        tf.keras.backend.clear_session()
        model = build_model(train_ds, numeric_cols, categorical_cols, pga_col)

        ckpt_path = str(OUTPUT_DIR / f"fold-{fold}-model-0.keras")
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            save_freq="epoch",
            verbose=0,
        )
        early = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )

        model.fit(
            train_ds,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=val_ds,
            class_weight=CLASS_WEIGHT,
            callbacks=[early, ckpt],
            verbose=2,
        )

        y_true = val_df["landslide"].to_numpy()
        y_pred = model.predict(val_ds, verbose=0).flatten()
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc)
        acc = sklearn.metrics.balanced_accuracy_score(y_true, y_pred > 0.5)
        print(f"Fold {fold}: AUC={auc:.4f}  BalAcc={acc:.4f}")

    print(f"\nMean AUC across folds: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")


if __name__ == "__main__":
    main()
