"""Data-driven baseline training, mirroring the v3 PINN evaluation surface.

Trains a plain MLP (no physics layers) on the same GA-EN feature subset used
by `train_rainfall_v3`, with per-fold SMOTENC oversampling applied to the
training partition only. Saves fold checkpoints + OOF predictions next to the
model save path so the notebook can resume in a fresh kernel without
re-running training.
"""
import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from imblearn.over_sampling import SMOTENC, SMOTE
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

from .data import (
    CategoricalEncoderLayer,
    NormalizationLayer,
    dataframe_to_dataset,
)

SKIP_NORMALIZATION = {"soil_texture_idx"}
OOF_FILENAME = "oof_preds_data_driven.npy"
CHECKPOINT_FMT = "fold-{fold}-model-data-driven.keras"


@tf.keras.utils.register_keras_serializable(package="data_driven")
class CastToFloat32(tf.keras.layers.Layer):
    """Pass-through cast layer used for inputs that skip normalization.

    Replaces a `Lambda(lambda t: tf.cast(t, tf.float32))` so the model can be
    reloaded from a `.keras` file without `safe_mode=False`.
    """

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

    def get_config(self):
        return super().get_config()


def _build_data_driven_model(
    numerical_cols: List[str],
    categorical_cols: List[str],
    train_ds: tf.data.Dataset,
    hidden_units: int = 64,
    depth: int = 12,
    dropout: float = 0.5,
) -> tf.keras.Model:
    """Functional MLP baseline.

    Same backbone width/depth as the v3 PINN trunk so the comparison isolates
    the physics chain (not the encoder capacity). No CohesionLayer / IFI /
    Newmark heads -- just a single sigmoid susceptibility output.
    """
    all_inputs, encoded_inputs = [], []

    for header in numerical_cols:
        inp = tf.keras.Input((1,), name=header)
        all_inputs.append(inp)
        if header in SKIP_NORMALIZATION:
            encoded_inputs.append(CastToFloat32(name=f"{header}_passthrough")(inp))
        else:
            encoded_inputs.append(NormalizationLayer(header, train_ds)(inp))

    for header in categorical_cols:
        inp = tf.keras.Input((1,), name=header, dtype="string")
        all_inputs.append(inp)
        encoded_inputs.append(CategoricalEncoderLayer(header, train_ds, "string")(inp))

    x = tf.keras.layers.Concatenate()(encoded_inputs)
    x = tf.keras.layers.Dense(hidden_units, activation="relu", name="trunk_0")(x)

    for i in range(1, depth + 1):
        x = tf.keras.layers.Dense(hidden_units, name=f"trunk_{i}")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Dropout(dropout)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="susceptibility")(x)

    return tf.keras.Model(inputs=all_inputs, outputs=output)


def _smote_oversample(
    train_df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    label_col: str = "landslide",
    random_state: int = 42,
) -> pd.DataFrame:
    """Apply SMOTENC to the training partition; preserve original schema.

    `type` is string-encoded for SMOTENC, then restored. `soil_texture_idx` is
    treated as categorical so SMOTENC samples integer values rather than
    interpolating between texture classes.
    """
    feature_cols = numerical_cols + categorical_cols
    X = train_df[feature_cols].copy()
    y = train_df[label_col].copy()

    string_cat_mappings = {}
    for col in categorical_cols:
        if X[col].dtype == object:
            cats = sorted(X[col].astype(str).unique())
            mapping = {v: i for i, v in enumerate(cats)}
            X[col] = X[col].astype(str).map(mapping).astype(int)
            string_cat_mappings[col] = mapping

    cat_idx = [
        feature_cols.index(c)
        for c in feature_cols
        if c in categorical_cols or c in SKIP_NORMALIZATION
    ]

    if cat_idx:
        sampler = SMOTENC(
            categorical_features=cat_idx,
            random_state=random_state,
            sampling_strategy="minority",
        )
    else:
        sampler = SMOTE(random_state=random_state, sampling_strategy="minority")

    X_res, y_res = sampler.fit_resample(X, y)

    for col, mapping in string_cat_mappings.items():
        inverse = {i: v for v, i in mapping.items()}
        X_res[col] = X_res[col].astype(int).map(inverse)

    for col in SKIP_NORMALIZATION:
        if col in X_res.columns:
            X_res[col] = X_res[col].astype(int)

    resampled = X_res.copy()
    resampled[label_col] = y_res.values
    return resampled.reset_index(drop=True)


def train_data_driven(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    feature_cols: List[str],
    pga_col: Optional[str] = None,
    path: str = ".",
    epochs: int = 200,
    batch_size: int = 128,
    patience: int = 5,
    smote: bool = True,
    random_state: int = 42,
):
    """Train the data-driven MLP with 5-fold CV and per-fold SMOTE.

    `pga_col` is accepted for signature parity with `train_model_rainfall_v3`
    but is treated like any other numerical feature -- there is no physics
    chain that needs PGA isolated.
    """
    os.makedirs(path, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    predictions = np.zeros(df.shape[0])
    aucs: List[float] = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs: List[np.ndarray] = []
    fold = 1

    plt.figure(figsize=(8, 6))

    for train_idx, val_idx in skf.split(df, df["landslide"]):
        train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

        if smote:
            n_before = len(train_df)
            train_df = _smote_oversample(
                train_df,
                numerical_cols,
                categorical_cols,
                random_state=random_state,
            )
            n_after = len(train_df)
            pos_after = int(train_df["landslide"].sum())
            print(
                f"  Fold {fold} SMOTE: {n_before} -> {n_after} rows "
                f"(positives={pos_after}/{n_after})"
            )

        train_ds = dataframe_to_dataset(
            train_df[feature_cols].copy(), batch_size=batch_size, seed=random_state,
        )
        val_ds = dataframe_to_dataset(
            val_df[feature_cols].copy(), shuffle=False, batch_size=batch_size,
        )

        model = _build_data_driven_model(numerical_cols, categorical_cols, train_ds)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=10000,
            decay_rate=0.9,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
        )

        ckpt_path = os.path.join(path, CHECKPOINT_FMT.format(fold=fold))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_path,
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
                verbose=0,
            ),
        ]

        model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=2,
        )

        y_true = val_df["landslide"].values
        val_preds = model.predict(val_ds, verbose=0).flatten()
        predictions[val_idx] = val_preds

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, val_preds)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        acc = round(
            sklearn.metrics.balanced_accuracy_score(y_true, val_preds > 0.5), 2,
        )
        plt.plot(fpr, tpr, lw=1, alpha=0.4, label=f"Fold {fold} (AUC={auc:.3f}, Acc={acc})")

        fold += 1
        tf.keras.backend.clear_session()

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    plt.plot(mean_fpr, mean_tpr, color="black", lw=2,
             label=f"Mean ROC (AUC={np.mean(aucs):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Data-Driven Baseline -- 5-Fold ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    np.save(os.path.join(path, OOF_FILENAME), predictions)
    return predictions, aucs


def regenerate_oof_data_driven(
    df: pd.DataFrame,
    feature_cols: List[str],
    path: str,
    batch_size: int = 128,
    random_state: int = 42,
):
    """Rebuild OOF preds from saved fold-N-model-data-driven.keras checkpoints."""
    from tensorflow.keras.models import load_model

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    predictions = np.zeros(df.shape[0])

    for fold, (_, val_idx) in enumerate(skf.split(df, df["landslide"]), start=1):
        ckpt_path = os.path.join(path, CHECKPOINT_FMT.format(fold=fold))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Missing fold checkpoint: {ckpt_path}. Re-run training first."
            )
        val_ds = dataframe_to_dataset(
            df.iloc[val_idx][feature_cols].copy(),
            shuffle=False,
            batch_size=batch_size,
        )
        model = load_model(ckpt_path)
        predictions[val_idx] = model.predict(val_ds, verbose=0).flatten()
        del model
        tf.keras.backend.clear_session()

    np.save(os.path.join(path, OOF_FILENAME), predictions)
    return predictions


def load_or_regenerate_oof_data_driven(
    df: pd.DataFrame,
    feature_cols: List[str],
    path: str,
    batch_size: int = 128,
    random_state: int = 42,
):
    """Load OOF preds from disk; regenerate from fold checkpoints if missing."""
    oof_path = os.path.join(path, OOF_FILENAME)
    if os.path.exists(oof_path):
        return np.load(oof_path)
    return regenerate_oof_data_driven(
        df, feature_cols, path, batch_size=batch_size, random_state=random_state,
    )
