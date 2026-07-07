"""LSTM time-series forecaster (TensorFlow Functional API).

Predicts next-day Indonesia equity return (``r_IDN``) from a rolling window of
the multivariate news/returns series in ``mnews_garch_dataset.csv``.

Design choices that match the rest of this repo:
- Built with the Keras **Functional API** (not Sequential / subclassing).
- A ``keras.layers.Normalization`` layer is **adapted on the training split
  only**, so validation/test data never leak into the scaler — the same
  train/serve-skew concern called out in ``CLAUDE.md`` for the PINN pipeline.
- The split is **chronological** (no shuffling) because this is a time series:
  the model must never see future rows when predicting the past.

Run as a script for a quick end-to-end demo:

    python -m py_files.lstm_timeseries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# --- Constants -------------------------------------------------------------

DATASET_PATH = Path(__file__).resolve().parent.parent / "mnews_garch_dataset.csv"
TARGET_COL = "r_IDN"
DATE_COL = "date"

DEFAULT_WINDOW = 30  # look-back length in trading days
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_LSTM_UNITS = (64, 32)
DEFAULT_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 1e-3
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15  # remaining 0.15 is the test split


@dataclass(frozen=True)
class WindowedData:
    """Immutable container for the windowed train/val/test tensors."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_features(self) -> int:
        return self.x_train.shape[-1]

    @property
    def window(self) -> int:
        return self.x_train.shape[1]


# --- Data preparation ------------------------------------------------------


def load_series(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the time series sorted ascending by date.

    Fails fast if the file or the target column is missing — never trust the
    on-disk schema silently.
    """
    if not path.exists():
        raise FileNotFoundError(f"Time-series dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=[DATE_COL])
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column {TARGET_COL!r} missing from {path}")

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    if df.isna().any().any():
        raise ValueError("Dataset contains NaNs; clean before training.")
    return df


def make_windows(
    matrix: np.ndarray,
    target: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a (T, F) matrix into (N, window, F) samples predicting next-step target.

    Sample i uses rows [i, i+window) to predict target at row i+window.
    Returns fresh arrays; the inputs are never mutated.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    n_samples = len(matrix) - window
    if n_samples <= 0:
        raise ValueError(
            f"Not enough rows ({len(matrix)}) for window={window}."
        )

    x = np.stack([matrix[i : i + window] for i in range(n_samples)], axis=0)
    y = target[window : window + n_samples]
    return x.astype("float32"), y.astype("float32")


def prepare_data(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> WindowedData:
    """Build chronologically-split windowed tensors.

    The split is applied to the *raw rows first*, then windows are built inside
    each split, so no window ever straddles a split boundary (no leakage).
    """
    feature_cols = tuple(c for c in df.columns if c != DATE_COL)
    values = df[list(feature_cols)].to_numpy(dtype="float32")
    target = df[TARGET_COL].to_numpy(dtype="float32")

    n = len(values)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))

    splits = {
        "train": (values[:train_end], target[:train_end]),
        "val": (values[train_end:val_end], target[train_end:val_end]),
        "test": (values[val_end:], target[val_end:]),
    }

    windowed = {
        name: make_windows(mat, tgt, window) for name, (mat, tgt) in splits.items()
    }

    return WindowedData(
        x_train=windowed["train"][0],
        y_train=windowed["train"][1],
        x_val=windowed["val"][0],
        y_val=windowed["val"][1],
        x_test=windowed["test"][0],
        y_test=windowed["test"][1],
        feature_names=feature_cols,
    )


# --- Model -----------------------------------------------------------------


def build_lstm_model(
    window: int,
    n_features: int,
    adapt_data: np.ndarray | None = None,
    lstm_units: tuple[int, ...] = DEFAULT_LSTM_UNITS,
    dropout: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """Build a stacked-LSTM regressor with the Keras Functional API.

    Architecture::

        Input (window, n_features)
          -> Normalization (adapted on training windows only)
          -> LSTM(64, return_sequences=True) -> Dropout
          -> LSTM(32)                         -> Dropout
          -> Dense(16, relu)
          -> Dense(1)  [next-day return]

    ``adapt_data`` should be the *training* windows; if provided, the
    Normalization layer learns per-feature mean/variance from it.
    """
    if not lstm_units:
        raise ValueError("lstm_units must contain at least one layer width.")

    inputs = keras.Input(shape=(window, n_features), name="window_input")

    # Per-feature normalization over the last axis; adapted on training data.
    normalizer = keras.layers.Normalization(axis=-1, name="feature_norm")
    if adapt_data is not None:
        normalizer.adapt(adapt_data)
    x = normalizer(inputs)

    last_idx = len(lstm_units) - 1
    for i, units in enumerate(lstm_units):
        return_sequences = i != last_idx
        x = keras.layers.LSTM(
            units,
            return_sequences=return_sequences,
            name=f"lstm_{i}",
        )(x)
        x = keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)

    x = keras.layers.Dense(16, activation="relu", name="dense_head")(x)
    outputs = keras.layers.Dense(1, name="next_day_return")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_returns_forecaster")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# --- Training --------------------------------------------------------------


def train(
    window: int = DEFAULT_WINDOW,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dataset_path: Path = DATASET_PATH,
) -> tuple[keras.Model, WindowedData, keras.callbacks.History]:
    """End-to-end: load -> window -> build -> fit, with early stopping."""
    df = load_series(dataset_path)
    data = prepare_data(df, window=window)

    model = build_lstm_model(
        window=data.window,
        n_features=data.n_features,
        adapt_data=data.x_train,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    history = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        shuffle=False,  # preserve temporal order
        verbose=2,
    )
    return model, data, history


def evaluate(model: keras.Model, data: WindowedData) -> dict[str, float]:
    """Report test-set loss/MAE and a naive-baseline comparison."""
    test_loss, test_mae = model.evaluate(data.x_test, data.y_test, verbose=0)

    # Naive baseline: predict tomorrow == today's return (last window value).
    naive_pred = data.x_test[:, -1, data.feature_names.index(TARGET_COL)]
    naive_mae = float(np.mean(np.abs(naive_pred - data.y_test)))

    return {
        "test_mse": float(test_loss),
        "test_mae": float(test_mae),
        "naive_mae": naive_mae,
    }


if __name__ == "__main__":
    model, data, _ = train()
    model.summary()
    metrics = evaluate(model, data)
    print("\n=== Test metrics ===")
    for key, value in metrics.items():
        print(f"{key:>10}: {value:.4f}")
