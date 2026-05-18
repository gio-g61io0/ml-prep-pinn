import math
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List
# from Landslidev2_Old import LandslideV2


#THIS MODULE CONTAINS DATA LOADING AND PREPROCESSING FUNCTIONS
#THIS IS CREATED TO MODULARIZE THE DATA PREPROCESSING AND VERSIONING


MISSINGNESS_SUFFIX = '_was_imputed'


def add_missingness_indicators(df, cols, suffix=MISSINGNESS_SUFFIX):
    """Add a binary indicator column for each input col, computed from NaN mask.

    Must be called BEFORE any fillna so the indicator reflects the original
    missingness. The new columns are named ``f"{col}{suffix}"`` and live in
    ``df`` alongside the original (still-NaN) columns; downstream callers are
    expected to fillna the originals after this.

    Returns (df, indicator_cols).
    """
    indicator_cols = []
    for col in cols:
        if col not in df.columns:
            continue
        name = f"{col}{suffix}"
        df[name] = df[col].isnull().astype(np.int8)
        indicator_cols.append(name)
    return df, indicator_cols


def apply_imputation_medians(df, medians):
    """Fill NaNs in ``df`` using training-derived medians from a manifest.

    ``medians`` is the ``{col: median_value}`` dict produced by
    ``preprocessing_v2(track_imputation=True)`` and persisted into each
    per-fold manifest. Columns absent from the dataframe are silently
    skipped so a single dict can cover datasets that don't have every
    training column (e.g. raw validation files missing some inputs).

    Replaces the previous practice of computing a fresh median from
    validation data, which collapsed near-fully-missing columns (e.g.
    ``Prc_mean``) to a value far from training's distribution.
    """
    for col, value in medians.items():
        if col not in df.columns:
            continue
        df[col] = df[col].fillna(float(value))
    return df


def apply_missingness_indicators(df, indicator_cols, source_suffix=MISSINGNESS_SUFFIX):
    """Inference-time replay: ensure each indicator col exists in ``df``.

    For each name in ``indicator_cols`` (formatted ``f"{source}{suffix}"``),
    derive the indicator from the corresponding source column's NaN mask if
    the indicator column isn't already populated. Source columns missing from
    the dataframe are silently skipped (the indicator is created as all-zeros
    to keep the model input shape stable).
    """
    for name in indicator_cols:
        if not name.endswith(source_suffix):
            continue
        source = name[: -len(source_suffix)]
        if source in df.columns and name not in df.columns:
            df[name] = df[source].isnull().astype(np.int8)
        elif source not in df.columns and name not in df.columns:
            df[name] = np.int8(0)
    return df

def preprocessing(df, columns_drop):
    df.drop(columns=columns_drop, inplace=True)
    df = df[df['Slope_mean'] >= 10]

    columns = list(df.columns)
    df.dropna(subset=list(columns), inplace=True) #cleans the dataframe by removing null rows for all columns

    columns = manipulate_cols(columns, remove_cols=['DN', 'BD_mean', 'geometry', 'PGA2_max', 'Soil Type', 'description', 'descriptio'])
    numeric_cols = [col for col in columns if col not in ['landslide', 'type', 'Landslide1', 'LITHODESC']]

    return df, columns, numeric_cols


def preprocessing_v2(df, columns_drop, label_col='landslide', track_imputation=False):
    """Enhanced preprocessing with dropped-row reporting.

    Same logic as ``preprocessing`` but prints how many rows (and how many
    positive labels) are lost at each filtering step.

    Missing values are handled the same way as the validation pipeline in
    ``notebooks/cotabato_new_slope_unit_v2-8.ipynb`` cell 20: numeric model
    inputs are median-imputed, and ``dropna`` is restricted to columns that
    cannot be safely imputed (the label and the categorical ``type``).
    Columns scheduled to be removed from the model inputs by
    ``manipulate_cols`` (e.g. ``BD_mean``, ``geometry``) no longer cause
    row drops for their NaNs.

    When ``track_imputation=True``, two extra outputs are returned:

    - ``indicator_cols``: list of ``{col}_was_imputed`` binary columns added
      for every numeric model input that had at least one NaN before the
      median fill. Useful when training itself contains missing values.
    - ``imputation_medians``: dict ``{col: median_value}`` capturing the
      training-derived median for every numeric model input. Persisted in
      each fold's manifest so the validation pipeline can replay the same
      fill value via ``apply_imputation_medians`` instead of computing a
      fresh (and usually divergent) median from its own data.

    Returns ``(df, columns, numeric_cols)`` by default, or
    ``(df, columns, numeric_cols, indicator_cols, imputation_medians)``
    when ``track_imputation``.
    """
    n_start = len(df)
    n_ls_start = int(df[label_col].sum()) if label_col in df.columns else None

    df.drop(columns=[c for c in columns_drop if c in df.columns], inplace=True)

    # Slope filter
    slope_mask = df['Slope_mean'] >= 10
    n_slope_drop = (~slope_mask).sum()
    ls_slope_drop = int(df.loc[~slope_mask, label_col].sum()) if label_col in df.columns else 0
    df = df[slope_mask]

    # Resolve the actual model-input column list BEFORE any null handling so
    # NaNs in soon-to-be-discarded columns can't take valid rows with them.
    columns = manipulate_cols(
        list(df.columns),
        remove_cols=['DN', 'BD_mean', 'geometry', 'PGA1_max', 'Soil Type', 'description', 'descriptio'],
    )
    numeric_cols = [col for col in columns if col not in ['landslide', 'type', 'Landslide1', 'LITHODESC']]

    # Inspect nulls on model inputs only, BEFORE imputing, so the printed
    # report reflects the real missingness rather than a post-fillna zero.
    null_counts = df[columns].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    imputed_per_col = {c: int(df[c].isnull().sum()) for c in numeric_cols if df[c].isnull().any()}

    # Capture missingness indicators BEFORE the median fill so they reflect
    # the real NaN mask rather than the post-imputation zeros.
    indicator_cols = []
    if track_imputation and imputed_per_col:
        df, indicator_cols = add_missingness_indicators(df, list(imputed_per_col.keys()))
        # Indicators become first-class model features.
        columns = columns + indicator_cols
        numeric_cols = numeric_cols + indicator_cols

    # Median-impute numeric model inputs (matches validation cell 20).
    imputation_medians = {}
    if numeric_cols:
        impute_targets = [c for c in numeric_cols if c not in indicator_cols]
        if impute_targets:
            medians_series = df[impute_targets].median(numeric_only=True)
            imputation_medians = {
                c: float(medians_series[c])
                for c in impute_targets
                if c in medians_series.index and pd.notna(medians_series[c])
            }
            df[impute_targets] = df[impute_targets].fillna(medians_series)

    # Drop rows only when a non-imputable column is missing.
    dropna_cols = [c for c in (label_col, 'type') if c in df.columns]
    n_before_dropna = len(df)
    if dropna_cols:
        df.dropna(subset=dropna_cols, inplace=True)
    n_na_drop = n_before_dropna - len(df)

    print("  Preprocessing report:")
    print(f"    Starting rows:        {n_start}  (landslide={n_ls_start})")
    print(f"    Dropped (Slope < 10): {n_slope_drop}  (landslide={ls_slope_drop})")
    if len(cols_with_nulls) > 0:
        print(f"    Columns with nulls (model inputs):   {dict(cols_with_nulls)}")
    if imputed_per_col:
        print(f"    Median-imputed numeric (per col):    {imputed_per_col}")
    if indicator_cols:
        print(f"    Missingness indicators added:        {indicator_cols}")
    print(f"    Dropped (NaN in {dropna_cols or 'n/a'}): {n_na_drop}")
    print(f"    Final rows:           {len(df)}  (landslide={int(df[label_col].sum()) if label_col in df.columns else '?'})")

    if track_imputation:
        return df, columns, numeric_cols, indicator_cols, imputation_medians
    return df, columns, numeric_cols


def log_transform_skewed(df, numeric_cols, skew_threshold=1.0, exclude=None):
    """Apply log1p to right-skewed numeric features.

    Parameters
    ----------
    df : DataFrame
    numeric_cols : list of str
    skew_threshold : float
        Only transform columns with |skewness| > this value.
    exclude : set, optional
        Column names to skip (e.g. target, soil index).

    Returns
    -------
    df : DataFrame (modified in-place)
    transformed_cols : list of str
    """
    if exclude is None:
        exclude = set()
    transformed_cols = []
    for col in numeric_cols:
        if col in exclude:
            continue
        skew = df[col].skew()
        if abs(skew) > skew_threshold and df[col].min() >= 0:
            df[col] = np.log1p(df[col])
            transformed_cols.append(col)
            print(f"    log1p({col})  skew was {skew:.2f}")
    return df, transformed_cols


def clip_outliers(df, numeric_cols, lower_pct=1, upper_pct=99, exclude=None):
    """Clip numeric features at percentile bounds.

    Returns (clipped DataFrame, thresholds) where thresholds is a dict
    {col: (lower, upper)} for every column that had at least one value
    outside the bounds and was actually clipped. Inference pipelines must
    persist this dict and replay it via `apply_clip_thresholds` instead of
    re-deriving percentiles on different data.
    """
    if exclude is None:
        exclude = set()
    thresholds = {}
    for col in numeric_cols:
        if col in exclude:
            continue
        lo = float(np.percentile(df[col], lower_pct))
        hi = float(np.percentile(df[col], upper_pct))
        n_clipped = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_clipped > 0:
            df[col] = df[col].clip(lo, hi)
            thresholds[col] = (lo, hi)
            print(f"    Clipped {col}: [{lo:.4f}, {hi:.4f}] ({n_clipped} values)")
    return df, thresholds


def apply_clip_thresholds(df, thresholds):
    """Apply pre-computed clip thresholds at inference time.

    `thresholds` must be the dict returned by `clip_outliers` at training
    time (typically loaded from a transform manifest JSON). Columns absent
    from the dataframe are silently skipped.
    """
    for col, bounds in thresholds.items():
        if col not in df.columns:
            continue
        lo, hi = bounds
        df[col] = df[col].clip(float(lo), float(hi))
    return df


def apply_log_transform(df, log_cols):
    """Apply log1p to a fixed list of columns at inference time.

    `log_cols` must be the list returned by `log_transform_skewed` at
    training time (typically loaded from a transform manifest JSON).
    Columns absent from the dataframe are silently skipped; columns with
    any negative values are skipped with a warning to mirror training's
    `df[col].min() >= 0` guard.
    """
    for col in log_cols:
        if col not in df.columns:
            continue
        if df[col].min() < 0:
            print(f"    [warn] {col} has negatives; skipping log1p (would error)")
            continue
        df[col] = np.log1p(df[col])
    return df


def check_feature_correlation(df, numeric_cols, threshold=0.9):
    """Flag highly correlated feature pairs.

    Returns a DataFrame of pairs with |correlation| > threshold.
    """
    corr = df[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if val > threshold:
                pairs.append({"feature_1": idx, "feature_2": col, "correlation": val})
    result = pd.DataFrame(pairs).sort_values("correlation", ascending=False)
    if len(result) > 0:
        print(f"  Highly correlated pairs (|r| > {threshold}):")
        print(result.to_string(index=False))
    else:
        print(f"  No feature pairs with |r| > {threshold}")
    return result

def manipulate_cols(columns, remove_cols) -> List:
    return [col for col in columns if col not in remove_cols]

def dataframe_to_input_list(df, sampling_columns) -> List[np.ndarray]:
    return [df[col].values.reshape(-1, 1) for col in sampling_columns]


def dataframe_to_dataset_multi(df, shuffle=True, batch_size=128, seed=None):
    """
        Transforms a dataframe into ({dict}, labels) Dataset
    """

    labels = df.pop('Landslide1')
    encoded_labels = encode_ordinal(labels)

    print(f"Encoded labels: {encoded_labels}")

    ds = tf.data.Dataset.from_tensor_slices((dict(df), encoded_labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed)
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def dataframe_to_dataset(df, shuffle=True, batch_size=32, seed=None):
    labels = df.pop('landslide')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def dataframe_to_dataset_no_pga(df, shuffle=True, batch_size=32, seed=None):
    labels = df.pop('landslide')

    pga = df.pop('PGA1_max')

    features = dict(df)

    inputs = {
        "features":features,
        "pga":pga
    }

    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

class NormalizationLayerNoPga(tf.keras.layers.Layer):
    def __init__(self, name, dataset) -> None:
        super().__init__()
        self.name = name
        values = []
        for inputs , labels in dataset:
            col = inputs['features'][name].numpy()
            values.append(col)

        values = np.concatenate(values, axis=0)

        self.normalizer = tf.keras.layers.Normalization(axis=None)
        self.normalizer.adapt(values)  # the layers learns to normalize input data
    
    def __call__(self, feature):
        return self.normalizer(feature)

    def get_config(self):
        config = super().get_config()
        config.update({"name": self.name})
        return config
    
class NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, name, dataset) -> None:
        super().__init__()
        self.name = name
        values = []
        for features, labels in dataset:
            col = features[name].numpy()
            values.append(col)

        values = np.concatenate(values, axis=0)

        self.normalizer = tf.keras.layers.Normalization(axis=None)
        self.normalizer.adapt(values)  # the layers learns to normalize input data

    def __call__(self, feature):
        return self.normalizer(feature)

    def get_config(self):
        config = super().get_config()
        config.update({"name": self.name})
        return config
    
class CategoricalEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, name, dataset, dtype, max_tokens=None):
        super().__init__()
        self.name = name
        values = []
        if dtype == "string":
            print("dtype is a string")
            self.index = tf.keras.layers.StringLookup(max_tokens=max_tokens)
        else:
            self.index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)

        for features, labels in dataset:
            feature_col = features[self.name]
            values.append(feature_col)

        values = np.concatenate(values, axis=0)
        self.index.adapt(values)
        self.encoder = tf.keras.layers.CategoryEncoding(
            num_tokens=self.index.vocabulary_size()
        )

    def __call__(self, feature):
        return self.encoder(self.index(feature))

    def get_config(self):
        config = super().get_config()
        config.update({"name": self.name})
        return config


@tf.keras.utils.register_keras_serializable()
class EmbeddingEncoderLayer(tf.keras.layers.Layer):
    """Dense embedding alternative to ``CategoricalEncoderLayer``.

    Replaces ``StringLookup``/``IntegerLookup`` -> ``CategoryEncoding`` (one-hot)
    with ``StringLookup``/``IntegerLookup`` -> ``Embedding``. OOV indices map
    to a dedicated learnable embedding instead of an all-zero one-hot, which
    is useful when the validation vocabulary diverges from training (e.g. the
    Merged_PINN_Features_2 ``type`` column has 18 levels never seen during
    training).

    ``embed_dim`` defaults to ``max(2, ceil(sqrt(vocab_size)))`` (a common
    rule of thumb for moderate-cardinality categoricals).
    """

    def __init__(self, name, dataset=None, dtype='string', max_tokens=None, embed_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.layer_name = name
        self.dtype_kind = dtype
        self.max_tokens = max_tokens

        if dtype == 'string':
            self.index = tf.keras.layers.StringLookup(max_tokens=max_tokens)
        else:
            self.index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)

        if dataset is not None:
            values = []
            for features, _labels in dataset:
                values.append(features[name])
            values = np.concatenate(values, axis=0)
            self.index.adapt(values)

        vocab_size = self.index.vocabulary_size()
        if embed_dim is None:
            embed_dim = max(2, math.ceil(math.sqrt(max(vocab_size, 1))))
        self.embed_dim = int(embed_dim)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=self.embed_dim, mask_zero=False,
        )
        # Embedding output is (batch, 1, embed_dim) when the input is (batch, 1).
        # Flatten collapses it back to (batch, embed_dim) so the encoder's output
        # is shape-compatible with the one-hot CategoryEncoding it replaces. Avoids
        # `emb.shape.rank` which is a tuple (not TensorShape) under TF 2.19 / Keras 3.
        self.flatten = tf.keras.layers.Flatten()

    def __call__(self, feature):
        idx = self.index(feature)
        emb = self.embedding(idx)
        return self.flatten(emb)

    def get_config(self):
        config = super().get_config()
        config.update({
            'name': self.layer_name,
            'dtype': self.dtype_kind,
            'max_tokens': self.max_tokens,
            'embed_dim': self.embed_dim,
        })
        return config
    
# def bootstrap_geotech_resampling(df, columns,numerical_cols, filepath, n_bootstrap=50):

#     pga_column = "PGA1_max"
#     categorical_cols = ['type']

#     for i in range(1, n_bootstrap + 1):
#         all_inputs = []
#         encoded_features = []

#         train_df = resample(df[columns], random_state=None, n_samples=10_000, replace=False)
#         test_df = df[~df.type.isin(train_df.type)]
#         print(f"Number of train set{len(train_df)} and number of test set{len(test_df)}")

#         train_ds = dataframe_to_dataset(train_df[columns], batch_size=32)
#         test_ds = dataframe_to_dataset(test_df[columns], batch_size=32)
#         y_test = test_df['landslide'].to_numpy()
       
#         for header in numerical_cols:
#             numerical_col = tf.keras.Input((1,),name=header)
#             if header == pga_column:
#                 pga_input = numerical_col
#                 continue
#             normalization_layer = NormalizationLayer(header, train_ds)
#             encoded_numerical_col = normalization_layer(numerical_col)
            
#             all_inputs.append(numerical_col)
#             encoded_features.append(encoded_numerical_col)


#         #For categorical columns
#         for header in categorical_cols:
#             categorical_col = tf.keras.Input((1,), name=header, dtype='string')

#             encoder = CategoricalEncoderLayer(header, train_ds, dtype='string', max_tokens=9)

#             encoded_categorical_col = encoder(categorical_col)
#             all_inputs.append(categorical_col)
#             encoded_features.append(encoded_categorical_col)
#         model = LandslideV2("leaky", "adam", 0.2)
#         model.get_classification_model_no_pga(all_inputs, pga_input, encoded_features)
#         model.get_optimizer()
#         model.compile_model()
#         trainmodel_geotech(model.model, train_ds, test_ds)
#         del model.model, model

#         model = tf.keras.models.load_model("geotechmodel.keras")

#         all_data = dataframe_to_dataset(df[columns], shuffle=False)
#         cohesion_geotech = tf.keras.Model(inputs=model.input, outputs=model.get_layer("cohesion_clip").output)
#         cohesion_geotech_preds = cohesion_geotech.predict(all_data)

#         ifi_geotech = tf.keras.Model(inputs=model.input, outputs=model.get_layer("ifi_clip").output)
#         ifi_geotech_preds = ifi_geotech.predict(all_data)

#         np.save(f"{filepath}/cohesion_geotech_preds_{i}.npy", cohesion_geotech_preds)
#         np.save(f"{filepath}/ifi_geotech_preds_{i}.npy", ifi_geotech_preds)
#         del cohesion_geotech
#         del ifi_geotech
#         del model
#         tf.keras.backend.clear_session()

def ensure_2d(features, labels):
    for k, v in features.items():
        if v.shape.rank == 1:
            features[k] = tf.expand_dims(v, axis=-1)
    return features, labels


def encode_ordinal(y, num_classes=4):

    """
        This encodes the ordinal labels into a binary matrix.
    """
    y = np.array(y)

    k_1 = num_classes - 1

    encoded = np.zeros((len(y), k_1))

    for i in range(k_1):
        encoded[:, i] = (y > i).astype(int)
    return encoded