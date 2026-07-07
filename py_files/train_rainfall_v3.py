import os
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import tensorflow as tf
from . import LandslideRainfall_v3 as _lr3
from .LandslideRainfall_v3 import LandslideRainFallV3

from .data import (
    CategoricalEncoderLayer,
    EmbeddingEncoderLayer,
    NormalizationLayer,
    dataframe_to_dataset,
    log_transform_skewed,
    clip_outliers,
    apply_log_transform,
    apply_clip_thresholds,
)
import sklearn
from matplotlib import pyplot as plt

SKIP_NORMALIZATION = {'soil_texture_idx'}
OOF_FILENAME = 'oof_preds.npy'
FOLD_MANIFEST_TEMPLATE = 'v1_cotabato_transforms_fold{fold}.json'
PRODUCTION_MANIFEST = 'v1_cotabato_transforms_production.json'
PRODUCTION_MODEL_FILENAME = 'production-model-v3.keras'


def _manifest_payload(
    version,
    fold,
    log_cols,
    thresholds,
    skew_threshold,
    clip_lower_pct,
    clip_upper_pct,
    physics_features,
    imputed_indicator_cols=None,
    imputation_medians=None,
):
    """Build the transform-manifest dict shared by fold and production writers."""
    return {
        "version": version,
        "fold": fold,
        "skew_threshold": skew_threshold,
        "clip_lower_pct": clip_lower_pct,
        "clip_upper_pct": clip_upper_pct,
        "physics_features_excluded": sorted(physics_features),
        "log_transformed_cols": list(log_cols),
        "clip_thresholds": {col: list(bounds) for col, bounds in thresholds.items()},
        "imputed_indicator_cols": list(imputed_indicator_cols or []),
        "imputation_medians": dict(imputation_medians or {}),
    }


def _write_manifest(transforms_dir, filename, payload):
    """Persist a transform manifest to ``transforms_dir/filename``."""
    transforms_dir = Path(transforms_dir)
    transforms_dir.mkdir(parents=True, exist_ok=True)
    out_path = transforms_dir / filename
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def _write_fold_manifest(
    transforms_dir,
    fold,
    log_cols,
    thresholds,
    skew_threshold,
    clip_lower_pct,
    clip_upper_pct,
    physics_features,
    imputed_indicator_cols=None,
    imputation_medians=None,
):
    """Write the per-fold transform decisions for inference replay."""
    payload = _manifest_payload(
        f"v1_cotabato_fold{fold}", fold, log_cols, thresholds,
        skew_threshold, clip_lower_pct, clip_upper_pct, physics_features,
        imputed_indicator_cols, imputation_medians,
    )
    return _write_manifest(
        transforms_dir, FOLD_MANIFEST_TEMPLATE.format(fold=fold), payload,
    )


def _derive_and_apply_transforms(
    train_df, holdout_df, numerical_cols, physics_features,
    skew_threshold, clip_lower_pct, clip_upper_pct,
):
    """Derive log/clip decisions from ``train_df`` only, replay on ``holdout_df``.

    Physics features and integer-index columns (``soil_texture_idx``) are
    excluded from both transforms — log1p / clipping would corrupt the real
    units the Newmark physics consumes and the one-hot routing of the
    hydraulic-conductivity layer. Returns
    ``(train_df, holdout_df, log_cols, thresholds)``; ``holdout_df`` is ``None``
    when no replay frame is supplied (whole-dataset training).
    """
    transform_exclude = set(physics_features) | SKIP_NORMALIZATION

    train_df, log_cols = log_transform_skewed(
        train_df, numerical_cols, skew_threshold=skew_threshold,
        exclude=transform_exclude,
    )
    outlier_cols = [c for c in numerical_cols if c not in transform_exclude]
    train_df, thresholds = clip_outliers(
        train_df, outlier_cols, lower_pct=clip_lower_pct, upper_pct=clip_upper_pct,
    )

    if holdout_df is not None:
        holdout_df = apply_log_transform(holdout_df, log_cols)
        holdout_df = apply_clip_thresholds(holdout_df, thresholds)

    return train_df, holdout_df, log_cols, thresholds


def _build_input_graph(numerical_cols, categorical_cols, pga_col, train_ds, categorical_encoder):
    """Build the (all_inputs, pga_input, soil_idx_input, encoded_inputs) tuple.

    Shared by the cross-validation fold loop and the production trainer so both
    construct the physics-layer inputs in exactly the same order (the physics
    layers index ``numeric_cols`` positionally, so ordering must not drift).
    """
    all_inputs, encoded_inputs = [], []
    pga_input, soil_idx_input = None, None

    for header in numerical_cols:
        numerical_col = tf.keras.Input((1,), name=header)
        if header == pga_col:
            pga_input = tf.keras.Input((1,), name=header)
            continue
        if header in SKIP_NORMALIZATION:
            soil_idx_input = numerical_col
            continue
        norm = NormalizationLayer(header, train_ds)
        encoded_numeric = norm(numerical_col)
        all_inputs.append(numerical_col)
        encoded_inputs.append(encoded_numeric)

    for header in categorical_cols:
        categorical_col = tf.keras.Input((1,), name=header, dtype="string")
        if categorical_encoder == 'embedding':
            cat_norm = EmbeddingEncoderLayer(header, train_ds, dtype="string")
        else:
            cat_norm = CategoricalEncoderLayer(header, train_ds, "string")
        encoded_cat = cat_norm(categorical_col)
        all_inputs.append(categorical_col)
        encoded_inputs.append(encoded_cat)

    if pga_input is None:
        raise ValueError("PGA input is none")

    return all_inputs, pga_input, soil_idx_input, encoded_inputs


def _load_fold_manifest(transforms_dir, fold):
    path = Path(transforms_dir) / FOLD_MANIFEST_TEMPLATE.format(fold=fold)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing per-fold transform manifest: {path}. "
            f"Re-run training (cell 14) to regenerate."
        )
    with open(path) as f:
        meta = json.load(f)
    log_cols = meta.get("log_transformed_cols", [])
    thresholds = {col: tuple(bounds) for col, bounds in meta.get("clip_thresholds", {}).items()}
    return log_cols, thresholds


def train_model_rainfall_v3(
    df,
    numerical_cols,
    categorical_cols,
    feature_cols,
    pga_col,
    path: str,
    epochs=200,
    batch_size=128,
    *,
    physics_features=None,
    skew_threshold=1.0,
    clip_lower_pct=1,
    clip_upper_pct=99,
    transforms_dir=None,
    categorical_encoder='onehot',
    imputed_indicator_cols=None,
    imputation_medians=None,
    use_rainfall=True,
):
    """
        Trains rainfall PINN model v3 using stratified KFold.
        Unconstrained coh/ifi + HydraulicConductivityLayerV3 for wetness
        with 12 USDA soil texture classes.

        When ``physics_features`` is provided, log-transform and 1/99-percentile
        clipping are derived **inside** each fold from the fold's own training
        rows (no CV leakage). The validation rows replay the same decisions via
        ``apply_log_transform`` / ``apply_clip_thresholds``. If
        ``transforms_dir`` is also provided, each fold's decisions are written
        to ``v1_cotabato_transforms_fold{N}.json`` for inference replay.

        When ``physics_features`` is ``None``, no transforms are applied — the
        caller is assumed to have already transformed ``df``.
    """
    # Keep the v3 module's canonical numeric_cols in sync with the caller-
    # supplied list so physics layers' numeric_cols.index(name) calls resolve
    # against the SAME ordering used to build all_inputs / encoded_inputs.
    _lr3.numeric_cols = list(numerical_cols)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    predictions = np.zeros(df.shape[0])
    mean_fpr = np.linspace(0, 1, 100)
    aucs, tprs = [], []
    fold = 1

    class_weight = {0: 1, 1: 5}

    for train_idx, val_idx in skf.split(df, df['landslide']):
        # `.copy()` so the in-place mutations inside `log_transform_skewed` /
        # `clip_outliers` never reach the caller's `df`.
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        if physics_features is not None:
            # Categorical / integer-index columns (e.g. soil_texture_idx) are
            # excluded inside _derive_and_apply_transforms: log1p on an integer
            # index silently corrupts one-hot routing inside layers like
            # HydraulicConductivityLayerV3 (cast(int) on log1p(7)=2.08 → 2),
            # collapsing all samples into a single bucket and zeroing gradient
            # for every other bucket.
            train_df, val_df, log_cols, thresholds = _derive_and_apply_transforms(
                train_df, val_df, numerical_cols, physics_features,
                skew_threshold, clip_lower_pct, clip_upper_pct,
            )

            if transforms_dir is not None:
                manifest_path = _write_fold_manifest(
                    transforms_dir,
                    fold,
                    log_cols,
                    thresholds,
                    skew_threshold,
                    clip_lower_pct,
                    clip_upper_pct,
                    physics_features,
                    imputed_indicator_cols=imputed_indicator_cols,
                    imputation_medians=imputation_medians,
                )
                print(f"  Fold {fold}: wrote transform manifest -> {manifest_path}")

        # Single-label dataset is used to adapt normalizers (NormalizationLayer
        # iterates `(features, labels)`); the multi-output wrapper is only
        # passed to `fit`/`predict`.
        train_ds = dataframe_to_dataset(train_df[feature_cols], batch_size=batch_size)
        val_ds = dataframe_to_dataset(val_df[feature_cols], shuffle=False, batch_size=batch_size)

        all_inputs, pga_input, soil_idx_input, encoded_inputs = _build_input_graph(
            numerical_cols, categorical_cols, pga_col, train_ds, categorical_encoder,
        )

        model = LandslideRainFallV3(use_rainfall=use_rainfall)
        model.classification_model(all_inputs, pga_input, soil_idx_input, encoded_inputs)
        model.get_optimizer()
        model.compile_model_dce()

        # Wrap into dual-head datasets with per-sample weights (multi-output
        # models don't support the class_weight kwarg of fit()).
        train_ds_mo = LandslideRainFallV3.to_multi_output_ds(train_ds, class_weight=class_weight)
        val_ds_mo = LandslideRainFallV3.to_multi_output_ds(val_ds)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            f"{path}/fold-{fold}-model-v3.keras",
            monitor="val_final_head_auc",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            save_freq="epoch",
            verbose=0,
        )

        model.model.fit(
            train_ds_mo,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_ds_mo,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_final_head_auc",
                    mode="max",
                    patience=5,
                    restore_best_weights=True,
                ),
                model_checkpoint_callback,
            ],
        )
        y_true = val_df['landslide']
        validation_preds = model.model.predict(val_ds_mo)["final_head"].flatten()
        predictions[val_idx] = validation_preds

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, validation_preds)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        acc = round(sklearn.metrics.balanced_accuracy_score(y_true, validation_preds > 0.5), 2)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Fold {fold} (AUC={auc:.2f}, Acc={acc})")

        fold += 1

    np.save(os.path.join(path, OOF_FILENAME), predictions)

    return predictions, aucs


def train_production_rainfall_v3(
    df,
    numerical_cols,
    categorical_cols,
    feature_cols,
    pga_col,
    path: str,
    epochs=200,
    batch_size=128,
    *,
    physics_features=None,
    skew_threshold=1.0,
    clip_lower_pct=1,
    clip_upper_pct=99,
    transforms_dir=None,
    categorical_encoder='embedding',
    imputed_indicator_cols=None,
    imputation_medians=None,
    val_frac=0.15,
    random_state=42,
    model_filename=PRODUCTION_MODEL_FILENAME,
):
    """Train a single production PINN v3 model on the full v2-8 training set.

    Unlike ``train_model_rainfall_v3`` (which fits five models on 5-fold CV
    splits for evaluation), this fits ONE deployable model on all the data,
    using the identical architecture, class weights, optimizer, loss, and
    early-stopping recipe.

    Early stopping still needs a validation signal, so a small stratified
    holdout (``val_frac``) is carved off purely to drive
    ``monitor="val_final_head_auc"`` — the remaining ~85% is the training set.
    This is a more faithful reproduction of the v2-8 recipe than fixed-epoch
    training (same monitor, same patience, same ``restore_best_weights``) while
    using more data than any single CV fold.

    The log/clip transforms are derived from the training portion only and
    persisted to ``v1_cotabato_transforms_production.json`` (same schema as the
    per-fold manifests) so inference scripts can replay them unchanged. The
    model is saved as ``{path}/{model_filename}``.

    Returns ``(model_path, holdout_auc)``.
    """
    # Keep the v3 module's canonical numeric_cols aligned with the caller's
    # ordering (physics layers index it positionally) — same as the fold loop.
    _lr3.numeric_cols = list(numerical_cols)

    class_weight = {0: 1, 1: 5}

    # Stratified holdout for the early-stopping signal only.
    train_df, holdout_df = train_test_split(
        df,
        test_size=val_frac,
        stratify=df['landslide'],
        random_state=random_state,
    )
    train_df = train_df.copy()
    holdout_df = holdout_df.copy()
    print(
        f"Production split: {len(train_df):,} train / {len(holdout_df):,} holdout "
        f"(val_frac={val_frac}); "
        f"landslide rate train={train_df['landslide'].mean():.3f} "
        f"holdout={holdout_df['landslide'].mean():.3f}"
    )

    log_cols, thresholds = [], {}
    if physics_features is not None:
        train_df, holdout_df, log_cols, thresholds = _derive_and_apply_transforms(
            train_df, holdout_df, numerical_cols, physics_features,
            skew_threshold, clip_lower_pct, clip_upper_pct,
        )
        if transforms_dir is not None:
            payload = _manifest_payload(
                "v1_cotabato_production", "production", log_cols, thresholds,
                skew_threshold, clip_lower_pct, clip_upper_pct, physics_features,
                imputed_indicator_cols, imputation_medians,
            )
            manifest_path = _write_manifest(transforms_dir, PRODUCTION_MANIFEST, payload)
            print(f"  wrote production transform manifest -> {manifest_path}")

    train_ds = dataframe_to_dataset(train_df[feature_cols], batch_size=batch_size)
    holdout_ds = dataframe_to_dataset(
        holdout_df[feature_cols], shuffle=False, batch_size=batch_size,
    )

    all_inputs, pga_input, soil_idx_input, encoded_inputs = _build_input_graph(
        numerical_cols, categorical_cols, pga_col, train_ds, categorical_encoder,
    )

    model = LandslideRainFallV3()
    model.classification_model(all_inputs, pga_input, soil_idx_input, encoded_inputs)
    model.get_optimizer()
    model.compile_model_dce()

    train_ds_mo = LandslideRainFallV3.to_multi_output_ds(train_ds, class_weight=class_weight)
    holdout_ds_mo = LandslideRainFallV3.to_multi_output_ds(holdout_ds)

    model_path = os.path.join(path, model_filename)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_final_head_auc",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        save_freq="epoch",
        verbose=0,
    )

    model.model.fit(
        train_ds_mo,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=holdout_ds_mo,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_final_head_auc",
                mode="max",
                patience=5,
                restore_best_weights=True,
            ),
            model_checkpoint_callback,
        ],
    )

    holdout_preds = model.model.predict(holdout_ds_mo)["final_head"].flatten()
    fpr, tpr, _ = sklearn.metrics.roc_curve(holdout_df['landslide'], holdout_preds)
    holdout_auc = sklearn.metrics.auc(fpr, tpr)
    print(f"\nProduction model holdout AUC: {holdout_auc:.4f}")
    print(f"Saved production model -> {model_path}")

    return model_path, holdout_auc


def regenerate_oof_predictions(df, feature_cols, path: str, batch_size=128, *, transforms_dir=None):
    """Regenerate OOF predictions from already-trained fold checkpoints.

    Used when the notebook is re-opened in a fresh kernel and `oof_preds` is no
    longer in scope but the fold-{N}-model-v3.keras files still exist on disk.
    Saves the resulting array next to the checkpoints as `oof_preds.npy` so
    later cells can load it without recomputing.

    When ``transforms_dir`` is provided, each fold's validation slice is
    transformed using that fold's per-fold manifest (``v1_cotabato_transforms_
    fold{N}.json``) before prediction. This is required when ``df`` is the
    raw frame (no pre-fold log/clip) — otherwise the fold checkpoint, which
    was trained on transformed features, receives raw inputs and produces
    meaningless predictions.
    """
    from tensorflow.keras.models import load_model
    from py_files.GallenModel_v1 import NewmarkActivation

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.zeros(df.shape[0])

    for fold, (_, val_idx) in enumerate(skf.split(df, df['landslide']), start=1):
        ckpt_path = os.path.join(path, f"fold-{fold}-model-v3.keras")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Missing fold checkpoint: {ckpt_path}. Re-run training (cell 14)."
            )
        val_df = df.iloc[val_idx].copy()

        if transforms_dir is not None:
            log_cols, thresholds = _load_fold_manifest(transforms_dir, fold)
            val_df = apply_log_transform(val_df, log_cols)
            val_df = apply_clip_thresholds(val_df, thresholds)

        val_ds = dataframe_to_dataset(
            val_df[feature_cols], shuffle=False, batch_size=batch_size,
        )
        val_ds_mo = LandslideRainFallV3.to_multi_output_ds(val_ds)
        model = load_model(
            ckpt_path,
            custom_objects={"NewmarkActivation": NewmarkActivation},
        )
        predictions[val_idx] = model.predict(val_ds_mo)["final_head"].flatten()
        del model
        tf.keras.backend.clear_session()

    np.save(os.path.join(path, OOF_FILENAME), predictions)
    return predictions


def load_or_regenerate_oof(df, feature_cols, path: str, batch_size=128, *, transforms_dir=None):
    """Load OOF predictions from disk; regenerate from fold checkpoints if missing.

    ``transforms_dir`` is forwarded to ``regenerate_oof_predictions`` so that
    raw ``df`` can be transformed with each fold's manifest before prediction.
    """
    oof_path = os.path.join(path, OOF_FILENAME)
    if os.path.exists(oof_path):
        return np.load(oof_path)
    return regenerate_oof_predictions(
        df, feature_cols, path, batch_size=batch_size, transforms_dir=transforms_dir,
    )
