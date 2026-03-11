from os import name
from sklearn.model_selection import StratifiedKFold
import numpy as np
import tensorflow as tf
from .LandslideRainfall_v3 import LandslideRainFallV3

from .data import CategoricalEncoderLayer, NormalizationLayer, dataframe_to_dataset
import sklearn
from matplotlib import pyplot as plt

SKIP_NORMALIZATION = {'soil_texture_idx'}


def train_model_rainfall_v3(df, numerical_cols, categorical_cols, feature_cols, pga_col, path: str, epochs=200, batch_size=128):
    """
        Trains rainfall PINN model v3 using stratified KFold.
        Unconstrained coh/ifi + HydraulicConductivityLayerV3 for wetness
        with 12 USDA soil texture classes.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    predictions = np.zeros(df.shape[0])
    mean_fpr = np.linspace(0, 1, 100)
    aucs, tprs = [], []
    fold = 1

    for train_idx, val_idx in skf.split(df, df['landslide']):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        pga_input, soil_idx_input = None, None

        train_ds = dataframe_to_dataset(train_df[feature_cols], batch_size=batch_size)
        val_ds = dataframe_to_dataset(val_df[feature_cols], shuffle=False, batch_size=batch_size)

        all_inputs, encoded_inputs = [], []

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

            cat_norm = CategoricalEncoderLayer(header, train_ds, "string")
            encoded_cat = cat_norm(categorical_col)

            all_inputs.append(categorical_col)
            encoded_inputs.append(encoded_cat)

        if pga_input is None:
            raise Exception("PGA input is none")

        print(all_inputs)
        model = LandslideRainFallV3()
        model.classification_model(all_inputs, pga_input, soil_idx_input, encoded_inputs)
        model.get_optimizer()
        model.compile_model_dce()

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            f"{path}/fold-{fold}-model-v3.keras",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            save_freq="epoch",
            verbose=0
        )

        hist = model.model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_ds,
            class_weight={0: 1, 1: 5},
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                model_checkpoint_callback,
            ]
        )
        y_true = val_df['landslide']
        validation_preds = model.model.predict(val_ds)
        predictions[val_idx] = validation_preds.flatten()

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, validation_preds)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        acc = round(sklearn.metrics.balanced_accuracy_score(y_true, validation_preds > 0.5), 2)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Fold {fold} (AUC={auc:.2f}, Acc={acc})")

        fold += 1
