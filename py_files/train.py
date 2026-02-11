from py_files import data, metrics
import importlib
importlib.reload(data)
importlib.reload(metrics)
import tensorflow as tf
from py_files.data import dataframe_to_dataset, dataframe_to_dataset_multi, NormalizationLayer, CategoricalEncoderLayer, ensure_2d
from py_files.metrics import roc_auc_score_multiclass
from sklearn.model_selection import KFold, StratifiedKFold
from py_files.Landslidev2_Old import LandslideV2, LandslideV4
import sklearn
# from metrics import find_best_threshold
import contextily as cx
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt


def find_best_threshold(y_true, y_pred_probs):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred_probs)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh, fpr, tpr

def train_model_folds_multi_class(df, numerical_cols, categorical_cols, feature_cols, idx, activation, optimizer,pga_column, epochs=200, batch_size=128, path=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    losses = []
    predictions = np.zeros(df.shape[0])

    mean_fpr = np.linspace(0, 1, 100)
    aucs, tprs = [], []
    fold = 1
    for train_idx, val_idx, in skf.split(df, df['Landslide1']):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_ds = dataframe_to_dataset_multi(train_df[feature_cols], batch_size=batch_size)
        val_ds = dataframe_to_dataset_multi(val_df[feature_cols], shuffle=False, batch_size=batch_size)

        # train_ds = train_ds.map(ensure_2d)
        # val_ds = val_ds.map(ensure_2d)

        pga_input = None
        all_inputs = []
        encoded_features = []

        #For numerical columns
        for header in numerical_cols:
            numerical_col = tf.keras.Input((1,),name=header)
            if header == pga_column:
                pga_input = numerical_col
                continue
            normalization_layer = NormalizationLayer(header, train_ds)
            encoded_numerical_col = normalization_layer(numerical_col)
            
            all_inputs.append(numerical_col)
            encoded_features.append(encoded_numerical_col)


        #For categorical columns
        for header in categorical_cols:
            categorical_col = tf.keras.Input((1,), name=header, dtype='string')
            if header == 'Geomorphology':
                encoder = CategoricalEncoderLayer(header, train_ds,dtype='float', max_tokens=9)
            else:
                encoder = CategoricalEncoderLayer(header, train_ds, dtype='string', max_tokens=9)

            encoded_categorical_col = encoder(categorical_col)
            all_inputs.append(categorical_col)
            encoded_features.append(encoded_categorical_col)


        if activation == "leaky":
            model = LandslideV2(activation, optimizer, leaky_alpha=0.2)
        else:
            model = LandslideV2(activation, optimizer)

        print(f"All inputs: {all_inputs}")
        print(f"Len inputs: {len(all_inputs)}")
        model.get_multi_classification_model_no_pga(all_inputs, pga_input, encoded_features)
        model.get_optimizer()
        model.compile_multiclass_model()
        

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            f"{path}/fold-{fold}-model-{idx}.keras",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            save_freq="epoch",
            # options=None,
            verbose=0
        )

        hist = model.model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_ds,
            # class_weight = {0: 1, 1: 5},
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                model_checkpoint_callback,
            ]
        )

        losses.append(hist.history['loss'])
        fold += 1
        #Use model to predict using validation data
    #     test_y = val_df['Landslide1'].to_numpy()
    #     preds = model.model.predict(val_ds)
    #     roc_auc_dict = roc_auc_score_multiclass(test_y, preds)

    #     for key, value in roc_auc_dict.items():
    #         fpr, tpr, auc = value
    #         interp_tpr = np.interp(mean_fpr, fpr, tpr)
    #         interp_tpr[0] = 0.0
    #         tprs.append(interp_tpr)
    #         plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Fold {fold} (AUC={auc:.2f})")
        
    #     fold += 1
        
    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    
    # plt.plot(mean_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUC = {mean_auc:.2f})", color='blue')
    # plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    # plt.title(f"PINN NO PGA PREDICTOR ROC Curve")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.legend()
    
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    return losses


def train_model_folds_no_pga_dce(df, numerical_cols, categorical_cols, feature_cols, idx, activation, optimizer,pga_column, epochs=200, batch_size=128, path=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Numeric Cols: {numerical_cols}")
    losses = []
    predictions = np.zeros(df.shape[0])

    mean_fpr = np.linspace(0, 1, 100)
    aucs, tprs = [], []
    fold = 1
    for train_idx, val_idx, in skf.split(df, df['landslide']):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_ds = dataframe_to_dataset(train_df[feature_cols])
        val_ds = dataframe_to_dataset(val_df[feature_cols], shuffle=False)

        pga_input = None
        all_inputs = []
        encoded_features = []

        #For numerical columns
        for header in numerical_cols:
            numerical_col = tf.keras.Input((1,),name=header)
            if header == pga_column:
                pga_input = numerical_col
                continue
            normalization_layer = NormalizationLayer(header, train_ds)
            encoded_numerical_col = normalization_layer(numerical_col)
            
            all_inputs.append(numerical_col)
            encoded_features.append(encoded_numerical_col)


        #For categorical columns
        for header in categorical_cols:
            categorical_col = tf.keras.Input((1,), name=header, dtype='string')

            encoder = CategoricalEncoderLayer(header, train_ds, dtype='string', max_tokens=9)

            encoded_categorical_col = encoder(categorical_col)
            all_inputs.append(categorical_col)
            encoded_features.append(encoded_categorical_col)


        if activation == "leaky":
            model = LandslideV4(activation, optimizer, leaky_alpha=0.2)
        else:
            model = LandslideV4(activation, optimizer)

        print(f"All inputs: {all_inputs}")
        model.classification_model(all_inputs, pga_input, encoded_features)
        model.get_optimizer()
        model.compile_model_dce()
          
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            f"{path}/fold-{fold}-model-{idx}.keras",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            save_freq="epoch",
            # options=None,
            verbose=0
        )

        hist = model.model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_ds,
            class_weight = {0: 1, 1: 5},
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                model_checkpoint_callback,
            ]
        )

        losses.append(hist.history['loss'])
        #Use model to predict using validation data
        test_y = val_df['landslide'].to_numpy()
        preds = model.model.predict(val_ds)
        predictions[val_idx] = preds.flatten()
        best_threshold, fpr, tpr = find_best_threshold(test_y, preds)
        # [fpr, tpr, threshold] = sklearn.metrics.roc_curve(test_y, preds)
        print(f"Best thresholds:{best_threshold}")
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        acc = round(sklearn.metrics.balanced_accuracy_score(test_y, preds > 0.5), 2)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Fold {fold} (AUC={auc:.2f}, Acc={acc})")
        fold += 1
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUC = {mean_auc:.2f})", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"PINN NO PGA PREDICTOR ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return losses


#THIS MODULE CONTAINS THE TRAINING SCRIPT FOR THE MODEL.
#THIS IS CREATED TO MODULARIZE THE TRAINING PROCESS AND VERSIONING
def train_model_folds_no_pga(df, numerical_cols, feature_cols, idx, activation, optimizer, pga_column, epochs=200, batch_size=128, path=None):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    losses = []
    predictions = np.zeros(df.shape[0])

    mean_fpr = np.linspace(0, 1, 100)
    aucs, tprs = [], []
    fold = 1
    for train_idx, val_idx, in skf.split(df, df['landslide']):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_ds = dataframe_to_dataset(train_df[feature_cols])
        val_ds = dataframe_to_dataset(val_df[feature_cols], shuffle=False)

        pga_input = None
        all_inputs = []
        encoded_features = []

        #For numerical columns
        for header in numerical_cols:
            numerical_col = tf.keras.Input((1,),name=header)
            if header == pga_column:
                pga_input = numerical_col
                continue
            normalization_layer = NormalizationLayer(header, train_ds)
            encoded_numerical_col = normalization_layer(numerical_col)
            
            all_inputs.append(numerical_col)
            encoded_features.append(encoded_numerical_col)


        if activation == "leaky":
            model = LandslideV2(activation, optimizer, leaky_alpha=0.2)
        else:
            model = LandslideV2(activation, optimizer)

        print(f"All inputs: {all_inputs}")
        model.get_classification_model_no_pga(all_inputs, pga_input, encoded_features)
        model.get_optimizer()
        model.compile_model()
          
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            f"{path}/fold-{fold}-model-{idx}.keras",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            save_freq="epoch",
            # options=None,
            verbose=0
        )

        hist = model.model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_ds,
            # class_weight = {0: 1, 1: 5},
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                model_checkpoint_callback,
            ]
        )

        losses.append(hist.history['loss'])
        #Use model to predict using validation data
        test_y = val_df['landslide'].to_numpy()
        preds = model.model.predict(val_ds)
        predictions[val_idx] = preds.flatten()
        best_threshold, fpr, tpr = find_best_threshold(test_y, preds)
        # [fpr, tpr, threshold] = sklearn.metrics.roc_curve(test_y, preds)
        print(f"Best thresholds:{best_threshold}")
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        acc = round(sklearn.metrics.balanced_accuracy_score(test_y, preds > 0.5), 2)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Fold {fold} (AUC={auc:.2f}, Acc={acc})")
        fold += 1
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUC = {mean_auc:.2f})", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"PINN NO PGA PREDICTOR ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return losses
        

def train_model_folds(df, numerical_cols, feature_cols, idx,activation, optimizer, folds=10, epochs=100, batch_size=128, path=None):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    fold = 1
    scores = []
    aucs, tprs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    losses = []

    predictions = np.zeros(df.shape[0])
    # for train_idx, val_idx in kf.split(df):
    for train_idx, val_idx in skf.split(df, df['landslide']):
        
        #split data set
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    
        train_ds = dataframe_to_dataset(train_df[feature_cols])
        val_ds = dataframe_to_dataset(val_df[feature_cols], shuffle=False)
        all_inputs = []
        encoded_features = []

        #normalize feature inputs
        for header in numerical_cols:
            numerical_col = tf.keras.Input((1,),name=header)
            normalization_layer = NormalizationLayer(header, train_ds)
            encoded_numerical_col = normalization_layer(numerical_col)
            all_inputs.append(numerical_col)
            encoded_features.append(encoded_numerical_col)
        if activation == "leaky":
            model = LandslideV2(activation, optimizer, leaky_alpha=0.2)
        else:
            model = LandslideV2(activation, optimizer)

        model.get_classification_model(all_inputs, encoded_features)
        model.get_optimizer()
        model.compile_model()

        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            f"{path}/fold-{fold}-model-{idx}.keras",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            save_freq="epoch",
            # options=None,
            verbose=0
        )

        hist = model.model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_ds,
            class_weight = {0: 1, 1: 5},
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                model_checkpoint_callback,
            ]
        )

        losses.append(hist.history['loss'])
        #Use model to predict using validation data
        test_y = val_df['landslide'].to_numpy()
        preds = model.model.predict(val_ds)
        predictions[val_idx] = preds.flatten()
        best_threshold, fpr, tpr = find_best_threshold(test_y, preds)
        # [fpr, tpr, threshold] = sklearn.metrics.roc_curve(test_y, preds)
        print(f"Best thresholds:{best_threshold}")
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        acc = round(sklearn.metrics.balanced_accuracy_score(test_y, preds > 0.5), 2)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Fold {fold} (AUC={auc:.2f}, Acc={acc})")
        fold += 1
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUC = {mean_auc:.2f})", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"PINN ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return losses
def trainmodel(model,train_ds,val_ds, idx):
    NUMBER_EPOCHS = 200
    filepath=f'/Users/giogonzales/Documents/ml-prep/ML-PREP-2025/learn/trainedWeights/trainedCotabatoPhase7/PINN-{idx}.keras'
    BATCH_SIZE=128
    
    model_checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor="val_auc",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        save_freq="epoch",
    )
    # print(type(train_ds))
    hist = model.fit(train_ds,
                     epochs=NUMBER_EPOCHS,
                     batch_size=BATCH_SIZE,
                     validation_data=val_ds,
                    #  validation_split=0.2,#auto validate using 20% of random samples at each epoch
                     class_weight = {0: 1, 1: 5},
                     callbacks=[
                            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                            model_checkpoint_callback,
                        ]
                    )

    return hist
def predict_model(model, val_ds):
    predictions = model.predict(val_ds)
    return predictions
    
def plot_susceptibility_map(gdf, predictions, label_name):
    gdf[f'sus_pinn_{label_name}'] = predictions
    norm = mcolors.Normalize(vmin=0, vmax=1.0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Remove legend=True here
    gdf.plot(column=f'sus_pinn_{label_name}', cmap='plasma_r', ax=ax, norm=norm)
    # gdf.plot(column=f'Landslide', cmap='plasma_r', ax=ax, norm=norm)

    # Add single custom colorbar
    sm = plt.cm.ScalarMappable(cmap='plasma_r', norm=norm)
    # sm._A = []  
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    cbar.set_ticklabels(["0.0", "0.125", "0.375", "0.625", "0.875", "1.0"])

    ax.set_title(f"CNN Susceptibility Map - {label_name}")
    cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
    plt.tight_layout()
    plt.show()

