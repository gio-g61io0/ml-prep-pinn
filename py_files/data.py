import tensorflow as tf
import numpy as np
from typing import List
from py_files.GallenModel_v1 import  NewmarkActivation, DisplacementLayer, LandslideActivationLayer, CohesionLayer, InternalFrictionLayer, ClipLayer
# from Landslidev2_Old import LandslideV2
from sklearn.utils import resample


#THIS MODULE CONTAINS DATA LOADING AND PREPROCESSING FUNCTIONS
#THIS IS CREATED TO MODULARIZE THE DATA PREPROCESSING AND VERSIONING

def preprocessing(df, columns_drop):
    df.drop(columns=columns_drop, inplace=True)
    df['type'].head()
    df = df[df['Slope_mean'] >= 10]

    columns = list(df.columns)
    df.dropna(subset=list(columns), inplace=True) #cleans the dataframe by removing null rows for all columns
    
    columns = manipulate_cols(columns, ['DN', 'BD_mean', 'geometry', 'PGA2_max', 'Soil Type', 'description', 'descriptio'])
    numeric_cols = [col for col in columns if col not in ['landslide', 'type', 'Landslide1', 'LITHO', 'Lithology', 'Geomorphology']]
    
    return df, columns, numeric_cols

def manipulate_cols(columns, remove_cols) -> List:
    return [col for col in columns if col not in remove_cols]

def dataframe_to_input_list(df, sampling_columns) -> List[np.ndarray]:
    return [df[col].values.reshape(-1, 1) for col in sampling_columns]


def dataframe_to_dataset_multi(df, shuffle=True, batch_size=128):
    """
        Transforms a dataframe into ({dict}, labels) Dataset
    """
    
    labels = df.pop('Landslide1')
    encoded_labels = encode_ordinal(labels)

    print(f"Encoded labels: {encoded_labels}")
    
    ds = tf.data.Dataset.from_tensor_slices((dict(df), encoded_labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def dataframe_to_dataset(df, shuffle=True, batch_size=32):
    labels = df.pop('landslide')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def dataframe_to_dataset(df, shuffle=True, batch_size=32):
    labels = df.pop('landslide')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def dataframe_to_dataset_no_pga(df, shuffle=True, batch_size=32):
    labels = df.pop('landslide')

    pga = df.pop('PGA1_max')

    features = dict(df)

    inputs = {
        "features":features,
        "pga":pga
    }

    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
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