import numpy as np
from typing import Optional
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import layers, optimizers, losses, metrics, Model, Input
import sklearn
from matplotlib import pyplot as plt
from GallenModel import (
    CriticalAcceleration,
    DisplacementIntermediate,
    DisplacementLayerEvent,
    DisplacementLayerNepal,
    ExedanceCotabatoLayer,
    NewmarkActivation,
    CohesionLayer,
    InternalFrictionLayer,
    DisplacementLayerV2,
    FosLayer,
    DisplacementIntermediate,
    CohesionCotabatoLayer,
    NormalizationLayer,
    NewmarkActivationV2
)
from data import dataframe_to_dataset

numeric_cols_nepal = [
    "Est_m",
    "Nrt_m",
    "HC_m",
    "VC_m",
    "Slp_m",
    "Prc_m",
    "NDVI_m",
    'estimated_pga',
    # 'PGA_Usgs',
    "Sand_m",
    "Silt_m",
    "Clay_m",
    "Bdod_m",
    'Distf_m',
    'GLG'
]  # numeric cols of nepal

numeric_col_cotabato = [
    "Clay_mean",
    "Sand_mean",
    "Silt_mean",
    "Est_mean",
    "Nrt_mean",
    "HorCurv_me",
    "VertCurv_m",
    "Slope_mean",
    "PGA_mean",
    "NDVI_mean",
    "Prc_mean",
    "blk-unit_m",
    "wtr-cont_m",
    "Elev_mean",
    "Gmrp_median",
    "disr_mean",
    "driv_mean",
]

numeric_cols_cotabato_v2 = [
    "Clay_mean",
    "Sand_mean",
    "Silt_mean",
    "NDVI_mean",
    "Est_mean",
    "Nrt_mean",
    "HorCurv_mean",
    "VertCurv_mean",
    "Slope_mean",
    "Elev_mean",
    "SoilThc_mean",
    "DistFlt_min",
    "LULC_majority",
    "TWI_mean",
    # "PGA2_max",
    "PGA1_max",
    "Prc_mean",
    "Distrv_min",
    "distrd_min",
    "Soil Type",
    "BUK_mean",
]


def calculate_newmark(coh, ifi, pga, bulk_unit_weight, slope):

        # coh = 0
        # ifi = -29.604 + 34.220 * (bulk_unit_weight / 9.81)
        slope *= 0.017453292519943295
        pga *= 10.0
        coh *= 1000.0
        # bulk_density *= 1000.0  # g/cm^3 to kg/m^3
        slope_normal_thickness = 3.33  # m

        # cohesion_t = coh
        # bulk_density = tf.expand_dims(bulk_density, 1)

        #NOTE:: Cohesion Term of the FOS
        cohesion_term = coh / ((bulk_unit_weight * slope_normal_thickness) * tf.math.sin(slope))

        #NOTE:: Internal Friction Angle Term of the FOS
        friction_angle_term = tf.math.tan(ifi) / tf.math.tan(slope)

        safety_factor = cohesion_term + friction_angle_term

        safety_factor = tf.clip_by_value(
            safety_factor, 1.2, 15.0
        )  # NOTE:: Performance changes when adding this line

        ac = (
            (safety_factor - 1) * 9.81 * tf.math.sin(slope)
        )  # NOTE::Critical Acceleration
        acpg = ac / pga

        acpg = tf.clip_by_value(acpg, 0.001, 0.999)

        powcomp = tf.math.pow((1 - acpg), 2.341) * tf.math.pow(acpg, -1.438)
        logds = 0.215 + tf.math.log(powcomp) + 0.51  # NOTE:: Newmark Displacement


        return tf.math.exp(logds), safety_factor

@tf.keras.utils.register_keras_serializable()
class CotabatoModel(tf.keras.Model):
    def __init__(self, features, **kwargs) -> None:

        super().__init__(**kwargs)

        self.features = features
        self.depth = 12
        # self.normalizer = {col: NormalizationLayer(col, train_ds) for col in features}

        # self.normalizer = {col: tf.keras.layers.Normalization(axis=None) for col in features}
        self.normalizers = [tf.keras.layers.Normalization(axis=None) for col in features]
        
        self.concat = layers.Concatenate()

        self.first_layer = layers.Dense(units=64, activation="relu",kernel_initializer="random_normal", bias_initializer="random_normal")

        self.hidden_layers = [layers.Dense(units=64, name=f"Sus_{i}",
                activation='relu', 
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
            ) for i in range(1, self.depth + 1)]

        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.5)
        self.geo_layer = layers.Dense(units=2, name="geotechnical",
                activation='relu',
                kernel_initializer="random_normal",
                bias_initializer="random_normal", )

        self.cohesion_layer = CohesionLayer("relu")

        self.internal_friction_layer = InternalFrictionLayer()

        self.displacement_layer = layers.Dense(units=1, name="displacement", activation=None)
        self.sigmoid = NewmarkActivationV2(threshold=3.0)
        # self.sigmoid = tf.keras.activations.Sig()


    def adapt_normalizers(self, input_list):
        for values, norm in zip(input_list, self.normalizers):
            norm.adapt(values)


    def call(self, inputs):

        # normalized_inputs = [self.normalizer[col](tensor) for col, tensor in inputs.items()]
        normalized_inputs = [norm(inputs[col]) for col, norm in zip(self.features, self.normalizers)]

        x = self.concat(normalized_inputs)

        x = self.first_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.batch_norm(x)
        x = self.dropout(x)
        geo = self.geo_layer(x)

        coh = self.cohesion_layer(geo)
        ifi = self.internal_friction_layer(geo)

        coh = tf.expand_dims(coh, axis=1)   # shape: [batch_size, 1]
        ifi = tf.expand_dims(ifi, axis=1)   # shape: [batch_size, 1]

        combined = tf.concat([coh, ifi], axis=1)

        displacement = self.displacement_layer(combined)

        sus = self.sigmoid(displacement)

       
        
        return sus
    
    @classmethod
    def from_config(cls, config):  # For deserialization purpose
        return cls(**config)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "features":self.features,
            # "input_dict":self.input_dict,
        })
        return config

class ModifiedNepalModel:
    def __init__(
        self, activation, optimizer, leaky_alpha: Optional[float] = None
    ) -> None:
        self.depth = 12
        self.activation = activation
        self.optimizer = optimizer
        self.leaky_alpha = leaky_alpha

    def get_classification_model(self, all_inputs, encoded_features):

        all_features = tf.keras.layers.concatenate(encoded_features)
        features_only = all_features
        slope = all_inputs[numeric_cols_nepal.index("Slp_m")]
        pga = all_inputs[numeric_cols_nepal.index("estimated_pga")]
        # pga = all_inputs[numeric_cols_nepal.index("PGA_Usgs")]
        bulk_density = all_inputs[numeric_cols_nepal.index("Bdod_m")]

        x = layers.Dense(
            units=64,
            name="Sus_0",
            kernel_initializer="random_normal",
            bias_initializer="random_normal",
        )(features_only)

        for i in range(1, self.depth + 1):
            x = layers.Dense(
                units=64,
                name=f"Sus_{i}",
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
            )(x)
            x = layers.BatchNormalization()(x)
            if self.activation == "prelu":
                x = layers.PReLU()(x)
            elif self.activation == "relu":
                x = layers.ReLU()(x)
            elif self.activation == "leaky":
                if self.leaky_alpha == None:
                    raise Exception("Missing leaky ReLU alpha value")
                x = layers.LeakyReLU(negative_slope=self.leaky_alpha)(x)
            else:
                x = layers.ReLU()(x)  # defaults to ReLU activation if none of the above

        x = layers.Dropout(0.1)(x)  # NOTE:: adding dropout layer for reguralization
        x = layers.Dense(units=2, name="geotechnical_param")(x)

        if self.activation == "prelu":
            x = layers.PReLU()(x)
        elif self.activation == "relu":
            x = layers.ReLU()(x)
        elif self.activation == "leaky":
            if self.leaky_alpha == None:
                raise Exception("Missing leaky ReLU alpha value")
            x = layers.LeakyReLU(negative_slope=self.leaky_alpha)(x)
        else:
            print("default activation")
            x = layers.ReLU()(x)  # defaults to ReLU activation if none of the above

        # if self.activation == "leaky":
        #     coh = CohesionLayer(self.activation, self.leaky_alpha)(x)
        # else:
        #     coh = CohesionLayer(self.activation)(x)
        coh = CohesionLayer("relu")(x)

        ifi = InternalFrictionLayer()(x)

        ds, safety_factor, ac = DisplacementLayerV2()(
            [coh, ifi, slope, pga, bulk_density]
        )

        ds = DisplacementIntermediate()(ds)
        safety_factor = FosLayer()(safety_factor)

        ac = CriticalAcceleration()(ac)
        exedance = ExedanceCotabatoLayer()(ac, pga)

        if self.activation == "prelu":
            ds = layers.PReLU()(ds)
        elif self.activation == "relu":
            ds = layers.ReLU()(ds)
        elif self.activation == "leaky":
            if self.leaky_alpha == None:
                raise Exception("Missing leaky ReLU alpha value")
            ds = layers.LeakyReLU(negative_slope=self.leaky_alpha)(ds)
        else:
            print("default")
            ds = layers.ReLU()(ds)  # defaults to ReLU activation if none of the above

        sus = NewmarkActivation(threshold=3.0)(ds, safety_factor, ac, exedance)
        # sus = layers.Activation("sigmoid")(ds)

        self.model = Model(inputs=all_inputs, outputs=sus)

    # NOTE::No momentum yet
    # def get_optimizer(self, lr=2.9064680405339487e-05):
    def get_optimizer(self, lr=1e-05):
        decay_steps = 10000
        decay_rate = 0.9
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate
        )

        if self.optimizer == "adam":
            self.optimizer_instance = optimizers.Adam(learning_rate=lr_schedule)
        elif self.optimizer == "rmsprop":
            self.optimizer_instance = optimizers.RMSprop(learning_rate=lr_schedule)
        if self.optimizer == "sgd":
            self.optimizer_instance = optimizers.SGD(learning_rate=lr_schedule)

    def compile_model(self):
        self.model.compile(
            optimizer=self.optimizer_instance,
            loss=losses.BinaryCrossentropy(),
            metrics=[
                metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5),
                metrics.AUC(),
                "accuracy",
            ],
        )


class NepalModel:
    def __init__(
        self, activation, optimizer, leaky_alpha: Optional[float] = None
    ) -> None:
        self.depth = 12
        self.activation = activation
        self.optimizer = optimizer
        self.leaky_alpha = leaky_alpha

    def get_classification_model(self, all_inputs, encoded_features):

        all_features = tf.keras.layers.concatenate(encoded_features)
        features_only = all_features
        slope = all_inputs[numeric_cols_nepal.index("Slp_m")]
        pga = all_inputs[numeric_cols_nepal.index("PGA_Usgs")]

        x = layers.Dense(
            units=64,
            name="Sus_0",
            kernel_initializer="random_normal",
            bias_initializer="random_normal",
        )(features_only)

        for i in range(1, self.depth + 1):
            x = layers.Dense(
                units=64,
                name=f"Sus_{i}",
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
            )(x)
            x = layers.BatchNormalization()(x)
            if self.activation == "prelu":
                x = layers.PReLU()(x)
            elif self.activation == "relu":
                x = layers.ReLU()(x)
            elif self.activation == "leaky":
                if self.leaky_alpha == None:
                    raise Exception("Missing leaky ReLU alpha value")
                x = layers.LeakyReLU(negative_slope=self.leaky_alpha)(x)
            else:
                x = layers.ReLU()(x)  # defaults to ReLU activation if none of the above

        x = layers.Dense(units=2, name="geotechnical_param")(x)

        if self.activation == "prelu":
            x = layers.PReLU()(x)
        elif self.activation == "relu":
            x = layers.ReLU()(x)
        elif self.activation == "leaky":
            if self.leaky_alpha == None:
                raise Exception("Missing leaky ReLU alpha value")
            x = layers.LeakyReLU(negative_slope=self.leaky_alpha)(x)
        else:
            print("default activation")
            x = layers.ReLU()(x)  # defaults to ReLU activation if none of the above

        if self.activation == "leaky":
            coh = CohesionLayer(self.activation, self.leaky_alpha)(x)
        else:
            coh = CohesionLayer(self.activation)(x)

        ifi = InternalFrictionLayer()(x)

        ds, safety_factor = DisplacementLayerNepal()([coh, ifi, slope, pga])
        ds = DisplacementIntermediate()(ds)
        safety_factor = FosLayer()(safety_factor)

        if self.activation == "prelu":
            ds = layers.PReLU()(ds)
        elif self.activation == "relu":
            ds = layers.ReLU()(ds)
        elif self.activation == "leaky":
            if self.leaky_alpha == None:
                raise Exception("Missing leaky ReLU alpha value")
            ds = layers.LeakyReLU(negative_slope=self.leaky_alpha)(ds)
        else:
            print("default")
            ds = layers.ReLU()(ds)  # defaults to ReLU activation if none of the above

        ds = NewmarkActivation(threshold=5.0)(ds, safety_factor)
        sus = layers.Activation("sigmoid")(ds)

        self.model = Model(inputs=all_inputs, outputs=sus)

    # NOTE::No momentum yet
    # def get_optimizer(self, lr=2.9064680405339487e-05):
    def get_optimizer(self, lr=1e-05):
        decay_steps = 10000
        decay_rate = 0.9
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate
        )

        if self.optimizer == "adam":
            self.optimizer_instance = optimizers.Adam(learning_rate=lr_schedule)
        elif self.optimizer == "rmsprop":
            self.optimizer_instance = optimizers.RMSprop(learning_rate=lr_schedule)
        if self.optimizer == "sgd":
            self.optimizer_instance = optimizers.SGD(learning_rate=lr_schedule)

    def compile_model(self):
        self.model.compile(
            optimizer=self.optimizer_instance,
            loss=losses.BinaryCrossentropy(),
            metrics=[
                metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5),
                metrics.AUC(),
                "accuracy",
            ],
        )


class LandslideV2:
    def __init__(
        self, activation, optimizer, leaky_alpha: Optional[float] = None
    ) -> None:
        self.depth = 12
        self.activation = activation
        self.optimizer = optimizer
        self.leaky_alpha = leaky_alpha
    

    def get_classification_model_no_pga_predictor(self, all_inputs, pga_input, encoded_features):
        units = [32, 64, 8, 64, 32, 8, 32, 8]
        all_features = tf.keras.layers.concatenate(encoded_features)
        features_only = all_features
        slope = all_inputs[numeric_cols_cotabato_v2.index("Slope_mean")]
        pga = pga_input
        bulk_density = all_inputs[numeric_cols_cotabato_v2.index("BUK_mean")]

        x = layers.Dense(
            units=64,
            name="Sus_0",
            kernel_initializer="random_normal",
            bias_initializer="random_normal",
        )(features_only)

        for i in range(1, self.depth + 1):
            # x = layers.Dense(
            #     units=units[i - 1],
            #     name=f"Sus_{i}",
            #     kernel_initializer="random_normal",
            #     bias_initializer="random_normal",
            # )(x)
            x = layers.Dense(
                units=64,
                name=f"Sus_{i}",
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
            )(x)
            x = layers.BatchNormalization()(x)
            if self.activation == "prelu":
                x = layers.PReLU()(x)
            elif self.activation == "relu":
                x = layers.ReLU()(x)
            elif self.activation == "leaky":
                if self.leaky_alpha == None:
                    raise Exception("Missing leaky ReLU alpha value")
                x = layers.LeakyReLU(negative_slope=self.leaky_alpha)(x)
            else:
                x = layers.ReLU()(x)  # defaults to ReLU activation if none of the above

        x = layers.Dropout(0.1)(x)  # NOTE:: adding dropout layer for reguralization
        x = layers.Dense(units=2, name="geotechnical_param")(x)

        if self.activation == "prelu":
            x = layers.PReLU()(x)
        elif self.activation == "relu":
            x = layers.ReLU()(x)
        elif self.activation == "leaky":
            if self.leaky_alpha == None:
                raise Exception("Missing leaky ReLU alpha value")
            x = layers.LeakyReLU(negative_slope=self.leaky_alpha)(x)
        else:
            print("default activation")
            x = layers.ReLU()(x)  # defaults to ReLU activation if none of the above

        coh = CohesionCotabatoLayer("relu")(x)
        ifi = InternalFrictionLayer()(x)

        ds, safety_factor, ac = DisplacementLayerEvent()(
            [coh, ifi, slope, pga, bulk_density]
        )

        ac = CriticalAcceleration()(ac)

        ex = ExedanceCotabatoLayer()(ac, pga)  # NOTE:: Excendance computation
        ds = DisplacementIntermediate()(ds)
        safety_factor = FosLayer()(safety_factor)

        if self.activation == "prelu":
            ds = layers.PReLU()(ds)
        elif self.activation == "relu":
            ds = layers.ReLU()(ds)
        elif self.activation == "leaky":
            if self.leaky_alpha == None:
                raise Exception("Missing leaky ReLU alpha value")
            ds = layers.LeakyReLU(negative_slope=self.leaky_alpha)(ds)
        else:
            print("default")
            ds = layers.ReLU()(ds)  # defaults to ReLU activation if none of the above

        # ds = layers.Lambda(lambda ds: tf.clip_by_value(ds, 0, 60.0))(ds)
        sus = NewmarkActivation(threshold=3.0)(ds, safety_factor, ac, ex)
        # sus = layers.Activation("sigmoid")(ds)

        self.model = Model(inputs=all_inputs, outputs=sus)

    def get_classification_model(self, all_inputs, encoded_features):
        units = [32, 64, 8, 64, 32, 8, 32, 8]
        all_features = tf.keras.layers.concatenate(encoded_features)
        features_only = all_features
        slope = all_inputs[numeric_cols_cotabato_v2.index("Slope_mean")]
        pga = all_inputs[numeric_cols_cotabato_v2.index("PGA1_max")]
        bulk_density = all_inputs[numeric_cols_cotabato_v2.index("BUK_mean")]

        x = layers.Dense(
            units=64,
            name="Sus_0",
            kernel_initializer="random_normal",
            bias_initializer="random_normal",
        )(features_only)

        for i in range(1, self.depth + 1):
            # x = layers.Dense(
            #     units=units[i - 1],
            #     name=f"Sus_{i}",
            #     kernel_initializer="random_normal",
            #     bias_initializer="random_normal",
            # )(x)
            x = layers.Dense(
                units=64,
                name=f"Sus_{i}",
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
            )(x)
            x = layers.BatchNormalization()(x)
            if self.activation == "prelu":
                x = layers.PReLU()(x)
            elif self.activation == "relu":
                x = layers.ReLU()(x)
            elif self.activation == "leaky":
                if self.leaky_alpha == None:
                    raise Exception("Missing leaky ReLU alpha value")
                x = layers.LeakyReLU(negative_slope=self.leaky_alpha)(x)
            else:
                x = layers.ReLU()(x)  # defaults to ReLU activation if none of the above

        x = layers.Dropout(0.1)(x)  # NOTE:: adding dropout layer for reguralization
        x = layers.Dense(units=2, name="geotechnical_param")(x)

        if self.activation == "prelu":
            x = layers.PReLU()(x)
        elif self.activation == "relu":
            x = layers.ReLU()(x)
        elif self.activation == "leaky":
            if self.leaky_alpha == None:
                raise Exception("Missing leaky ReLU alpha value")
            x = layers.LeakyReLU(negative_slope=self.leaky_alpha)(x)
        else:
            print("default activation")
            x = layers.ReLU()(x)  # defaults to ReLU activation if none of the above

        coh = CohesionCotabatoLayer("relu")(x)
        ifi = InternalFrictionLayer()(x)

        ds, safety_factor, ac = DisplacementLayerEvent()(
            [coh, ifi, slope, pga, bulk_density]
        )

        ac = CriticalAcceleration()(ac)

        ex = ExedanceCotabatoLayer()(ac, pga)  # NOTE:: Excendance computation
        ds = DisplacementIntermediate()(ds)
        safety_factor = FosLayer()(safety_factor)

        if self.activation == "prelu":
            ds = layers.PReLU()(ds)
        elif self.activation == "relu":
            ds = layers.ReLU()(ds)
        elif self.activation == "leaky":
            if self.leaky_alpha == None:
                raise Exception("Missing leaky ReLU alpha value")
            ds = layers.LeakyReLU(negative_slope=self.leaky_alpha)(ds)
        else:
            print("default")
            ds = layers.ReLU()(ds)  # defaults to ReLU activation if none of the above

        # ds = layers.Lambda(lambda ds: tf.clip_by_value(ds, 0, 60.0))(ds)
        sus = NewmarkActivation(threshold=3.0)(ds, safety_factor, ac, ex)
        # sus = layers.Activation("sigmoid")(ds)

        self.model = Model(inputs=all_inputs, outputs=sus)

    # NOTE::No momentum yet
    # def get_optimizer(self, lr=2.9064680405339487e-05):
    def get_optimizer(self, lr=1e-05):
        decay_steps = 10000
        decay_rate = 0.9
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate
        )

        if self.optimizer == "adam":
            self.optimizer_instance = optimizers.Adam(learning_rate=lr_schedule)
        elif self.optimizer == "rmsprop":
            self.optimizer_instance = optimizers.RMSprop(learning_rate=lr_schedule)
        if self.optimizer == "sgd":
            self.optimizer_instance = optimizers.SGD(learning_rate=lr_schedule)

    def compile_model(self):
        self.model.compile(
            optimizer=self.optimizer_instance,
            loss=losses.BinaryCrossentropy(),
            metrics=[
                metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5),
                metrics.AUC(),
                "accuracy",
            ],
        )


def plot_auc(
    y_true,
    y_preds,
    threshold,
    activation: Optional[str] = None,
    optimizer: Optional[str] = None,
    title: str = "ROC Curves North Cotabato Dataset",
):

    print(threshold)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_preds)
    auc = sklearn.metrics.auc(fpr, tpr)
    acc = round(sklearn.metrics.balanced_accuracy_score(y_true, y_preds > threshold), 2)
    plt.plot(
        fpr,
        tpr,
        lw=1,
        alpha=0.3,
        label=f"(AUC={auc:.2f}, Acc={acc}) Activation: {activation} Optimizer: {optimizer}",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


def plot_confusion_matrix(preds, test_y, threshold):
    print(threshold)
    y_pred_classes = (preds > threshold).astype(
        "int32"
    )  # threshold for binary classification
    cm = confusion_matrix(test_y, y_pred_classes)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
