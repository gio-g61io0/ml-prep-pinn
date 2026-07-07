import os
import json
from tensorflow.keras.callbacks import CSVLogger
from sys import setdlopenflags
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model, Input
from sklearn.model_selection import KFold, StratifiedKFold
from keras_tuner import RandomSearch
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

FTP = tf.keras.metrics.TruePositives()
FFP = tf.keras.metrics.FalsePositives()
FFN = tf.keras.metrics.FalseNegatives()

numeric_cols = [
    "BD_mean",
    "BD_std",
    "Clay_mean",
    "Clay_std",
    "Sand_mean",
    "Sand_std",
    "Silt_mean",
    "Silt_std",
    "Est_mean",
    "Est_std",
    "Nrt_mean",
    "Nrt_std",
    "blk-unit_mean",
    "blk-unit_stdev",
    "wtr-cont_mean",
    "wtr-cont_stdev",
    "Slope_mean",
    "Slope_std",
    "PGA_mean",
    "PGA_std",
    "NDVI_mean",
    "NDVI_std",
    "Prc_mean",
    "Prc_std",
]

@tf.keras.utils.register_keras_serializable()
class DisplacementLayerRainFall(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DisplacementLayerRainFall, self).__init__(**kwargs)


    def call(self, inputs):
        cohesion_t, friction_angle, slope, pga, bulk_density, m = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],

        )
        slope *= 0.017453292519943295

        pga *= 100
        cohesion_t *= 1000.0 #kPa -> Pa
        bulk_density *= 1000.0  # kN/m^3 to N/m^3
        slope_normal_thickness = 3.33  # m
        cohesion_t = tf.expand_dims(cohesion_t, 1)
        friction_angle = tf.expand_dims(friction_angle, 1)
        m = tf.reshape(m, [-1])          # flatten to (batch,) in case it arrives as (batch,1)
        m = tf.expand_dims(m, 1)

        water_unit_weight = 9810.0 #This is 9.81 N/m^3
       
        safety_factor = (cohesion_t / ((bulk_density * slope_normal_thickness) * tf.math.sin(slope))) + (tf.math.tan(friction_angle) / tf.math.tan(slope)) - ((m * water_unit_weight * tf.math.tan(friction_angle)) / (bulk_density * tf.math.tan(slope)))

        ac = (
            (safety_factor - 1) * 9.81 * tf.math.sin(slope)
        )  # NOTE::Critical Acceleration
        
        ac = layers.ReLU()(ac)

        acpg = ac / pga
        # acpg = a_c / PGA. Low values mean slope yields easily (high displacement),
        # high values mean slope barely yields (low displacement).
        # Lower bound 0.05 -> caps max displacement at ~135 cm (physically realistic
        # for severe landslides; previous 0.001 lower bound let it explode to ~42,500 cm
        # at the Jibson regression singularity).
        # Upper bound 0.75 -> stable slopes can still go down to ~0.12 cm displacement.
        acpg = tf.clip_by_value(acpg, 0.05, 0.75)

        powcomp = tf.math.pow((1 - acpg), 2.341) * tf.math.pow(acpg, -1.438)
        logds = 0.215 + tf.math.log(powcomp) + 0.51  # NOTE:: Newmark Displacement

        return tf.math.exp(logds), safety_factor, ac, acpg

def build_model_pinn(train_df):
    def model_fn(hp):
        all_inputs = []
        encoded_features = []
        for header in numeric_cols:
            numerical_col = keras.Input((1,), name=header)
            normalization_layer = NormalizationLayer(header, train_df)
            encoded = normalization_layer(numerical_col)
            all_inputs.append(numerical_col)
            encoded_features.append(encoded)

        all_features = layers.concatenate(encoded_features)
        features_only = all_features
        slope = all_inputs[numeric_cols.index("Slope_mean")]
        pga = all_inputs[numeric_cols.index("PGA_mean")]
        bulk_dense = all_inputs[numeric_cols.index("blk-unit_mean")]

        x = layers.Dense(
            units=64,
            kernel_initializer="random_normal",
            bias_initializer="random_normal",
            name="Sus_0",
        )(features_only)

        for i in range(1, hp.Choice("depths", [8, 12, 24, 32]) + 1):
            x = layers.Dense(
                units=hp.Choice(f"units_{i}", [8, 16, 32, 64]),
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
                name=f"Sus_{str(i)}",
            )(x)

            x = layers.BatchNormalization()(x)
            activation = hp.Choice("inner_activation", ["leaky", "prelu", "relu"])
            if activation == "leaky":
                x = layers.LeakyReLU(negative_slope=0.2)(x)
            elif activation == "prelu":
                x = layers.PReLU()(x)
            elif activation == "relu":
                x = layers.ReLU()(x)
            else:
                x = layers.PReLU()(x)  # NOTE:: defaults to parametric ReLU

        x = layers.Dense(units=2, name="geotechnical_params")(x)
        # x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = layers.PReLU()(x)

        coh = CohesionLayer()(x)
        ifi = InternalFrictionLayer()(x)

        # coh = layers.Lambda(lambda x: tf.clip_by_value(x, 5.0, 40.0))(coh)
        # ifi = layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 60.0))(ifi)

        ds = DisplacementLayer()([coh, ifi, slope, pga, bulk_dense])
        ds = layers.PReLU()(ds)
        # ds = tf.keras.layers.Activation('relu')(ds)

        # sus = LandslideActivationLayer()(ds)
        sus = NewmarkActivation()(ds)
        # sus = tf.keras.layers.Activation('sigmoid')(sus)
        lr = hp.Choice("lr", [1e-5, 1e-4, 1e-3, 1e-2])
        optimizer_name = hp.Choice("optimizer", ["adam", "sgd", "rmsprop"])
        if optimizer_name == "adam":
            optimizer = optimizers.Adam(learning_rate=lr)
        elif optimizer_name == "sgd":
            optimizer = optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif optimizer_name == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=lr)
        else:
            optimizer = optimizers.Adam(
                learning_rate=lr
            )  # NOTE:: its just here to fix "possible unbound error"

        model = Model(inputs=all_inputs, outputs=sus)
        model.compile(
            optimizer=optimizer,
            loss=losses.BinaryCrossentropy(),
            metrics=[
                metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5),
                metrics.AUC(),
                "accuracy",
            ],
        )
        os.makedirs("tuner_logs", exist_ok=True)
        # log_filename = os.path.join(
        #     "tuner_logs",
        #     f"trial_{hp.values['optimizer']}_{hp.values.get('lr', '')}.csv",
        # )
        trial_id = hp.values["Trial-id"] if "Trial-id" in hp.values else "trial_unknown"
        log_file = f"tuner_logs/{trial_id}_{optimizer}.csv"
        csv_logger = CSVLogger(log_file)

        # csv_logger = CSVLogger(log_filename)

        return model

    return model_fn


class LoggingRandomSearch(RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        # history = super().run_trial(trial, *args, **kwargs)
        model = self.hypermodel.build(trial.hyperparameters)

        history = model.fit(*args, **kwargs)

        trial_id = trial.trial_id
        trial_dir = os.path.join(self.project_dir, trial_id)
        print(trial_dir)
        os.makedirs(trial_dir, exist_ok=True)
        # 1. Save history as JSON
        history_path = os.path.join(trial_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history.history, f)

        # 2. Save a quick plot of metrics
        plt.figure(figsize=(8, 5))
        plt.plot(history.history.get("loss", []), label="loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="val_loss")
        plt.title(f"Trial {trial_id} Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(trial_dir, "metrics.png")
        plt.savefig(fig_path)
        plt.close()

        return history  # return history to preserve behavior

@tf.keras.utils.register_keras_serializable()
class HydraulicConductivityLayer(tf.keras.layers.Layer):
    """Learnable hydraulic conductivity (K) per soil type.

    Input: soil_type_idx (int, 0-2)
    Output: K in m/hr

    Soil types: 0=Sandy Clay Loam, 1=Loam, 2=Undifferentiated
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "hydraulic_conductivity")
        super(HydraulicConductivityLayer, self).__init__(**kwargs)
        # K ranges in cm/s per soil type
        self.k_min = tf.constant([1e-5, 1e-7, 1e-7], dtype=tf.float32)
        self.k_max = tf.constant([1e-3, 1e-5, 1e-3], dtype=tf.float32)

    def build(self, input_shape):
        self.u_k = self.add_weight(
            name="u_k",
            shape=(3,),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: soil_type_idx (batch,1) float — flatten to (batch,) int
        soil_idx = tf.cast(tf.reshape(inputs, [-1]), tf.int32)
        # K in cm/s via sigmoid scaling
        k_cms = self.k_min + (self.k_max - self.k_min) * tf.nn.sigmoid(self.u_k)
        # Use one_hot + matmul instead of tf.gather to avoid
        # UnsortedSegmentSum shape errors in the backward pass
        one_hot = tf.one_hot(soil_idx, depth=3, dtype=tf.float32)
        k_per_sample = tf.reduce_sum(one_hot * k_cms, axis=-1)
        # Convert cm/s -> m/hr: × 0.01 × 3600 = × 36
        k_mhr = k_per_sample * 36.0
        return k_mhr

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class WetnessLayer(tf.keras.layers.Layer):
    """Computes empirical wetness m = λ * (R * log1p(A)) / (T * sin θ), clamped to [0, 1].

    Input: [precipitation (mm/month), contributing_area (m²), soil_thickness (m), slope (deg), k (m/hr)]
    Output: m ∈ [0, 1]  (now a wetness *index* — see note below)

    The learnable scale factor λ ∈ [lambda_min, lambda_max] (default [0.01, 1.0]) lets
    the model calibrate the formula. log1p is applied to contributing_area to compress
    its wide dynamic range (m² values often span 3 orders of magnitude); without it,
    high-accumulation pixels saturate at m=1 and block K's gradient through the clip.

    NOTE on physical meaning: with log1p on A, m is no longer dimensionally a
    saturation ratio — it's a wetness index in the spirit of TOPMODEL's TWI, ranging
    in [0,1] by construction. The downstream pore-pressure term in
    DisplacementLayerRainFall still uses m as if it were a saturation ratio; treat
    that as a hazard-index proxy, not literal Mohr-Coulomb pore pressure.
    """
    def __init__(self, lambda_min=0.01, lambda_max=1000.0, u_lambda_init=-2.0, **kwargs):
        kwargs.setdefault("name", "wetness_layer")
        super(WetnessLayer, self).__init__(**kwargs)
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.u_lambda_init = u_lambda_init

    def build(self, input_shape):
        # Single learnable scalar; sigmoid(u_lambda) is rescaled into [lambda_min, lambda_max].
        # Default u_lambda_init = -2 -> sigmoid(-2) ≈ 0.12 -> lambda ≈ 0.13 with default bounds,
        # which puts the median pixel's pre-clip m at ~0.3 instead of >2 (saturated).
        self.u_lambda = self.add_weight(
            name="u_lambda",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.u_lambda_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        precipitation, contributing_area, soil_thickness, slope, k = (
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        )
        # k comes from HydraulicConductivityLayer as (batch,);
        # expand to (batch,1) to match the other (batch,1) inputs
        k = tf.expand_dims(k, -1)
        # Convert precipitation mm/month -> m/hr
        r = precipitation * (0.001 / 720.0)
        # Convert slope degrees -> radians
        slope_rad = slope * 0.017453292519943295
        # Transmissivity T = K * soil_thickness (m²/hr)
        t = k * soil_thickness
        t = tf.maximum(t, 1e-10)
        sin_slope = tf.maximum(tf.math.sin(slope_rad), 1e-6)
        # Learnable scale, bounded via sigmoid into [lambda_min, lambda_max]
        scale = self.lambda_min + (self.lambda_max - self.lambda_min) * tf.nn.sigmoid(self.u_lambda)
        # Wetness ratio — log1p compresses the contributing-area dynamic range so the
        # formula doesn't saturate on high-accumulation pixels (which previously
        # pegged ~22% of m at 1.0 and blocked K's gradient through clip_by_value).
        # This makes m a wetness *index* rather than a literal saturation ratio.
        m = scale * (r * tf.math.log1p(contributing_area)) / (t * sin_slope)
        m = tf.clip_by_value(m, 0.0, 1.0)
        return m

    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "u_lambda_init": self.u_lambda_init,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ClipLayer(tf.keras.layers.Layer):
    def __init__(self, min_val, max_val,name, **kwargs):

        kwargs.setdefault("name", name)
        super().__init__(**kwargs)
        self.name = name
        self.min_val = min_val
        self.max_val = max_val

    def call(self, x):
        return tf.clip_by_value(x, self.min_val, self.max_val)

    def get_config(self):
        config = super().get_config()
        config.update({
            "min_val": self.min_val,
            "max_val": self.max_val,
            "name":self.name
        })
        return config


@tf.keras.utils.register_keras_serializable()
class ProportionSlabThicknessLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):

        kwargs.setdefault("name", "proportion_slab_thickness_layer")  # default name if not provided
        super(ProportionSlabThicknessLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.relu(inputs[..., 2])

    @classmethod
    def from_config(cls, config):  # For deserialization purpose
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class CohesionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):

        kwargs.setdefault("name", "cohesion_layer")  # default name if not provided
        super(CohesionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.relu(inputs[..., 0])

    @classmethod
    def from_config(cls, config):  # For deserialization purpose
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class InternalFrictionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "internal_friction")  # default name if not provided
        super(InternalFrictionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        #This puts the ifi to radians 0-1
        return tf.nn.sigmoid(inputs[..., 1])

    @classmethod
    def from_config(cls, config):  # For deserialization purpose
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class IFIClipLayer(tf.keras.layers.Layer):
    """Hard clip on the internal friction angle (in radians).

    Used as a drop-in replacement for `Lambda(tf.clip_by_value, ...)`, which
    breaks on reload because the lambda closure doesn't carry `tf` into the
    deserialization namespace. Bounds are stored on the layer and survive
    serialization.
    """

    def __init__(self, low_rad, high_rad, **kwargs):
        super().__init__(**kwargs)
        self.low_rad = float(low_rad)
        self.high_rad = float(high_rad)

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.low_rad, self.high_rad)

    def get_config(self):
        config = super().get_config()
        config.update({"low_rad": self.low_rad, "high_rad": self.high_rad})
        return config


@tf.keras.utils.register_keras_serializable()
class LandslideActivationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LandslideActivationLayer, self).__init__(**kwargs)

    def call(self, x):
        return x - 5.0

    @classmethod
    def from_config(cls, config):  # For deserialization purpose
        return cls(**config)


# SHAP support (optional)
def run_shap_analysis(trained_model, background_inputs, test_inputs, feature_names):
    import shap

    explainer = shap.GradientExplainer(model=trained_model, data=background_inputs)
    shap_values = explainer.shap_values(test_inputs)
    shap.summary_plot(shap_values, test_inputs, feature_names=feature_names)


from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model, Input
import numpy as np

FTP = tf.keras.metrics.TruePositives()
FFP = tf.keras.metrics.FalsePositives()
FFN = tf.keras.metrics.FalseNegatives()

@tf.keras.utils.register_keras_serializable()
class DisplacementLayerFOSMakilala(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DisplacementLayerFOSMakilala, self).__init__(**kwargs)

    def call(self, inputs):
        cohesion_t, friction_angle, slope, pga, bulk_unit_weight, m = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],

        )
        slope *= 0.017453292519943295

        pga *= 9.81
        cohesion_t *= 1000.0 #kPa -> Pa
        bulk_unit_weight *= 1000.0  # kN/m^3 to N/m^3
        slope_normal_thickness = 3.33  # m
        
        cohesion_t = tf.expand_dims(cohesion_t, 1)
        friction_angle = tf.expand_dims(friction_angle, 1)
        m = tf.expand_dims(m, 1)

        unit_weight_freshwater = 9.81

        safety_factor = (cohesion_t + ((bulk_unit_weight - (m * unit_weight_freshwater)) * slope_normal_thickness * (tf.math.cos(slope) ** 2) * tf.math.tan(friction_angle))) / (((bulk_unit_weight * slope_normal_thickness) * tf.math.sin(slope)*tf.math.cos(slope)) + ((0.5 * (pga / 9.81)) * (bulk_unit_weight * slope_normal_thickness) * (tf.math.cos(slope) ** 2))) 
        ac = (
            (safety_factor - 1) * 9.81 * tf.math.sin(slope)
        )  # NOTE::Critical Acceleration

        # ac = 9.81 * ((cohesion_t / bulk_unit_weight * slope_normal_thickness * tf.math.cos(slope) ** 2) + ((1 - ((m * unit_weight_freshwater) / bulk_unit_weight)) * tf.math.tan(friction_angle)) - tf.math.tan(slope))
        # ac /= 9.81
        
        acpg = ac / pga
        acpg = tf.clip_by_value(acpg, 0.001, 0.999)

        powcomp = tf.math.pow((1 - acpg), 2.341) * tf.math.pow(acpg, -1.438)
        logds = 0.215 + tf.math.log(powcomp) + 0.51  # NOTE:: Newmark Displacement

        return tf.math.exp(logds), safety_factor, ac
    
@tf.keras.utils.register_keras_serializable()
class DisplacementLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DisplacementLayer, self).__init__(**kwargs)

    def call(self, inputs):
        cohesion_t, friction_angle, slope, pga, bulk_density = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
        )
        slope *= 0.017453292519943295

        # pga *= 9.81
        cohesion_t *= 1000.0 #kPa -> Pa
        bulk_density *= 1000.0  # kN/m^3 to N/m^3
        slope_normal_thickness = 3.33  # m
        cohesion_t = tf.expand_dims(cohesion_t, 1)
        friction_angle = tf.expand_dims(friction_angle, 1)

       
        safety_factor = (cohesion_t / ((bulk_density * slope_normal_thickness) * tf.math.sin(slope))) + tf.math.tan(friction_angle) / tf.math.tan(slope)

        # safety_factor = (
        #     cohesion_t
        #     * (1 / (bulk_density * slope_normal_thickness) * (tf.math.sin(slope)))
        # ) + (
        #     tf.math.tan(friction_angle) / tf.math.tan(slope)
        # )  # NOTE::Factory of Safety

        ac = (
            (safety_factor - 1) * 9.81 * tf.math.sin(slope)
        )  # NOTE::Critical Acceleration
        
        ac /= 9.81
        
        acpg = ac / pga
        acpg = tf.clip_by_value(acpg, 0.001, 0.75)

        powcomp = tf.math.pow((1 - acpg), 2.341) * tf.math.pow(acpg, -1.438)
        logds = 0.215 + tf.math.log(powcomp) + 0.51  # NOTE:: Newmark Displacement

        return tf.math.exp(logds)
    
@tf.keras.utils.register_keras_serializable()
class ModifiedFosDisplacementLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "modified_fos_displacement")
        super(ModifiedFosDisplacementLayer, self).__init__(**kwargs)

        #Matric suction as a trainable/learned variable in this layer
        self.matric_suction = self.add_weight(
            name="matric_suction",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(-15_000.0),
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(min_value=-15_000.0, max_value=-50.0)
        )
    #Internal Friction angle is in radians before feeding into this layer
    def call(self, inputs):
        cohesion_t, friction_angle, slope, pga, bulk_unit_weight = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
        )

        slope *= 0.017453292519943295 #Angle to Radians
        pga *= 9.81 #NOTE kh 0.5 * (PGA / g)
        cohesion_t *= 1000.0 #kPa -> Pa 
        # matric_suction = (-15) #Constant matric suction assumed to be -15 kPa for unsaturated soils
        # matric_suction *= 1000 #kPa -> Pa

        matric_suction = self.matric_suction
        slope_normal_thickness = 3.33  # m
        cohesion_t = tf.expand_dims(cohesion_t, 1)
        friction_angle = tf.expand_dims(friction_angle, 1)

        safety_factor  = (((cohesion_t - (matric_suction * tf.math.tan(friction_angle))) / (bulk_unit_weight * slope_normal_thickness * tf.math.sin(slope))) + (tf.math.tan(friction_angle) / tf.math.tan(slope)))
        
        #Clip safety factor

        ac = (
            (safety_factor - 1) * 9.81 * tf.math.sin(slope)
        )  # NOTE::Critical Acceleration
        
        # ac /= 9.81

        acpg = ac / pga

        acpg = tf.clip_by_value(acpg, 0.001, 0.75)

        powcomp = tf.math.pow((1 - acpg), 2.341) * tf.math.pow(acpg, -1.438)
        logds = 0.215 + tf.math.log(powcomp) + 0.51  # NOTE:: Newmark Displacement

        return tf.math.exp(logds), safety_factor
    
    @classmethod
    def from_config(cls, config):  # For deserialization purpose
        return cls(**config)
    
@tf.keras.utils.register_keras_serializable()
class DisplacementLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DisplacementLayer, self).__init__(**kwargs)

    def call(self, inputs):
        cohesion_t, friction_angle, slope, pga, bulk_density = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
        )
        slope *= 0.017453292519943295

        # pga *= 9.81
        cohesion_t *= 1000.0 #kPa -> Pa
        bulk_density *= 1000.0  # kN/m^3 to N/m^3
        slope_normal_thickness = 3.33  # m
        cohesion_t = tf.expand_dims(cohesion_t, 1)
        friction_angle = tf.expand_dims(friction_angle, 1)

       
        safety_factor = (cohesion_t / ((bulk_density * slope_normal_thickness) * tf.math.sin(slope))) + tf.math.tan(friction_angle) / tf.math.tan(slope)

        # safety_factor = (
        #     cohesion_t
        #     * (1 / (bulk_density * slope_normal_thickness) * (tf.math.sin(slope)))
        # ) + (
        #     tf.math.tan(friction_angle) / tf.math.tan(slope)
        # )  # NOTE::Factory of Safety

        ac = (
            (safety_factor - 1) * 9.81 * tf.math.sin(slope)
        )  # NOTE::Critical Acceleration
        
        ac /= 9.81
        
        acpg = ac / pga
        acpg = tf.clip_by_value(acpg, 0.001, 0.75)

        powcomp = tf.math.pow((1 - acpg), 2.341) * tf.math.pow(acpg, -1.438)
        logds = 0.215 + tf.math.log(powcomp) + 0.51  # NOTE:: Newmark Displacement

        return tf.math.exp(logds)
    
@tf.keras.utils.register_keras_serializable()
class DisplacementConditionalLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DisplacementConditionalLayer, self).__init__(**kwargs)

    def call(self, inputs):
        cohesion_t, friction_angle, slope, pga, bulk_density, soil_type = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
        )

        #Conditions that based on soil type
        COH_LIMITS = tf.constant([
        [10,90], #Loam
        [10, 90],  #Silt Loam
        [10, 105], #Clay Loam and Silty Clay Loam
        [10, 105], #Clay Loam and Silty Clay Loam
        [10, 105], #Clay Loam and Silty Clay Loam
        [10, 75], #Sandy Loam
        ])

        limits = tf.gather(COH_LIMITS, soil_type)
        coh_min = limits[:, 0]
        coh_max = limits[:, 1]

        cohesion_t = tf.clip_by_value(cohesion_t, coh_min, coh_max)

        slope *= 0.017453292519943295
        pga *= 10.0
        cohesion_t *= 1000.0
        # bulk_density *= 1000.0  # g/cm^3 to kg/m^3
        slope_normal_thickness = 3.33  # m
        cohesion_t = tf.expand_dims(cohesion_t, 1)
        friction_angle = tf.expand_dims(friction_angle, 1)

        # NOTE:: Change 2300 this (slope unit)
        # safety_factor = (cohesion_t * (1 / (2300 * 9.81 * tf.math.sin(slope)))) + (
        #     tf.math.tan(friction_angle) / tf.math.tan(slope)
        # )

        safety_factor = (
            cohesion_t
            * (1 / (bulk_density * slope_normal_thickness) * (tf.math.sin(slope)))
        ) + (
            tf.math.tan(friction_angle) / tf.math.tan(slope)
        )  # NOTE::Factory of Safety

        # safety_factor = tf.clip_by_value(safety_factor, 1.2, 15.0)
        ac = (
            (safety_factor - 1) * 9.81 * tf.math.sin(slope)
        )  # NOTE::Critical Acceleration
        acpg = ac / pga
        acpg = tf.clip_by_value(acpg, 0.001, 0.999)

        powcomp = tf.math.pow((1 - acpg), 2.341) * tf.math.pow(acpg, -1.438)
        logds = 0.215 + tf.math.log(powcomp) + 0.51  # NOTE:: Newmark Displacement

        return tf.math.exp(logds)

@tf.keras.utils.register_keras_serializable()
class NewmarkActivation(tf.keras.layers.Layer):
    def __init__(self, threshold=5.0, **kwargs):
        super(NewmarkActivation, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, inputs, safety_factor=None, ac=None, acpg=None):
        return 1.0 / (
            1.0 + tf.exp(self.threshold - inputs)
        )  # The activation function based on the paper

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
        print(values.shape)
        self.index.adapt(values)
        print(values)
        self.encoder = tf.keras.layers.CategoryEncoding(
            num_tokens=self.index.vocabulary_size()
        )

    def __call__(self, feature):
        return self.encoder(self.index(feature))


class LandslideModel:
    def __init__(self):
        self.depth = 12

    def landslide_activation(self, x):
        return x - 5.0

    def cohesion_activation(self, x):
        return tf.nn.relu(x)

    def friction_activation(self, x):
        return tf.nn.sigmoid(x)

    def get_classification_model(
        self, all_inputs, encoded_features, in_num=17, out_num=1
    ):
        all_features = tf.keras.layers.concatenate(encoded_features)
        slope = all_inputs[numeric_cols.index("Slope_mean")]
        pga = all_inputs[numeric_cols.index("PGA_mean")]
        bulk_dense = all_inputs[numeric_cols.index("blk-unit_mean")]

        x = layers.Dense(
            units=64,
            name="Sus_0",
            kernel_initializer="random_normal",
            bias_initializer="random_normal",
        )(all_features)
        for i in range(1, self.depth + 1):
            x = layers.Dense(
                units=64,
                name=f"Sus_{i}",
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

        x = layers.Dense(units=2, activation="relu", name="geotechnical_param")(x)

        # ✅ Fix serialization issue: avoid slicing with ...
        coh = layers.Lambda(lambda x: tf.nn.relu(x[:, 0]), name="cohesion")(x)
        ifi = layers.Lambda(lambda x: tf.nn.sigmoid(x[:, 1]), name="internalFriction")(
            x
        )  # save best coh and ifi to see if what is the final predicted coh and ifi  (per slope unit)

        ds = DisplacementLayer()([coh, ifi, slope, pga])
        ds = layers.Activation("relu")(ds)
        sus = layers.Lambda(lambda x: x - 5.0)(ds)
        sus = layers.Activation("sigmoid")(sus)

        self.model = Model(inputs=all_inputs, outputs=sus)

    def get_optimizer(
        self, opt=tf.keras.optimizers.Adam, lr=1e-4, decay_steps=10000, decay_rate=0.9
    ):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate
        )
        self.optimizer = opt(learning_rate=lr_schedule)

    def dataframe_to_dataset(self, df, shuffle=True, batch_size=32):
        labels = df.pop("landslide")
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    # Trains the model using StratifiedKFold technique
    def run_model_folds(
        self, df, numerical_cols, feature_cols, folds=10, epochs=100, batch_size=128
    ):
        # kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        fold = 1
        scores = []
        aucs = []
        predictions = np.zeros(df.shape[0])
        # for train_idx, val_idx in kf.split(df):
        for train_idx, val_idx in skf.split(df, df["landslide"]):

            # split data set
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

            train_ds = self.dataframe_to_dataset(train_df[feature_cols])
            val_ds = self.dataframe_to_dataset(val_df[feature_cols], shuffle=False)
            all_inputs = []
            encoded_features = []

            # normalize feature inputs
            for header in numerical_cols:
                numerical_col = tf.keras.Input((1,), name=header)
                normalization_layer = NormalizationLayer(header, train_ds)
                encoded_numerical_col = normalization_layer(numerical_col)
                all_inputs.append(numerical_col)
                encoded_features.append(encoded_numerical_col)
            model = LandslideModel()
            self.get_classification_model(
                all_inputs, encoded_features, in_num=len(all_inputs), out_num=1
            )
            self.get_optimizer()
            self.compile_model()

            hist = self.model.fit(
                train_ds,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_ds,
                class_weight={0: 1, 1: 5},
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=5, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        f"./TrainedModels/fold_{fold}_best.keras", save_best_only=True
                    ),
                ],
            )
            # Use model to predict using validation data
            test_y = val_df["landslide"].to_numpy()
            preds = model.model.predict(val_ds)
            predictions[val_idx] = preds.flatten()
            best_threshold, fpr, tpr = find_best_threshold(test_y, preds)
            # [fpr, tpr, threshold] = sklearn.metrics.roc_curve(test_y, preds)
            print(f"Best thresholds:{best_threshold}")
            auc = sklearn.metrics.auc(fpr, tpr)
            aucs.append(auc)
            # plt.text(0.61, 0.15,f"Accuracy={round(sklearn.metrics.balanced_accuracy_score(test_y, preds>0.5),2)} fold: {fold}")
            # plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Fold {fold} (AUC = {auc:.2f})")
            acc = round(sklearn.metrics.balanced_accuracy_score(test_y, preds > 0.5), 2)
            plt.plot(
                fpr,
                tpr,
                lw=1,
                alpha=0.3,
                label=f"Fold {fold} (AUC={auc:.2f}, Acc={acc})",
            )
            fold += 1
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return predictions

    def compile_model(self, weights=None):
        self.model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5),
                tf.keras.metrics.AUC(),
                tf.keras.metrics.BinaryAccuracy(),
                "accuracy",
            ],
        )


def find_best_threshold(y_true, y_pred_probs):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred_probs)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh, fpr, tpr


# SHAP support (optional)
def run_shap_analysis(trained_model, background_inputs, test_inputs, feature_names):
    import shap

    explainer = shap.GradientExplainer(model=trained_model, data=background_inputs)
    shap_values = explainer.shap_values(test_inputs)
    shap.summary_plot(shap_values, test_inputs, feature_names=feature_names)
