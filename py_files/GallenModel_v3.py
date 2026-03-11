import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


NUM_SOIL_TYPES = 12


@tf.keras.utils.register_keras_serializable()
class HydraulicConductivityLayerV3(tf.keras.layers.Layer):
    """Learnable hydraulic conductivity (K) per USDA soil texture class.

    Input: soil_texture_idx (int, 0-11)
    Output: K in m/hr

    Uses Min / Max Ksat from literature as bounds.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "hydraulic_conductivity_v3")
        super(HydraulicConductivityLayerV3, self).__init__(**kwargs)
        # K ranges in cm/h per soil texture (Min, Max from literature)
        self.k_min = tf.constant([
             0.01,  # 0  Sand
             0.01,  # 1  Loamy Sand
             0.00,  # 2  Sandy Loam
             0.00,  # 3  Silt Loam
             0.01,  # 4  Loam
             0.27,  # 5  Silt
             0.00,  # 6  Sandy Clay Loam
             0.01,  # 7  Clay Loam
             0.01,  # 8  Silty Clay Loam
             0.00,  # 9  Sandy Clay
             0.00,  # 10 Silty Clay
             0.00,  # 11 Clay
        ], dtype=tf.float32)
        self.k_max = tf.constant([
           841.00,  # 0  Sand
           189.00,  # 1  Loamy Sand
           504.00,  # 2  Sandy Loam
            53.90,  # 3  Silt Loam
            52.60,  # 4  Loam
           213.00,  # 5  Silt
           405.00,  # 6  Sandy Clay Loam
            38.20,  # 7  Clay Loam
           159.00,  # 8  Silty Clay Loam
            60.60,  # 9  Sandy Clay
            21.00,  # 10 Silty Clay
           421.00,  # 11 Clay
        ], dtype=tf.float32)

    def build(self, input_shape):
        self.u_k = self.add_weight(
            name="u_k",
            shape=(NUM_SOIL_TYPES,),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        soil_idx = tf.cast(tf.reshape(inputs, [-1]), tf.int32)
        # K in cm/h via sigmoid scaling
        k_cmh = self.k_min + (self.k_max - self.k_min) * tf.nn.sigmoid(self.u_k)
        one_hot = tf.one_hot(soil_idx, depth=NUM_SOIL_TYPES, dtype=tf.float32)
        k_per_sample = tf.reduce_sum(one_hot * k_cmh, axis=-1)
        # Convert cm/h -> m/hr: × 0.01
        # k_mhr = k_per_sample * 0.01
      
        k_mhr = k_per_sample / 100

        return k_mhr

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class SoilConditionedGeotechLayerV3(tf.keras.layers.Layer):
    """Per-soil-type learnable baselines for cohesion and internal friction.

    Same architecture as SoilConditionedGeotechLayer (v2) but expanded to
    12 USDA soil texture classes instead of 3.

    Input: [soil_texture_idx (batch,1), dense_output (batch,2)]
    Output: [coh (batch,), ifi (batch,)] — bounded per soil type, with dense residual

    Also adds a variance regularization loss to prevent collapsed (constant)
    intermediate values.
    """

    # Physical ranges per soil texture class
    # Cohesion (c') in kPa, internal friction angle (φ') in radians
    # Effective shear strength parameters for shallow landslide analysis
    COH_RANGES = {
        0:  ( 0.0,   2.0),  # Sand             (0-2 kPa)
        1:  ( 0.0,   3.0),  # Loamy Sand       (0-3 kPa)
        2:  ( 0.0,   5.0),  # Sandy Loam       (0-5 kPa)
        3:  ( 0.0,  10.0),  # Silt Loam        (0-10 kPa)
        4:  ( 5.0,  15.0),  # Loam             (5-15 kPa)
        5:  ( 0.0,   5.0),  # Silt             (0-5 kPa)
        6:  ( 0.0,   5.0),  # Sandy Clay Loam  (0-5 kPa)
        7:  (15.0,  30.0),  # Clay Loam        (15-30 kPa)
        8:  (10.0,  25.0),  # Silty Clay Loam  (10-25 kPa)
        9:  ( 5.0,  20.0),  # Sandy Clay       (5-20 kPa)
        10: (20.0,  40.0),  # Silty Clay       (20-40 kPa)
        11: (25.0,  50.0),  # Clay             (25-50 kPa)
    }
    IFI_RANGES = {
        0:  (0.5236, 0.6632),  # Sand             (30-38°)
        1:  (0.5236, 0.6283),  # Loamy Sand       (30-36°)
        2:  (0.4887, 0.5934),  # Sandy Loam       (28-34°)
        3:  (0.4189, 0.5585),  # Silt Loam        (24-32°)
        4:  (0.4538, 0.5585),  # Loam             (26-32°)
        5:  (0.4363, 0.5585),  # Silt             (25-32°)
        6:  (0.4538, 0.5934),  # Sandy Clay Loam  (26-34°)
        7:  (0.4363, 0.5585),  # Clay Loam        (25-32°)
        8:  (0.3491, 0.4887),  # Silty Clay Loam  (20-28°)
        9:  (0.4363, 0.5585),  # Sandy Clay       (25-32°)
        10: (0.3142, 0.4538),  # Silty Clay       (18-26°)
        11: (0.3142, 0.4363),  # Clay             (18-25°)
    }

    def __init__(self, lambda_var=0.01, residual_scale_coh=5.0, residual_scale_ifi=0.05, **kwargs):
        kwargs.setdefault("name", "soil_geotech_v3")
        super(SoilConditionedGeotechLayerV3, self).__init__(**kwargs)
        self.lambda_var = lambda_var
        self.residual_scale_coh = residual_scale_coh
        self.residual_scale_ifi = residual_scale_ifi

        # Build constant tensors for ranges
        self.coh_min = tf.constant(
            [self.COH_RANGES[i][0] for i in range(NUM_SOIL_TYPES)], dtype=tf.float32
        )
        self.coh_max = tf.constant(
            [self.COH_RANGES[i][1] for i in range(NUM_SOIL_TYPES)], dtype=tf.float32
        )
        self.ifi_min = tf.constant(
            [self.IFI_RANGES[i][0] for i in range(NUM_SOIL_TYPES)], dtype=tf.float32
        )
        self.ifi_max = tf.constant(
            [self.IFI_RANGES[i][1] for i in range(NUM_SOIL_TYPES)], dtype=tf.float32
        )

    def build(self, input_shape):
        self.u_coh = self.add_weight(
            name="u_coh",
            shape=(NUM_SOIL_TYPES,),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
        )
        self.u_ifi = self.add_weight(
            name="u_ifi",
            shape=(NUM_SOIL_TYPES,),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        soil_type_idx, dense_output = inputs[0], inputs[1]

        # --- Base values per soil type via sigmoid bounding ---
        base_coh_all = self.coh_min + (self.coh_max - self.coh_min) * tf.nn.sigmoid(self.u_coh)
        base_ifi_all = self.ifi_min + (self.ifi_max - self.ifi_min) * tf.nn.sigmoid(self.u_ifi)

        # Select per-sample base values using one_hot
        soil_idx = tf.cast(tf.reshape(soil_type_idx, [-1]), tf.int32)
        one_hot = tf.one_hot(soil_idx, depth=NUM_SOIL_TYPES, dtype=tf.float32)

        base_coh = tf.reduce_sum(one_hot * base_coh_all, axis=-1)  # (batch,)
        base_ifi = tf.reduce_sum(one_hot * base_ifi_all, axis=-1)  # (batch,)

        # --- Dense residual (bounded via tanh) ---
        raw_coh_residual = dense_output[:, 0]  # (batch,)
        raw_ifi_residual = dense_output[:, 1]  # (batch,)

        coh_residual = self.residual_scale_coh * tf.math.tanh(raw_coh_residual)
        ifi_residual = self.residual_scale_ifi * tf.math.tanh(raw_ifi_residual)

        # --- Combine base + residual ---
        coh = base_coh + coh_residual
        ifi = base_ifi + ifi_residual

        # --- Final soft bounding via sigmoid to stay within physical range ---
        per_sample_coh_min = tf.reduce_sum(one_hot * self.coh_min, axis=-1)
        per_sample_coh_max = tf.reduce_sum(one_hot * self.coh_max, axis=-1)
        per_sample_ifi_min = tf.reduce_sum(one_hot * self.ifi_min, axis=-1)
        per_sample_ifi_max = tf.reduce_sum(one_hot * self.ifi_max, axis=-1)

        # Rescale into [0,1] relative to range, apply sigmoid, rescale back
        coh_normalized = (coh - per_sample_coh_min) / (per_sample_coh_max - per_sample_coh_min + 1e-8)
        coh = per_sample_coh_min + (per_sample_coh_max - per_sample_coh_min) * tf.nn.sigmoid(
            tf.math.log(tf.clip_by_value(coh_normalized, 1e-6, 1.0 - 1e-6) /
                        (1.0 - tf.clip_by_value(coh_normalized, 1e-6, 1.0 - 1e-6)))
        )

        ifi_normalized = (ifi - per_sample_ifi_min) / (per_sample_ifi_max - per_sample_ifi_min + 1e-8)
        ifi = per_sample_ifi_min + (per_sample_ifi_max - per_sample_ifi_min) * tf.nn.sigmoid(
            tf.math.log(tf.clip_by_value(ifi_normalized, 1e-6, 1.0 - 1e-6) /
                        (1.0 - tf.clip_by_value(ifi_normalized, 1e-6, 1.0 - 1e-6)))
        )

        # --- Variance regularization loss ---
        var_loss = -self.lambda_var * (
            tf.math.reduce_variance(coh) + tf.math.reduce_variance(ifi)
        )
        self.add_loss(var_loss)

        return coh, ifi

    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda_var": self.lambda_var,
            "residual_scale_coh": self.residual_scale_coh,
            "residual_scale_ifi": self.residual_scale_ifi,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
