import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


@tf.keras.utils.register_keras_serializable()
class SoilConditionedGeotechLayer(tf.keras.layers.Layer):
    """Per-soil-type learnable baselines for cohesion and internal friction.

    Replaces CohesionLayer + InternalFrictionLayer + ClipLayer with a single
    layer that produces physically bounded, soil-type-conditioned coh and ifi.

    Input: [soil_type_idx (batch,1), dense_output (batch,2)]
    Output: [coh (batch,), ifi (batch,)] — bounded per soil type, with dense residual

    Soil types: 0=Sandy Clay Loam, 1=Loam, 2=Undifferentiated

    Also adds a variance regularization loss to prevent collapsed (constant)
    intermediate values.
    """

    # Physical ranges per soil type
    # Cohesion in kPa, internal friction angle in radians
    COH_RANGES = {
        0: (5.0, 30.0),    # Sandy Clay Loam
        1: (2.0, 20.0),    # Loam
        2: (2.0, 35.0),    # Undifferentiated
    }
    IFI_RANGES = {
        0: (0.20, 0.60),   # Sandy Clay Loam
        1: (0.15, 0.50),   # Loam
        2: (0.15, 0.65),   # Undifferentiated
    }

    def __init__(self, lambda_var=0.01, residual_scale_coh=5.0, residual_scale_ifi=0.05, **kwargs):
        kwargs.setdefault("name", "soil_geotech")
        super(SoilConditionedGeotechLayer, self).__init__(**kwargs)
        self.lambda_var = lambda_var
        self.residual_scale_coh = residual_scale_coh
        self.residual_scale_ifi = residual_scale_ifi

        # Build constant tensors for ranges
        self.coh_min = tf.constant(
            [self.COH_RANGES[i][0] for i in range(3)], dtype=tf.float32
        )
        self.coh_max = tf.constant(
            [self.COH_RANGES[i][1] for i in range(3)], dtype=tf.float32
        )
        self.ifi_min = tf.constant(
            [self.IFI_RANGES[i][0] for i in range(3)], dtype=tf.float32
        )
        self.ifi_max = tf.constant(
            [self.IFI_RANGES[i][1] for i in range(3)], dtype=tf.float32
        )

    def build(self, input_shape):
        # Learnable unconstrained parameters for per-soil-type baselines
        self.u_coh = self.add_weight(
            name="u_coh",
            shape=(3,),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
        )
        self.u_ifi = self.add_weight(
            name="u_ifi",
            shape=(3,),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        soil_type_idx, dense_output = inputs[0], inputs[1]

        # --- Base values per soil type via sigmoid bounding ---
        base_coh_all = self.coh_min + (self.coh_max - self.coh_min) * tf.nn.sigmoid(self.u_coh)
        base_ifi_all = self.ifi_min + (self.ifi_max - self.ifi_min) * tf.nn.sigmoid(self.u_ifi)

        # Select per-sample base values using one_hot (avoids tf.gather gradient issues)
        soil_idx = tf.cast(tf.reshape(soil_type_idx, [-1]), tf.int32)
        one_hot = tf.one_hot(soil_idx, depth=3, dtype=tf.float32)

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
        # Get per-sample min/max
        per_sample_coh_min = tf.reduce_sum(one_hot * self.coh_min, axis=-1)
        per_sample_coh_max = tf.reduce_sum(one_hot * self.coh_max, axis=-1)
        per_sample_ifi_min = tf.reduce_sum(one_hot * self.ifi_min, axis=-1)
        per_sample_ifi_max = tf.reduce_sum(one_hot * self.ifi_max, axis=-1)

        # Rescale into [0,1] relative to range, apply sigmoid, rescale back
        # This provides soft bounding with gradient everywhere
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
