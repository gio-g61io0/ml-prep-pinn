import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, metrics, losses
from py_files.GallenModel import CriticalAcceleration, DisplacementIntermediate, FosLayer
from py_files.GallenModel_v1 import DisplacementLayerRainFall, NewmarkActivation, WetnessLayer, CohesionLayer, InternalFrictionLayer
from py_files.GallenModel_v3 import FOSConsistencyLoss, HydraulicConductivityLayerV3, WetnessRatioLayer
from py_files.Landslidev2_Old import DiceCrossEntropyLoss, FOSPhysicsLoss


# NOTE: this module-level list defines the canonical input order for the
# physics layers. To train on a reduced feature subset (e.g. via GA-EN
# feature selection), reassign `py_files.LandslideRainfall_v3.numeric_cols`
# to the reduced list BEFORE calling `classification_model()`.
numeric_cols = ['Clay_mean',
  'Sand_mean',
  'Silt_mean',
  'NDVI_mean',
  'Est_mean',
  'Nrt_mean',
  'HorCurv_mean',
  'VertCurv_mean',
  'Slope_mean',
  'Elev_mean',
  'SoilThc_mean',
  'DistFlt_min',
  'LULC_majority',
  'TWI_mean',
  'Prc_mean',
  'Distrv_min',
  'distrd_min',
  'BUK_mean',
  'ContributingFactor_mean',
  'type',
  'soil_texture_idx',
  ]

PHYSICS_REQUIRED_COLS = (
    "Slope_mean", "BUK_mean", "Prc_mean",
    "ContributingFactor_mean", "SoilThc_mean",
)


@tf.keras.utils.register_keras_serializable()
class LogitLayer(tf.keras.layers.Layer):
    """Numerically stable logit: log(p / (1 - p)) with clipping."""
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, p):
        p_clip = tf.clip_by_value(p, self.eps, 1.0 - self.eps)
        return tf.math.log(p_clip) - tf.math.log(1.0 - p_clip)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


class LandslideRainFallV3():
    """Rainfall PINN model v3 with 12 USDA soil texture classes.

    Changes from v1 (LandslideRainFall):
    1. Unconstrained CohesionLayer + InternalFrictionLayer (model learns freely).
    2. HydraulicConductivityLayerV3 with 12 USDA soil types for wetness only.
    3. Soil type index derived from USDA texture classification of
       clay/silt/sand g/kg values rather than the 'type' column.
    4. Hybrid output: additive-logit residual + auxiliary physics_prob
       supervision to prevent physics collapse.
    """

    def __init__(self, depth=8, aux_weight=0.7, residual_scale=3.0, use_rainfall=True,
                 wetness_mode="conductivity"):
        self.depth = depth
        self.aux_weight = aux_weight
        # Caps the residual head in logit space: residual = scale * tanh(dense_out)
        # so |residual| <= residual_scale. Forces physics to carry most of the signal;
        # residual can only nudge by up to ~sigmoid(scale) - 0.5 in probability space.
        self.residual_scale = residual_scale
        # When False, builds a "pure earthquake-induced landslide" (EIL) model:
        # precipitation is fully disconnected from BOTH downstream paths — the
        # wetness/pore-pressure term (m := 0 -> dry static FoS) AND the residual
        # DNN branch (which otherwise sees Prc_mean via all_features). Use for
        # inventories where rainfall does not drive WHERE slides occur (e.g. a
        # purely seismic trigger). Default True = original rainfall PINN.
        self.use_rainfall = use_rainfall
        # Selects how the wetness index m is produced:
        #   "conductivity" (default) — learn K per USDA soil texture
        #       (HydraulicConductivityLayerV3), derive T = K·soil_thickness, and
        #       read R from Prc_mean inside WetnessLayer. Preserves the trained
        #       v2-8 architecture and checkpoints.
        #   "rt_ratio" — estimate the recharge/transmissivity ratio R/T DIRECTLY
        #       as one learnable scalar (WetnessRatioLayer). Precipitation drops
        #       out of the wetness formula; wetness varies only with contributing
        #       area and slope. Classic TOPMODEL lumped calibration parameter.
        if wetness_mode not in ("conductivity", "rt_ratio"):
            raise ValueError(
                f"wetness_mode must be 'conductivity' or 'rt_ratio', got {wetness_mode!r}"
            )
        self.wetness_mode = wetness_mode

    def classification_model(self, all_inputs, pga_input, soil_idx_input, encoded_features):
        """
            Builds the graph for PINN Model v3
            Unconstrained coh/ifi + soil-conditioned K for wetness
            Additive-logit hybrid head + auxiliary physics_prob output
        """
        for required in PHYSICS_REQUIRED_COLS:
            if required not in numeric_cols:
                raise ValueError(
                    f"Physics-required feature '{required}' missing from numeric_cols. "
                    "Re-point py_files.LandslideRainfall_v3.numeric_cols to a list "
                    "that includes all PHYSICS_REQUIRED_COLS before building the model."
                )

        units = [32, 64, 8, 64, 32, 8, 32, 8]
        all_features = tf.keras.layers.concatenate(encoded_features)
        features_only = all_features

        # Geotechnical MLP must NOT see Prc_mean — rainfall reaches FOS only through
        # WetnessLayer → m → DisplacementLayerRainFall. Without this cut, the Dense head
        # learns a shortcut (Prc → coh/IFI) that bypasses the physics and fails to
        # generalize to gentle-slope, high-rainfall pixels (the FN cluster).
        prc_idx = numeric_cols.index("Prc_mean")
        geotech_features = tf.keras.layers.concatenate(
            [f for i, f in enumerate(encoded_features) if i != prc_idx]
        )

        # Pure-EIL: also keep rainfall out of the residual DNN branch (it would
        # otherwise re-enter via `features_only = all_features`). Fall back to the
        # Prc-excluded concat so precipitation has NO path to the output.
        if not self.use_rainfall:
            features_only = geotech_features

        slope = all_inputs[numeric_cols.index("Slope_mean")]
        bulk_unit_weight = all_inputs[numeric_cols.index("BUK_mean")]
        precipitation = all_inputs[numeric_cols.index("Prc_mean")]
        contributing_area = all_inputs[numeric_cols.index("ContributingFactor_mean")]
        soil_thickness = all_inputs[numeric_cols.index("SoilThc_mean")]

        x = layers.Dense(
            units=64,
            name="Sus_0",
            kernel_initializer="random_normal",
            bias_initializer="random_normal",
        )(geotech_features)

        for i in range(1, self.depth + 1):
            x = layers.Dense(
                units=units[i - 1],
                name=f"Sus_{i}",
                kernel_initializer="random_normal",
                bias_initializer="random_normal",

            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)

        x = layers.Dense(units=2, name="geotechnical_param")(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

        # Unconstrained cohesion and internal friction
        coh = CohesionLayer()(x)
        ifi = InternalFrictionLayer()(x)

        # --- Wetness index m ---------------------------------------------------
        # Two interchangeable parameterizations (self.wetness_mode). Both must keep
        # EVERY declared input connected to an output — Keras functional models
        # require it — hence the zero-scaled sinks in the rt_ratio branch.
        if self.wetness_mode == "rt_ratio":
            # Per-pixel soil-topographic index. Recharge R is a FREE learned per-pixel
            # field (small Dense head, bounded to a physical range inside the layer),
            # and transmissivity T = K(soil)·SoilThc is derived per pixel by reusing
            # HydraulicConductivityLayerV3. Transmissivity varies ~89× across the area,
            # so it must NOT be lumped into a single scalar (the old design).
            k = HydraulicConductivityLayerV3()(soil_idx_input)
            r_head = layers.Dense(
                8,
                name="recharge_head_1",
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
            )(features_only)
            r_head = layers.LeakyReLU(negative_slope=0.2)(r_head)
            r_raw = layers.Dense(1, name="recharge_head")(r_head)  # per-pixel recharge logit
            m = WetnessRatioLayer()([r_raw, contributing_area, soil_thickness, slope, k])
            # Precip now enters via features_only -> recharge_head; soil_idx via k;
            # soil_thickness via T. In pure-EIL (use_rainfall=False) features_only
            # excludes precip, so precipitation would be graph-disconnected — keep it
            # connected (gradient-dead) with a zero-scale sink only in that case.
            if not self.use_rainfall:
                precip_sink = layers.Rescaling(scale=0.0, name="precip_sink")(precipitation)
                m = layers.Add(name="m_rt_sinks")([m, precip_sink])
        else:
            # Default: soil-conditioned K for wetness only.
            k = HydraulicConductivityLayerV3()(soil_idx_input)
            # Always build the wetness subgraph so every declared input (precipitation,
            # contributing area, soil thickness, soil-conductivity index) stays connected
            # to the output — Keras functional models require it. For pure-EIL we then zero
            # the wetness index with Rescaling(scale=0): (a) the pore-pressure term in FoS
            # vanishes (dry static FoS) and (b) the gradient back to precipitation is killed,
            # so rainfall carries no learnable influence. Rescaling is a serializable built-in
            # (a Python-lambda Lambda would break load_model's safe deserialization).
            m = WetnessLayer()([precipitation, contributing_area, soil_thickness, slope, k])
        if not self.use_rainfall:
            m = layers.Rescaling(scale=0.0, name="m_zero")(m)
        # WetnessLayer already clips m to [0, 1]; pass through a linear "Activation"
        # only to preserve the "m_clip" layer name for downstream diagnostic code.
        # The previous sigmoid here was a bug: sigmoid([0,1]) -> [0.5, 0.731], which
        # compressed physical wetness variation and blocked the gradient to K.
        m = layers.Activation("linear", name="m_clip")(m)
        ds, fos, critical_acceleration, acpg = DisplacementLayerRainFall()([coh, ifi, slope, pga_input, bulk_unit_weight, m])
        fos = FosLayer()(fos)
        ac, acpg = CriticalAcceleration()(critical_acceleration, acpg)
        ds = DisplacementIntermediate()(ds)

        # Physics-only probability (auxiliary output)
        physics_prob = NewmarkActivation(threshold=10.0, name="physics_prob")(ds, fos, ac, acpg)

        # Residual DNN branch (unregularized; allows symmetric corrections)
        res = layers.Dense(
            16,
            name="residual_dense1",
            kernel_initializer="random_normal",
            bias_initializer="random_normal",
        )(features_only)
        res = layers.LeakyReLU(negative_slope=0.2)(res)
        res = layers.Dense(
            1,
            name="residual_dense2",
            kernel_initializer="random_normal",
            bias_initializer="random_normal",
        )(res)
        # Bound residual to [-residual_scale, +residual_scale] in logit space.
        # Prevents the DNN branch from overriding the physics layer (e.g. residual
        # logits of +10 fully drowning out physics_logit).
        res_bounded = layers.Activation("tanh", name="residual_tanh")(res)
        res_scaled = layers.Rescaling(
            scale=self.residual_scale, name="residual_scaled"
        )(res_bounded)

        # Option A: additive residual in logit space
        # final_logit = logit(physics_prob) + residual
        physics_logit = LogitLayer(name="physics_logit")(physics_prob)
        combined_logit = layers.Add(name="combined_logit")([physics_logit, res_scaled])
        final = layers.Activation("sigmoid", name="final_head")(combined_logit)

        # Option B: multi-output for auxiliary supervision on physics_prob
        self.model = Model(
            inputs=all_inputs + [pga_input, soil_idx_input],
            outputs={"final_head": final, "physics_prob": physics_prob, "fos":fos},
        )

    @staticmethod
    def build_residual_extractor(model):
        """Wrap a trained LandslideRainFallV3 model to expose intermediate signals.

        Returns a Keras Model with the same inputs as `model` but named
        outputs per sample:
          - residual:      logit-space nudge actually added to physics_logit
                           (after tanh + scale bounding). This is what matters
                           for interpreting how the residual moved the prediction.
          - residual_raw:  unbounded output of residual_dense2 (pre-tanh) — useful
                           for diagnosing saturation of the bounded residual.
          - physics_logit: logit(physics_prob), the physics branch in logit space
          - physics_prob:  physics-only probability
          - final_head:    final combined probability
                           (sigmoid(physics_logit + bounded_residual))

        Usage:
            extractor = LandslideRainFallV3.build_residual_extractor(trained_model)
            preds = extractor.predict(inference_ds)
            residual = preds["residual"].squeeze()
        """
        outputs = {
            "residual_raw":  model.get_layer("residual_dense2").output,
            "physics_logit": model.get_layer("physics_logit").output,
            "physics_prob":  model.get_layer("physics_prob").output,
            "final_head":    model.get_layer("final_head").output,
        }
        # Newer models (with bounded residual) expose residual_scaled; fall back
        # to residual_dense2 for backward compatibility with older checkpoints.
        try:
            outputs["residual"] = model.get_layer("residual_scaled").output
        except ValueError:
            outputs["residual"] = model.get_layer("residual_dense2").output
        return Model(inputs=model.inputs, outputs=outputs)

    @staticmethod
    def to_multi_output_ds(ds, class_weight=None):
        """Replicate single-label dataset into dict labels for dual-head training.

        If class_weight is provided (e.g. {0: 1, 1: 5}) the dataset also emits
        per-sample weights, which is the multi-output-safe substitute for the
        `class_weight` kwarg of `fit()` (that kwarg is not supported with
        multi-output models).
        """
        if class_weight is None:
            return ds.map(lambda x, y: (x, {"final_head": y, "physics_prob": y}))

        w0 = tf.constant(float(class_weight[0]), dtype=tf.float32)
        w1 = tf.constant(float(class_weight[1]), dtype=tf.float32)

        def _attach(x, y):
            y_f = tf.cast(y, tf.float32)
            sw = y_f * w1 + (1.0 - y_f) * w0
            return (
                x,
                {"final_head": y, "physics_prob": y},
                {"final_head": sw, "physics_prob": sw},
            )

        return ds.map(_attach)

    def get_optimizer(self, lr=1e-4):
        self.optimizer_instance = optimizers.Adam(learning_rate=lr)

    def compile_model_dce(self):
        self.model.compile(
            optimizer=self.optimizer_instance,
            loss={
                "final_head": DiceCrossEntropyLoss(),
                "physics_prob": losses.BinaryCrossentropy(),
            },
            loss_weights={"final_head": 1.0, "physics_prob": self.aux_weight},
            metrics={
                "final_head": [
                    metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5),
                    metrics.AUC(name="auc"),
                    "accuracy",
                ],
                "physics_prob": [metrics.AUC(name="auc")],
            },
        )

    def compile_model_bce(self):
        self.model.compile(
            optimizer=self.optimizer_instance,
            loss={
                "final_head": losses.BinaryCrossentropy(),
                "physics_prob": losses.BinaryCrossentropy(),
            },
            loss_weights={"final_head": 1.0, "physics_prob": self.aux_weight},
            metrics={
                "final_head": [
                    metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5),
                    metrics.AUC(name="auc"),
                    "accuracy",
                ],
                "physics_prob": [metrics.AUC(name="auc")],
            },
        )


#Wrapper for the functional model
class PhysicsModel(tf.keras.Model):
    def __init__(self, base_model, aux_weight=0.4, fos_weight=0.6, **kwargs):
        super(PhysicsModel, self).__init__(**kwargs)
        self.base_model = base_model
        self.aux_weight = aux_weight
        self.fos_weight = fos_weight
        self.dice_loss = DiceCrossEntropyLoss()
        self.bce_loss = losses.BinaryCrossentropy()

        #Trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.final_loss_tracker = tf.keras.metrics.Mean(name="final_head_loss")
        self.physics_loss_tracker = tf.keras.metrics.Mean(name="physics_prob_loss")
        self.fos_loss_tracker = tf.keras.metrics.Mean(name="fos_loss")  



    def train_step(self, data):
        x, y, sample_weight = data
        with tf.GradientTape() as tape:
            pred = self.base_model(x, training=True)
            final=pred["final_head"]
            physics=pred["physics_prob"]
            fos=pred["fos"]

            final_loss = self.dice_loss(
                y["final_head"], final, sample_weight=sample_weight['final_head'],
            )
            # FOS teaches final: fos is the (frozen) target, final learns toward it.
            # reshape both to (batch, 1) so BinaryCrossentropy's strict equal-rank
            # check can't fail on a shape/rank mismatch between fos and final.
            fos_t = tf.reshape(fos, [-1, 1])
            final_t = tf.reshape(final, [-1, 1])
            fos_loss = self.bce_loss(tf.stop_gradient(fos_t), final_t)
            total_loss = (
                (final_loss * self.aux_weight) + (self.fos_weight * fos_loss)
            )

        self.loss_tracker.update_state(total_loss)
        self.final_loss_tracker.update_state(final_loss)
        self.fos_loss_tracker.update_state(fos_loss)

        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {
            "loss": self.loss_tracker.result(),
            "final_head_loss": self.final_loss_tracker.result(),
            "fos_loss": self.fos_loss_tracker.result(),
        }
    

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def test_step(self, data):
        x, y, sample_weight = data

        pred = self.base_model(x, training=False)

        final = pred["final_head"]
        fos = pred["fos"]

        final_loss = self.dice_loss(
            y["final_head"],
            final,
            sample_weight=sample_weight["final_head"],
        )

        fos_t = tf.reshape(fos, [-1, 1])
        final_t = tf.reshape(final, [-1, 1])
        fos_loss = self.bce_loss(tf.stop_gradient(fos_t), final_t)

        total_loss = final_loss + self.fos_weight * fos_loss

        self.loss_tracker.update_state(total_loss)
        self.final_loss_tracker.update_state(final_loss)
        self.fos_loss_tracker.update_state(fos_loss)

        return {
            "loss": self.loss_tracker.result(),
            "final_head_loss": self.final_loss_tracker.result(),
            "fos_loss": self.fos_loss_tracker.result(),
        }
    def get_config(self):
        config = super().get_config()
        config.update({
            "base_model": self.base_model,
            "aux_weight": self.aux_weight,
            "fos_weight": self.fos_weight,
        })
        return config
    
    
class BaseModelCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        self.model.base_model.save(self.filepath)


class NewmarkPhysicsModel(tf.keras.Model):
    def __init__(self):
        pass