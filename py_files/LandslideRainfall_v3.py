import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, metrics, losses
from py_files.GallenModel import CriticalAcceleration, DisplacementIntermediate, FosLayer
from py_files.GallenModel_v1 import DisplacementLayerRainFall, NewmarkActivation, WetnessLayer, ClipLayer, CohesionLayer, InternalFrictionLayer
from py_files.GallenModel_v3 import HydraulicConductivityLayerV3
from py_files.Landslidev2_Old import DiceCrossEntropyLoss


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
  'soil_type_idx',
  ]

class LandslideRainFallV3():
    """Rainfall PINN model v3 with 12 USDA soil texture classes.

    Changes from v1 (LandslideRainFall):
    1. Unconstrained CohesionLayer + InternalFrictionLayer (model learns freely).
    2. HydraulicConductivityLayerV3 with 12 USDA soil types for wetness only.
    3. Soil type index derived from USDA texture classification of
       clay/silt/sand g/kg values rather than the 'type' column.
    """

    def __init__(self, depth=8):
        self.depth = depth

    def classification_model(self, all_inputs, pga_input, soil_idx_input, encoded_features):
        """
            Builds the graph for PINN Model v3
            Unconstrained coh/ifi + soil-conditioned K for wetness
        """
        units = [32, 64, 8, 64, 32, 8, 32, 8]
        all_features = tf.keras.layers.concatenate(encoded_features)
        features_only = all_features

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
        )(features_only)

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

        # Soil-conditioned K for wetness only
        k = HydraulicConductivityLayerV3()(soil_idx_input)
        m = WetnessLayer()([precipitation, contributing_area, soil_thickness, slope, k])
        m = ClipLayer(0, 0.7, name="m_clip")(m)

        ds, fos, critical_acceleration, acpg = DisplacementLayerRainFall()([coh, ifi, slope, pga_input, bulk_unit_weight, m])
        fos = FosLayer()(fos)
        ac, acpg = CriticalAcceleration()(critical_acceleration, acpg)
        ds = DisplacementIntermediate()(ds)

        sus = NewmarkActivation(threshold=2.0)(ds, fos, ac, acpg)
        self.model = Model(inputs=all_inputs + [pga_input, soil_idx_input], outputs=sus)

    def get_optimizer(self, lr=1e-05):
        self.optimizer_instance = optimizers.Adam(learning_rate=lr)

    def compile_model_dce(self):
        dce_loss = DiceCrossEntropyLoss()
        self.model.compile(
            optimizer=self.optimizer_instance,
            loss=dce_loss,
            metrics=[
                metrics.BinaryIoU(target_class_ids=[0,1], threshold=0.5),
                metrics.AUC(),
                "accuracy",
            ],
        )

    def compile_model_bce(self):
        self.model.compile(
            optimizer=self.optimizer_instance,
            loss=losses.BinaryCrossentropy(),
            metrics=[
                metrics.BinaryIoU(target_class_ids=[0,1], threshold=0.5),
                metrics.AUC(),
                "accuracy",
            ],
        )
