

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, metrics, losses
from py_files.GallenModel import CriticalAcceleration, DisplacementIntermediate, FosLayer
from py_files.GallenModel_v1 import ClipLayer, DisplacementLayerRainFall, NewmarkActivation, CohesionLayer, InternalFrictionLayer, HydraulicConductivityLayer, WetnessLayer
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
#   'LITHODESC',
#   'Geomorphology'
  ]

class LandslideRainFall():

    def __init__(self, depth=8):
        self.depth = depth

    def classification_model(self, all_inputs, pga_input, soil_idx_input, encoded_features):
        """
            Builds the graph for PINN Model
            1. Starts with defining the layers of the network
            2. Was defined as 8 hidden layers
            3. Physics layer starts at CohesionLayer, InternalFrictionLayer, and ProportionSlab.
            4. DisplacementLayer defines the physics equation for Newmark-Based Displacement
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


        coh = CohesionLayer()(x)
        ifi = InternalFrictionLayer()(x)

        coh = ClipLayer(0, 40, name="cohesion_clip")(coh)
        ifi = ClipLayer(0.15, 0.75, name="ifi_clip")(ifi)

        k = HydraulicConductivityLayer()(soil_idx_input)
        m = WetnessLayer()([precipitation, contributing_area, soil_thickness, slope, k])
        m = ClipLayer(0, 0.5, name="m_clip")(m)

        ds, fos, critical_acceleration, acpg = DisplacementLayerRainFall()([coh, ifi, slope, pga_input, bulk_unit_weight, m])
        fos = FosLayer()(fos)
        ac, acpg = CriticalAcceleration()(critical_acceleration, acpg)
        ds = DisplacementIntermediate()(ds)

        sus = NewmarkActivation(threshold=2.0)(ds, fos, ac, acpg)
        self.model = Model(inputs = all_inputs + [pga_input, soil_idx_input], outputs = sus)

    def get_optimizer(self, lr=1e-05):
        """
            Uses Adam Optimizer
        """
        self.optimizer_instance = optimizers.Adam(learning_rate=lr)

    def compile_model_dce(self):
        """
            Compiles the model using Weighted Dice CrossEntropy Loss function
            Uses BinaryIoU, AUC, and accuracy for visual metrics during training of the model
        """

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

