import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


NUM_SOIL_TYPES = 12

# cm/s -> m/hr.  length: cm -> m is /100 (x 0.01);  time: per-second -> per-hour is
# x 3600 (a rate per hour is 3600x the same speed expressed per second).
# Combined: 0.01 * 3600 = 36.  So 1 cm/s = 36 m/hr.
CM_S_TO_M_HR = 36.0

# Representative saturated hydraulic conductivity per USDA texture class, in cm/s.
# Index order matches helpers.SOIL_TEXTURE_TO_IDX.
#
# "geotech" -- tiles the material bands of the project's Table 1 ("Typical values of
#   hydraulic conductivity for various soils based on past literatures"), which gives K
#   directly in cm/sec. These are the classical geotechnical-textbook bands (cf. Freeze
#   & Cherry 1979; Terzaghi & Peck), and the same convention
#   GallenModel_v1.HydraulicConductivityLayer already uses:
#
#       Gravel                     1e-1           cm/s   "Very High"
#       Coarse Sand, Fine Sand     1e-3 .. 1e-1   cm/s   "High to Medium"
#       Silty Sand                 1e-5 .. 1e-3   cm/s   "Low"
#       Silt, Silty Clay           1e-7 .. 1e-5   cm/s   "Very Low"
#       Clay                       < 1e-7         cm/s   "Tends to Impermeable"
#
#   The bands are contiguous -- they share the edges 1e-1, 1e-3, 1e-5, 1e-7 and tile the
#   space -- so a class sitting near an edge whose +/- spread crosses into the neighbour
#   is physically harmless, not an out-of-range error.
#
#   Each USDA texture class is assigned to the band of its dominant material, then placed
#   at an evenly log-spaced position within that band by relative fineness:
#       Coarse/Fine Sand   -> Sand, Loamy Sand
#       Silty Sand         -> Sandy Loam, Loam, Silt Loam, Sandy Clay Loam
#       Silt, Silty Clay   -> Clay Loam, Silty Clay Loam, Sandy Clay, Silt, Silty Clay
#       Clay               -> Clay
#   The Gravel row is unused: no USDA texture class is gravel.
#   Clay's band is open-ended below 1e-7, so its representative is set to 1e-7 / 3, which
#   makes the upper bound land exactly on the cited 1e-7 ceiling.
#
#   The per-class placement WITHIN a band is interpolation, NOT a tabulated value -- Table 1
#   is defined by material, not by USDA texture, so only the five band edges are citable.
#
#   SILT IS A DELIBERATE EXCEPTION to the coarse -> fine ordering, in both tables. Pure
#   silt is far less permeable than its texture position suggests, because it has neither
#   sand's large inter-grain pores nor clay's aggregation / shrink-swell cracks. Table 1
#   itself groups "Silt, Silty Clay" into one row, and Carsel & Parrish likewise put Silt
#   (6.0 cm/day) below Sandy Clay Loam (31.44). Silt is therefore placed low in that band,
#   next to Silty Clay. Every OTHER class is strictly monotonic coarse -> fine. Silt has
#   zero units in the Cotabato dataset, so this placement is untested here.
#
#   CAVEAT (measured, not theoretical): these bands describe REMOULDED / CONSOLIDATED
#   material at depth, not structured near-surface field soil with roots, peds and
#   macropores. Applied to the Cotabato units they drive the steady-state saturation
#   ratio to m = 1 for ~94% of units at the default lambda, and reproducing a median m of
#   0.5 requires lambda ~ 0.007 (a 0.7% recharge efficiency, an order of magnitude below
#   the physical 0.1-0.5 band). Use "rawls" if you need the wetness term to carry spatial
#   signal at a physical lambda.
#
# "rawls" -- Rawls, Brakensiek & Saxton (1982) representative Ksat, the table
#   conventional in the shallow-landslide / SHALSTAB literature, converted from
#   cm/hr by /3600. Strictly monotonic. The published table has 11 classes (no Silt);
#   the Silt entry here is interpolated between Silt Loam and Sandy Clay Loam and is
#   flagged below. With this table lambda ~ 0.089 reproduces a median m of 0.5, which
#   sits inside the physical recharge-efficiency band.
#
# VERIFY BEFORE PUBLICATION: these values are transcribed from the literature and
# should be checked against the primary sources before they appear in a paper.
KSAT_REPRESENTATIVE_CM_S = {
    # Table 1 bands, tiled. Band / log-spaced slot shown per class.
    "geotech": [
        3.00e-2,  # 0  Sand              Coarse+Fine Sand  1e-3..1e-1, upper slot
        3.00e-3,  # 1  Loamy Sand        Coarse+Fine Sand  1e-3..1e-1, lower slot
        4.00e-4,  # 2  Sandy Loam        Silty Sand        1e-5..1e-3, slot 1/4
        6.30e-5,  # 3  Silt Loam         Silty Sand        1e-5..1e-3, slot 3/4
        1.60e-4,  # 4  Loam              Silty Sand        1e-5..1e-3, slot 2/4
        4.70e-7,  # 5  Silt              Silt/Silty Clay   1e-7..1e-5, slot 4/5 (see note)
        2.50e-5,  # 6  Sandy Clay Loam   Silty Sand        1e-5..1e-3, slot 4/4
        4.70e-6,  # 7  Clay Loam         Silt/Silty Clay   1e-7..1e-5, slot 1/5
        2.10e-6,  # 8  Silty Clay Loam   Silt/Silty Clay   1e-7..1e-5, slot 2/5
        1.00e-6,  # 9  Sandy Clay        Silt/Silty Clay   1e-7..1e-5, slot 3/5
        2.10e-7,  # 10 Silty Clay        Silt/Silty Clay   1e-7..1e-5, slot 5/5
        3.33e-8,  # 11 Clay              Clay  < 1e-7      (rep = 1e-7/3 -> max hits 1e-7)
    ],
    "rawls": [
        5.833e-3,  # 0  Sand              21.0   cm/hr
        1.697e-3,  # 1  Loamy Sand         6.11  cm/hr
        7.194e-4,  # 2  Sandy Loam         2.59  cm/hr
        1.889e-4,  # 3  Silt Loam          0.68  cm/hr
        3.667e-4,  # 4  Loam               1.32  cm/hr
        6.400e-5,  # 5  Silt              INTERPOLATED — absent from the 11-class table;
                   #                      placed level with Clay Loam per Carsel & Parrish

        1.194e-4,  # 6  Sandy Clay Loam    0.43  cm/hr
        6.389e-5,  # 7  Clay Loam          0.23  cm/hr
        4.167e-5,  # 8  Silty Clay Loam    0.15  cm/hr
        3.333e-5,  # 9  Sandy Clay         0.12  cm/hr
        2.500e-5,  # 10 Silty Clay         0.09  cm/hr
        1.667e-5,  # 11 Clay               0.06  cm/hr
    ],
}


@tf.keras.utils.register_keras_serializable()
class HydraulicConductivityLayerCms(tf.keras.layers.Layer):
    """Learnable K per USDA texture class, tabulated in cm/s, output in m/hr.

    Replaces ``HydraulicConductivityLayerV3`` for the dimensionally consistent
    wetness path. Three differences, each deliberate:

      1. THE TABLE IS IN cm/s AND CONVERTED WITH ``* 36``, matching the v1 layer's
         convention. ``HydraulicConductivityLayerV3`` instead carries a table labelled
         cm/hr converted with ``/ 100``; its values look like OBSERVED min-max extremes
         from a soil database (6 of 12 classes have min = 0.00, maxima reach 841) rather
         than representative conductivities, which is why its arithmetic midpoint lands
         ~2 orders of magnitude high (Clay: 2.105 m/hr).
      2. BOUNDS ARE LOG-UNIFORM, not linear:
             K = exp( log(k_lo) + (log(k_hi) - log(k_lo)) * sigmoid(u_k) )
         K spans orders of magnitude, so a linear sweep puts sigmoid(0) = 0.5 at the
         ARITHMETIC mean of the bounds -- which on a log-scale quantity sits essentially
         at the maximum. Log-uniform puts it at the GEOMETRIC mean, i.e. exactly the
         tabulated representative value. This matches what ``WetnessRatioLayer`` already
         does for recharge R in this same module.
      3. BOUNDS ARE DERIVED, not tabulated: k_lo = rep / spread_factor,
         k_hi = rep * spread_factor. With the default spread_factor = 3 each class gets
         about one order of magnitude of freedom end to end -- enough to calibrate,
         tight enough that a trained K is still reportable as a physical conductivity.

    Input:  soil_texture_idx (batch, 1) -- int 0-11, per helpers.SOIL_TEXTURE_TO_IDX
    Output: K in m/hr (batch,) -- the unit ``WetnessLayerDimensional`` expects, where
            transmissivity T = K * H then carries m^2/hr.

    ``k_table`` selects the source: "geotech" (geotechnical-textbook bands, the v1
    convention) or "rawls" (Rawls et al. 1982). See KSAT_REPRESENTATIVE_CM_S for the
    provenance of each and for the measured consequence of choosing "geotech".
    """

    def __init__(self, k_table="geotech", spread_factor=3.0, trainable_k=True, **kwargs):
        kwargs.setdefault("name", "hydraulic_conductivity_cms")
        super(HydraulicConductivityLayerCms, self).__init__(**kwargs)
        if k_table not in KSAT_REPRESENTATIVE_CM_S:
            raise ValueError(
                f"k_table must be one of {sorted(KSAT_REPRESENTATIVE_CM_S)}, got {k_table!r}"
            )
        if spread_factor <= 1.0:
            raise ValueError(f"spread_factor must be > 1, got {spread_factor}")
        self.k_table = k_table
        self.spread_factor = spread_factor
        # trainable_k=False freezes u_k at its 0.0 initializer, which makes
        # K = rep * spread_factor**(2*sigmoid(0) - 1) = rep * spread_factor**0 = rep
        # EXACTLY — i.e. the tabulated literature value, with spread_factor then
        # having no effect at all. Use this when K should be a CITED CONSTANT rather
        # than a fitted parameter.
        #
        # Why that is often the honest choice: m depends only on the ratio
        # lambda*R / (K*H*sin θ), so a global lambda and a per-class K are NOT
        # separately identifiable from landslide labels. A "learned" K is then just
        # one arbitrary split of that ratio, not a measured conductivity — measured
        # here as a 26% spread in Clay Loam K across the 5 CV folds (u_k[7] = +0.094
        # in fold 1 vs +0.421 in fold 5). Freezing K leaves lambda as the single free
        # scalar, which has a checkable physical band (0.1-0.5), so a lambda that
        # must leave that band becomes evidence about the table / slab thickness /
        # DEM cell size instead of being silently absorbed into K.
        self.trainable_k = trainable_k
        rep = np.asarray(KSAT_REPRESENTATIVE_CM_S[k_table], dtype=np.float32)
        # Work in log space: bounds are a fixed ratio either side of the
        # representative value, so log(rep) is exactly the midpoint.
        self.log_k_lo = tf.constant(np.log(rep / spread_factor), dtype=tf.float32)
        self.log_k_hi = tf.constant(np.log(rep * spread_factor), dtype=tf.float32)

    def build(self, input_shape):
        # sigmoid(0) = 0.5 -> the geometric mean of the bounds -> exactly the
        # tabulated representative Ksat for every class.
        self.u_k = self.add_weight(
            name="u_k",
            shape=(NUM_SOIL_TYPES,),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=self.trainable_k,
        )
        super().build(input_shape)

    def call(self, inputs):
        soil_idx = tf.cast(tf.reshape(inputs, [-1]), tf.int32)
        # Log-uniform interpolation between the per-class bounds, in cm/s.
        log_k = self.log_k_lo + (self.log_k_hi - self.log_k_lo) * tf.nn.sigmoid(self.u_k)
        k_cms = tf.exp(log_k)
        # one_hot + reduce_sum rather than tf.gather: avoids UnsortedSegmentSum shape
        # errors in the backward pass (same workaround as the v1/v3 layers).
        one_hot = tf.one_hot(soil_idx, depth=NUM_SOIL_TYPES, dtype=tf.float32)
        k_cms_per_sample = tf.reduce_sum(one_hot * k_cms, axis=-1)
        # cm/s -> m/hr: x 0.01 (cm->m) x 3600 (per s -> per hr) = x 36.
        return k_cms_per_sample * CM_S_TO_M_HR

    def get_config(self):
        config = super().get_config()
        config.update({
            "k_table": self.k_table,
            "spread_factor": self.spread_factor,
            "trainable_k": self.trainable_k,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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
class FOSConsistencyLoss(tf.keras.layers.Layer):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def call(self, inputs):
        final, fos = inputs

        loss = self.bce(final, fos)
        self.add_loss(self.weight * loss)
        
        # track separately
        self.add_metric(loss, name="fos_loss", aggregation="mean")
        return final
    
@tf.keras.utils.register_keras_serializable()
class WetnessRatioLayer(tf.keras.layers.Layer):
    """Wetness index from a PER-PIXEL recharge-to-transmissivity ratio (R/T).

    Implements the TOPMODEL saturation form
        m = clip( (R/T) * A' / sin θ , 0, 1 )
    but — unlike the earlier version that used ONE global learnable R/T scalar —
    both R and T now vary per pixel (the *soil-topographic index* form,
    ln(a/(T₀·tanβ)), that TOPMODEL uses once transmissivity is not uniform):

      - ``R`` (recharge, m/hr) is a FREE learned per-pixel field: the caller passes
        a raw per-pixel logit ``r_raw`` (from a small Dense head), and this layer
        bounds it log-uniformly into a physical recharge range [r_min, r_max]:
            R = exp( log(r_min) + (log(r_max) - log(r_min)) * sigmoid(r_raw) ).
        The bound keeps R free/per-pixel but physically scaled (no runaway values),
        mirroring the sigmoid-range trick used for K in HydraulicConductivityLayerV3.
      - ``T`` (transmissivity, m²/hr) is derived per pixel as ``K · soil_thickness``,
        with K from HydraulicConductivityLayerV3 (learnable per USDA soil texture).

    Motivation: in the Cotabato study area transmissivity varies ~89× across pixels
    (soil thickness 0–40 m × a ~10× K contrast between soil classes) while recharge
    varies only ~1.5×, so lumping T into a single scalar is physically indefensible.

    This layer holds NO trainable weights of its own — R comes in pre-computed
    (bounded here) and T comes from ``k`` and ``soil_thickness``.

    Input:  [r_raw (batch,1), contributing_area (m²), soil_thickness (m),
             slope (deg), k (m/hr from HydraulicConductivityLayerV3)]
    Output: m ∈ [0, 1]

    NOTE: with ``use_log_area=True`` (default) the contributing area is compressed
    with log1p — matching WetnessLayer, whose docstring explains that raw A pegs
    high-accumulation pixels at m=1 and blocks the gradient through the clip. This
    makes m a wetness *index* (TWI-like), not a literal saturation ratio. Set
    ``use_log_area=False`` to use A directly as written in the formula.
    """

    def __init__(self, r_min=1e-5, r_max=1e-1, use_log_area=True, **kwargs):
        kwargs.setdefault("name", "wetness_ratio_layer")
        super(WetnessRatioLayer, self).__init__(**kwargs)
        self.r_min = r_min
        self.r_max = r_max
        self.use_log_area = use_log_area

    def call(self, inputs):
        r_raw, contributing_area, soil_thickness, slope, k = (
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        )
        # k arrives as (batch,) from HydraulicConductivityLayerV3; match (batch,1).
        k = tf.expand_dims(k, -1)
        # Convert slope degrees -> radians; floor sin to avoid divide-by-zero on flats.
        slope_rad = slope * 0.017453292519943295
        sin_slope = tf.maximum(tf.math.sin(slope_rad), 1e-6)
        # Per-pixel recharge R (m/hr): free logit bounded log-uniformly into
        # [r_min, r_max] so equal steps in r_raw are equal ratio steps in R.
        log_min = tf.math.log(tf.constant(self.r_min, dtype=tf.float32))
        log_max = tf.math.log(tf.constant(self.r_max, dtype=tf.float32))
        r = tf.exp(log_min + (log_max - log_min) * tf.nn.sigmoid(r_raw))
        # Per-pixel transmissivity T = K · soil_thickness (m²/hr); floor to avoid /0.
        t = tf.maximum(k * soil_thickness, 1e-10)
        area = tf.math.log1p(contributing_area) if self.use_log_area else contributing_area
        m = (r / t) * area / sin_slope
        m = tf.clip_by_value(m, 0.0, 1.0)
        return m

    def get_config(self):
        config = super().get_config()
        config.update({
            "r_min": self.r_min,
            "r_max": self.r_max,
            "use_log_area": self.use_log_area,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Recharge unit -> m/hr. R and K must be expressed per the SAME unit of time (m
# depends only on the ratio R/K), and K is in m/hr throughout this module, so every
# recharge quantity is converted to m/hr before it reaches the saturation formula.
# "mm/month" uses a 30-day month and "mm/yr" a 365-day year, matching the /720 and
# /8760 hour counts respectively.
RECHARGE_UNIT_TO_M_PER_HR = {
    "m/hr": 1.0,
    "mm/hr": 1e-3,
    "mm/day": 1e-3 / 24.0,
    "mm/month": 1e-3 / 720.0,
    "mm/yr": 1e-3 / 8760.0,
}


def steady_state_saturation(r, contributing_area, soil_thickness, slope,
                            k, cell_size_m, min_soil_thickness_m, scale):
    """Steady-state saturation ratio m = h_w / H, shared by the dimensional wetness layers.

    Implements ``m = clip(scale * R * a / (T * sin θ), 0, 1)`` — see
    ``WetnessLayerDimensional``'s docstring for the derivation and the dimensional
    bookkeeping. Factored out so the fitted-lambda variant
    (``WetnessLayerDimensional``, R read per unit from precipitation) and the
    user-supplied-recharge variant (``WetnessLayerFixedRecharge``, R a global
    constant) cannot drift apart in their physics.

    Args:
        r: recharge in m/hr — a scalar or a (batch, 1) tensor.
        contributing_area: total upslope area A in m^2, shape (batch, 1).
        soil_thickness: column thickness H in m, shape (batch, 1).
        slope: slope θ in DEGREES, shape (batch, 1).
        k: saturated conductivity in m/hr, shape (batch,) as emitted by the
            HydraulicConductivityLayer family.
        cell_size_m: contour width b in m (the DEM cell size) converting A to the
            specific catchment area a = A / b.
        min_soil_thickness_m: floor on H, so zero-thickness units give a shallow
            column rather than an infinitely resistive one (T -> 0 => m -> 1).
        scale: dimensionless recharge efficiency lambda.

    Returns:
        m in [0, 1], shape (batch, 1).
    """
    # k arrives as (batch,) from the conductivity layers; match (batch, 1).
    k = tf.expand_dims(k, -1)
    # Slope deg -> rad; floor sin θ so flats do not divide by zero.
    slope_rad = slope * 0.017453292519943295
    sin_slope = tf.maximum(tf.math.sin(slope_rad), 1e-6)
    # Specific catchment area a = A / b  [m^2 / m = m]. NO log — the linear
    # proportionality to a is the physics.
    a = contributing_area / cell_size_m
    # Transmissivity T = K * H  [m^2/hr], with H floored to a physical minimum.
    thickness = tf.maximum(soil_thickness, min_soil_thickness_m)
    t = k * thickness
    # Saturation ratio h_w/H. Every factor's units cancel: (m/hr * m) / (m^2/hr).
    m = scale * (r * a) / (t * sin_slope)
    # m > 1 is physically "saturated to the surface", so clipping is the model,
    # not a rescaling trick.
    return tf.clip_by_value(m, 0.0, 1.0)


@tf.keras.utils.register_keras_serializable()
class WetnessLayerDimensional(tf.keras.layers.Layer):
    """Dimensionally consistent TOPMODEL wetness: m = h_w / H, a TRUE saturation ratio.

    This is the physically closed replacement for ``GallenModel_v1.WetnessLayer``,
    which computed ``lambda * R * log1p(A) / (T sin θ)``. That form has two defects
    this layer fixes:

      1. ``log1p(A)`` takes the logarithm of a dimensioned quantity, leaving the
         result with units of 1/m — so it could not be a saturation ratio, and its
         absolute value was meaningless. It also destroyed the proportionality to
         upslope area that IS the physical content of the steady-state saturation
         formula (in the Cotabato data a 1.8e6-fold range in A compressed to 5.4-fold).
      2. ``lambda`` was bounded to [0.01, 1000] — an unbounded fitting scalar that
         silently absorbed the leftover dimensions (fold-5 trained it to 88.4).

    Formula (steady-state Montgomery & Dietrich 1994 / SHALSTAB saturation ratio):

        a = A / b                        specific catchment area   [m^2/m = m]
        T = K * H                        transmissivity            [m/hr * m = m^2/hr]
        m = clip( lambda * R * a / (T * sin θ), 0, 1 )             [dimensionless]

    WHY THE AREA IS DIVIDED BY b (i.e. why `a` is a LENGTH, not an area) -- the
    balance is between two DISCHARGES PER UNIT CONTOUR WIDTH, both m^2/hr:

        supply from upslope, through a contour of width b:
            Q = R * A          [m/hr * m^2 = m^3/hr]      total discharge
            q = Q / b = R * a   [m^3/hr / m = m^2/hr]      per unit width
        Darcy drainage through the saturated thickness h_w, slope-parallel flow:
            q_darcy = K * h_w * sin θ  [m/hr * m = m^2/hr] per unit width

        q = q_darcy  =>  h_w = R * a / (K * sin θ)  =>  m = h_w / H = R*a/(T sin θ)

    Darcy's law here is a per-unit-width discharge, so the supply side must be too.
    Using total A (m^3/hr) against K*h_w*sinθ (m^2/hr) mismatches by one length --
    that missing length IS b. This is also why b must be the DEM cell size: the
    contributing-area raster accumulates flow through one cell face.

    Dimensional bookkeeping, term by term:
        R      mm/month * (1e-3 m/mm) / (720 hr/month)  ->  m/hr      (a flux velocity)
        a      m^2 / m                                  ->  m
        K      m/hr                                     ->  m/hr      (a velocity)
        T      K * H = m/hr * m                         ->  m^2/hr    (NOT m/hr)
        R*a/T  (m/hr * m) / (m^2/hr)                    ->  1         (dimensionless)
        lambda dimensionless recharge efficiency in [0, 1]

    NOTE on sin θ vs tan β: this uses sin θ, following Montgomery & Dietrich /
    SHALSTAB, whose infinite-slope FoS is the same one DisplacementLayerRainFall
    implements. Classic TOPMODEL writes tan β instead; the two differ by cos θ
    (~7% at 20 deg, ~23% at 40 deg). sin θ is the internally consistent choice here.

    NOTE on R and K sharing a time unit: m depends only on the RATIO R/K, so both
    must be expressed per the same unit of time. Verified: scaling R and K together
    by any factor leaves m bit-identical. A K mis-converted from cm/s to m/hr as if
    it were cm/hr is therefore a clean factor-3600 error in m (it saturates).

    ``lambda`` is retained as the ONLY free scalar but is now bounded to [0, 1] and
    carries a physical meaning: the fraction of precipitation that becomes steady
    subsurface recharge after evapotranspiration and overland runoff (literature
    range roughly 0.1-0.5). Because it is dimensionless it cannot absorb a unit
    error, so the output stays a genuine saturation ratio whatever the network learns.

    m = 1 now means a physically saturated column (water table at the ground surface,
    i.e. the cell generates saturation-excess overland flow), which is the correct
    reading for valley-bottom / channel cells with very large ``a``. The clip is
    physics, not a range hack.

    Input:  [precipitation (mm/month), contributing_area (m^2), soil_thickness (m),
             slope (deg), k (m/hr from HydraulicConductivityLayerV3)]
    Output: m in [0, 1]  -- consumed by DisplacementLayerRainFall's pore-pressure
            term ``-(m * gamma_w * tan phi) / (gamma * tan theta)``, which requires
            exactly this quantity.

    IMPORTANT -- ``cell_size_m`` must match the DEM the contributing-area raster was
    derived from; it is the contour width b that converts total upslope area to
    specific catchment area. The Cotabato slope-unit rasters are on a 5 m grid
    (29% of slope-unit polygons are exactly 25 m^2 = one cell, and min(A)/25 = 1.04
    cells), hence the 5.0 default. A wrong b rescales m by a constant factor.

    IMPORTANT -- this layer is only meaningful with a physically plausible K, and
    ``HydraulicConductivityLayerV3``'s table is currently NOT one. Three separate
    defects, all of which this layer now exposes instead of hiding:

      (a) UNIT PROVENANCE IS UNDOCUMENTED AND INCONSISTENT WITH v1. The v3 table is
          labelled cm/hr and converted with ``/100``; the v1 layer
          (``GallenModel_v1.HydraulicConductivityLayer``) uses cm/s and converts with
          ``*36``. For the shared Sandy Clay Loam class v3's maximum is 112x v1's; for
          Loam it is 1461x. They cannot both be right. The v3 magnitudes are at least
          self-consistent as cm/hr -- read as cm/s they would give 15,138 m/hr (4 m/s)
          for Sand -- so do NOT "fix" v3 by applying v1's *36. The numbers, not the
          conversion, are the problem.
      (b) THE ORDERING IS NON-PHYSICAL. Ksat must fall monotonically from coarse to
          fine, but the table ranks Clay (421) 3rd HIGHEST -- above Loam (52.6) and
          Silt Loam (53.9) -- and puts Loamy Sand (189) below Silt (213). No choice of
          unit repairs an ordering violation. Clay must be the MINIMUM. This matters
          directly here: Cotabato uses only Clay Loam (class 7) and Clay (class 11),
          so 13% of units draw their K from the single most corrupted entry.
      (c) u_k = 0 -> sigmoid = 0.5 -> the MIDPOINT of very wide bounds, e.g. 2.105 m/hr
          for Clay (1.2e-1 cm/s -- roughly 6 orders too high for clay).

    Under the old log1p form the oversized K was silently cancelled by the oversized
    lambda (fold-5: K ~ 199 cm/hr with lambda ~ 88). Here it is not. With realistic
    Ksat and lambda = 1 roughly 90% of Cotabato units saturate at m = 1, so K must be
    constrained (or lambda learned small) for m to retain spatial variation.
    """

    def __init__(self, cell_size_m=5.0, lambda_min=0.0, lambda_max=1.0,
                 u_lambda_init=-2.3, min_soil_thickness_m=0.5, **kwargs):
        kwargs.setdefault("name", "wetness_layer_dimensional")
        super(WetnessLayerDimensional, self).__init__(**kwargs)
        self.cell_size_m = cell_size_m
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.u_lambda_init = u_lambda_init
        # SoilThc_mean contains exact zeros (57 rows in SU_17_training_v3) which would
        # send T -> 0 and m -> 1 for those units. Floor the THICKNESS (a physical
        # minimum soil depth) rather than flooring T at 1e-10, so the failure mode is
        # a shallow column rather than an infinitely resistive one.
        self.min_soil_thickness_m = min_soil_thickness_m

    def build(self, input_shape):
        # Single learnable scalar: recharge efficiency, sigmoid-bounded into
        # [lambda_min, lambda_max]. u_lambda_init = -2.3 -> sigmoid = 0.091 -> lambda
        # ~ 0.09 with the default [0, 1] bounds: the low end of the physical 0.1-0.5
        # band, chosen so the MEDIAN Cotabato unit starts near m = 0.5 under the "rawls"
        # Ksat table. Initializing at lambda = 0.5 instead would start 80-100% of units
        # clipped at m = 1, where clip_by_value has ZERO gradient — so lambda (and K)
        # could only learn from the unclipped minority. Starting off the clip keeps the
        # whole batch on the gradient path.
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
        # Recharge R: mm/month -> m/hr  (1e-3 m/mm, 720 hr per 30-day month).
        r = precipitation * RECHARGE_UNIT_TO_M_PER_HR["mm/month"]
        # Dimensionless recharge efficiency.
        scale = self.lambda_min + (self.lambda_max - self.lambda_min) * tf.nn.sigmoid(self.u_lambda)
        return steady_state_saturation(
            r, contributing_area, soil_thickness, slope, k,
            self.cell_size_m, self.min_soil_thickness_m, scale,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "cell_size_m": self.cell_size_m,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "u_lambda_init": self.u_lambda_init,
            "min_soil_thickness_m": self.min_soil_thickness_m,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class WetnessLayerFixedRecharge(tf.keras.layers.Layer):
    """Same steady-state saturation as ``WetnessLayerDimensional``, but R is a GLOBAL
    USER-SUPPLIED EFFECTIVE RECHARGE rather than a per-unit precipitation column.

        m = clip( R * a / (T * sin θ), 0, 1 )        a = A/b,  T = K*H

    WHY THIS EXISTS. ``WetnessLayerDimensional`` reads R from ``Prc_mean``, which is a
    zonal-mean PRECIPITATION statistic, not a recharge. Two separate objections:

      1. Precipitation is not recharge. What drives a water table is the fraction of
         rainfall that infiltrates and reaches the saturated zone after interception,
         evapotranspiration and overland runoff. ``WetnessLayerDimensional`` handles
         this with a fitted ``lambda`` in [0,1], which is defensible but leaves the
         actual recharge implicit and unreportable.
      2. Its SPATIAL variation is largely uninformative. Across the Cotabato slope
         units ``Prc_mean`` spans only ~1.5x (177-263), typically an artifact of
         interpolating a sparse gauge network, while ``a`` spans ~6 orders of
         magnitude and K ~3. Per-unit precipitation therefore contributes almost
         nothing to WHERE the model predicts saturation but does inject a covariate
         the network can latch onto.

    Fixing R to one number the modeller chooses and can cite makes the wetness field a
    function of topography, soil depth and conductivity ONLY, and turns "how wet is the
    scenario" into an explicit, sweepable input (a design storm, a wet-season mean
    recharge, a water-balance estimate) rather than a fitted scalar.

    NO FREE SCALAR. There is deliberately no ``lambda`` here: R is already the EFFECTIVE
    recharge, so an extra efficiency factor would be unidentifiable against it. That
    also means this layer has NO trainable weights — with ``trainable_k=False`` the whole
    wetness field becomes a fixed, fully reproducible physical forcing. If you want a
    fitted efficiency instead, use ``wetness_mode="dimensional"``.

    CHOOSING R. Because m scales linearly in R up to the clip, the value matters. Some
    anchors, all as steady effective recharge:
        ~50-200 mm/yr    long-term recharge in a drained tropical upland
        ~10-50 mm/day    wet-season / antecedent-storm conditions
        ~50-200 mm/day   an extreme triggering storm
    Sweep it and check the resulting m distribution: if ~everything clips at 1 the layer
    is saturated and its gradient to K is dead, and if ~everything sits near 0 the
    pore-pressure term never engages. Both are visible in the ``m_clip`` diagnostics.

    RECHARGE MUST BE STEADY. The formula is a steady-state balance, so R is a sustained
    rate, not a total depth. "300 mm in 24 h" enters as ``recharge=300,
    recharge_unit="mm/day"``, not as 300 mm/yr.

    Input:  [precipitation, contributing_area (m^2), soil_thickness (m), slope (deg),
             k (m/hr)] — SAME five-tensor signature as ``WetnessLayerDimensional`` so
            the two are drop-in interchangeable. ``precipitation`` is accepted and
            DELIBERATELY IGNORED: passing it keeps the ``Prc_mean`` Input reachable in
            the functional graph (Keras requires every declared Input to reach an
            output) while contributing exactly zero gradient — the same "connected but
            gradient-dead" idiom the pure-EIL path uses, without an extra sink layer.
    Output: m in [0, 1].
    """

    def __init__(self, recharge, recharge_unit, cell_size_m=5.0,
                 min_soil_thickness_m=0.5, **kwargs):
        kwargs.setdefault("name", "wetness_layer_fixed_recharge")
        super(WetnessLayerFixedRecharge, self).__init__(**kwargs)
        # Both are required with no default: a wrong recharge unit rescales m by up to
        # 8760x, and silently defaulting one would be exactly the class of unit error
        # this layer family keeps tripping over.
        if recharge is None or float(recharge) <= 0.0:
            raise ValueError(f"recharge must be a positive number, got {recharge!r}")
        if recharge_unit not in RECHARGE_UNIT_TO_M_PER_HR:
            raise ValueError(
                f"recharge_unit must be one of {sorted(RECHARGE_UNIT_TO_M_PER_HR)}, "
                f"got {recharge_unit!r}"
            )
        self.recharge = float(recharge)
        self.recharge_unit = recharge_unit
        self.cell_size_m = cell_size_m
        self.min_soil_thickness_m = min_soil_thickness_m
        # Converted once at construction; stored so diagnostics can read the value
        # the physics actually used rather than re-deriving the conversion.
        self.recharge_m_per_hr = self.recharge * RECHARGE_UNIT_TO_M_PER_HR[recharge_unit]

    def call(self, inputs):
        # inputs[0] is precipitation — present for graph connectivity only (see the
        # class docstring); the recharge comes from the constructor instead.
        contributing_area, soil_thickness, slope, k = (
            inputs[1], inputs[2], inputs[3], inputs[4]
        )
        return steady_state_saturation(
            self.recharge_m_per_hr, contributing_area, soil_thickness, slope, k,
            self.cell_size_m, self.min_soil_thickness_m, scale=1.0,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "recharge": self.recharge,
            "recharge_unit": self.recharge_unit,
            "cell_size_m": self.cell_size_m,
            "min_soil_thickness_m": self.min_soil_thickness_m,
        })
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
