"""
Configuration objects for the standalone PEGPN framework.

Every physically meaningful constant, unit assumption and modelling choice is
surfaced here so that the "minimum reporting checklist" of the methodology
report can be produced automatically from a single object.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence, Tuple
import json


# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
@dataclass
class PhysicsConfig:
    """Constants and operators of the infinite-slope / Newmark chain."""

    gamma_w: float = 9.80665          # unit weight of water [kN/m^3]
    g: float = 9.80665                # gravity [m/s^2]

    # Reference-stress parameterisation (Sec. 6.1 of the report)
    sigma_ref: float = 50.0           # sigma'_0, fixed reference effective normal stress [kPa]
    R0_min: float = 5.0               # lower bound of reference resistance [kPa]
    R0_max: float = 150.0             # upper bound of reference resistance [kPa]
    psi0: float = 0.30                # prior mean of the resistance partition

    # Slope-angle guard rails (tan(beta) appears in a denominator)
    beta_min_deg: float = 1.0
    beta_max_deg: float = 85.0

    # Jibson (2007) coseismic displacement regression, D_n in cm
    jibson_k: Tuple[float, float, float] = (0.215, 2.341, -1.438)
    jibson_sigma_log10: float = 0.510  # residual scatter of the regression

    # Newmark operator numerics
    tau: float = 0.05                 # smoothing temperature of the soft yield boundary
    eps_r: float = 1.0e-3             # numerical bound on the acceleration ratio
    D_cap_cm: float = 1000.0          # hard cap on reported displacement [cm]
    log1p_cap: float = 8.0            # smooth cap applied to log1p(D_soft) during training

    # Displacement-based instability index
    D0_cm: float = 5.0

    def __post_init__(self) -> None:
        if not (0.0 < self.psi0 < 1.0):
            raise ValueError("psi0 must lie strictly inside (0, 1).")
        if self.R0_min <= 0 or self.R0_max <= self.R0_min:
            raise ValueError("Require 0 < R0_min < R0_max.")
        if self.sigma_ref <= 0:
            raise ValueError("sigma_ref must be positive.")
        if not (0.0 < self.eps_r < 0.5):
            raise ValueError("eps_r must be a small positive number.")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Dataset location, column mapping, unit policy and derived-variable rules."""

    gpkg_path: str
    layer: Optional[str] = None

    # ---- column mapping (resolved exactly first, see columns.py) ----
    label_col: str = "landslide"
    group_col: str = "brgy_name"
    slope_col: str = "Slope_mean"
    depth_col: str = "SoilThc_mean"
    density_col: str = "BUK_mean"
    pga_col: str = "PGA2_max"

    material_numeric: Sequence[str] = ("Clay_mean", "Sand_mean", "Silt_mean")
    material_categorical: Sequence[str] = ("Lithology", "soil_texture_idx")
    texture_name_col: Optional[str] = "soil_texture"
    litho_name_col: Optional[str] = "LITHO"

    # ---- groundwater ratio m = h_w / z ----
    use_uniform_m: bool = True
    m_uniform: float = 0.18
    m_col: Optional[str] = None       # set together with use_uniform_m=False
    m_min: float = 0.0
    m_max: float = 1.0

    # ---- unit resolution ----
    # 'auto' inspects the median and matches it against known conventions.
    density_units: str = "auto"       # auto | g/cm3 | cg/cm3 | kg/m3
    depth_units: str = "auto"         # auto | m | cm
    pga_units: str = "auto"           # auto | g | m/s2 | gal
    slope_units: str = "deg"          # deg | rad | percent

    # ---- dry bulk density -> total unit weight ----
    # BUK_mean is a DRY bulk density, not a total moist unit weight. The
    # conversion is explicit and reported.
    unit_weight_mode: str = "sat_weighted"   # dry | moist | sat_weighted
    specific_gravity: float = 2.65           # G_s for porosity-based saturation
    gravimetric_water_content: float = 0.15  # w, used by 'moist' and above the water table

    # ---- depth-to-bedrock -> potential failure depth ----
    # SoilThc_mean is a depth to bedrock, not a failure-plane depth. The
    # translation rule is explicit and sensitivity-tested.
    z_mode: str = "cap"               # raw | cap | scale | fixed
    z_cap_m: float = 3.0
    z_scale: float = 0.5
    z_fixed_m: float = 2.0
    z_min_m: float = 0.3

    # ---- fatal validation ----
    fatal_validation: bool = True
    admissible: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "beta_deg": (0.0, 90.0),
        "z_m": (0.05, 60.0),
        "gamma_kNm3": (10.0, 26.0),
        "m_ratio": (0.0, 1.0),
        "pga_g": (0.0, 3.0),
    })

    def __post_init__(self) -> None:
        if not self.use_uniform_m and not self.m_col:
            raise ValueError("use_uniform_m=False requires m_col to be set.")
        if not (0.0 <= self.m_uniform <= 1.0):
            raise ValueError("m_uniform must lie in [0, 1].")
        if self.unit_weight_mode not in {"dry", "moist", "sat_weighted"}:
            raise ValueError(f"Unknown unit_weight_mode: {self.unit_weight_mode}")
        if self.z_mode not in {"raw", "cap", "scale", "fixed"}:
            raise ValueError(f"Unknown z_mode: {self.z_mode}")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Optimisation, regularisation and ensembling settings."""

    hidden: Sequence[int] = (64, 32)
    emb_dim_litho: int = 8
    emb_dim_texture: int = 6
    dropout: float = 0.10
    activation: str = "tanh"

    learning_rate: float = 3.0e-3
    epochs: int = 120
    batch_size: int = 1024
    patience: int = 15
    min_delta: float = 1.0e-4
    seeds: Sequence[int] = (0, 1, 2)

    # class imbalance (~11% positives)
    class_weighting: str = "balanced"   # none | balanced | focal
    focal_gamma: float = 2.0

    # loss weights
    lambda_prior: float = 1.0
    lambda_psi: float = 0.10
    lambda_anchor: float = 1.0
    lambda_mono: float = 0.0            # monotonicity is architecturally guaranteed; tested post hoc
    l2: float = 1.0e-5

    calibration: str = "isotonic"       # none | platt | isotonic
    verbose: int = 0

    def __post_init__(self) -> None:
        if self.class_weighting not in {"none", "balanced", "focal"}:
            raise ValueError(f"Unknown class_weighting: {self.class_weighting}")
        if self.calibration not in {"none", "platt", "isotonic"}:
            raise ValueError(f"Unknown calibration: {self.calibration}")


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
@dataclass
class CVConfig:
    """Nested, spatially aware validation design."""

    scheme: str = "group"     # group (barangay) | block (KMeans on centroids)
    n_outer: int = 5
    n_inner: int = 3
    n_blocks: int = 15        # only for scheme='block'
    random_state: int = 42
    select_threshold_on_inner: bool = True

    def __post_init__(self) -> None:
        if self.scheme not in {"group", "block"}:
            raise ValueError(f"Unknown CV scheme: {self.scheme}")
        if self.n_outer < 2 or self.n_inner < 2:
            raise ValueError("Need at least 2 outer and 2 inner folds.")


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
@dataclass
class UncertaintyConfig:
    """Joint Monte-Carlo propagation through the full physical chain."""

    n_draws: int = 200
    chunk: int = 20                 # draws evaluated per vectorised chunk

    cv_z: float = 0.30              # lognormal CV of failure depth
    cv_gamma: float = 0.07          # lognormal CV of unit weight
    cv_pga: float = 0.25            # lognormal CV of PGA
    sd_m: float = 0.10              # additive sd of the groundwater ratio (truncated to [0,1])
    include_jibson_residual: bool = True
    include_parameter_uncertainty: bool = True   # sample across ensemble members

    random_state: int = 0
    quantiles: Tuple[float, float] = (0.05, 0.95)


# --------------------------------------------------------------------------
# Master
# --------------------------------------------------------------------------
@dataclass
class PEGPNConfig:
    data: DataConfig
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)

    # ablation switches (Sec. 16)
    use_psi: bool = True              # False -> psi fixed at psi0
    use_priors: bool = True
    use_anchor: bool = True
    use_water: bool = True            # False -> m forced to 0
    use_embeddings: bool = True
    pga_in_encoder: bool = False      # True reproduces the shortcut defect on purpose
    smooth_training: bool = True      # False -> train on the exact piecewise operator

    def to_dict(self) -> dict:
        return {
            "data": asdict(self.data),
            "physics": asdict(self.physics),
            "train": asdict(self.train),
            "cv": asdict(self.cv),
            "uncertainty": asdict(self.uncertainty),
            "ablation": {
                "use_psi": self.use_psi,
                "use_priors": self.use_priors,
                "use_anchor": self.use_anchor,
                "use_water": self.use_water,
                "use_embeddings": self.use_embeddings,
                "pga_in_encoder": self.pga_in_encoder,
                "smooth_training": self.smooth_training,
            },
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        txt = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            with open(path, "w") as fh:
                fh.write(txt)
        return txt
