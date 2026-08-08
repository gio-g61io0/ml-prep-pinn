"""Amortized Earthquake-Induced Landslide (EIL) displacement PINN for Makilala.

The network IS the solution family u(z, t; rho, k, eps_y, H, pga); training drives the
physics residuals (PDE, Bouc-Wen ODE, BC, IC) to zero via automatic differentiation --
no displacement labels. Body-force formulation: the earthquake enters the momentum PDE as
the inertial source +rho*a_base(t) (SPECFEM acceleration record, PGA-scaled per unit) with a
FIXED base BC; ``u`` is displacement relative to the base.
"""
from .config import (
    CharacteristicScales,
    ColumnParams,
    GlobalConsts,
    load_column_params,
    load_global_consts,
    load_norm_ranges,
    load_scales,
    load_unit_table,
)
from .data import UnitSampler
from .model import AmortizedNormalization, EILPinn, build_pinn
from .physics import (
    bc_residual,
    boucwen_residual,
    ic_residual,
    mse,
    pde_residual,
)
from .seismic import (
    real_base_acceleration,
    real_base_motion,
    ricker_base_motion,
    zero_base_motion,
)

__all__ = [
    "GlobalConsts",
    "CharacteristicScales",
    "ColumnParams",
    "load_global_consts",
    "load_scales",
    "load_norm_ranges",
    "load_unit_table",
    "load_column_params",
    "UnitSampler",
    "AmortizedNormalization",
    "EILPinn",
    "build_pinn",
    "pde_residual",
    "boucwen_residual",
    "bc_residual",
    "ic_residual",
    "mse",
    "real_base_motion",
    "real_base_acceleration",
    "ricker_base_motion",
    "zero_base_motion",
]
