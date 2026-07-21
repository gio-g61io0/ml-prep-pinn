"""Amortized Earthquake-Induced Landslide (EIL) displacement PINN for Makilala.

The network IS the solution family u(z, t; rho, k, eps_y, H, pga); training drives the
physics residuals (PDE, Bouc-Wen ODE, BC, IC) to zero via automatic differentiation --
no displacement labels. Base motion is synthetic (PGA-scaled Ricker) until the SPECFEM
base motion exists.
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
from .model import AmortizedNormalization, build_pinn
from .physics import (
    bc_residual,
    boucwen_residual,
    ic_residual,
    mse,
    pde_residual,
)
from .seismic import real_base_motion, ricker_base_motion, zero_base_motion

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
    "build_pinn",
    "pde_residual",
    "boucwen_residual",
    "bc_residual",
    "ic_residual",
    "mse",
    "real_base_motion",
    "ricker_base_motion",
    "zero_base_motion",
]
