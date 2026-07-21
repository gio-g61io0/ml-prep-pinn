"""Configuration for the Makilala EIL PINN.

Loads the physical constants and per-unit training data from the data-prep specs
already produced in ``mlprep/pinn_wave/data/`` -- it does NOT recompute them.

Provenance (see the pipeline plan):
  * ``rho, H``   -- read directly from the Makilala dataset (measured/mapped).
  * ``k, eps_y`` -- estimated ONCE during data-prep via the (provisional) calibration
                    formulas in ``build_domain_params.py`` and persisted to the parquet.
  * ``alpha, A, beta, gamma, n, g, T`` -- globals from ``makilala_domain_params.json``.
  * ``pga``      -- per unit (``PGA 2019``), used to scale the (synthetic, for now) base motion.
  * ``u_scale``  -- characteristic displacement U; physical u = u_scale * net_u. The real
                    scale comes from the SPECFEM base motion later; here we derive a synthetic
                    U from the max PGA (see ``characteristic_displacement``).

NO SPECFEM data is used yet -- the base motion is synthetic (see ``seismic.py``).
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --- Spec locations ---------------------------------------------------------
PINN_WAVE_DATA = Path(__file__).resolve().parent.parent / "pinn_wave" / "data"
DOMAIN_PARAMS_JSON = PINN_WAVE_DATA / "makilala_domain_params.json"
NORM_SPEC_JSON = PINN_WAVE_DATA / "makilala_norm_spec.json"
BOUC_WEN_PARQUET = PINN_WAVE_DATA / "makilala_bouc_wen_params.parquet"
CONDITIONING_PARQUET = PINN_WAVE_DATA / "makilala_pinn_conditioning.parquet"

# Synthetic base-motion central frequency [Hz] (FKMODEL site fundamental ~2.1 Hz).
SEISMIC_F0 = 2.0

# Conditioning features the amortized network sees (besides z, t), in input order.
COND_FEATURES = ("rho", "k", "eps_y", "H", "pga")


@dataclass(frozen=True)
class GlobalConsts:
    """Constants shared by every soil column (SI units)."""

    alpha: float        # post-yield stiffness ratio [-]
    A: float            # Bouc-Wen shape [-]
    beta: float         # Bouc-Wen shape [-]
    gamma: float        # Bouc-Wen shape [-]
    n: float            # Bouc-Wen shape exponent [-]
    g: float            # gravity [m/s^2]
    T: float            # simulation duration [s]
    u_scale: float      # characteristic displacement U [m]; physical u = u_scale * net_u


@dataclass(frozen=True)
class CharacteristicScales:
    """Per-term residual scales used to non-dimensionalize the training losses.

    Each residual is divided by its scale so all loss terms are O(1) and comparable.
    Derived from the dataset-mean column and the displacement scale U.
    """

    U: float            # displacement scale [m]
    U_dot: float        # velocity scale U/T [m/s]
    S_pde: float        # momentum residual scale [Pa/m]
    S_bw: float         # Bouc-Wen residual scale [1/s]
    S_surf: float       # surface-stress residual scale [Pa]


@dataclass(frozen=True)
class ColumnParams:
    """Single-column bundle (kept for the manufactured-solution correctness test)."""

    rho: float
    k: float
    eps_y: float
    H: float
    alpha: float
    A: float
    beta: float
    gamma: float
    n: float
    g: float
    T: float
    u_scale: float
    unit_id: int | None = None


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing spec file: {path}")
    with open(path) as fh:
        return json.load(fh)


# --- Normalization ranges ---------------------------------------------------
def load_norm_ranges() -> dict[str, tuple[float, float]]:
    """(min, max) for each conditioning feature, for min-max input normalization.

    rho, k, eps_y, H come from domain_params per_unit_ranges; pga from norm_spec.
    """
    domain = _load_json(DOMAIN_PARAMS_JSON)
    norm_spec = _load_json(NORM_SPEC_JSON)
    pur = domain["per_unit_ranges"]
    pga = norm_spec["inputs"]["pga"]
    return {
        "rho": (float(pur["rho"]["min"]), float(pur["rho"]["max"])),
        "k": (float(pur["k_pa"]["min"]), float(pur["k_pa"]["max"])),
        "eps_y": (float(pur["eps_y"]["min"]), float(pur["eps_y"]["max"])),
        "H": (float(pur["H_m"]["min"]), float(pur["H_m"]["max"])),
        "pga": (float(pga["min"]), float(pga["max"])),
    }


def characteristic_displacement(f0: float = SEISMIC_F0) -> float:
    """Synthetic displacement scale U from the max PGA (harmonic disp<-acc conversion).

    U = PGA_max[g] * g / (2*pi*f0)^2. Replaced by the SPECFEM base-motion peak later.
    """
    domain = _load_json(DOMAIN_PARAMS_JSON)
    norm_spec = _load_json(NORM_SPEC_JSON)
    g = float(domain["global"]["g_m_s2"])
    pga_max = float(norm_spec["inputs"]["pga"]["max"])  # in g
    w0 = 2.0 * math.pi * f0
    return pga_max * g / (w0 ** 2)


def load_global_consts(u_scale: float | None = None) -> GlobalConsts:
    """Load global constants.

    u_scale precedence: explicit arg > norm_spec output.u.by (real base-motion peak) >
    synthetic characteristic U (fallback, with a warning).
    """
    domain = _load_json(DOMAIN_PARAMS_JSON)
    bwc = domain["bouc_wen_constants"]
    if u_scale is None:
        by = _load_json(NORM_SPEC_JSON).get("output", {}).get("u", {}).get("by")
        if by is not None:
            u_scale = float(by)
        else:
            u_scale = characteristic_displacement()
            warnings.warn(
                "norm_spec output.u.by is null (base motion not extracted); using synthetic "
                f"u_scale={u_scale:.4e} m from max PGA. Replace once makilala_base_motion.npz exists.",
                stacklevel=2,
            )
    return GlobalConsts(
        alpha=float(bwc["alpha"]), A=float(bwc["A"]), beta=float(bwc["beta"]),
        gamma=float(bwc["gamma"]), n=float(bwc["n"]),
        g=float(domain["global"]["g_m_s2"]), T=float(domain["global"]["T_s"]),
        u_scale=float(u_scale),
    )


def dataset_means() -> dict[str, float]:
    """Mean rho, k, eps_y, H over all units (for characteristic scales)."""
    bw = pd.read_parquet(BOUC_WEN_PARQUET)
    return {
        "rho": float(bw["rho"].mean()), "k": float(bw["k_pa"].mean()),
        "eps_y": float(bw["eps_y"].mean()), "H": float(bw["H_m"].mean()),
    }


def load_scales(consts: GlobalConsts) -> CharacteristicScales:
    """Compute per-term residual scales from dataset means + displacement scale U."""
    m = dataset_means()
    U = consts.u_scale
    return CharacteristicScales(
        U=U,
        U_dot=U / consts.T,
        # momentum: gravity/stiffness dominate at this scale (both ~ rho*g, k*U/H^2).
        S_pde=m["rho"] * consts.g,
        # Bouc-Wen rhs ~ (1/eps_y) * eps_dot, eps_dot ~ (U/H)/T.
        S_bw=(U / m["H"]) / consts.T / m["eps_y"],
        # surface stress ~ (1-alpha) k eps_y.
        S_surf=(1.0 - consts.alpha) * m["k"] * m["eps_y"],
    )


# --- Per-unit training table ------------------------------------------------
def load_unit_table() -> pd.DataFrame:
    """Combine conditioning + Bouc-Wen params into the real Makilala training rows.

    The two parquets are ROW-ALIGNED (same unit order, verified: identical unit_id
    sequence, matching rho/H_m), but ``unit_id`` is NOT unique (16309 unique of 18145
    rows) -- so they must be joined POSITIONALLY, never merged on unit_id (which would
    explode into a many-to-many join).

    Returns columns: unit_id, rho, k, eps_y, H, pga, lon, lat (SI units; pga in g).
    """
    bw = pd.read_parquet(BOUC_WEN_PARQUET)[["k_pa", "eps_y"]].reset_index(drop=True)
    cond = pd.read_parquet(CONDITIONING_PARQUET)[
        ["unit_id", "rho", "H_m", "pga", "lon", "lat"]].reset_index(drop=True)
    if len(bw) != len(cond):
        raise ValueError(f"Row count mismatch: bouc_wen={len(bw)} conditioning={len(cond)}")
    df = cond.rename(columns={"H_m": "H"})
    df["k"] = bw["k_pa"].to_numpy()
    df["eps_y"] = bw["eps_y"].to_numpy()
    return df[["unit_id", "rho", "k", "eps_y", "H", "pga", "lon", "lat"]]


# --- Single-column loader (manufactured-solution test) ----------------------
def load_column_params(unit_id: int | None = None, u_scale: float = 1.0) -> ColumnParams:
    """One representative column (dataset means by default). u_scale=1.0 for the
    manufactured-solution correctness test (so physical u == analytic u)."""
    domain = _load_json(DOMAIN_PARAMS_JSON)
    bw = pd.read_parquet(BOUC_WEN_PARQUET)
    if unit_id is not None:
        row = bw.loc[bw["unit_id"] == unit_id]
        if row.empty:
            raise ValueError(f"unit_id {unit_id} not found in {BOUC_WEN_PARQUET}")
        row = row.iloc[0]
        rho, k, eps_y, H = float(row["rho"]), float(row["k_pa"]), float(row["eps_y"]), float(row["H_m"])
    else:
        rho, k, eps_y, H = (
            float(bw["rho"].mean()), float(bw["k_pa"].mean()),
            float(bw["eps_y"].mean()), float(bw["H_m"].mean()),
        )
    bwc = domain["bouc_wen_constants"]
    return ColumnParams(
        rho=rho, k=k, eps_y=eps_y, H=H,
        alpha=float(bwc["alpha"]), A=float(bwc["A"]), beta=float(bwc["beta"]),
        gamma=float(bwc["gamma"]), n=float(bwc["n"]),
        g=float(domain["global"]["g_m_s2"]), T=float(domain["global"]["T_s"]),
        u_scale=u_scale, unit_id=unit_id,
    )
