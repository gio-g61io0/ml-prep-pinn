"""
Dataset ingestion, mandatory unit audit and construction of the physical state.

Two dataset-specific definitions are handled explicitly rather than assumed:

``BUK_mean``    is a **dry bulk density**, not a total moist unit weight. It is
                converted through an declared moisture/porosity model.
``SoilThc_mean`` is a **depth to bedrock**, not a failure-plane depth. It is
                translated through a declared rule (cap / scale / fixed) whose
                sensitivity must be reported, because the cohesive share of the
                factor of safety scales as 1/z.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

from .columns import resolve_column, resolve_many
from .config import DataConfig, PhysicsConfig


class PhysicalDataError(ValueError):
    """Raised when an input cannot be interpreted as a valid physical quantity."""


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------
@dataclass
class UnitDecision:
    variable: str
    source_column: str
    detected: str
    factor_to_target: float
    target_units: str
    median_raw: float
    median_converted: float
    note: str = ""


@dataclass
class AuditReport:
    decisions: List[UnitDecision] = field(default_factory=list)
    violations: Dict[str, int] = field(default_factory=dict)
    clipped: Dict[str, int] = field(default_factory=dict)
    n_rows: int = 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([d.__dict__ for d in self.decisions])

    def summary(self) -> str:
        lines = [f"Unit audit over {self.n_rows} slope units", "-" * 72]
        for d in self.decisions:
            lines.append(
                f"{d.variable:<12s} <- {d.source_column:<16s} detected={d.detected:<8s} "
                f"median {d.median_raw:.4g} -> {d.median_converted:.4g} {d.target_units}"
                + (f"  [{d.note}]" if d.note else "")
            )
        if self.clipped:
            lines.append("-" * 72)
            for k, v in self.clipped.items():
                lines.append(f"clipped to admissible range: {k}: {v} rows")
        if self.violations:
            lines.append("-" * 72)
            for k, v in self.violations.items():
                lines.append(f"FATAL range violation: {k}: {v} rows")
        return "\n".join(lines)


def _detect_density_units(med: float) -> Tuple[str, float]:
    """Return (label, factor to g/cm^3)."""
    if 0.5 <= med <= 3.0:
        return "g/cm3", 1.0
    if 50.0 <= med <= 300.0:
        return "cg/cm3", 0.01          # SoilGrids bdod convention
    if 500.0 <= med <= 3000.0:
        return "kg/m3", 0.001
    raise PhysicalDataError(
        f"Dry bulk density median {med:.4g} matches no known unit convention "
        "(g/cm3, cg/cm3, kg/m3). Set DataConfig.density_units explicitly."
    )


def _detect_depth_units(med: float) -> Tuple[str, float]:
    if med <= 50.0:
        return "m", 1.0
    return "cm", 0.01


def _detect_pga_units(med: float) -> Tuple[str, float]:
    if med <= 3.0:
        return "g", 1.0
    if med <= 30.0:
        return "m/s2", 1.0 / 9.80665
    return "gal", 1.0 / 980.665


# --------------------------------------------------------------------------
# prepared dataset
# --------------------------------------------------------------------------
@dataclass
class PreparedData:
    frame: pd.DataFrame                 # attribute table (no geometry)
    gdf: object                         # GeoDataFrame kept for mapping
    y: np.ndarray
    groups: np.ndarray
    coords: np.ndarray                  # centroid x/y for spatial blocking

    beta_rad: np.ndarray
    z_m: np.ndarray
    gamma_kNm3: np.ndarray
    m_ratio: np.ndarray
    pga_g: np.ndarray

    X_material: np.ndarray              # raw numeric material descriptors
    material_names: List[str]
    litho_codes: np.ndarray             # raw category labels
    texture_codes: np.ndarray

    audit: AuditReport
    columns: Dict[str, str]

    # optional laboratory anchors
    c_obs: Optional[np.ndarray] = None
    phi_obs_deg: Optional[np.ndarray] = None
    anchor_mask: Optional[np.ndarray] = None
    sigma_c: Optional[np.ndarray] = None
    sigma_phi: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.y)

    def state_matrix(self) -> np.ndarray:
        """[beta_rad, z, gamma, m, pga] as one float64 array."""
        return np.column_stack([self.beta_rad, self.z_m, self.gamma_kNm3,
                                self.m_ratio, self.pga_g]).astype(np.float64)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_gdf(path: str, layer: Optional[str] = None):
    import geopandas as gpd
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if len(gdf) == 0:
        raise PhysicalDataError(f"No features read from {path}")
    return gdf


def _centroids(gdf) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cen = gdf.geometry.centroid
    return np.column_stack([cen.x.to_numpy(), cen.y.to_numpy()])


def dry_density_to_unit_weight(
    rho_dry_gcm3: np.ndarray,
    m_ratio: np.ndarray,
    cfg: DataConfig,
    phys: PhysicsConfig,
) -> Tuple[np.ndarray, str]:
    """Convert dry bulk density to a total unit weight [kN/m^3].

    dry           gamma = rho_dry * g
    moist         gamma = rho_dry * g * (1 + w)
    sat_weighted  gamma = (1-m) gamma_moist + m gamma_sat, with
                  gamma_sat = gamma_dry + n gamma_w and n = 1 - rho_dry / G_s
    """
    g = phys.g
    gamma_dry = rho_dry_gcm3 * g
    if cfg.unit_weight_mode == "dry":
        return gamma_dry, "gamma = rho_dry * g (no moisture correction)"

    gamma_moist = gamma_dry * (1.0 + cfg.gravimetric_water_content)
    if cfg.unit_weight_mode == "moist":
        return gamma_moist, f"gamma = rho_dry * g * (1 + w), w={cfg.gravimetric_water_content}"

    porosity = np.clip(1.0 - rho_dry_gcm3 / cfg.specific_gravity, 0.05, 0.70)
    gamma_sat = gamma_dry + porosity * phys.gamma_w
    gamma = (1.0 - m_ratio) * gamma_moist + m_ratio * gamma_sat
    note = (f"depth-weighted moist/saturated profile, w={cfg.gravimetric_water_content}, "
            f"G_s={cfg.specific_gravity}")
    return gamma, note


def depth_to_failure_depth(d2b_m: np.ndarray, cfg: DataConfig) -> Tuple[np.ndarray, str]:
    """Translate depth-to-bedrock into a potential failure-plane depth."""
    if cfg.z_mode == "raw":
        z = d2b_m
        note = "z = depth to bedrock (raw); interpret with caution"
    elif cfg.z_mode == "cap":
        z = np.clip(d2b_m, cfg.z_min_m, cfg.z_cap_m)
        note = f"z = clip(depth-to-bedrock, {cfg.z_min_m}, {cfg.z_cap_m}) m"
    elif cfg.z_mode == "scale":
        z = np.clip(cfg.z_scale * d2b_m, cfg.z_min_m, cfg.z_cap_m)
        note = f"z = clip({cfg.z_scale} * depth-to-bedrock, {cfg.z_min_m}, {cfg.z_cap_m}) m"
    else:
        z = np.full_like(d2b_m, cfg.z_fixed_m)
        note = f"z fixed at {cfg.z_fixed_m} m"
    return np.maximum(z, cfg.z_min_m), note


def prepare_data(
    cfg: DataConfig,
    phys: Optional[PhysicsConfig] = None,
    gdf=None,
    verbose: bool = True,
) -> PreparedData:
    """Load, audit, convert and validate the dataset."""
    phys = phys or PhysicsConfig()
    if gdf is None:
        gdf = load_gdf(cfg.gpkg_path, cfg.layer)

    cols = list(gdf.columns)
    audit = AuditReport(n_rows=len(gdf))

    # ---- resolve columns exactly ----
    core = {
        "label": cfg.label_col, "group": cfg.group_col, "slope": cfg.slope_col,
        "depth": cfg.depth_col, "density": cfg.density_col, "pga": cfg.pga_col,
    }
    resolved = {k: resolve_column(v, cols) for k, v in core.items()}
    mat_map = resolve_many(list(cfg.material_numeric), cols)
    material_names = [mat_map[k] for k in cfg.material_numeric]

    litho_col = resolve_column(cfg.litho_name_col, cols, required=False) if cfg.litho_name_col else None
    tex_col = resolve_column(cfg.texture_name_col, cols, required=False) if cfg.texture_name_col else None
    if litho_col is None:
        litho_col = resolve_column("Lithology", cols, required=False)
    if tex_col is None:
        tex_col = resolve_column("soil_texture_idx", cols, required=False)
    resolved["lithology"] = litho_col or ""
    resolved["texture"] = tex_col or ""

    # ---- slope ----
    beta_raw = gdf[resolved["slope"]].to_numpy(dtype=np.float64)
    if cfg.slope_units == "deg":
        beta_deg = beta_raw
    elif cfg.slope_units == "rad":
        beta_deg = np.degrees(beta_raw)
    elif cfg.slope_units == "percent":
        beta_deg = np.degrees(np.arctan(beta_raw / 100.0))
    else:
        raise PhysicalDataError(f"Unknown slope_units {cfg.slope_units}")
    audit.decisions.append(UnitDecision(
        "beta", resolved["slope"], cfg.slope_units, 1.0, "deg",
        float(np.nanmedian(beta_raw)), float(np.nanmedian(beta_deg))))

    n_clip = int(((beta_deg < phys.beta_min_deg) | (beta_deg > phys.beta_max_deg)).sum())
    if n_clip:
        audit.clipped["beta_deg"] = n_clip
    beta_deg = np.clip(beta_deg, phys.beta_min_deg, phys.beta_max_deg)
    beta_rad = np.radians(beta_deg)

    # ---- groundwater ratio (needed before the unit-weight conversion) ----
    if cfg.use_uniform_m:
        m_ratio = np.full(len(gdf), cfg.m_uniform, dtype=np.float64)
        m_note = f"uniform m = {cfg.m_uniform} (assumption; switchable via m_col)"
        m_src = "<uniform>"
    else:
        m_src = resolve_column(cfg.m_col, cols)
        m_ratio = gdf[m_src].to_numpy(dtype=np.float64)
        m_note = f"m from column {m_src}"
    bad_m = int((~np.isfinite(m_ratio)).sum())
    if bad_m:
        m_ratio = np.where(np.isfinite(m_ratio), m_ratio, cfg.m_uniform)
        audit.clipped["m_ratio_nonfinite"] = bad_m
    n_clip = int(((m_ratio < cfg.m_min) | (m_ratio > cfg.m_max)).sum())
    if n_clip:
        audit.clipped["m_ratio"] = n_clip
    m_ratio = np.clip(m_ratio, cfg.m_min, cfg.m_max)
    audit.decisions.append(UnitDecision(
        "m_ratio", m_src, "ratio", 1.0, "-",
        float(np.nanmedian(m_ratio)), float(np.nanmedian(m_ratio)), m_note))

    # ---- dry bulk density -> total unit weight ----
    rho_raw = gdf[resolved["density"]].to_numpy(dtype=np.float64)
    med = float(np.nanmedian(rho_raw))
    if cfg.density_units == "auto":
        lab, fac = _detect_density_units(med)
    else:
        lab = cfg.density_units
        fac = {"g/cm3": 1.0, "cg/cm3": 0.01, "kg/m3": 0.001}[lab]
    rho_gcm3 = rho_raw * fac
    gamma, gnote = dry_density_to_unit_weight(rho_gcm3, m_ratio, cfg, phys)
    audit.decisions.append(UnitDecision(
        "gamma", resolved["density"], lab, fac, "kN/m3",
        med, float(np.nanmedian(gamma)),
        "DRY bulk density -> total unit weight: " + gnote))

    # ---- depth to bedrock -> failure depth ----
    d_raw = gdf[resolved["depth"]].to_numpy(dtype=np.float64)
    med_d = float(np.nanmedian(d_raw))
    if cfg.depth_units == "auto":
        lab_d, fac_d = _detect_depth_units(med_d)
    else:
        lab_d = cfg.depth_units
        fac_d = {"m": 1.0, "cm": 0.01}[lab_d]
    d2b_m = d_raw * fac_d
    z_m, znote = depth_to_failure_depth(d2b_m, cfg)
    audit.decisions.append(UnitDecision(
        "z", resolved["depth"], lab_d, fac_d, "m", med_d, float(np.nanmedian(z_m)),
        "DEPTH-TO-BEDROCK -> failure depth: " + znote))

    # ---- PGA ----
    p_raw = gdf[resolved["pga"]].to_numpy(dtype=np.float64)
    med_p = float(np.nanmedian(p_raw))
    if cfg.pga_units == "auto":
        lab_p, fac_p = _detect_pga_units(med_p)
    else:
        lab_p = cfg.pga_units
        fac_p = {"g": 1.0, "m/s2": 1.0 / 9.80665, "gal": 1.0 / 980.665}[lab_p]
    pga_g = p_raw * fac_p
    audit.decisions.append(UnitDecision(
        "PGA", resolved["pga"], lab_p, fac_p, "g", med_p, float(np.nanmedian(pga_g))))

    # ---- fatal validation ----
    # Ranges are checked on the *unclipped* derived values. Clipping first would
    # silently repair exactly the violations this step exists to catch.
    checks = {"beta_deg": beta_deg, "z_m": z_m, "gamma_kNm3": gamma,
              "m_ratio": m_ratio, "pga_g": pga_g}
    for name, arr in checks.items():
        lo, hi = cfg.admissible[name]
        bad = int((~np.isfinite(arr)).sum()
                  + ((arr < lo) | (arr > hi))[np.isfinite(arr)].sum())
        if bad:
            audit.violations[name] = bad
    if audit.violations and cfg.fatal_validation:
        raise PhysicalDataError(
            "Impossible or unit-incompatible physical inputs; model fitting is "
            "terminated by design. Resolve the definitions or set "
            "DataConfig.fatal_validation=False to clip instead.\n" + audit.summary()
        )

    # non-fatal mode: salvage the run by clipping, and say so
    for name, arr in [("gamma_kNm3", gamma), ("z_m", z_m), ("pga_g", pga_g)]:
        lo, hi = cfg.admissible[name]
        n_out = int(((arr < lo) | (arr > hi) | ~np.isfinite(arr)).sum())
        if n_out:
            audit.clipped[name] = n_out
    gamma = np.clip(np.nan_to_num(gamma, nan=float(np.nanmedian(gamma))),
                    *cfg.admissible["gamma_kNm3"])
    z_m = np.clip(np.nan_to_num(z_m, nan=float(np.nanmedian(z_m))), *cfg.admissible["z_m"])
    pga_g = np.clip(np.nan_to_num(pga_g, nan=float(np.nanmedian(pga_g))), *cfg.admissible["pga_g"])

    # ---- targets, groups, material descriptors ----
    y = gdf[resolved["label"]].to_numpy().astype(np.int32)
    groups = gdf[resolved["group"]].astype(str).fillna("<missing>").to_numpy()
    X_material = gdf[material_names].to_numpy(dtype=np.float64)
    litho_codes = (gdf[litho_col].astype(str).to_numpy() if litho_col
                   else np.full(len(gdf), "NA"))
    tex_codes = (gdf[tex_col].astype(str).to_numpy() if tex_col
                 else np.full(len(gdf), "NA"))

    frame = pd.DataFrame({
        "y": y, "group": groups,
        "beta_deg": beta_deg, "z_m": z_m, "gamma_kNm3": gamma,
        "m_ratio": m_ratio, "pga_g": pga_g,
        "lithology": litho_codes, "texture": tex_codes,
    })
    for nm, j in zip(material_names, range(X_material.shape[1])):
        frame[nm] = X_material[:, j]

    prepared = PreparedData(
        frame=frame, gdf=gdf, y=y, groups=groups, coords=_centroids(gdf),
        beta_rad=beta_rad, z_m=z_m, gamma_kNm3=gamma, m_ratio=m_ratio, pga_g=pga_g,
        X_material=X_material, material_names=material_names,
        litho_codes=litho_codes, texture_codes=tex_codes,
        audit=audit, columns=resolved,
    )
    if verbose:
        print(audit.summary())
        pos = float(y.mean())
        print("-" * 72)
        print(f"positives: {y.sum()} / {len(y)} ({100*pos:.2f}%)   "
              f"groups: {len(np.unique(groups))}")
    return prepared


def attach_anchors(
    data: PreparedData,
    c_obs: np.ndarray,
    phi_obs_deg: np.ndarray,
    sigma_c: float | np.ndarray = 5.0,
    sigma_phi: float | np.ndarray = 3.0,
) -> PreparedData:
    """Attach measured cohesion / friction with per-observation uncertainty.

    Rows with NaN are masked out; only masked rows contribute to L_anchor.
    """
    c_obs = np.asarray(c_obs, dtype=np.float64)
    phi_obs = np.asarray(phi_obs_deg, dtype=np.float64)
    mask = np.isfinite(c_obs) & np.isfinite(phi_obs)
    data.c_obs = np.nan_to_num(c_obs)
    data.phi_obs_deg = np.nan_to_num(phi_obs)
    data.anchor_mask = mask.astype(np.float64)
    data.sigma_c = np.broadcast_to(np.asarray(sigma_c, dtype=np.float64), c_obs.shape).copy()
    data.sigma_phi = np.broadcast_to(np.asarray(sigma_phi, dtype=np.float64), phi_obs.shape).copy()
    return data
