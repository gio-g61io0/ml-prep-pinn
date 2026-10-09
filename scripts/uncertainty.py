"""
Joint uncertainty propagation.

Percentile widths are *not* combined through a root-sum-square formula. Every
draw resamples the network member, the geotechnical latent parameters, the
hydrological ratio, the seismic loading and the Jibson residual, and then
recomputes the entire chain

    (R0, psi) -> (c, phi) -> Fs -> a_y -> D_n -> p_occ

with the exact operators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from . import physics as P
from .config import PhysicsConfig, UncertaintyConfig
from .model import PEGPN, TrainArrays, predict


@dataclass
class MCResult:
    quantiles: Dict[str, Dict[str, np.ndarray]]
    summary: pd.DataFrame
    n_draws: int
    idx: np.ndarray

    def band(self, key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = self.quantiles[key]
        return q["lo"], q["median"], q["hi"]

    def width(self, key: str) -> np.ndarray:
        q = self.quantiles[key]
        return q["hi"] - q["lo"]


def _lognormal_multiplier(rng, cv: float, size) -> np.ndarray:
    """Multiplicative noise with unit median and coefficient of variation ~cv."""
    if cv <= 0:
        return np.ones(size)
    sigma = np.sqrt(np.log1p(cv ** 2))
    return np.exp(rng.normal(0.0, sigma, size=size))


def propagate(models: Sequence[PEGPN], arrays: TrainArrays, phys: PhysicsConfig,
              ucfg: UncertaintyConfig, *, subset: Optional[np.ndarray] = None,
              calibrator=None, quantities: Sequence[str] = ("Fs", "ID", "p", "D_hard_cm"),
              verbose: bool = True) -> MCResult:
    rng = np.random.default_rng(ucfg.random_state)
    n_all = len(arrays.y)
    idx = np.arange(n_all) if subset is None else np.asarray(subset)
    A = arrays.subset(idx)
    N = len(idx)

    # latent parameters per ensemble member (computed once)
    member_pred = [predict(m, A, phys) for m in models]
    R0_m = np.stack([mp["R0"] for mp in member_pred]).astype(np.float64)
    psi_m = np.stack([mp["psi"] for mp in member_pred]).astype(np.float64)
    b_m = np.array([float(tf.convert_to_tensor(m.b).numpy()) for m in models])
    s_m = np.array([float(tf.math.softplus(m.s_raw).numpy()) for m in models])
    M = len(models)

    beta = A.state[:, 0].astype(np.float64)
    z0 = A.state[:, 1].astype(np.float64)
    g0 = A.state[:, 2].astype(np.float64)
    m0 = A.state[:, 3].astype(np.float64)
    pga0 = A.state[:, 4].astype(np.float64)

    store = {q: np.empty((ucfg.n_draws, N), dtype=np.float32) for q in quantities}

    done = 0
    while done < ucfg.n_draws:
        k = min(ucfg.chunk, ucfg.n_draws - done)
        mem = rng.integers(0, M, size=k) if ucfg.include_parameter_uncertainty else np.zeros(k, int)
        R0 = R0_m[mem]                                   # (k, N)
        psi = psi_m[mem]
        b = b_m[mem][:, None]
        s = s_m[mem][:, None]

        z = z0[None, :] * _lognormal_multiplier(rng, ucfg.cv_z, (k, N))
        gam = g0[None, :] * _lognormal_multiplier(rng, ucfg.cv_gamma, (k, N))
        pga = pga0[None, :] * _lognormal_multiplier(rng, ucfg.cv_pga, (k, N))
        mm = np.clip(m0[None, :] + rng.normal(0.0, ucfg.sd_m, (k, N)), 0.0, 1.0)
        gam = np.clip(gam, 11.0, 24.0)
        z = np.clip(z, 0.1, 60.0)

        c, tan_phi, _ = P.resistance_to_c_phi(R0, psi, phys.sigma_ref)
        ch = P.forward_chain_numpy(c, tan_phi, np.broadcast_to(beta, (k, N)),
                                   z, gam, mm, pga, phys)

        D = ch["D_hard_cm"]
        if ucfg.include_jibson_residual:
            resid = np.power(10.0, rng.normal(0.0, phys.jibson_sigma_log10, (k, N)))
            D = np.minimum(D * resid, phys.D_cap_cm)
        ID = P.instability_index(D, ch["regime"], phys.D0_cm)

        u = np.log1p(np.where(np.isfinite(D), D, phys.D_cap_cm))
        p = 1.0 / (1.0 + np.exp(-(b + s * u)))
        if calibrator is not None and getattr(calibrator, "fitted", False):
            p = calibrator.transform(p.ravel()).reshape(p.shape)

        pool = {"Fs": ch["Fs"], "ID": ID, "p": p, "D_hard_cm": D,
                "ay_g": ch["ay_g"], "c": c, "phi_deg": np.degrees(np.arctan(tan_phi))}
        for q in quantities:
            store[q][done:done + k] = pool[q].astype(np.float32)
        done += k
        if verbose:
            print(f"  Monte Carlo {done}/{ucfg.n_draws} draws", end="\r")
    if verbose:
        print()

    lo_q, hi_q = ucfg.quantiles
    quant: Dict[str, Dict[str, np.ndarray]] = {}
    rows = []
    for q, arr in store.items():
        with np.errstate(invalid="ignore"):
            lo = np.nanpercentile(arr, 100 * lo_q, axis=0)
            md = np.nanpercentile(arr, 50.0, axis=0)
            hi = np.nanpercentile(arr, 100 * hi_q, axis=0)
        quant[q] = {"lo": lo, "median": md, "hi": hi}
        w = hi - lo
        rows.append({"quantity": q, "median_width": float(np.nanmedian(w)),
                     "mean_width": float(np.nanmean(w)),
                     "median_value": float(np.nanmedian(md))})
    return MCResult(quantiles=quant, summary=pd.DataFrame(rows),
                    n_draws=ucfg.n_draws, idx=idx)


def uncertainty_by_regime(mc: MCResult, regime: np.ndarray, key: str = "p") -> pd.DataFrame:
    reg = np.asarray(regime)[mc.idx]
    w = mc.width(key)
    rows = []
    for k in (P.STATIC, P.YIELDING, P.NON_YIELDING):
        sel = reg == k
        rows.append({"regime": P.REGIME_NAMES[k], "n": int(sel.sum()),
                     "median_interval_width": float(np.nanmedian(w[sel])) if sel.any() else np.nan})
    return pd.DataFrame(rows)


def bin_level_coverage(mc: MCResult, y: np.ndarray, key: str = "p",
                       n_bins: int = 10) -> pd.DataFrame:
    """Does the observed landslide rate fall inside the propagated band?

    A per-row interval cannot be checked against a binary outcome, so coverage is
    assessed at the level of predicted-probability bins.
    """
    lo, md, hi = mc.band(key)
    yy = np.asarray(y)[mc.idx]
    edges = np.quantile(md, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    b = np.clip(np.digitize(md, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        sel = b == i
        if not sel.any():
            continue
        obs = float(yy[sel].mean())
        l, h = float(np.mean(lo[sel])), float(np.mean(hi[sel]))
        rows.append({"bin": i, "n": int(sel.sum()), "median_pred": float(np.mean(md[sel])),
                     "lo": l, "hi": h, "observed_rate": obs,
                     "covered": bool(l <= obs <= h)})
    df = pd.DataFrame(rows)
    if len(df):
        df.attrs["coverage"] = float(df["covered"].mean())
    return df
