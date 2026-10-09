"""
Mandatory diagnostics (Sec. 14 of the methodology report).

These are the evidence items that separate a physics-embedded model from a
black box wearing physical notation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from . import physics as P
from .config import PhysicsConfig
from .model import DTYPE, PEGPN, TrainArrays, predict


# --------------------------------------------------------------------------
# smooth vs exact operator
# --------------------------------------------------------------------------
def soft_hard_gap(model: PEGPN, arrays: TrainArrays, phys: PhysicsConfig,
                  D_tol_cm: float = 0.01) -> Dict[str, float]:
    """Quantify the approximation introduced by training on the smooth operator.

    Two points of care. First, the exact operator assigns *exactly* zero to every
    non-yielding unit, so a rank correlation over all rows is dominated by a huge
    tie block; the informative statistic is the correlation restricted to
    yielding units, and both are reported. Second, the smooth surrogate is
    strictly below 1 by construction, so "is r_soft < 1" is trivially true
    everywhere; yield-state agreement is therefore judged against a physically
    negligible displacement tolerance ``D_tol_cm``.
    """
    from scipy.stats import spearmanr

    pred = predict(model, arrays, phys)
    r = pred["ratio"]

    _, D_soft, _ = P.newmark_soft(r, phys.tau, phys.eps_r, phys.jibson_k, phys.log1p_cap)
    D_soft = np.minimum(D_soft, phys.D_cap_cm)
    ID_soft = 1.0 - np.exp(-D_soft / phys.D0_cm)
    ID_hard = pred["ID"]

    finite = np.isfinite(ID_soft) & np.isfinite(ID_hard)
    yield_hard = pred["regime"] == P.YIELDING
    yield_soft = D_soft > D_tol_cm

    def _rho(sel):
        return (float(spearmanr(ID_soft[sel], ID_hard[sel]).statistic)
                if sel.sum() > 2 else np.nan)

    return {
        "tau": phys.tau,
        "D_tol_cm": D_tol_cm,
        "spearman_ID_yielding_only": _rho(finite & yield_hard),
        "spearman_ID_all_rows": _rho(finite),
        "mean_abs_dID": float(np.mean(np.abs(ID_soft[finite] - ID_hard[finite]))),
        "max_abs_dID": float(np.max(np.abs(ID_soft[finite] - ID_hard[finite]))),
        "yield_state_disagreement": float(np.mean(yield_soft != yield_hard)),
        "soft_yields_hard_does_not": float(np.mean(yield_soft & ~yield_hard)),
        "hard_yields_soft_does_not": float(np.mean(~yield_soft & yield_hard)),
        "frac_static": float(np.mean(pred["regime"] == P.STATIC)),
    }


def gradient_health(phys: PhysicsConfig, taus: Sequence[float] = (0.2, 0.1, 0.05, 0.02, 0.01),
                    r_grid: Optional[np.ndarray] = None) -> pd.DataFrame:
    """d log(1+D)/dr across the yield boundary for each smoothing temperature."""
    r_grid = r_grid if r_grid is not None else np.linspace(0.2, 1.8, 161)
    rows = []
    for tau in taus:
        rt = tf.constant(r_grid, DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(rt)
            u, _, _ = P.newmark_soft(rt, tau, phys.eps_r, phys.jibson_k, phys.log1p_cap)
        g = tape.gradient(u, rt).numpy()
        near = np.abs(r_grid - 1.0) < 0.1
        beyond = r_grid > 1.05
        rows.append({
            "tau": tau,
            "frac_zero_gradient": float(np.mean(np.abs(g) < 1e-12)),
            "mean_abs_grad_near_boundary": float(np.mean(np.abs(g[near]))),
            "mean_abs_grad_beyond_boundary": float(np.mean(np.abs(g[beyond]))),
            "max_abs_grad": float(np.max(np.abs(g))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# identifiability
# --------------------------------------------------------------------------
def identifiability_scan(pred: Dict[str, np.ndarray], state: np.ndarray,
                         phys: PhysicsConfig,
                         psi_grid: Sequence[float] = (0.05, 0.15, 0.3, 0.5, 0.7, 0.9),
                         sample: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Hold R0 fixed, sweep psi, and record what the data can and cannot see.

    A large swing in c and phi accompanied by a small swing in Fs is the direct
    numerical signature of the non-identified resistance direction.
    """
    rng = np.random.default_rng(seed)
    n = len(pred["R0"])
    idx = rng.choice(n, size=min(sample, n), replace=False)
    R0 = pred["R0"][idx]
    beta, z, gamma, m, pga = (state[idx, 0], state[idx, 1], state[idx, 2],
                              state[idx, 3], state[idx, 4])

    base = None
    rows = []
    for psi in psi_grid:
        c, tan_phi, phi = P.resistance_to_c_phi(R0, np.full_like(R0, psi), phys.sigma_ref)
        ch = P.forward_chain_numpy(c, tan_phi, beta, z, gamma, m, pga, phys)
        rec = {"psi": psi, "mean_c_kPa": float(np.mean(c)),
               "mean_phi_deg": float(np.mean(np.degrees(phi))),
               "mean_Fs": float(np.nanmean(ch["Fs"])),
               "mean_ID": float(np.nanmean(ch["ID"])),
               "frac_static": float(np.mean(ch["regime"] == P.STATIC))}
        if base is None:
            base = rec
        rec["rel_dFs_vs_psi0"] = abs(rec["mean_Fs"] - base["mean_Fs"]) / max(base["mean_Fs"], 1e-9)
        rec["rel_dc_vs_psi0"] = abs(rec["mean_c_kPa"] - base["mean_c_kPa"]) / max(base["mean_c_kPa"], 1e-9)
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# regimes, caps, categories
# --------------------------------------------------------------------------
def regime_occupancy(regime: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    n = len(regime)
    for key in (P.STATIC, P.YIELDING, P.NON_YIELDING):
        sel = regime == key
        rows.append({"regime": P.REGIME_NAMES[key], "n": int(sel.sum()),
                     "fraction": float(sel.sum()) / max(n, 1),
                     "observed_landslide_rate": float(y[sel].mean()) if sel.any() else np.nan})
    return pd.DataFrame(rows)


def cap_diagnostics(D_hard: np.ndarray, phys: PhysicsConfig) -> Dict[str, float]:
    finite = np.isfinite(D_hard)
    at_cap = finite & (D_hard >= phys.D_cap_cm - 1e-6)
    return {"D_cap_cm": phys.D_cap_cm,
            "frac_at_cap": float(at_cap.sum()) / max(finite.sum(), 1),
            "n_at_cap": int(at_cap.sum()),
            "median_D_cm": float(np.nanmedian(D_hard)),
            "p95_D_cm": float(np.nanpercentile(D_hard[finite], 95)) if finite.any() else np.nan}


def category_diagnostics(cv_result, data, key: str = "texture") -> pd.DataFrame:
    """Out-of-fold performance by lithology or texture class."""
    from .metrics import classification_metrics
    labels = data.texture_codes if key == "texture" else data.litho_codes
    df = cv_result.oof
    rows = []
    for lab in np.unique(labels):
        sel = labels == lab
        if sel.sum() < 30:
            continue
        m = classification_metrics(df["y"].to_numpy()[sel], df["p_cal"].to_numpy()[sel])
        m.update({key: lab, "n": int(sel.sum())})
        rows.append(m)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# --------------------------------------------------------------------------
# physical sensitivity sweeps
# --------------------------------------------------------------------------
def groundwater_sensitivity(pred: Dict[str, np.ndarray], state: np.ndarray,
                            phys: PhysicsConfig,
                            m_grid: Sequence[float] = (0.0, 0.09, 0.18, 0.36, 0.6, 1.0)) -> pd.DataFrame:
    """Recompute the entire chain under alternative uniform groundwater ratios."""
    c = pred["c"]; tan_phi = np.tan(np.radians(pred["phi_deg"]))
    beta, z, gamma, pga = state[:, 0], state[:, 1], state[:, 2], state[:, 4]
    rows = []
    for m in m_grid:
        ch = P.forward_chain_numpy(c, tan_phi, beta, z, gamma,
                                   np.full_like(beta, m, dtype=float), pga, phys)
        rec = {"m": m, "mean_Fs": float(np.nanmean(ch["Fs"])),
               "median_Fs": float(np.nanmedian(ch["Fs"])),
               "mean_ID": float(np.nanmean(ch["ID"]))}
        rec.update(P.regime_fractions(ch["regime"]))
        rows.append(rec)
    return pd.DataFrame(rows)


def depth_sensitivity(pred: Dict[str, np.ndarray], state: np.ndarray, phys: PhysicsConfig,
                      z_values: Sequence[float] = (1.0, 2.0, 3.0, 5.0, 10.0),
                      y: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Sensitivity to the depth-to-bedrock -> failure-depth translation.

    Cohesion enters Fs as c / (gamma z sin beta cos beta), so its share collapses
    as z grows. This table is the quantitative statement of that concern.
    """
    from sklearn.metrics import roc_auc_score
    c = pred["c"]; tan_phi = np.tan(np.radians(pred["phi_deg"]))
    beta, gamma, m, pga = state[:, 0], state[:, 2], state[:, 3], state[:, 4]
    rows = []
    for z in z_values:
        zz = np.full_like(beta, z, dtype=float)
        ch = P.forward_chain_numpy(c, tan_phi, beta, zz, gamma, m, pga, phys)
        A = ch["A"]; B = ch["B"]
        share = (A * c) / np.maximum(A * c + B * tan_phi, 1e-9)
        rec = {"z_m": z, "mean_Fs": float(np.nanmean(ch["Fs"])),
               "median_cohesion_share": float(np.nanmedian(share)),
               "mean_ID": float(np.nanmean(ch["ID"]))}
        rec.update(P.regime_fractions(ch["regime"]))
        if y is not None:
            try:
                rec["ID_roc_auc"] = float(roc_auc_score(y, ch["ID"]))
            except ValueError:
                rec["ID_roc_auc"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def monotonicity_tests(model: PEGPN, arrays: TrainArrays, phys: PhysicsConfig,
                       sample: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Post-hoc verification of the three required monotone responses."""
    rng = np.random.default_rng(seed)
    n = len(arrays.y)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    sub = arrays.subset(idx)
    base = predict(model, sub, phys)

    rows = []

    hot = sub.subset(np.arange(len(sub.y)))
    hot.state = sub.state.copy(); hot.state[:, 4] *= 1.25
    ph = predict(model, hot, phys)
    ok = ph["ID"] >= base["ID"] - 1e-9
    rows.append({"relation": "PGA up => D_n up", "frac_satisfied": float(np.mean(ok))})

    wet = sub.subset(np.arange(len(sub.y)))
    wet.state = sub.state.copy(); wet.state[:, 3] = np.minimum(sub.state[:, 3] + 0.2, 1.0)
    pw = predict(model, wet, phys)
    ok = pw["Fs"] <= base["Fs"] + 1e-9
    rows.append({"relation": "m up => Fs down", "frac_satisfied": float(np.mean(ok))})

    c2, t2, _ = P.resistance_to_c_phi(base["R0"] * 1.2, base["psi"], phys.sigma_ref)
    Fs2, _, _ = P.factor_of_safety(c2, t2, sub.state[:, 0], sub.state[:, 1],
                                   sub.state[:, 2], sub.state[:, 3], phys.gamma_w)
    ok = Fs2 >= base["Fs"] - 1e-9
    rows.append({"relation": "R0 up => Fs up", "frac_satisfied": float(np.mean(ok))})
    return pd.DataFrame(rows)


def latent_parameter_summary(pred: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for key, unit in [("R0", "kPa"), ("psi", "-"), ("c", "kPa"), ("phi_deg", "deg"),
                      ("Fs", "-"), ("ay_g", "g"), ("D_hard_cm", "cm"), ("ID", "-")]:
        if key not in pred:
            continue
        v = np.asarray(pred[key], dtype=float)
        f = v[np.isfinite(v)]
        if f.size == 0:
            continue
        rows.append({"quantity": key, "units": unit, "n_finite": int(f.size),
                     "min": float(f.min()), "p05": float(np.percentile(f, 5)),
                     "median": float(np.median(f)), "mean": float(f.mean()),
                     "p95": float(np.percentile(f, 95)), "max": float(f.max())})
    return pd.DataFrame(rows)
