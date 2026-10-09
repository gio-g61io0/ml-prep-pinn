"""
Infinite-slope, critical-acceleration and Newmark operators.

The same source code is executed by NumPy (exact scoring, Monte-Carlo
propagation) and by TensorFlow (differentiable training). A tiny operator shim
dispatches on the argument type, which removes the classic failure mode of
maintaining two drifting copies of the physics.

Sign and unit conventions
-------------------------
c        cohesion                              [kPa]
phi      internal friction angle               [rad] (tan_phi dimensionless)
beta     slope angle                           [rad]
z        potential failure depth (vertical)    [m]
gamma    total unit weight of soil             [kN/m^3]
gamma_w  unit weight of water                  [kN/m^3]
m        groundwater ratio h_w / z             [-]
PGA      peak ground acceleration              [g]
a_y      critical (yield) acceleration         [g]
D_n      Newmark permanent displacement        [cm]
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

try:  # TensorFlow is optional for the pure-NumPy code paths
    import tensorflow as tf
    _HAS_TF = True
except Exception:  # pragma: no cover
    tf = None
    _HAS_TF = False

LN10 = float(np.log(10.0))

STATIC, YIELDING, NON_YIELDING = 0, 1, 2
REGIME_NAMES = {STATIC: "static instability", YIELDING: "seismically yielding",
                NON_YIELDING: "non-yielding"}


# --------------------------------------------------------------------------
# backend shim
# --------------------------------------------------------------------------
_NP_OPS: Dict[str, object] = dict(
    sin=np.sin, cos=np.cos, tan=np.tan, atan=np.arctan,
    exp=np.exp, log=np.log, log10=np.log10, sqrt=np.sqrt, tanh=np.tanh,
    abs=np.abs, maximum=np.maximum, minimum=np.minimum, where=np.where,
    softplus=lambda x: np.logaddexp(0.0, x),
    sigmoid=lambda x: 0.5 * (1.0 + np.tanh(0.5 * x)),
    clip=lambda x, lo, hi: np.clip(x, lo, hi),
    zeros_like=np.zeros_like,
)

if _HAS_TF:
    _TF_OPS: Dict[str, object] = dict(
        sin=tf.sin, cos=tf.cos, tan=tf.tan, atan=tf.atan,
        exp=tf.exp, log=tf.math.log, log10=lambda x: tf.math.log(x) / tf.cast(LN10, x.dtype),
        sqrt=tf.sqrt, tanh=tf.tanh, abs=tf.abs,
        maximum=lambda a, b: tf.maximum(a, tf.cast(b, a.dtype)),
        minimum=lambda a, b: tf.minimum(a, tf.cast(b, a.dtype)),
        where=tf.where,
        softplus=tf.math.softplus,
        sigmoid=tf.sigmoid,
        clip=lambda x, lo, hi: tf.clip_by_value(x, tf.cast(lo, x.dtype), tf.cast(hi, x.dtype)),
        zeros_like=tf.zeros_like,
    )
else:  # pragma: no cover
    _TF_OPS = {}


def ops(x):
    """Return the operator table matching the backend of ``x``."""
    if _HAS_TF and tf.is_tensor(x):
        return _TF_OPS
    return _NP_OPS


# --------------------------------------------------------------------------
# reference-stress parameterisation
# --------------------------------------------------------------------------
def resistance_to_c_phi(R0, psi, sigma_ref: float):
    """Map (R0, psi) at a fixed reference effective normal stress to (c, tan phi).

    R0 = c + sigma_ref * tan(phi);  c = psi * R0;  tan(phi) = (1-psi) R0 / sigma_ref.
    Neither slope geometry nor PGA appears here, which is the point of the
    reparameterisation.
    """
    o = ops(R0)
    c = psi * R0
    tan_phi = (1.0 - psi) * R0 / sigma_ref
    phi = o["atan"](tan_phi)
    return c, tan_phi, phi


# --------------------------------------------------------------------------
# infinite-slope operator
# --------------------------------------------------------------------------
def infinite_slope_terms(beta_rad, z, gamma, m, gamma_w: float):
    """Return the geometry coefficients A and B of Fs = A c + B tan(phi)."""
    o = ops(beta_rad)
    sin_b = o["sin"](beta_rad)
    cos_b = o["cos"](beta_rad)
    tan_b = sin_b / o["maximum"](cos_b, 1e-9)
    denom = o["maximum"](gamma * z * sin_b * cos_b, 1e-9)
    A = 1.0 / denom
    B = (1.0 - m * gamma_w / gamma) / o["maximum"](tan_b, 1e-9)
    return A, B


def factor_of_safety(c, tan_phi, beta_rad, z, gamma, m, gamma_w: float):
    """Independently computed factor of safety (Eq. 15-18 of the report)."""
    A, B = infinite_slope_terms(beta_rad, z, gamma, m, gamma_w)
    Fs = A * c + B * tan_phi
    return Fs, A, B


# --------------------------------------------------------------------------
# critical acceleration
# --------------------------------------------------------------------------
def critical_acceleration_g(Fs, beta_rad):
    """a_y / g = max{(Fs - 1) sin(beta), 0}."""
    o = ops(Fs)
    return o["maximum"]((Fs - 1.0) * o["sin"](beta_rad), 0.0)


def acceleration_ratio(ay_g, pga_g, eps: float = 1e-6):
    o = ops(ay_g)
    return ay_g / o["maximum"](pga_g, eps)


# --------------------------------------------------------------------------
# Newmark operators
# --------------------------------------------------------------------------
def soft_ratio(r, tau: float, eps: float):
    """Differentiable surrogate of min(r, 1) that never reaches the kink.

    Only a lower clamp is applied. Clamping the upper side at ``1 - eps`` would
    flatten the surrogate for every r beyond roughly ``1 + 4 tau`` and kill the
    gradient for all strongly non-yielding slope units; ``newmark_soft`` instead
    handles the ``log10(1 - r)`` term analytically.
    """
    o = ops(r)
    rs = 1.0 - tau * o["softplus"]((1.0 - r) / tau)
    return o["maximum"](rs, eps)


def _log_softplus(x, o):
    """log(softplus(x)), stable as x -> -inf where softplus(x) ~ exp(x)."""
    return o["where"](x < -20.0, x, o["log"](o["softplus"](o["maximum"](x, -20.0))))


def jibson_log10_displacement(r, k: Tuple[float, float, float]):
    """log10 D_n = k0 + k1 log10(1-r) + k2 log10(r), valid for 0 < r < 1."""
    o = ops(r)
    k0, k1, k2 = k
    return k0 + k1 * o["log10"](1.0 - r) + k2 * o["log10"](r)


def newmark_soft(r, tau: float, eps: float, k, log1p_cap: float):
    """Smooth training branch.

    Returns ``(u, D_soft, r_soft)`` where ``u = log(1 + D)`` is computed as
    ``softplus(log10 D * ln 10)``. Building u directly from the log-displacement
    avoids overflow and keeps a live gradient on both sides of the yield
    boundary; the smooth ``tanh`` cap replaces a hard clip that would zero the
    gradient for statically unstable units.
    """
    o = ops(r)
    k0, k1, k2 = k

    w = (1.0 - r) / tau
    r_s = o["maximum"](1.0 - tau * o["softplus"](w), eps)

    # 1 - r_soft = tau * softplus(w) exactly, so the (1 - r) term is evaluated
    # in log space instead of through a clip that would zero the gradient.
    log10_one_minus = float(np.log10(tau)) + _log_softplus(w, o) / LN10
    log10_r = o["log10"](r_s)

    log10_D = k0 + k1 * log10_one_minus + k2 * log10_r
    u = o["softplus"](log10_D * LN10)               # = log1p(D), overflow-safe
    u = log1p_cap * o["tanh"](u / log1p_cap)        # smooth saturation
    D_soft = o["exp"](o["minimum"](u, 30.0)) - 1.0
    return u, D_soft, r_s


def newmark_hard(Fs, r, k, eps: float, D_cap: float):
    """Exact piecewise operator used for scoring (NumPy only).

    Static instability is *not* converted into a very large displacement; it is
    returned as NaN and reported as its own regime.
    """
    Fs = np.asarray(Fs, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    regime = np.full(Fs.shape, YIELDING, dtype=np.int8)
    regime[Fs <= 1.0] = STATIC
    regime[(Fs > 1.0) & (r >= 1.0)] = NON_YIELDING

    r_safe = np.clip(r, eps, 1.0 - eps)
    D = np.power(10.0, jibson_log10_displacement(r_safe, k))
    capped = (regime == YIELDING) & (D > D_cap)
    D = np.minimum(D, D_cap)

    D = np.where(regime == YIELDING, D, np.where(regime == NON_YIELDING, 0.0, np.nan))
    return D, regime, capped


def instability_index(D_cm, regime, D0: float):
    """I_D = 1 - exp(-D/D0); statically unstable units are assigned 1.0.

    This is a *displacement-based instability index*, not a probability.
    """
    D = np.asarray(D_cm, dtype=np.float64)
    idx = 1.0 - np.exp(-np.where(np.isfinite(D), D, 0.0) / D0)
    idx = np.where(regime == STATIC, 1.0, idx)
    return idx


def regime_fractions(regime) -> Dict[str, float]:
    regime = np.asarray(regime)
    n = max(regime.size, 1)
    return {REGIME_NAMES[key]: float((regime == key).sum()) / n
            for key in (STATIC, YIELDING, NON_YIELDING)}


# --------------------------------------------------------------------------
# convenience: full NumPy forward chain from (c, phi) onwards
# --------------------------------------------------------------------------
def forward_chain_numpy(c, tan_phi, beta_rad, z, gamma, m, pga_g, phys) -> Dict[str, np.ndarray]:
    """Run c, phi -> Fs -> a_y -> D_n -> I_D with the exact operators."""
    Fs, A, B = factor_of_safety(c, tan_phi, beta_rad, z, gamma, m, phys.gamma_w)
    ay = critical_acceleration_g(Fs, beta_rad)
    r = acceleration_ratio(ay, pga_g)
    D, regime, capped = newmark_hard(Fs, r, phys.jibson_k, phys.eps_r, phys.D_cap_cm)
    return {
        "A": A, "B": B, "Fs": Fs, "ay_g": ay, "ratio": r,
        "D_hard_cm": D, "regime": regime, "capped": capped,
        "ID": instability_index(D, regime, phys.D0_cm),
    }
