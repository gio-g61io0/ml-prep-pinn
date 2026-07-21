"""Physics residuals for the amortized Makilala EIL PINN (TensorFlow ``tf.GradientTape``).

Each residual encodes a governing equation rearranged to equal zero. A perfect solution
drives every residual to 0; training minimizes their (scaled) mean squares. No
displacement labels -- the physics IS the supervision.

Governing equations (physical units), with per-unit coefficients rho, k, eps_y, H, pga
carried in ``batch`` and globals (alpha, A, beta, gamma, n, g, T, u_scale) in ``consts``:

  strain    eps   = du/dz
  stress    sigma = alpha*k*eps + (1-alpha)*k*eps_y*Phi
  momentum  r_pde = rho*u_tt - sigma_z - rho*g,
                    sigma_z = alpha*k*u_zz + (1-alpha)*k*eps_y*Phi_z   (k, eps_y const in z)
  Bouc-Wen  r_bw  = Phi_t - (1/eps_y)*[A*eps_dot - beta*|eps_dot|*|Phi|^(n-1)*Phi
                                       - gamma*eps_dot*|Phi|^n],  eps_dot = d(u_z)/dt
  BC base   (z=0): u(0,t) = u_input(t, pga)     [synthetic Ricker x PGA -> seismic.py]
  BC surf   (z=H): sigma(H,t) = 0
  IC        (t=0): u=0, u_t=0, Phi=0

The tape watches PHYSICAL z, t; the model normalizes internally, so autodiff returns
physical derivatives directly. Conditioning coeffs (rho, k, eps_y, H, pga) are treated
as constants at each point (not differentiated).
"""
from __future__ import annotations

from typing import Callable

import tensorflow as tf

from .config import GlobalConsts
from .seismic import real_base_motion

_EPS = 1e-8  # guard for |Phi|^(n-1) near Phi=0


def _col(x) -> tf.Tensor:
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if x.shape.rank == 1:
        x = tf.reshape(x, (-1, 1))
    return x


def _coeffs(batch: dict) -> dict[str, tf.Tensor]:
    """Coerce the per-point conditioning coefficients to (N, 1) float32 tensors."""
    return {c: _col(batch[c]) for c in ("rho", "k", "eps_y", "H", "pga")}


def _model_input(z: tf.Tensor, t: tf.Tensor, c: dict) -> tf.Tensor:
    """Assemble the physical input vector [z, t, rho, k, eps_y, H, pga]."""
    return tf.concat([z, t, c["rho"], c["k"], c["eps_y"], c["H"], c["pga"]], axis=1)


def _forward(model, consts: GlobalConsts, z, t, c):
    """Evaluate the model; return (u_physical, Phi)."""
    out = model(_model_input(z, t, c), training=False)
    return consts.u_scale * out["u"], out["phi"]


def mse(residual: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(residual))


# --- Interior residuals -----------------------------------------------------
def pde_residual(model, consts: GlobalConsts, batch: dict) -> tf.Tensor:
    """Momentum residual r_pde = rho*u_tt - sigma_z - rho*g."""
    c = _coeffs(batch)
    z, t = _col(batch["z"]), _col(batch["t"])
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(z)
        tape2.watch(t)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(z)
            tape1.watch(t)
            u, phi = _forward(model, consts, z, t, c)
        u_z = tape1.gradient(u, z)
        u_t = tape1.gradient(u, t)
        phi_z = tape1.gradient(phi, z)
        del tape1
    u_zz = tape2.gradient(u_z, z)
    u_tt = tape2.gradient(u_t, t)
    del tape2

    sigma_z = consts.alpha * c["k"] * u_zz \
        + (1.0 - consts.alpha) * c["k"] * c["eps_y"] * phi_z
    return c["rho"] * u_tt - sigma_z - c["rho"] * consts.g


def boucwen_residual(model, consts: GlobalConsts, batch: dict) -> tf.Tensor:
    """Bouc-Wen hysteretic ODE residual."""
    c = _coeffs(batch)
    z, t = _col(batch["z"]), _col(batch["t"])
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(z)
        tape2.watch(t)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(z)
            tape1.watch(t)
            u, phi = _forward(model, consts, z, t, c)
        u_z = tape1.gradient(u, z)
        phi_t = tape1.gradient(phi, t)
        del tape1
    eps_dot = tape2.gradient(u_z, t)
    del tape2

    phi_abs = tf.abs(phi) + _EPS
    rhs = (1.0 / c["eps_y"]) * (
        consts.A * eps_dot
        - consts.beta * tf.abs(eps_dot) * tf.pow(phi_abs, consts.n - 1.0) * phi
        - consts.gamma * eps_dot * tf.pow(phi_abs, consts.n)
    )
    return phi_t - rhs


# --- Boundary residuals -----------------------------------------------------
def bc_residual(
    model,
    consts: GlobalConsts,
    batch: dict,
    u_input: Callable = real_base_motion,
) -> dict[str, tf.Tensor]:
    """Boundary residuals along t.

    base (z=0):    u(0, t) - u_input(t, pga)   [default: real 2019 M6.7 base motion]
    surface (z=H): sigma(H, t) = alpha*k*u_z + (1-alpha)*k*eps_y*Phi
    """
    c = _coeffs(batch)
    t = _col(batch["t"])
    n = tf.shape(t)[0]

    # base: prescribed displacement (synthetic Ricker x PGA)
    z_base = tf.zeros((n, 1), dtype=tf.float32)
    u_base, _ = _forward(model, consts, z_base, t, c)
    r_base = u_base - u_input(t, c["pga"])

    # surface: zero stress (needs u_z at z=H, per-unit H)
    z_surf = c["H"]
    with tf.GradientTape() as tape:
        tape.watch(z_surf)
        u_surf, phi_surf = _forward(model, consts, z_surf, t, c)
    u_z_surf = tape.gradient(u_surf, z_surf)
    sigma_surf = consts.alpha * c["k"] * u_z_surf \
        + (1.0 - consts.alpha) * c["k"] * c["eps_y"] * phi_surf

    return {"base": r_base, "surface": sigma_surf}


# --- Initial residuals ------------------------------------------------------
def ic_residual(model, consts: GlobalConsts, batch: dict) -> dict[str, tf.Tensor]:
    """Initial-condition residuals at t=0: u=0, u_t=0, Phi=0."""
    c = _coeffs(batch)
    z = _col(batch["z"])
    t0 = tf.zeros_like(z)
    with tf.GradientTape() as tape:
        tape.watch(t0)
        u, phi = _forward(model, consts, z, t0, c)
    u_t = tape.gradient(u, t0)
    return {"u": u, "u_t": u_t, "phi": phi}
