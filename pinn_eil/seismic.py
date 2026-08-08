"""Base-motion time histories from the SPECFEM record, for the two PINN formulations.

Both are extracted from the 2019-12-15 M6.7 Cotabato SPECFEM run
(`pinn_wave/data/makilala_base_motion.npz`) and scaled per unit by that unit's PGA relative
to the record's own PGA -- the standard "scale record to target PGA" convention.

`real_base_motion(t, PGA)` -- base DISPLACEMENT (boundary-input formulation). Uses the
normalized shape `disp_norm` (peak 1); physical base displacement:

    u_input(t; PGA) = (PGA / PGA_ref) * peak_disp * disp_norm(t)

`real_base_acceleration(t, PGA)` -- base ACCELERATION a_base(t) [m/s^2] (body-force /
relative-displacement formulation). Uses the physical `acc` record (peak = PGA_ref * g),
so the per-unit physical acceleration is simply:

    a_base(t; PGA) = (PGA / PGA_ref) * acc(t)

and the momentum PDE carries the inertial body force +rho*a_base(t) while the base BC becomes
fixed (u(0, t) = 0). See `physics.pde_residual` / `physics.bc_residual`.

Interpolation is done in-graph via index math on the uniform time grid (dt constant), so both
work inside `tf.function`.

FALLBACKS: `ricker_base_motion` (synthetic pulse, pre-SPECFEM) and `zero_base_motion` (used
for the manufactured-solution test and as the fixed-base BC target).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import SEISMIC_F0

DEFAULT_T0 = 2.0  # Ricker pulse center [s] (synthetic fallback)
BASE_MOTION_NPZ = Path(__file__).resolve().parent.parent / "pinn_wave" / "data" / "makilala_base_motion.npz"

_G = 9.81


def _interp_uniform(values: tf.Tensor, t0: float, dt: float, n: int, t) -> tf.Tensor:
    """Linear interpolation of a uniform-grid signal `values` at times `t` (in-graph)."""
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    idx = (t - t0) / dt                              # fractional sample index
    idx = tf.clip_by_value(idx, 0.0, float(n - 1))
    i0 = tf.floor(idx)
    frac = idx - i0
    i0 = tf.cast(i0, tf.int32)
    i1 = tf.minimum(i0 + 1, n - 1)
    s0 = tf.gather(values, tf.reshape(i0, [-1]))
    s1 = tf.gather(values, tf.reshape(i1, [-1]))
    return tf.reshape(s0 + (s1 - s0) * tf.reshape(frac, [-1]), tf.shape(t))


class _RealBaseSignal:
    """Lazily-loaded, in-graph, PGA-scaled interpolator for a real base-motion channel.

    mode="disp": interpolate the normalized shape `disp_norm` (peak 1) and scale to physical
                 metres via (PGA/PGA_ref) * peak_disp.
    mode="acc":  interpolate the physical `acc` record [m/s^2] (peak = PGA_ref*g) and scale by
                 (PGA/PGA_ref) so the per-unit peak acceleration is PGA*g.
    """

    def __init__(self, key: str, mode: str):
        self.key = key      # npz array to interpolate ("disp_norm" | "acc")
        self.mode = mode    # amplitude convention ("disp" | "acc")
        self._loaded = False

    def _load(self):
        d = np.load(BASE_MOTION_NPZ)
        self.dt = float(d["dt"])
        self.t0 = float(d["t"][0])                 # = 0 after re-zeroing
        self.n = int(d[self.key].shape[0])
        self.peak_disp = float(d["peak_disp"])
        self.pga_ref = float(np.max(np.abs(d["acc"])) / _G)   # PGA of the reference record [g]
        # init_scope lifts the constant into the eager/outer context so it is reusable
        # even when _load is first triggered from inside a tf.function graph.
        with tf.init_scope():
            self.values = tf.constant(d[self.key], dtype=tf.float32)
        self._loaded = True

    def __call__(self, t, pga):
        if not self._loaded:
            self._load()
        pga = tf.convert_to_tensor(pga, dtype=tf.float32)
        shape = _interp_uniform(self.values, self.t0, self.dt, self.n, t)
        # per-unit amplitude: scale record to this unit's PGA
        amp = (pga / self.pga_ref) * (self.peak_disp if self.mode == "disp" else 1.0)
        return amp * shape


real_base_motion = _RealBaseSignal("disp_norm", "disp")          # base displacement [m]
real_base_acceleration = _RealBaseSignal("acc", "acc")           # base acceleration [m/s^2]


def ricker_base_motion(t, pga, f0: float = SEISMIC_F0, t0: float = DEFAULT_T0,
                       g: float = _G) -> tf.Tensor:
    """Synthetic Ricker displacement pulse [m], scaled by PGA [g] (pre-SPECFEM fallback)."""
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    pga = tf.convert_to_tensor(pga, dtype=tf.float32)
    w0 = 2.0 * math.pi * f0
    arg = (math.pi * f0 * (t - t0)) ** 2
    ricker = (1.0 - 2.0 * arg) * tf.exp(-arg)
    return (pga * g / (w0 ** 2)) * ricker


def zero_base_motion(t, pga=None) -> tf.Tensor:
    """u_input(t) = 0 (manufactured-solution correctness test)."""
    return tf.zeros_like(tf.convert_to_tensor(t, dtype=tf.float32))
