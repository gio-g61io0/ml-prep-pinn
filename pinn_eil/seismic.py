"""Base-motion boundary input u_input(t) for the base BC.

PRIMARY (`real_base_motion`): the actual rock-outcrop waveform from the 2019-12-15 M6.7
Cotabato SPECFEM run, extracted to `pinn_wave/data/makilala_base_motion.npz`. We use the
normalized shape `disp_norm` (peak 1) and scale it per unit by that unit's PGA relative to
the base-motion's own PGA -- the standard "scale record to target PGA" convention. Physical
base displacement:

    u_input(t; PGA) = (PGA / PGA_ref) * peak_disp * disp_norm(t)

`peak_disp` also sets `u_scale` (norm_spec output.u.by), so in the loss (which divides by U =
peak_disp) the non-dimensional base target is simply `(PGA/PGA_ref) * disp_norm(t)`.

Interpolation is done in-graph via index math on the uniform time grid (dt constant), so it
works inside `tf.function`.

FALLBACKS: `ricker_base_motion` (synthetic pulse, pre-SPECFEM) and `zero_base_motion` (for
the manufactured-solution test).
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


class _RealBaseMotion:
    """Lazily-loaded, in-graph interpolator for the real base-motion shape."""

    def __init__(self):
        self._loaded = False

    def _load(self):
        d = np.load(BASE_MOTION_NPZ)
        self.dt = float(d["dt"])
        self.t0 = float(d["t"][0])                 # = 0 after re-zeroing
        self.n = int(d["disp_norm"].shape[0])
        self.peak_disp = float(d["peak_disp"])
        self.pga_ref = float(np.max(np.abs(d["acc"])) / _G)   # PGA of the reference record [g]
        # init_scope lifts the constant into the eager/outer context so it is reusable
        # even when _load is first triggered from inside a tf.function graph.
        with tf.init_scope():
            self.shape = tf.constant(d["disp_norm"], dtype=tf.float32)  # peak 1, signed
        self._loaded = True

    def __call__(self, t, pga):
        if not self._loaded:
            self._load()
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        pga = tf.convert_to_tensor(pga, dtype=tf.float32)

        # linear interpolation on the uniform grid: idx = (t - t0)/dt
        idx = (t - self.t0) / self.dt
        idx = tf.clip_by_value(idx, 0.0, float(self.n - 1))
        i0 = tf.floor(idx)
        frac = idx - i0
        i0 = tf.cast(i0, tf.int32)
        i1 = tf.minimum(i0 + 1, self.n - 1)
        s0 = tf.gather(self.shape, tf.reshape(i0, [-1]))
        s1 = tf.gather(self.shape, tf.reshape(i1, [-1]))
        shape = tf.reshape(s0 + (s1 - s0) * tf.reshape(frac, [-1]), tf.shape(t))

        # per-unit amplitude: scale record to this unit's PGA, in physical metres
        amp = (pga / self.pga_ref) * self.peak_disp
        return amp * shape


real_base_motion = _RealBaseMotion()


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
