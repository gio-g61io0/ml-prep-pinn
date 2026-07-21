"""Amortized network architecture for the Makilala EIL PINN.

The network IS the solution family: NN(z, t, rho, k, eps_y, H, pga) -> [u_norm, Phi].
One trained network covers all slope units; at inference you query it per unit.

Design choices (see plan):
  * Activation ``tanh`` + ``glorot_normal`` init -- a residual PINN needs a smooth C^2
    activation so second derivatives (u_tt, u_zz) exist. ``relu`` would give u_tt == 0.
  * Inputs are PHYSICAL; the model normalizes internally (z/H per-sample, t/T, min-max on
    the conditioning) so a GradientTape watching physical z, t yields physical derivatives.
  * Output ``u`` is NORMALIZED; physical u = u_scale * u_norm is applied in the residuals.

Input column order (must match physics._model_input): [z, t, rho, k, eps_y, H, pga].
Follows the repo serialization convention (register_keras_serializable + get_config).
"""
from __future__ import annotations

import tensorflow as tf

DEFAULT_DEPTH = 5
DEFAULT_WIDTH = 64
N_INPUTS = 7  # z, t, rho, k, eps_y, H, pga

# Index of each column in the physical input vector.
_IZ, _IT, _IRHO, _IK, _IEPS, _IH, _IPGA = range(N_INPUTS)


@tf.keras.utils.register_keras_serializable(package="pinn_eil")
class AmortizedNormalization(tf.keras.layers.Layer):
    """Normalize the physical input vector to ~[0, 1].

    z -> z/H (per-sample H, col 5), t -> t/T, conditioning -> min-max over dataset ranges.
    Kept as a layer so the tape can watch physical z, t and let autodiff apply the 1/H,
    1/T chain-rule factors automatically.
    """

    def __init__(self, T: float, ranges: dict, **kwargs):
        kwargs.setdefault("name", "amortized_normalization")
        super().__init__(**kwargs)
        self.T = float(T)
        # store as plain lists for JSON serialization
        self.ranges = {k: [float(v[0]), float(v[1])] for k, v in ranges.items()}

    def call(self, x):
        z = x[:, _IZ:_IZ + 1]
        t = x[:, _IT:_IT + 1]
        rho = x[:, _IRHO:_IRHO + 1]
        k = x[:, _IK:_IK + 1]
        eps_y = x[:, _IEPS:_IEPS + 1]
        H = x[:, _IH:_IH + 1]
        pga = x[:, _IPGA:_IPGA + 1]

        def mm(v, key):
            lo, hi = self.ranges[key]
            return (v - lo) / (hi - lo)

        z_n = z / H
        t_n = t / self.T
        return tf.concat(
            [z_n, t_n, mm(rho, "rho"), mm(k, "k"), mm(eps_y, "eps_y"),
             mm(H, "H"), mm(pga, "pga")],
            axis=1,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"T": self.T, "ranges": self.ranges})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_pinn(
    ranges: dict,
    T: float,
    depth: int = DEFAULT_DEPTH,
    width: int = DEFAULT_WIDTH,
    bound_phi: bool = False,
    name: str = "pinn_eil",
) -> tf.keras.Model:
    """Build NN(z, t, rho, k, eps_y, H, pga) -> {"u": u_norm, "phi": Phi}.

    Args:
        ranges:    min/max per conditioning feature (from config.load_norm_ranges()).
        T:         simulation duration [s], for t-normalization.
        depth:     number of hidden tanh layers.
        width:     units per hidden layer.
        bound_phi: if True, squash Phi through tanh (~[-1, 1]); else linear.
    """
    inp = tf.keras.Input(shape=(N_INPUTS,), name="physical_inputs")
    x = AmortizedNormalization(T=T, ranges=ranges)(inp)

    for i in range(depth):
        x = tf.keras.layers.Dense(
            width, activation="tanh", kernel_initializer="glorot_normal",
            bias_initializer="zeros", name=f"hidden_{i}",
        )(x)

    u = tf.keras.layers.Dense(
        1, activation=None, kernel_initializer="glorot_normal",
        bias_initializer="zeros", name="u",
    )(x)
    phi = tf.keras.layers.Dense(
        1, activation="tanh" if bound_phi else None,
        kernel_initializer="glorot_normal", bias_initializer="zeros", name="phi",
    )(x)

    return tf.keras.Model(inputs=inp, outputs={"u": u, "phi": phi}, name=name)
