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


class EILPinn:
    """Config-holding builder for the amortized Makilala EIL PINN (Functional API).

    Follows the repo's production-model paradigm (cf. ``py_files/LandslideRainfall_v3.py::
    LandslideRainFallV3``): a plain class (NOT a ``tf.keras.Model``) whose ``__init__`` holds
    the configuration, ``build_model()`` assembles the Functional graph onto ``self.model``,
    and ``get_optimizer()`` stores the optimizer on ``self.optimizer_instance``. Usage:

        builder = EILPinn(ranges, T)
        builder.build_model()          # -> self.model
        builder.get_optimizer(lr=1e-3) # -> self.optimizer_instance

    The graph is identical to the one produced by the ``build_pinn`` free function (which is
    now a thin wrapper around this class): single stacked ``Input(shape=(7,))`` ->
    ``AmortizedNormalization`` -> tanh Dense stack -> ``{"u": u_norm, "phi": Phi}`` heads.
    """

    def __init__(
        self,
        ranges: dict,
        T: float,
        depth: int = DEFAULT_DEPTH,
        width: int = DEFAULT_WIDTH,
        bound_phi: bool = False,
        name: str = "pinn_eil",
    ):
        """Store config.

        Args:
            ranges:    min/max per conditioning feature (from config.load_norm_ranges()).
            T:         simulation duration [s], for t-normalization.
            depth:     number of hidden tanh layers.
            width:     units per hidden layer.
            bound_phi: if True, squash Phi through tanh (~[-1, 1]); else linear.
            name:      Keras model name.
        """
        self.ranges = ranges
        self.T = float(T)
        self.depth = int(depth)
        self.width = int(width)
        self.bound_phi = bool(bound_phi)
        self.name = name
        self.model: tf.keras.Model | None = None
        self.optimizer_instance: tf.keras.optimizers.Optimizer | None = None

    def build_model(self) -> tf.keras.Model:
        """Assemble NN(z, t, rho, k, eps_y, H, pga) -> {"u": u_norm, "phi": Phi} onto self.model.

        Analog of the cotabato model's ``classification_model()``; named ``build_model`` since
        the PINN outputs a displacement field rather than a classification.
        """
        inp = tf.keras.Input(shape=(N_INPUTS,), name="physical_inputs")
        x = AmortizedNormalization(T=self.T, ranges=self.ranges)(inp)

        for i in range(self.depth):
            x = tf.keras.layers.Dense(
                self.width, activation="tanh", kernel_initializer="glorot_normal",
                bias_initializer="zeros", name=f"hidden_{i}",
            )(x)

        u = tf.keras.layers.Dense(
            1, activation=None, kernel_initializer="glorot_normal",
            bias_initializer="zeros", name="u",
        )(x)
        phi = tf.keras.layers.Dense(
            1, activation="tanh" if self.bound_phi else None,
            kernel_initializer="glorot_normal", bias_initializer="zeros", name="phi",
        )(x)

        self.model = tf.keras.Model(inputs=inp, outputs={"u": u, "phi": phi}, name=self.name)
        return self.model

    def get_optimizer(
        self, lr: float = 1e-3, decay_steps: int = 10000, decay_rate: float = 0.9,
    ) -> tf.keras.optimizers.Optimizer:
        """Build an Adam + ExponentialDecay optimizer onto self.optimizer_instance.

        Reuses ``train.make_optimizer`` (imported lazily to avoid an import cycle) so the
        schedule matches the standalone training path exactly.
        """
        from .train import make_optimizer  # lazy: avoid model<->train import cycle at load

        self.optimizer_instance = make_optimizer(
            lr=lr, decay_steps=decay_steps, decay_rate=decay_rate)
        return self.optimizer_instance


def build_pinn(
    ranges: dict,
    T: float,
    depth: int = DEFAULT_DEPTH,
    width: int = DEFAULT_WIDTH,
    bound_phi: bool = False,
    name: str = "pinn_eil",
) -> tf.keras.Model:
    """Build NN(z, t, rho, k, eps_y, H, pga) -> {"u": u_norm, "phi": Phi}.

    Thin backward-compatible wrapper around ``EILPinn`` (preserves existing callers such as
    run_training.py, check_residuals.py, predict_map.py). See ``EILPinn`` for arg details.
    """
    return EILPinn(ranges, T, depth=depth, width=width,
                   bound_phi=bound_phi, name=name).build_model()
