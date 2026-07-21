"""Training loop for the amortized Makilala EIL PINN.

Physics-only training (no displacement labels): minimize the weighted, non-dimensionalized
sum of the four residual losses over collocation / boundary / initial points drawn from
REAL Makilala units.

  L = w_pde*L_pde + w_bw*L_bw + w_bc*(L_base + L_surf) + w_ic*(L_u + L_ut + L_phi)

Each residual is divided by its characteristic scale (config.load_scales) so the terms are
comparable. Optimizer + schedule follow the repo convention (Adam + ExponentialDecay);
gradient clipping and periodic resampling follow the pipeline PDF.

NOTE: the base motion is SYNTHETIC (Ricker x PGA) and the displacement scale U is derived
from max PGA -- both are placeholders until makilala_base_motion.npz exists. Full convergence
on this stiff, oscillatory problem needs curriculum + tuning (PDF Step 7); this loop provides
the working, correctly-scaled machinery and a smoke-trainable baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from . import physics
from .config import CharacteristicScales, GlobalConsts


@dataclass
class LossWeights:
    pde: float = 1.0
    bw: float = 1.0
    bc: float = 10.0   # BCs are critical -> weighted higher (PDF Step 7)
    ic: float = 10.0


def make_optimizer(lr: float = 1e-3, decay_steps: int = 10000,
                   decay_rate: float = 0.9) -> tf.keras.optimizers.Optimizer:
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate)
    return tf.keras.optimizers.Adam(learning_rate=schedule)


def compute_losses(model, consts: GlobalConsts, scales: CharacteristicScales,
                   col: dict, bc: dict, ic: dict) -> dict[str, tf.Tensor]:
    """All (scaled) loss terms for one batch. Kept tape-transparent for the trainer."""
    r_pde = physics.pde_residual(model, consts, col) / scales.S_pde
    r_bw = physics.boucwen_residual(model, consts, col) / scales.S_bw
    b = physics.bc_residual(model, consts, bc)
    i = physics.ic_residual(model, consts, ic)

    l_pde = physics.mse(r_pde)
    l_bw = physics.mse(r_bw)
    l_base = physics.mse(b["base"] / scales.U)
    l_surf = physics.mse(b["surface"] / scales.S_surf)
    l_u = physics.mse(i["u"] / scales.U)
    l_ut = physics.mse(i["u_t"] / scales.U_dot)
    l_phi = physics.mse(i["phi"])
    return {
        "pde": l_pde, "bw": l_bw, "base": l_base, "surf": l_surf,
        "ic_u": l_u, "ic_ut": l_ut, "ic_phi": l_phi,
    }


def total_loss(parts: dict[str, tf.Tensor], w: LossWeights) -> tf.Tensor:
    return (w.pde * parts["pde"] + w.bw * parts["bw"]
            + w.bc * (parts["base"] + parts["surf"])
            + w.ic * (parts["ic_u"] + parts["ic_ut"] + parts["ic_phi"]))


def train(
    model,
    consts: GlobalConsts,
    scales: CharacteristicScales,
    sampler,
    epochs: int = 2000,
    n_col: int = 4000,
    n_bc: int = 1000,
    n_ic: int = 1000,
    resample_every: int = 200,
    weights: LossWeights | None = None,
    clip_norm: float = 1.0,
    log_every: int = 100,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
    t_max: float | None = None,
    epoch_offset: int = 0,
) -> tuple[list[dict], tf.keras.optimizers.Optimizer]:
    """Run the physics-only training loop over t in [0, t_max] (default full T).

    Returns (history, optimizer) so a curriculum can reuse the optimizer state.
    """
    weights = weights or LossWeights()
    optimizer = optimizer or make_optimizer()
    history: list[dict] = []
    col = bc = ic = None

    @tf.function
    def step(col, bc, ic):
        with tf.GradientTape() as tape:
            parts = compute_losses(model, consts, scales, col, bc, ic)
            loss = total_loss(parts, weights)
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, parts

    for epoch in range(epochs):
        if epoch % resample_every == 0:
            col = sampler.collocation(n_col, t_max=t_max)
            bc = sampler.boundary(n_bc, t_max=t_max)
            ic = sampler.initial(n_ic)
        loss, parts = step(col, bc, ic)

        if epoch % log_every == 0 or epoch == epochs - 1:
            rec = {"epoch": epoch_offset + epoch, "loss": float(loss)}
            rec.update({k: float(v) for k, v in parts.items()})
            history.append(rec)
            print(f"epoch {rec['epoch']:6d} | L={rec['loss']:.4e} | pde={rec['pde']:.3e} "
                  f"bw={rec['bw']:.3e} base={rec['base']:.3e} surf={rec['surf']:.3e} "
                  f"ic_u={rec['ic_u']:.3e} ic_ut={rec['ic_ut']:.3e} ic_phi={rec['ic_phi']:.3e}")
    return history, optimizer


def train_curriculum(
    model,
    consts: GlobalConsts,
    scales: CharacteristicScales,
    sampler,
    stages: list[tuple[float, int]] | None = None,
    weights: LossWeights | None = None,
    **kwargs,
) -> list[dict]:
    """Time-window curriculum: train on growing time windows, sharing optimizer state.

    Args:
        stages: list of (t_max, epochs). Default expands 5 -> 10 -> 20 -> full T.
    Returns the concatenated history across all stages.
    """
    if stages is None:
        T = consts.T
        stages = [(min(5.0, T), 1000), (min(10.0, T), 1000),
                  (min(20.0, T), 1000), (T, 1500)]
    optimizer = make_optimizer()
    history: list[dict] = []
    offset = 0
    for i, (t_max, epochs) in enumerate(stages):
        print(f"\n=== curriculum stage {i + 1}/{len(stages)}: "
              f"t in [0, {t_max:g}s], {epochs} epochs ===")
        hist, optimizer = train(
            model, consts, scales, sampler, epochs=epochs, weights=weights,
            optimizer=optimizer, t_max=t_max, epoch_offset=offset, **kwargs)
        history.extend(hist)
        offset += epochs
    return history
