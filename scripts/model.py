"""
The PEGPN model.

Architecture (Sec. 4 of the methodology report)::

    material descriptors  -> encoder -> (R0, psi)
                                     -> reference-stress mapping -> (c, phi)
    slope state (beta,z,gamma,m)     -> infinite-slope operator  -> Fs
                                     -> critical acceleration    -> a_y
    earthquake loading (PGA)         -> Newmark operator         -> D_n
                                     -> monotonic head           -> p_occ

PGA and slope geometry never enter the material encoder (unless the
``pga_in_encoder`` ablation is switched on deliberately), so the encoder cannot
reconstruct the factor of safety by algebraic identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from . import physics as P
from .config import PhysicsConfig, TrainConfig

DTYPE = tf.float32


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1.0 - p)))


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------
class MaterialEncoder(tf.keras.Model):
    """Maps material descriptors to a positive reference resistance and a partition."""

    def __init__(self, n_numeric: int, n_litho: int, n_texture: int,
                 train_cfg: TrainConfig, phys: PhysicsConfig,
                 use_embeddings: bool = True, use_psi: bool = True,
                 param_mode: str = "R0_psi", **kw):
        super().__init__(**kw)
        if param_mode not in {"R0_psi", "direct_cphi"}:
            raise ValueError(f"Unknown param_mode: {param_mode}")
        self.phys = phys
        self.use_embeddings = use_embeddings
        self.use_psi = use_psi
        self.param_mode = param_mode
        self.psi0_logit = _logit(phys.psi0)

        if use_embeddings:
            self.emb_l = tf.keras.layers.Embedding(max(n_litho, 1), train_cfg.emb_dim_litho,
                                                   name="litho_emb")
            self.emb_t = tf.keras.layers.Embedding(max(n_texture, 1), train_cfg.emb_dim_texture,
                                                   name="texture_emb")
        self.blocks: List[tf.keras.layers.Layer] = []
        for i, h in enumerate(train_cfg.hidden):
            self.blocks.append(tf.keras.layers.Dense(h, activation=train_cfg.activation,
                                                     name=f"dense_{i}"))
            if train_cfg.dropout > 0:
                self.blocks.append(tf.keras.layers.Dropout(train_cfg.dropout, name=f"drop_{i}"))
        # small-scale init keeps R0 near mid-range and psi near psi0 at step 0
        # while still delivering a non-zero gradient to the hidden stack.
        self.head = tf.keras.layers.Dense(
            2, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer="zeros", name="head")

    def call(self, x_num, litho_idx, tex_idx, training: bool = False):
        feats = [tf.cast(x_num, DTYPE)]
        if self.use_embeddings:
            feats.append(self.emb_l(litho_idx))
            feats.append(self.emb_t(tex_idx))
        h = tf.concat(feats, axis=-1)
        for layer in self.blocks:
            h = layer(h, training=training) if isinstance(layer, tf.keras.layers.Dropout) else layer(h)
        raw = self.head(h)

        r_raw, q_raw = raw[:, 0], raw[:, 1]

        if self.param_mode == "direct_cphi":
            # Unconstrained, independently predicted c and phi: the closest
            # neural-Newmark predecessor, kept as a matched baseline.
            c = tf.sigmoid(r_raw) * self.phys.R0_max
            phi_deg = 10.0 + tf.sigmoid(q_raw) * 35.0
            tan_phi = tf.tan(phi_deg * tf.constant(np.pi / 180.0, DTYPE))
            R0 = c + self.phys.sigma_ref * tan_phi
            psi = c / tf.maximum(R0, 1e-6)
            return R0, psi

        R0 = self.phys.R0_min + tf.sigmoid(r_raw) * (self.phys.R0_max - self.phys.R0_min)
        if self.use_psi:
            psi = tf.sigmoid(q_raw + self.psi0_logit)
        else:
            psi = tf.fill(tf.shape(r_raw), tf.constant(self.phys.psi0, DTYPE))
        return R0, psi


# --------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------
class PEGPN(tf.keras.Model):
    def __init__(self, n_numeric: int, n_litho: int, n_texture: int,
                 train_cfg: TrainConfig, phys: PhysicsConfig,
                 base_rate: float = 0.11, use_embeddings: bool = True,
                 use_psi: bool = True, smooth_training: bool = True,
                 param_mode: str = "R0_psi", **kw):
        super().__init__(**kw)
        self.phys = phys
        self.train_cfg = train_cfg
        self.smooth_training = smooth_training
        self.param_mode = param_mode
        self.encoder = MaterialEncoder(n_numeric, n_litho, n_texture, train_cfg, phys,
                                       use_embeddings=use_embeddings, use_psi=use_psi,
                                       param_mode=param_mode)
        self.b = self.add_weight(name="head_bias", shape=(),
                                 initializer=tf.keras.initializers.Constant(_logit(base_rate)),
                                 trainable=True)
        # s = softplus(s_raw) > 0 guarantees monotonicity in displacement
        self.s_raw = self.add_weight(name="head_scale_raw", shape=(),
                                     initializer=tf.keras.initializers.Constant(0.5413),
                                     trainable=True)   # softplus(0.5413) ~ 1.0

    # -- physics chain -------------------------------------------------
    def forward(self, x_num, litho_idx, tex_idx, state, training: bool = False,
                smooth: Optional[bool] = None) -> Dict[str, tf.Tensor]:
        smooth = self.smooth_training if smooth is None else smooth
        ph = self.phys
        beta, z, gamma, m, pga = (state[:, 0], state[:, 1], state[:, 2],
                                  state[:, 3], state[:, 4])

        R0, psi = self.encoder(x_num, litho_idx, tex_idx, training=training)
        c, tan_phi, phi = P.resistance_to_c_phi(R0, psi, ph.sigma_ref)

        Fs, A, B = P.factor_of_safety(c, tan_phi, beta, z, gamma, m, ph.gamma_w)
        ay = P.critical_acceleration_g(Fs, beta)
        r = P.acceleration_ratio(ay, pga)

        if smooth:
            u, D_soft, r_used = P.newmark_soft(r, ph.tau, ph.eps_r, ph.jibson_k, ph.log1p_cap)
        else:  # exact piecewise branch: gradient dies outside (eps, 1-eps)
            r_used = tf.clip_by_value(r, ph.eps_r, 1.0 - ph.eps_r)
            log10_D = P.jibson_log10_displacement(r_used, ph.jibson_k)
            u = tf.math.softplus(log10_D * P.LN10)
            u = tf.where(r >= 1.0, tf.zeros_like(u), u)
            D_soft = tf.exp(tf.minimum(u, 30.0)) - 1.0

        s = tf.math.softplus(self.s_raw)
        logits = self.b + s * u
        p = tf.sigmoid(logits)
        return {"R0": R0, "psi": psi, "c": c, "tan_phi": tan_phi, "phi": phi,
                "A": A, "B": B, "Fs": Fs, "ay_g": ay, "ratio": r, "r_used": r_used,
                "u": u, "D_soft": D_soft, "logits": logits, "p": p}

    def call(self, inputs, training: bool = False):
        return self.forward(inputs["x_num"], inputs["litho"], inputs["tex"],
                            inputs["state"], training=training)


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------
def _hinge_norm(x, lo, hi):
    width = tf.maximum(hi - lo, 1e-6)
    return tf.square((tf.nn.relu(lo - x) + tf.nn.relu(x - hi)) / width)


def occurrence_loss(y, logits, cfg: TrainConfig, w_pos: float, w_neg: float):
    y = tf.cast(y, DTYPE)
    if cfg.class_weighting == "focal":
        p = tf.sigmoid(logits)
        pt = y * p + (1 - y) * (1 - p)
        alpha = y * w_pos + (1 - y) * w_neg
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        return tf.reduce_mean(alpha * tf.pow(1.0 - pt, cfg.focal_gamma) * ce)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    if cfg.class_weighting == "balanced":
        w = y * w_pos + (1 - y) * w_neg
        return tf.reduce_sum(w * ce) / tf.maximum(tf.reduce_sum(w), 1e-6)
    return tf.reduce_mean(ce)


def total_loss(model: PEGPN, batch: Dict[str, tf.Tensor], out: Dict[str, tf.Tensor],
               cfg: TrainConfig, phys: PhysicsConfig, w_pos: float, w_neg: float,
               use_priors: bool = True, use_anchor: bool = True) -> Dict[str, tf.Tensor]:
    parts: Dict[str, tf.Tensor] = {}
    parts["occ"] = occurrence_loss(batch["y"], out["logits"], cfg, w_pos, w_neg)

    if use_priors and cfg.lambda_prior > 0:
        phi_deg = out["phi"] * tf.constant(180.0 / np.pi, DTYPE)
        lp = tf.reduce_mean(_hinge_norm(out["c"], batch["c_low"], batch["c_high"])
                            + _hinge_norm(phi_deg, batch["phi_low"], batch["phi_high"]))
        parts["prior"] = cfg.lambda_prior * lp

    if cfg.lambda_psi > 0:
        parts["psi"] = cfg.lambda_psi * tf.reduce_mean(
            tf.square(out["psi"] - tf.constant(phys.psi0, DTYPE)))

    if use_anchor and cfg.lambda_anchor > 0 and "anchor_mask" in batch:
        mask = batch["anchor_mask"]
        denom = tf.maximum(tf.reduce_sum(mask), 1.0)
        phi_deg = out["phi"] * tf.constant(180.0 / np.pi, DTYPE)
        la = tf.reduce_sum(mask * (tf.square(out["c"] - batch["c_obs"]) / tf.square(batch["sigma_c"])
                                   + tf.square(phi_deg - batch["phi_obs"]) / tf.square(batch["sigma_phi"]))) / denom
        parts["anchor"] = cfg.lambda_anchor * la

    if cfg.lambda_mono > 0:
        st = batch["state"]
        st_wet = tf.concat([st[:, :3], tf.minimum(st[:, 3:4] * 1.10 + 0.02, 1.0), st[:, 4:]], axis=1)
        st_hot = tf.concat([st[:, :4], st[:, 4:] * 1.20], axis=1)
        o_wet = model.forward(batch["x_num"], batch["litho"], batch["tex"], st_wet, training=True)
        o_hot = model.forward(batch["x_num"], batch["litho"], batch["tex"], st_hot, training=True)
        pen = (tf.reduce_mean(tf.nn.relu(o_wet["Fs"] - out["Fs"]))
               + tf.reduce_mean(tf.nn.relu(out["u"] - o_hot["u"])))
        parts["mono"] = cfg.lambda_mono * pen

    if cfg.l2 > 0:
        parts["l2"] = cfg.l2 * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])

    parts["total"] = tf.add_n(list(parts.values()))
    return parts


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
@dataclass
class TrainArrays:
    x_num: np.ndarray
    litho: np.ndarray
    tex: np.ndarray
    state: np.ndarray
    y: np.ndarray
    c_low: np.ndarray
    c_high: np.ndarray
    phi_low: np.ndarray
    phi_high: np.ndarray
    anchor_mask: Optional[np.ndarray] = None
    c_obs: Optional[np.ndarray] = None
    phi_obs: Optional[np.ndarray] = None
    sigma_c: Optional[np.ndarray] = None
    sigma_phi: Optional[np.ndarray] = None

    def as_tensors(self) -> Dict[str, tf.Tensor]:
        d = {
            "x_num": tf.constant(self.x_num, DTYPE),
            "litho": tf.constant(self.litho, tf.int32),
            "tex": tf.constant(self.tex, tf.int32),
            "state": tf.constant(self.state, DTYPE),
            "y": tf.constant(self.y, DTYPE),
            "c_low": tf.constant(self.c_low, DTYPE),
            "c_high": tf.constant(self.c_high, DTYPE),
            "phi_low": tf.constant(self.phi_low, DTYPE),
            "phi_high": tf.constant(self.phi_high, DTYPE),
        }
        if self.anchor_mask is not None:
            d.update({
                "anchor_mask": tf.constant(self.anchor_mask, DTYPE),
                "c_obs": tf.constant(self.c_obs, DTYPE),
                "phi_obs": tf.constant(self.phi_obs, DTYPE),
                "sigma_c": tf.constant(self.sigma_c, DTYPE),
                "sigma_phi": tf.constant(self.sigma_phi, DTYPE),
            })
        return d

    def subset(self, idx) -> "TrainArrays":
        def take(a):
            return None if a is None else a[idx]
        return TrainArrays(*[take(getattr(self, f)) for f in
                             ["x_num", "litho", "tex", "state", "y", "c_low", "c_high",
                              "phi_low", "phi_high", "anchor_mask", "c_obs", "phi_obs",
                              "sigma_c", "sigma_phi"]])


def build_model(arrays: TrainArrays, n_litho: int, n_tex: int,
                train_cfg: TrainConfig, phys: PhysicsConfig,
                use_embeddings: bool = True, use_psi: bool = True,
                smooth_training: bool = True, seed: int = 0,
                param_mode: str = "R0_psi") -> PEGPN:
    tf.keras.utils.set_random_seed(seed)
    model = PEGPN(arrays.x_num.shape[1], n_litho, n_tex, train_cfg, phys,
                  base_rate=float(np.clip(arrays.y.mean(), 1e-3, 1 - 1e-3)),
                  use_embeddings=use_embeddings, use_psi=use_psi,
                  smooth_training=smooth_training, param_mode=param_mode)
    probe = arrays.subset(np.arange(min(8, len(arrays.y)))).as_tensors()
    _ = model.forward(probe["x_num"], probe["litho"], probe["tex"], probe["state"],
                      training=False)
    return model


def train_pegpn(model: PEGPN, train: TrainArrays, val: Optional[TrainArrays],
                cfg: TrainConfig, phys: PhysicsConfig,
                use_priors: bool = True, use_anchor: bool = True,
                seed: int = 0) -> Dict[str, object]:
    """Mini-batch training with grouped early stopping on validation ROC-AUC."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    pos = float(train.y.sum())
    neg = float(len(train.y) - pos)
    w_pos = (len(train.y) / (2.0 * max(pos, 1.0))) if cfg.class_weighting != "none" else 1.0
    w_neg = (len(train.y) / (2.0 * max(neg, 1.0))) if cfg.class_weighting != "none" else 1.0

    T = train.as_tensors()
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    @tf.function(reduce_retracing=True)
    def step(batch):
        with tf.GradientTape() as tape:
            out = model.forward(batch["x_num"], batch["litho"], batch["tex"],
                                batch["state"], training=True)
            parts = total_loss(model, batch, out, cfg, phys, w_pos, w_neg,
                               use_priors=use_priors, use_anchor=use_anchor)
        grads = tape.gradient(parts["total"], model.trainable_variables)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else g for g in grads]
        optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None])
        return parts["total"]

    @tf.function(reduce_retracing=True)
    def infer(batch):
        return model.forward(batch["x_num"], batch["litho"], batch["tex"],
                             batch["state"], training=False)["p"]

    n = len(train.y)
    bs = min(cfg.batch_size, n)
    V = val.as_tensors() if val is not None else None
    best_score, best_weights, best_epoch, waited = -np.inf, None, -1, 0
    history: List[Dict[str, float]] = []

    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, bs):
            sel = order[start:start + bs]
            if len(sel) < 8:
                continue
            idx = tf.constant(sel, tf.int32)
            batch = {k: tf.gather(v, idx) for k, v in T.items()}
            losses.append(float(step(batch)))

        rec = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else np.nan}
        if V is not None:
            pv = infer(V).numpy()
            try:
                rec["val_auc"] = float(roc_auc_score(val.y, pv))
            except ValueError:
                rec["val_auc"] = float("nan")
            score = rec["val_auc"] if np.isfinite(rec["val_auc"]) else -rec["loss"]
        else:
            score = -rec["loss"]
        history.append(rec)

        if score > best_score + cfg.min_delta:
            best_score, best_epoch, waited = score, epoch, 0
            best_weights = [w.copy() for w in model.get_weights()]
        else:
            waited += 1
            if waited >= cfg.patience:
                break
        if cfg.verbose and epoch % max(cfg.verbose, 1) == 0:
            print(f"  epoch {epoch:3d}  loss {rec['loss']:.4f}"
                  + (f"  val_auc {rec.get('val_auc', float('nan')):.4f}" if V is not None else ""))

    if best_weights is not None:
        model.set_weights(best_weights)
    return {"history": history, "best_epoch": best_epoch, "best_score": best_score,
            "w_pos": w_pos, "w_neg": w_neg}


# --------------------------------------------------------------------------
# prediction with the exact operators
# --------------------------------------------------------------------------
def predict(model: PEGPN, arrays: TrainArrays, phys: PhysicsConfig,
            batch_size: int = 8192) -> Dict[str, np.ndarray]:
    """Latent parameters from the network, everything downstream scored exactly."""
    n = len(arrays.y)
    keys = ["R0", "psi", "c", "tan_phi", "phi", "Fs", "ay_g", "ratio", "p", "D_soft", "u"]
    acc = {k: [] for k in keys}
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        out = model.forward(tf.constant(arrays.x_num[sl], DTYPE),
                            tf.constant(arrays.litho[sl], tf.int32),
                            tf.constant(arrays.tex[sl], tf.int32),
                            tf.constant(arrays.state[sl], DTYPE),
                            training=False, smooth=True)
        for k in keys:
            acc[k].append(out[k].numpy())
    res = {k: np.concatenate(v).astype(np.float64) for k, v in acc.items()}

    D_hard, regime, capped = P.newmark_hard(res["Fs"], res["ratio"], phys.jibson_k,
                                            phys.eps_r, phys.D_cap_cm)
    res["D_hard_cm"] = D_hard
    res["regime"] = regime
    res["capped"] = capped
    res["ID"] = P.instability_index(D_hard, regime, phys.D0_cm)
    res["phi_deg"] = np.degrees(res["phi"])
    return res


def ensemble_predict(models: Sequence[PEGPN], arrays: TrainArrays,
                     phys: PhysicsConfig) -> Dict[str, np.ndarray]:
    """Aggregate members without letting the regime structure corrupt the mean.

    ``p``, ``Fs``, ``c``, ``phi`` and ``R0`` are finite for every member and are
    averaged directly. ``D_hard`` is NaN exactly on statically unstable units, so
    it is averaged with a NaN-aware mean; a unit that is static for every member
    stays NaN instead of collapsing to zero. The regime is a majority vote and
    member disagreement is reported rather than hidden.
    """
    preds = [predict(m, arrays, phys) for m in models]
    out: Dict[str, np.ndarray] = {}

    for key in ["p", "Fs", "R0", "psi", "c", "phi_deg", "ay_g", "ratio", "ID", "u"]:
        stack = np.stack([pr[key] for pr in preds])
        out[key] = np.nanmean(stack, axis=0)
        out[key + "_sd"] = np.nanstd(stack, axis=0)

    Dstack = np.stack([pr["D_hard_cm"] for pr in preds])
    all_nan = np.all(~np.isfinite(Dstack), axis=0)
    with np.errstate(invalid="ignore"):
        Dmean = np.where(all_nan, np.nan, np.nanmean(np.where(np.isfinite(Dstack), Dstack, np.nan), axis=0))
    out["D_hard_cm"] = Dmean
    out["D_hard_cm_sd"] = np.where(all_nan, np.nan,
                                   np.nanstd(np.where(np.isfinite(Dstack), Dstack, np.nan), axis=0))

    Rstack = np.stack([pr["regime"] for pr in preds])
    counts = np.stack([(Rstack == k).sum(axis=0) for k in (P.STATIC, P.YIELDING, P.NON_YIELDING)])
    out["regime"] = np.argmax(counts, axis=0).astype(np.int8)
    out["regime_agreement"] = counts.max(axis=0) / float(len(preds))
    out["capped_frac"] = np.mean(np.stack([pr["capped"] for pr in preds]), axis=0)
    return out
