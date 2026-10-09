"""
Nested, spatially aware cross-validation.

Outer folds are whole barangays (or KMeans blocks of slope-unit centroids), so
no neighbouring slope unit is ever split across the training/test boundary. Every
learned operation - scaler, category vocabulary, network weights, calibrator and
decision threshold - is refitted inside each outer training fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import time

import numpy as np
import pandas as pd

from . import physics as P
from .calibration import Calibrator
from .config import CVConfig, PEGPNConfig, PhysicsConfig, TrainConfig
from .data import PreparedData
from .metrics import best_threshold, classification_metrics
from .model import TrainArrays, build_model, ensemble_predict, train_pegpn
from .preprocessing import FoldPreprocessor
from .priors import bounds_from_texture


# --------------------------------------------------------------------------
# fold construction
# --------------------------------------------------------------------------
def spatial_blocks(coords: np.ndarray, n_blocks: int, random_state: int = 42) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_blocks, n_init=10, random_state=random_state)
    return km.fit_predict(np.asarray(coords, dtype=float))


def balanced_group_kfold(groups: np.ndarray, n_splits: int,
                         random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Assign whole groups to folds, largest first, to equalise fold sizes."""
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(random_state)
    order = np.lexsort((rng.random(len(uniq)), -counts))
    loads = np.zeros(n_splits, dtype=np.int64)
    assign: Dict[object, int] = {}
    for gi in order:
        f = int(np.argmin(loads))
        assign[uniq[gi]] = f
        loads[f] += counts[gi]
    fold_of_row = np.array([assign[g] for g in groups])
    return [(np.where(fold_of_row != f)[0], np.where(fold_of_row == f)[0])
            for f in range(n_splits)]


def make_outer_folds(data: PreparedData, cfg: CVConfig) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    if cfg.scheme == "group":
        units = data.groups
    else:
        units = spatial_blocks(data.coords, cfg.n_blocks, cfg.random_state).astype(str)
    return balanced_group_kfold(units, cfg.n_outer, cfg.random_state), units


# --------------------------------------------------------------------------
# array assembly
# --------------------------------------------------------------------------
def make_state_matrix(data: PreparedData, use_water: bool = True) -> np.ndarray:
    m = data.m_ratio if use_water else np.zeros_like(data.m_ratio)
    return np.column_stack([data.beta_rad, data.z_m, data.gamma_kNm3, m,
                            data.pga_g]).astype(np.float32)


def build_arrays(data: PreparedData, idx: np.ndarray, prep: FoldPreprocessor,
                 bounds: Dict[str, np.ndarray], state: np.ndarray,
                 pga_in_encoder: bool = False) -> TrainArrays:
    x_num, litho, tex = prep.transform(data.X_material[idx],
                                       data.litho_codes[idx],
                                       data.texture_codes[idx])
    if pga_in_encoder:  # deliberate shortcut ablation
        x_num = np.column_stack([x_num, state[idx, 4]]).astype(np.float32)

    kw = {}
    if data.anchor_mask is not None:
        kw = {"anchor_mask": data.anchor_mask[idx].astype(np.float32),
              "c_obs": data.c_obs[idx].astype(np.float32),
              "phi_obs": data.phi_obs_deg[idx].astype(np.float32),
              "sigma_c": data.sigma_c[idx].astype(np.float32),
              "sigma_phi": data.sigma_phi[idx].astype(np.float32)}

    return TrainArrays(
        x_num=x_num, litho=litho.astype(np.int32), tex=tex.astype(np.int32),
        state=state[idx], y=data.y[idx].astype(np.float32),
        c_low=bounds["c_low"][idx].astype(np.float32),
        c_high=bounds["c_high"][idx].astype(np.float32),
        phi_low=bounds["phi_low"][idx].astype(np.float32),
        phi_high=bounds["phi_high"][idx].astype(np.float32), **kw)


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------
@dataclass
class CVResult:
    oof: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_of_row: np.ndarray
    preprocessor_reports: List[Dict[str, object]] = field(default_factory=list)
    thresholds: List[float] = field(default_factory=list)
    elapsed_s: float = 0.0
    models_last_fold: Optional[List[object]] = None
    prep_last_fold: Optional[FoldPreprocessor] = None

    def overall(self, threshold: Optional[float] = None) -> Dict[str, float]:
        thr = threshold if threshold is not None else float(np.median(self.thresholds or [0.5]))
        return classification_metrics(self.oof["y"].to_numpy(),
                                      self.oof["p_cal"].to_numpy(), thr)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_nested_cv(data: PreparedData, cfg: PEGPNConfig, *,
                  verbose: bool = True, keep_last_models: bool = True,
                  prior_widen: float = 1.0, param_mode: str = "R0_psi") -> CVResult:
    t0 = time.time()
    folds, _ = make_outer_folds(data, cfg.cv)
    bounds = bounds_from_texture(data.texture_codes, widen=prior_widen)
    state = make_state_matrix(data, use_water=cfg.use_water)

    n = data.n
    cols = ["p_raw", "p_cal", "Fs", "R0", "psi", "c", "phi_deg", "ay_g", "ratio",
            "D_hard_cm", "ID", "p_sd", "Fs_sd", "regime_agreement"]
    oof = {c: np.full(n, np.nan) for c in cols}
    oof_regime = np.full(n, -1, dtype=np.int8)
    fold_of_row = np.full(n, -1, dtype=np.int16)

    rows, reports, thresholds = [], [], []
    models_last, prep_last = None, None

    for f, (tr, te) in enumerate(folds):
        ft = time.time()
        prep = FoldPreprocessor().fit(data.X_material[tr], data.litho_codes[tr],
                                      data.texture_codes[tr])

        inner = balanced_group_kfold(data.groups[tr], cfg.cv.n_inner, cfg.cv.random_state + f)
        i_tr_rel, i_va_rel = inner[0]
        i_tr, i_va = tr[i_tr_rel], tr[i_va_rel]

        A_tr = build_arrays(data, i_tr, prep, bounds, state, cfg.pga_in_encoder)
        A_va = build_arrays(data, i_va, prep, bounds, state, cfg.pga_in_encoder)
        A_te = build_arrays(data, te, prep, bounds, state, cfg.pga_in_encoder)

        models = []
        for seed in cfg.train.seeds:
            mdl = build_model(A_tr, prep.litho.n_classes, prep.texture.n_classes,
                              cfg.train, cfg.physics,
                              use_embeddings=cfg.use_embeddings, use_psi=cfg.use_psi,
                              smooth_training=cfg.smooth_training, seed=seed,
                              param_mode=param_mode)
            train_pegpn(mdl, A_tr, A_va, cfg.train, cfg.physics,
                        use_priors=cfg.use_priors,
                        use_anchor=cfg.use_anchor and data.anchor_mask is not None,
                        seed=seed)
            models.append(mdl)

        pv = ensemble_predict(models, A_va, cfg.physics)
        cal = Calibrator(cfg.train.calibration).fit(pv["p"], data.y[i_va])
        thr = best_threshold(data.y[i_va], cal.transform(pv["p"])) if cfg.cv.select_threshold_on_inner else 0.5
        thresholds.append(thr)

        pt = ensemble_predict(models, A_te, cfg.physics)
        p_cal = cal.transform(pt["p"])

        oof["p_raw"][te] = pt["p"]; oof["p_cal"][te] = p_cal
        oof["Fs"][te] = pt["Fs"]; oof["R0"][te] = pt["R0"]; oof["psi"][te] = pt["psi"]
        oof["c"][te] = pt["c"]; oof["phi_deg"][te] = pt["phi_deg"]
        oof["ay_g"][te] = pt["ay_g"]; oof["ratio"][te] = pt["ratio"]
        oof["D_hard_cm"][te] = pt["D_hard_cm"]; oof["ID"][te] = pt["ID"]
        oof["p_sd"][te] = pt["p_sd"]; oof["Fs_sd"][te] = pt["Fs_sd"]
        oof["regime_agreement"][te] = pt["regime_agreement"]
        oof_regime[te] = pt["regime"]
        fold_of_row[te] = f

        m = classification_metrics(data.y[te], p_cal, thr)
        m.update({"fold": f, "n_test": len(te), "n_train": len(i_tr),
                  "seconds": time.time() - ft})
        m.update({f"frac_{k}": v for k, v in P.regime_fractions(pt["regime"]).items()})
        rows.append(m)
        reports.append(prep.report)

        if verbose:
            print(f"[fold {f}] n_te={len(te):6d}  ROC-AUC={m['roc_auc']:.4f}  "
                  f"PR-AUC={m['pr_auc']:.4f}  Brier={m['brier']:.4f}  "
                  f"ECE={m['ece']:.4f}  ({m['seconds']:.1f}s)")

        if keep_last_models and f == len(folds) - 1:
            models_last, prep_last = models, prep

    df = pd.DataFrame({**oof, "y": data.y, "group": data.groups,
                       "regime": oof_regime, "fold": fold_of_row})
    return CVResult(oof=df, fold_metrics=pd.DataFrame(rows), fold_of_row=fold_of_row,
                    preprocessor_reports=reports, thresholds=thresholds,
                    elapsed_s=time.time() - t0,
                    models_last_fold=models_last, prep_last_fold=prep_last)


# --------------------------------------------------------------------------
# final full-data model (mapping only, after all decisions are frozen)
# --------------------------------------------------------------------------
def fit_full_model(data: PreparedData, cfg: PEGPNConfig, *, verbose: bool = True,
                   prior_widen: float = 1.0, param_mode: str = "R0_psi"):
    """Refit on all rows for wall-to-wall mapping.

    An internal grouped split is still used for early stopping so the number of
    epochs is not chosen by looking at the mapped output.
    """
    bounds = bounds_from_texture(data.texture_codes, widen=prior_widen)
    state = make_state_matrix(data, use_water=cfg.use_water)
    prep = FoldPreprocessor().fit(data.X_material, data.litho_codes, data.texture_codes)

    split = balanced_group_kfold(data.groups, cfg.cv.n_inner, cfg.cv.random_state)
    tr, va = split[0]
    A_tr = build_arrays(data, tr, prep, bounds, state, cfg.pga_in_encoder)
    A_va = build_arrays(data, va, prep, bounds, state, cfg.pga_in_encoder)
    A_all = build_arrays(data, np.arange(data.n), prep, bounds, state, cfg.pga_in_encoder)

    models = []
    for seed in cfg.train.seeds:
        mdl = build_model(A_tr, prep.litho.n_classes, prep.texture.n_classes,
                          cfg.train, cfg.physics, use_embeddings=cfg.use_embeddings,
                          use_psi=cfg.use_psi, smooth_training=cfg.smooth_training, seed=seed,
                          param_mode=param_mode)
        info = train_pegpn(mdl, A_tr, A_va, cfg.train, cfg.physics,
                           use_priors=cfg.use_priors,
                           use_anchor=cfg.use_anchor and data.anchor_mask is not None,
                           seed=seed)
        if verbose:
            print(f"  seed {seed}: best epoch {info['best_epoch']}, "
                  f"val score {info['best_score']:.4f}")
        models.append(mdl)

    pred = ensemble_predict(models, A_all, cfg.physics)
    return {"models": models, "prep": prep, "arrays": A_all, "pred": pred,
            "bounds": bounds, "state": state}
