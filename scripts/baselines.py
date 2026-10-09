"""
Matched baselines (Sec. 15.1).

The decisive comparison is the separated-branch PEGPN against a direct MLP that
receives *identical inputs* with *comparable capacity*. Only that pairing can
show whether the embedded mechanics buy transferable predictive skill rather
than simply more parameters.

Every baseline is evaluated on the same outer folds with the same fold-safe
preprocessing, so paired fold differences are meaningful.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from . import physics as P
from .calibration import Calibrator
from .config import PEGPNConfig
from .cv import CVResult, balanced_group_kfold, build_arrays, make_outer_folds, make_state_matrix
from .data import PreparedData
from .metrics import best_threshold, classification_metrics
from .preprocessing import FoldPreprocessor
from .priors import DEFAULT_TEXTURE_BOUNDS, FALLBACK_BOUNDS, _norm, bounds_from_texture


# --------------------------------------------------------------------------
# shared feature assembly (identical information content)
# --------------------------------------------------------------------------
def _design_matrix(data: PreparedData, idx, prep: FoldPreprocessor,
                   state: np.ndarray, one_hot: bool = True) -> np.ndarray:
    x_num, litho, tex = prep.transform(data.X_material[idx], data.litho_codes[idx],
                                       data.texture_codes[idx])
    blocks = [x_num, state[idx]]
    if one_hot:
        blocks.append(np.eye(prep.litho.n_classes, dtype=np.float32)[litho])
        blocks.append(np.eye(prep.texture.n_classes, dtype=np.float32)[tex])
    return np.column_stack(blocks).astype(np.float32)


def _fit_predict_logistic(Xtr, ytr, Xte, seed=0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _fit_predict_mlp(Xtr, ytr, Xva, yva, Xte, cfg: PEGPNConfig, seed=0):
    """Capacity-matched dense network trained with the same protocol."""
    tf.keras.utils.set_random_seed(seed)
    tc = cfg.train
    layers = [tf.keras.layers.Input(shape=(Xtr.shape[1],))]
    for h in tc.hidden:
        layers.append(tf.keras.layers.Dense(h, activation=tc.activation))
        if tc.dropout > 0:
            layers.append(tf.keras.layers.Dropout(tc.dropout))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(tc.learning_rate),
                  loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    pos = max(float(ytr.sum()), 1.0)
    neg = max(float(len(ytr) - ytr.sum()), 1.0)
    cw = {0: len(ytr) / (2 * neg), 1: len(ytr) / (2 * pos)} if tc.class_weighting != "none" else None
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=tc.epochs,
              batch_size=tc.batch_size, class_weight=cw, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                          patience=tc.patience,
                                                          restore_best_weights=True)])
    return model.predict(Xte, verbose=0).ravel(), model.predict(Xva, verbose=0).ravel()


# --------------------------------------------------------------------------
# conventional Newmark with externally assigned parameters
# --------------------------------------------------------------------------
def conventional_newmark(data: PreparedData, cfg: PEGPNConfig) -> Dict[str, np.ndarray]:
    """Literature parameters by texture class, no learning at all."""
    b = bounds_from_texture(data.texture_codes)
    c = 0.5 * (b["c_low"] + b["c_high"])
    phi_deg = 0.5 * (b["phi_low"] + b["phi_high"])
    tan_phi = np.tan(np.radians(phi_deg))
    state = make_state_matrix(data, use_water=cfg.use_water).astype(np.float64)
    ch = P.forward_chain_numpy(c, tan_phi, state[:, 0], state[:, 1], state[:, 2],
                               state[:, 3], state[:, 4], cfg.physics)
    ch["c"] = c
    ch["phi_deg"] = phi_deg
    return ch


# --------------------------------------------------------------------------
# CV runner shared by all baselines
# --------------------------------------------------------------------------
def run_baseline_cv(data: PreparedData, cfg: PEGPNConfig, kind: str = "mlp",
                    verbose: bool = True) -> CVResult:
    """kind in {'logistic', 'mlp', 'newmark', 'direct_cphi'}."""
    if kind == "direct_cphi":
        from .cv import run_nested_cv
        return run_nested_cv(data, cfg, verbose=verbose, param_mode="direct_cphi")

    folds, _ = make_outer_folds(data, cfg.cv)
    state = make_state_matrix(data, use_water=cfg.use_water)
    n = data.n
    p_oof = np.full(n, np.nan)
    fold_of_row = np.full(n, -1, dtype=np.int16)
    rows, thresholds = [], []

    if kind == "newmark":
        ch = conventional_newmark(data, cfg)
        score = ch["ID"]

    for f, (tr, te) in enumerate(folds):
        prep = FoldPreprocessor().fit(data.X_material[tr], data.litho_codes[tr],
                                      data.texture_codes[tr])
        inner = balanced_group_kfold(data.groups[tr], cfg.cv.n_inner, cfg.cv.random_state + f)
        i_tr, i_va = tr[inner[0][0]], tr[inner[0][1]]

        if kind == "newmark":
            p_va, p_te = score[i_va], score[te]
        elif kind == "logistic":
            Xtr = _design_matrix(data, i_tr, prep, state)
            Xva = _design_matrix(data, i_va, prep, state)
            Xte = _design_matrix(data, te, prep, state)
            clf_p = _fit_predict_logistic(Xtr, data.y[i_tr], np.vstack([Xva, Xte]))
            p_va, p_te = clf_p[:len(Xva)], clf_p[len(Xva):]
        elif kind == "mlp":
            Xtr = _design_matrix(data, i_tr, prep, state)
            Xva = _design_matrix(data, i_va, prep, state)
            Xte = _design_matrix(data, te, prep, state)
            preds = [_fit_predict_mlp(Xtr, data.y[i_tr].astype(np.float32), Xva,
                                      data.y[i_va].astype(np.float32), Xte, cfg, seed=s)
                     for s in cfg.train.seeds]
            p_te = np.mean([a for a, _ in preds], axis=0)
            p_va = np.mean([b for _, b in preds], axis=0)
        else:
            raise ValueError(f"Unknown baseline kind: {kind}")

        cal = Calibrator(cfg.train.calibration).fit(p_va, data.y[i_va])
        thr = best_threshold(data.y[i_va], cal.transform(p_va))
        p_cal = cal.transform(p_te)
        p_oof[te] = p_cal
        fold_of_row[te] = f
        thresholds.append(thr)

        m = classification_metrics(data.y[te], p_cal, thr)
        m.update({"fold": f, "n_test": len(te), "model": kind})
        rows.append(m)
        if verbose:
            print(f"[{kind} fold {f}] ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")

    oof = pd.DataFrame({"y": data.y, "p_cal": p_oof, "p_raw": p_oof,
                        "group": data.groups, "fold": fold_of_row})
    return CVResult(oof=oof, fold_metrics=pd.DataFrame(rows), fold_of_row=fold_of_row,
                    thresholds=thresholds)


def compare_models(results: Dict[str, CVResult], metric: str = "roc_auc") -> pd.DataFrame:
    """Paired fold comparison of every baseline against PEGPN."""
    from .metrics import paired_fold_test
    if "PEGPN" not in results:
        raise KeyError("results must contain a 'PEGPN' entry to compare against.")
    ref = results["PEGPN"].fold_metrics.sort_values("fold")[metric].to_numpy()
    rows = []
    for name, res in results.items():
        vals = res.fold_metrics.sort_values("fold")[metric].to_numpy()
        rec = {"model": name, f"mean_{metric}": float(np.nanmean(vals)),
               f"sd_{metric}": float(np.nanstd(vals))}
        if name != "PEGPN":
            rec.update({f"vs_PEGPN_{k}": v for k, v in paired_fold_test(ref, vals).items()})
        rows.append(rec)
    return pd.DataFrame(rows)
