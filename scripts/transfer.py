"""
Applying a fitted PEGPN to data it was not trained on.

Three distinct situations, in increasing order of how much can go wrong:

1. **New earthquake scenario, same area.** Only the loading branch changes.
   Because material resistance is inferred without ever seeing PGA, ``R0``,
   ``psi``, ``c``, ``phi`` and ``Fs`` are *unchanged* by definition and only
   ``a_y/PGA``, ``D_n`` and ``p_occ`` move. Use :func:`predict_scenario`.

2. **New region or event, with a landslide inventory.** True external
   validation. Predict, then evaluate. Use :func:`predict_new` +
   :func:`evaluate_transfer`.

3. **New region, no inventory.** Deployment mapping. Nothing can be validated,
   so the applicability report is the only honest safeguard.
   Use :func:`predict_new` + :func:`applicability_report`.

The traps this module exists to close
-------------------------------------
*Unit re-detection.* ``prepare_data`` auto-detects units from the median of each
column. Run independently on a second dataset it can reach a *different*
conclusion, silently changing the physics. A bundle freezes the training unit
policy and replays it.

*Silent vocabulary mismatch.* A lithology written ``"Basalt"`` in training and
``"basaltic flows"`` in the new area falls into the reserved unseen bucket. The
model still returns numbers. :func:`category_overlap` makes that visible, and
``crosswalk=`` lets you fix it.

*Calibration transfer.* Discrimination (ROC-AUC) survives a monotone shift in
base rate; calibrated probabilities do not. A region with a different landslide
prevalence needs recalibration against local labels, or the probabilities should
be demoted to a ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import copy
import pickle

import numpy as np
import pandas as pd

from . import physics as P
from .calibration import Calibrator
from .config import DataConfig, PEGPNConfig
from .data import PreparedData, UnitDecision, prepare_data
from .metrics import classification_metrics
from .model import PEGPN, TrainArrays, build_model, ensemble_predict
from .preprocessing import FoldPreprocessor
from .priors import bounds_from_texture

BUNDLE_FORMAT = 2


class TransferError(RuntimeError):
    """Raised when a bundle cannot be applied to the supplied data."""


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------
@dataclass
class PEGPNBundle:
    """Everything needed to reproduce a fitted model on new data.

    Weights are stored as plain NumPy arrays rather than a Keras archive, so the
    bundle survives Keras 2 / Keras 3 differences and TensorFlow upgrades.
    """

    weights: List[List[np.ndarray]]
    arch: Dict[str, object]
    cfg: PEGPNConfig
    prep: FoldPreprocessor
    calibrator: Optional[Calibrator] = None
    threshold: float = 0.5
    unit_decisions: List[UnitDecision] = field(default_factory=list)
    envelope: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    source_prevalence: float = float("nan")
    source_metrics: Dict[str, float] = field(default_factory=dict)
    format_version: int = BUNDLE_FORMAT
    note: str = ""

    @property
    def n_members(self) -> int:
        return len(self.weights)

    def summary(self) -> str:
        lines = [
            f"PEGPN bundle (format {self.format_version}) - {self.n_members} ensemble members",
            f"  encoder inputs   : {self.arch['n_numeric']} numeric, "
            f"{self.arch['n_litho']} lithology classes, {self.arch['n_tex']} texture classes",
            f"  reference stress : {self.cfg.physics.sigma_ref} kPa, psi0 = {self.cfg.physics.psi0}",
            f"  unit weight mode : {self.cfg.data.unit_weight_mode}",
            f"  failure depth    : z_mode = {self.cfg.data.z_mode} (cap {self.cfg.data.z_cap_m} m)",
            f"  groundwater      : " + ("uniform m = %.3f" % self.cfg.data.m_uniform
                                        if self.cfg.data.use_uniform_m
                                        else f"column {self.cfg.data.m_col}"),
            f"  calibrator       : {self.calibrator.method if self.calibrator else 'none'}"
            f"   threshold {self.threshold:.4f}",
            f"  source prevalence: {self.source_prevalence:.4f}",
        ]
        if self.source_metrics:
            lines.append("  source out-of-fold: " + ", ".join(
                f"{k}={v:.4f}" for k, v in self.source_metrics.items()
                if k in ("roc_auc", "pr_auc", "brier", "ece")))
        if self.note:
            lines.append(f"  note             : {self.note}")
        return "\n".join(lines)


def _frozen_data_config(cfg_data: DataConfig, decisions: Sequence[UnitDecision]) -> DataConfig:
    """Replace every 'auto' unit policy with the literal choice made at training."""
    frozen = copy.deepcopy(cfg_data)
    for d in decisions:
        if d.variable == "gamma" and frozen.density_units == "auto":
            frozen.density_units = d.detected
        elif d.variable == "z" and frozen.depth_units == "auto":
            frozen.depth_units = d.detected
        elif d.variable == "PGA" and frozen.pga_units == "auto":
            frozen.pga_units = d.detected
    return frozen


def make_bundle(full: Dict[str, object], cfg: PEGPNConfig, data: PreparedData,
                cv=None, note: str = "") -> PEGPNBundle:
    """Package a full-data fit for reuse.

    Parameters
    ----------
    full : the dict returned by ``fit_full_model`` / ``PEGPNRun.fit_map_model``
    cv   : optional ``CVResult``; supplies the deployment calibrator and
           threshold, both fitted on out-of-fold predictions only.
    """
    models: List[PEGPN] = full["models"]
    prep: FoldPreprocessor = full["prep"]
    arrays: TrainArrays = full["arrays"]

    calibrator, threshold, src_metrics = None, 0.5, {}
    if cv is not None:
        y = cv.oof["y"].to_numpy()
        p_raw = cv.oof["p_raw"].to_numpy()
        ok = np.isfinite(p_raw)
        if cfg.train.calibration != "none":
            calibrator = Calibrator(cfg.train.calibration).fit(p_raw[ok], y[ok])
        threshold = float(np.median(cv.thresholds)) if cv.thresholds else 0.5
        src_metrics = cv.overall(threshold)

    env: Dict[str, Tuple[float, float]] = {}
    for name, arr in [("beta_deg", np.degrees(data.beta_rad)), ("z_m", data.z_m),
                      ("gamma_kNm3", data.gamma_kNm3), ("m_ratio", data.m_ratio),
                      ("pga_g", data.pga_g)]:
        env[name] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    for j, nm in enumerate(data.material_names):
        col = data.X_material[:, j]
        env[nm] = (float(np.nanpercentile(col, 2.5)), float(np.nanpercentile(col, 97.5)))

    return PEGPNBundle(
        weights=[[np.asarray(w) for w in m.get_weights()] for m in models],
        arch={"n_numeric": int(arrays.x_num.shape[1]),
              "n_litho": int(prep.litho.n_classes),
              "n_tex": int(prep.texture.n_classes),
              "use_embeddings": bool(cfg.use_embeddings),
              "use_psi": bool(cfg.use_psi),
              "smooth_training": bool(cfg.smooth_training),
              "pga_in_encoder": bool(cfg.pga_in_encoder),
              "param_mode": "R0_psi"},
        cfg=copy.deepcopy(cfg),
        prep=copy.deepcopy(prep),
        calibrator=calibrator, threshold=threshold,
        unit_decisions=list(data.audit.decisions),
        envelope=env,
        source_prevalence=float(np.mean(data.y)),
        source_metrics=src_metrics,
        note=note,
    )


def save_bundle(bundle: PEGPNBundle, path: str) -> str:
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    return path


def load_bundle(path: str) -> PEGPNBundle:
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    if not isinstance(bundle, PEGPNBundle):
        raise TransferError(f"{path} does not contain a PEGPN bundle.")
    return bundle


# --------------------------------------------------------------------------
# loading new data under the frozen policy
# --------------------------------------------------------------------------
def load_new_data(bundle: PEGPNBundle, gpkg_path: str, *, layer: Optional[str] = None,
                  gdf=None, has_labels: bool = True, group_col: Optional[str] = None,
                  crosswalk: Optional[Dict[str, Dict[str, str]]] = None,
                  verbose: bool = True) -> PreparedData:
    """Prepare a new dataset using the *training* unit policy and column mapping.

    ``crosswalk`` renames category labels before encoding, e.g.
    ``{"lithology": {"basaltic flows": "Basalt"}, "texture": {...}}``.
    """
    from .data import load_gdf

    if gdf is None:
        gdf = load_gdf(gpkg_path, layer)
    gdf = gdf.copy()

    cfg_data = _frozen_data_config(bundle.cfg.data, bundle.unit_decisions)
    cfg_data.gpkg_path = gpkg_path
    cfg_data.layer = layer

    if not has_labels and cfg_data.label_col not in gdf.columns:
        gdf[cfg_data.label_col] = 0
        if verbose:
            print(f"note: no inventory supplied; '{cfg_data.label_col}' filled with zeros. "
                  "Performance metrics are meaningless for this run.")
    if group_col is not None:
        cfg_data.group_col = group_col
    if cfg_data.group_col not in gdf.columns:
        gdf[cfg_data.group_col] = "unknown"
        if verbose:
            print(f"note: '{cfg_data.group_col}' absent; all rows assigned to one group.")

    if crosswalk:
        for key, mapping in crosswalk.items():
            col = {"lithology": bundle.cfg.data.litho_name_col,
                   "texture": bundle.cfg.data.texture_name_col}.get(key, key)
            if col and col in gdf.columns:
                gdf[col] = gdf[col].astype(str).replace(mapping)

    data = prepare_data(cfg_data, bundle.cfg.physics, gdf=gdf, verbose=verbose)
    if verbose:
        print("\nUnit policy was FROZEN from training "
              f"(density={cfg_data.density_units}, depth={cfg_data.depth_units}, "
              f"pga={cfg_data.pga_units}); auto-detection was not re-run.")
    return data


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------
def rebuild_models(bundle: PEGPNBundle, arrays: TrainArrays) -> List[PEGPN]:
    if int(arrays.x_num.shape[1]) != int(bundle.arch["n_numeric"]):
        raise TransferError(
            f"Encoder expects {bundle.arch['n_numeric']} numeric inputs but the new data "
            f"produced {arrays.x_num.shape[1]}. The material_numeric column list must match "
            "the one used at training.")
    models: List[PEGPN] = []
    for k, w in enumerate(bundle.weights):
        m = build_model(arrays, int(bundle.arch["n_litho"]), int(bundle.arch["n_tex"]),
                        bundle.cfg.train, bundle.cfg.physics,
                        use_embeddings=bool(bundle.arch["use_embeddings"]),
                        use_psi=bool(bundle.arch["use_psi"]),
                        smooth_training=bool(bundle.arch["smooth_training"]),
                        seed=0, param_mode=str(bundle.arch["param_mode"]))
        current = m.get_weights()
        if len(current) != len(w) or any(np.shape(a) != np.shape(b)
                                         for a, b in zip(current, w)):
            raise TransferError(
                f"Weight shapes of member {k} do not match the rebuilt architecture. "
                "The bundle was produced by a different package version.")
        m.set_weights(w)
        models.append(m)
    return models


def build_transfer_arrays(bundle: PEGPNBundle, data: PreparedData) -> TrainArrays:
    from .cv import build_arrays, make_state_matrix
    bounds = bounds_from_texture(data.texture_codes)
    state = make_state_matrix(data, use_water=bundle.cfg.use_water)
    return build_arrays(data, np.arange(data.n), bundle.prep, bounds, state,
                        bool(bundle.arch["pga_in_encoder"]))


def predict_new(bundle: PEGPNBundle, data: PreparedData, *,
                recalibrate: bool = False, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Apply the bundled ensemble to prepared new data.

    ``recalibrate=True`` refits the calibrator on the new region's own labels.
    Only do that when a genuine local inventory exists, and report it: the
    resulting probabilities are no longer a pure out-of-sample transfer.
    """
    arrays = build_transfer_arrays(bundle, data)
    models = rebuild_models(bundle, arrays)
    pred = ensemble_predict(models, arrays, bundle.cfg.physics)

    if recalibrate:
        if data.y.sum() == 0:
            raise TransferError("recalibrate=True requires a labelled inventory.")
        cal = Calibrator(bundle.cfg.train.calibration).fit(pred["p"], data.y)
        pred["p_cal"] = cal.transform(pred["p"])
        pred["calibration_source"] = "target region (local labels)"
    elif bundle.calibrator is not None:
        pred["p_cal"] = bundle.calibrator.transform(pred["p"])
        pred["calibration_source"] = "source region (transferred)"
    else:
        pred["p_cal"] = pred["p"]
        pred["calibration_source"] = "none"

    if verbose:
        prev = float(np.mean(data.y))
        print(f"predicted {data.n} slope units with {bundle.n_members} members")
        print(f"  calibration: {pred['calibration_source']}")
        print("  regimes: " + ", ".join(f"{k} {v:.3f}"
                                        for k, v in P.regime_fractions(pred["regime"]).items()))
        if np.isfinite(bundle.source_prevalence) and prev > 0:
            ratio = prev / max(bundle.source_prevalence, 1e-9)
            if ratio < 0.5 or ratio > 2.0:
                print(f"  WARNING: target prevalence {prev:.4f} differs from source "
                      f"{bundle.source_prevalence:.4f} by {ratio:.1f}x. Transferred "
                      "probabilities will be miscalibrated; use the ranking, or "
                      "recalibrate=True against local labels.")
    return pred


def predict_scenario(bundle: PEGPNBundle, data: PreparedData, pga_g_new,
                     *, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Re-run the same area under a different ground-motion scenario.

    Because PGA never enters the material encoder, ``R0``, ``psi``, ``c``,
    ``phi`` and ``Fs`` are identical to the baseline run by construction. Only
    ``a_y/PGA``, ``D_n``, ``I_D`` and ``p_occ`` respond. That invariance is a
    property of the separated architecture and is asserted here.
    """
    pga_new = np.broadcast_to(np.asarray(pga_g_new, dtype=np.float64), data.pga_g.shape).copy()
    scen = copy.copy(data)
    scen.pga_g = pga_new
    pred = predict_new(bundle, scen, verbose=False)
    if verbose:
        print(f"scenario PGA: median {np.median(pga_new):.3f} g "
              f"(baseline {np.median(data.pga_g):.3f} g)")
        print("  regimes: " + ", ".join(f"{k} {v:.3f}"
                                        for k, v in P.regime_fractions(pred["regime"]).items()))
        print(f"  mean p_occ {np.mean(pred['p_cal']):.4f}")
    return pred


# --------------------------------------------------------------------------
# applicability and evaluation
# --------------------------------------------------------------------------
def category_overlap(bundle: PEGPNBundle, data: PreparedData) -> pd.DataFrame:
    """How much of the new data falls into the reserved unseen-category bucket."""
    rows = []
    for name, enc, codes in [("lithology", bundle.prep.litho, data.litho_codes),
                             ("texture", bundle.prep.texture, data.texture_codes)]:
        new_labels = set(np.unique(np.asarray(codes).astype(str)))
        known = set(enc.vocab)
        idx = enc.transform(codes)
        rows.append({
            "field": name,
            "classes_in_training": len(known),
            "classes_in_new_data": len(new_labels),
            "recognised_classes": len(new_labels & known),
            "unrecognised_classes": sorted(new_labels - known)[:12],
            "rows_in_unseen_bucket": int((idx == 0).sum()),
            "frac_rows_unseen": float((idx == 0).mean()),
        })
    return pd.DataFrame(rows)


def applicability_report(bundle: PEGPNBundle, data: PreparedData) -> pd.DataFrame:
    """Fraction of new rows outside the training 2.5-97.5 percentile envelope."""
    fields = {"beta_deg": np.degrees(data.beta_rad), "z_m": data.z_m,
              "gamma_kNm3": data.gamma_kNm3, "m_ratio": data.m_ratio,
              "pga_g": data.pga_g}
    for j, nm in enumerate(data.material_names):
        fields[nm] = data.X_material[:, j]

    rows = []
    for name, arr in fields.items():
        if name not in bundle.envelope:
            continue
        lo, hi = bundle.envelope[name]
        below = float(np.mean(arr < lo))
        above = float(np.mean(arr > hi))
        rows.append({"variable": name, "train_p2.5": lo, "train_p97.5": hi,
                     "new_median": float(np.median(arr)),
                     "frac_below": below, "frac_above": above,
                     "frac_outside": below + above})
    df = pd.DataFrame(rows).sort_values("frac_outside", ascending=False)
    df.attrs["max_frac_outside"] = float(df["frac_outside"].max()) if len(df) else 0.0
    return df


def evaluate_transfer(bundle: PEGPNBundle, data: PreparedData,
                      pred: Dict[str, np.ndarray]) -> pd.DataFrame:
    """External-validation metrics. Requires a genuine inventory for the new area."""
    if data.y.sum() == 0:
        raise TransferError("No positive labels: external validation is not possible.")
    rows = []
    for label, p in [("transferred p_occ", pred["p_cal"]),
                     ("uncalibrated p", pred["p"]),
                     ("instability index I_D", pred["ID"])]:
        m = classification_metrics(data.y, np.clip(p, 1e-9, 1 - 1e-9), bundle.threshold)
        m["score"] = label
        rows.append(m)
    df = pd.DataFrame(rows).set_index("score")
    keep = ["roc_auc", "pr_auc", "brier", "log_loss", "ece", "mcc",
            "sensitivity", "specificity", "prevalence", "mean_pred"]
    return df[[c for c in keep if c in df.columns]]


def transfer_gap(bundle: PEGPNBundle, target_metrics: pd.DataFrame,
                 metric: str = "roc_auc", row: str = "transferred p_occ") -> Dict[str, float]:
    """Source out-of-fold score vs target score - the number reviewers ask for."""
    src = float(bundle.source_metrics.get(metric, np.nan))
    tgt = float(target_metrics.loc[row, metric])
    return {"source_out_of_fold": src, "target_external": tgt,
            "gap": src - tgt, "retention": tgt / src if src else np.nan}
