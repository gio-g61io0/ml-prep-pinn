"""
End-to-end orchestration and the minimum reporting checklist.

``PEGPNRun`` wires the stages together in the order required for a leakage-safe
study: audit -> nested spatial CV -> diagnostics -> uncertainty -> full-data map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import time

import numpy as np
import pandas as pd

from . import diagnostics as DG
from . import physics as P
from .calibration import Calibrator
from .config import PEGPNConfig
from .cv import CVResult, fit_full_model, run_nested_cv
from .data import PreparedData, prepare_data
from .metrics import classification_metrics
from .uncertainty import MCResult, propagate


@dataclass
class PEGPNRun:
    cfg: PEGPNConfig
    data: Optional[PreparedData] = None
    cv: Optional[CVResult] = None
    full: Optional[Dict[str, object]] = None
    mc: Optional[MCResult] = None
    notes: Dict[str, object] = field(default_factory=dict)

    # -- stages --------------------------------------------------------
    def prepare(self, gdf=None, verbose: bool = True) -> PreparedData:
        self.data = prepare_data(self.cfg.data, self.cfg.physics, gdf=gdf, verbose=verbose)
        return self.data

    def run_cv(self, verbose: bool = True) -> CVResult:
        assert self.data is not None, "call prepare() first"
        self.cv = run_nested_cv(self.data, self.cfg, verbose=verbose)
        return self.cv

    def fit_map_model(self, verbose: bool = True) -> Dict[str, object]:
        assert self.data is not None, "call prepare() first"
        self.full = fit_full_model(self.data, self.cfg, verbose=verbose)
        return self.full

    def propagate_uncertainty(self, subset: Optional[np.ndarray] = None,
                              verbose: bool = True) -> MCResult:
        assert self.full is not None, "call fit_map_model() first"
        cal = None
        if self.cv is not None and self.cfg.train.calibration != "none":
            cal = Calibrator(self.cfg.train.calibration).fit(
                self.cv.oof["p_raw"].to_numpy(), self.cv.oof["y"].to_numpy())
        self.mc = propagate(self.full["models"], self.full["arrays"], self.cfg.physics,
                            self.cfg.uncertainty, subset=subset, calibrator=cal,
                            verbose=verbose)
        return self.mc

    # -- reporting -----------------------------------------------------
    def diagnostics_bundle(self) -> Dict[str, object]:
        assert self.full is not None and self.cv is not None
        pred = self.full["pred"]
        state = self.full["state"].astype(np.float64)
        d: Dict[str, object] = {}
        d["latent_summary"] = DG.latent_parameter_summary(pred)
        d["regime_occupancy"] = DG.regime_occupancy(pred["regime"], self.data.y)
        d["cap"] = DG.cap_diagnostics(pred["D_hard_cm"], self.cfg.physics)
        d["soft_hard_gap"] = DG.soft_hard_gap(self.full["models"][0], self.full["arrays"],
                                              self.cfg.physics)
        d["gradient_health"] = DG.gradient_health(self.cfg.physics)
        d["identifiability"] = DG.identifiability_scan(pred, state, self.cfg.physics)
        d["groundwater"] = DG.groundwater_sensitivity(pred, state, self.cfg.physics)
        d["depth_sensitivity"] = DG.depth_sensitivity(pred, state, self.cfg.physics,
                                                      y=self.data.y)
        d["monotonicity"] = DG.monotonicity_tests(self.full["models"][0],
                                                  self.full["arrays"], self.cfg.physics)
        d["category_texture"] = DG.category_diagnostics(self.cv, self.data, "texture")
        return d

    def reporting_checklist(self) -> Dict[str, object]:
        """Machine-readable version of Sec. 21."""
        assert self.data is not None
        ph, tr, cv = self.cfg.physics, self.cfg.train, self.cfg.cv
        chk: Dict[str, object] = {
            "inputs": self.data.audit.to_frame().to_dict(orient="records"),
            "reference_stress_kPa": ph.sigma_ref,
            "R0_bounds_kPa": [ph.R0_min, ph.R0_max],
            "psi0": ph.psi0,
            "regularisation_weights": {"prior": tr.lambda_prior, "psi": tr.lambda_psi,
                                       "anchor": tr.lambda_anchor, "mono": tr.lambda_mono,
                                       "l2": tr.l2},
            "architecture": {"hidden": list(tr.hidden), "activation": tr.activation,
                             "dropout": tr.dropout, "emb_litho": tr.emb_dim_litho,
                             "emb_texture": tr.emb_dim_texture},
            "optimisation": {"optimizer": "Adam", "lr": tr.learning_rate,
                             "batch_size": tr.batch_size, "epochs": tr.epochs,
                             "patience": tr.patience, "seeds": list(tr.seeds),
                             "class_weighting": tr.class_weighting},
            "newmark": {"coefficients": list(ph.jibson_k),
                        "residual_sd_log10": ph.jibson_sigma_log10,
                        "tau": ph.tau, "eps_r": ph.eps_r, "D_cap_cm": ph.D_cap_cm},
            "instability_index_D0_cm": ph.D0_cm,
            "validation": {"scheme": cv.scheme, "n_outer": cv.n_outer,
                           "n_inner": cv.n_inner, "n_blocks": cv.n_blocks,
                           "calibration": tr.calibration},
            "ablation_switches": self.cfg.to_dict()["ablation"],
            "groundwater_assumption": ("uniform m = %.3f" % self.cfg.data.m_uniform
                                       if self.cfg.data.use_uniform_m
                                       else f"column {self.cfg.data.m_col}"),
            "unit_weight_mode": self.cfg.data.unit_weight_mode,
            "failure_depth_rule": {"mode": self.cfg.data.z_mode,
                                   "cap_m": self.cfg.data.z_cap_m,
                                   "scale": self.cfg.data.z_scale,
                                   "fixed_m": self.cfg.data.z_fixed_m},
        }
        if self.cv is not None:
            chk["out_of_fold_metrics"] = self.cv.overall()
            chk["fold_metrics"] = self.cv.fold_metrics.to_dict(orient="records")
            chk["thresholds"] = self.cv.thresholds
        if self.full is not None:
            chk["regime_fractions"] = P.regime_fractions(self.full["pred"]["regime"])
        if self.mc is not None:
            chk["uncertainty"] = {"n_draws": self.mc.n_draws,
                                  "summary": self.mc.summary.to_dict(orient="records")}
        return chk

    def save_checklist(self, path: str) -> str:
        with open(path, "w") as fh:
            json.dump(self.reporting_checklist(), fh, indent=2, default=str)
        return path

    def terminology_note(self) -> str:
        return (
            "Terminology used throughout the outputs:\n"
            "  R0, psi        prior-constrained latent resistance coordinates\n"
            "  c, phi         latent parameters derived at the fixed reference stress;\n"
            "                 NOT recovered fields unless anchored to measurements\n"
            "  Fs, a_y, D_n   physically computed quantities\n"
            "  I_D            displacement-based instability index, NOT a probability\n"
            "  p_occ          calibrated landslide-occurrence probability"
        )
