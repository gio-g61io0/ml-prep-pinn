"""
Prespecified ablation study (Sec. 16).

All variants share the outer folds, so paired fold differences are valid and
statistical significance can be separated from practical importance.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import PEGPNConfig
from .cv import CVResult, run_nested_cv
from .data import PreparedData
from .metrics import paired_fold_test

ABLATIONS: Dict[str, Dict[str, object]] = {
    "PEGPN-full":            {},
    "PEGPN-no-psi":          {"use_psi": False},
    "PEGPN-no-prior":        {"use_priors": False},
    "PEGPN-no-anchor":       {"use_anchor": False},
    "PEGPN-hard-only":       {"smooth_training": False},
    "PEGPN-no-water":        {"use_water": False},
    "PEGPN-no-embeddings":   {"use_embeddings": False},
    "PEGPN-PGA-in-encoder":  {"pga_in_encoder": True},
}


def make_variant(cfg: PEGPNConfig, overrides: Dict[str, object]) -> PEGPNConfig:
    new = copy.deepcopy(cfg)
    for k, v in overrides.items():
        if not hasattr(new, k):
            raise AttributeError(f"PEGPNConfig has no field '{k}'")
        setattr(new, k, v)
    return new


def run_ablations(data: PreparedData, cfg: PEGPNConfig,
                  names: Optional[Sequence[str]] = None,
                  verbose: bool = True) -> Dict[str, CVResult]:
    names = list(names) if names is not None else list(ABLATIONS)
    if data.anchor_mask is None and "PEGPN-no-anchor" in names:
        names.remove("PEGPN-no-anchor")  # nothing to ablate without measurements

    results: Dict[str, CVResult] = {}
    for name in names:
        if verbose:
            print(f"\n=== {name} ===")
        variant = make_variant(cfg, ABLATIONS[name])
        results[name] = run_nested_cv(data, variant, verbose=verbose,
                                      keep_last_models=False)
    return results


def ablation_table(results: Dict[str, CVResult], reference: str = "PEGPN-full",
                   metric: str = "roc_auc") -> pd.DataFrame:
    ref = results[reference].fold_metrics.sort_values("fold")[metric].to_numpy()
    rows = []
    for name, res in results.items():
        vals = res.fold_metrics.sort_values("fold")[metric].to_numpy()
        rec = {"variant": name, f"mean_{metric}": float(np.nanmean(vals)),
               f"sd_{metric}": float(np.nanstd(vals)),
               "mean_pr_auc": float(np.nanmean(res.fold_metrics["pr_auc"])),
               "mean_brier": float(np.nanmean(res.fold_metrics["brier"])),
               "mean_ece": float(np.nanmean(res.fold_metrics["ece"]))}
        if name != reference:
            t = paired_fold_test(vals, ref)
            rec.update({"delta_vs_ref": t["mean_diff"], "wilcoxon_p": t["wilcoxon_p"],
                        "sign_p": t["sign_p"], "n_wins": t["n_wins"]})
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(f"mean_{metric}", ascending=False)


def prior_sensitivity(data: PreparedData, cfg: PEGPNConfig,
                      psi0_grid: Sequence[float] = (0.15, 0.30, 0.50),
                      widen_grid: Sequence[float] = (1.0, 1.5),
                      lambda_grid: Sequence[float] = (0.5, 1.0, 2.0),
                      verbose: bool = True) -> pd.DataFrame:
    """Repeat fitting across plausible psi0, band widths and penalty weights."""
    rows = []
    for psi0 in psi0_grid:
        for widen in widen_grid:
            for lam in lambda_grid:
                v = copy.deepcopy(cfg)
                v.physics.psi0 = psi0
                v.train.lambda_prior = lam
                res = run_nested_cv(data, v, verbose=False, keep_last_models=False,
                                    prior_widen=widen)
                fm = res.fold_metrics
                rows.append({"psi0": psi0, "widen": widen, "lambda_prior": lam,
                             "roc_auc": float(fm["roc_auc"].mean()),
                             "pr_auc": float(fm["pr_auc"].mean()),
                             "brier": float(fm["brier"].mean()),
                             "mean_c_kPa": float(np.nanmean(res.oof["c"])),
                             "mean_phi_deg": float(np.nanmean(res.oof["phi_deg"])),
                             "mean_Fs": float(np.nanmean(res.oof["Fs"]))})
                if verbose:
                    print(f"psi0={psi0} widen={widen} lambda={lam} -> "
                          f"AUC {rows[-1]['roc_auc']:.4f}, mean c {rows[-1]['mean_c_kPa']:.2f} kPa")
    return pd.DataFrame(rows)
