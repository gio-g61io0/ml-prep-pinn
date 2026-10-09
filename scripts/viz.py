"""
Cartographic and diagnostic plotting.

Maps of latent parameters (c, phi, R0) are labelled as *prior-constrained latent
fields*, not recovered measurements, and the factor-of-safety map is drawn with a
diverging scale centred on Fs = 1 so the static-instability boundary is legible.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm

from . import physics as P

NAN_COLOR = "#d9d9d9"


def _prep_gdf(gdf, simplify_tolerance: Optional[float] = None):
    if simplify_tolerance:
        g = gdf.copy()
        g["geometry"] = g.geometry.simplify(simplify_tolerance, preserve_topology=False)
        return g
    return gdf


def map_field(gdf, values: np.ndarray, title: str, *, cmap: str = "viridis",
              label: str = "", ax=None, robust: Tuple[float, float] = (2, 98),
              norm=None, simplify_tolerance: Optional[float] = None,
              edgecolor: str = "none", figsize=(8, 8)):
    """Choropleth of one field; non-finite values are drawn in neutral grey."""
    g = _prep_gdf(gdf, simplify_tolerance).copy()
    v = np.asarray(values, dtype=float)
    g["_v"] = v
    finite = np.isfinite(v)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if norm is None and finite.any():
        lo, hi = np.nanpercentile(v[finite], robust)
        if hi <= lo:
            hi = lo + 1e-6
        norm = Normalize(vmin=lo, vmax=hi)

    if (~finite).any():
        g.loc[~finite].plot(ax=ax, color=NAN_COLOR, linewidth=0, edgecolor=edgecolor)
    g.loc[finite].plot(ax=ax, column="_v", cmap=cmap, norm=norm, linewidth=0,
                       edgecolor=edgecolor)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(label or title)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    return ax


def map_geotechnical_panel(gdf, pred: Dict[str, np.ndarray], *,
                           simplify_tolerance: Optional[float] = None,
                           figsize=(15, 14), suptitle: str = "PEGPN latent geotechnical fields"):
    """Cohesion, friction angle, factor of safety and reference shear resistance."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    map_field(gdf, pred["c"], "Latent cohesion $c$", cmap="YlOrBr",
              label="c [kPa]", ax=axes[0, 0], simplify_tolerance=simplify_tolerance)
    map_field(gdf, pred["phi_deg"], r"Latent friction angle $\varphi$", cmap="BuPu",
              label=r"$\varphi$ [deg]", ax=axes[0, 1], simplify_tolerance=simplify_tolerance)

    fs = np.asarray(pred["Fs"], dtype=float)
    hi = float(np.nanpercentile(fs[np.isfinite(fs)], 98)) if np.isfinite(fs).any() else 2.0
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=max(hi, 1.05))
    map_field(gdf, fs, "Factor of safety $F_s$ (static)", cmap="RdYlGn",
              label="$F_s$  (red: $F_s\\leq1$)", ax=axes[1, 0], norm=norm,
              simplify_tolerance=simplify_tolerance)

    map_field(gdf, pred["R0"], "Reference shear resistance $R_0$", cmap="cividis",
              label=r"$R_0$ [kPa] at $\sigma'_0$", ax=axes[1, 1],
              simplify_tolerance=simplify_tolerance)

    fig.suptitle(suptitle + "\n(c, $\\varphi$ and $R_0$ are prior-constrained latent "
                 "parameters, not measured fields)", fontsize=12)
    fig.tight_layout()
    return fig, axes


def map_susceptibility_panel(gdf, pred: Dict[str, np.ndarray], *,
                             simplify_tolerance: Optional[float] = None,
                             figsize=(15, 14)):
    """Occurrence probability, Newmark displacement, instability index and regime."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    map_field(gdf, pred["p"], "Calibrated occurrence probability $p_{occ}$",
              cmap="magma", label="probability", ax=axes[0, 0],
              norm=Normalize(0, 1), simplify_tolerance=simplify_tolerance)

    D = np.asarray(pred["D_hard_cm"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        logD = np.log10(np.where(D > 0, D, np.nan))
    map_field(gdf, logD, "Newmark displacement $D_n$ (exact operator)", cmap="inferno",
              label=r"$\log_{10} D_n$ [cm]  (grey: $D_n=0$ or static)",
              ax=axes[0, 1], simplify_tolerance=simplify_tolerance)

    map_field(gdf, pred["ID"], "Displacement-based instability index $I_D$",
              cmap="cividis", label="$I_D$ (not a probability)", ax=axes[1, 0],
              norm=Normalize(0, 1), simplify_tolerance=simplify_tolerance)

    map_regimes(gdf, pred["regime"], ax=axes[1, 1], simplify_tolerance=simplify_tolerance)
    fig.tight_layout()
    return fig, axes


def map_regimes(gdf, regime: np.ndarray, ax=None, simplify_tolerance: Optional[float] = None,
                figsize=(8, 8)):
    g = _prep_gdf(gdf, simplify_tolerance).copy()
    g["_r"] = np.asarray(regime, dtype=float)
    colors = ["#b2182b", "#ef8a62", "#2166ac"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    g.plot(ax=ax, column="_r", cmap=cmap, norm=norm, linewidth=0)
    handles = [mpl.patches.Patch(color=c, label=P.REGIME_NAMES[k])
               for k, c in zip((P.STATIC, P.YIELDING, P.NON_YIELDING), colors)]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True)
    ax.set_title("Physical regime", fontsize=11)
    ax.set_axis_off()
    return ax


def map_uncertainty(gdf, mc, key: str = "p", ax=None,
                    simplify_tolerance: Optional[float] = None, figsize=(8, 8)):
    lo, md, hi = mc.band(key)
    full = np.full(len(gdf), np.nan)
    full[mc.idx] = hi - lo
    return map_field(gdf, full, f"Propagated 90% interval width of {key}",
                     cmap="plasma", label="interval width", ax=ax,
                     simplify_tolerance=simplify_tolerance, figsize=figsize)


# --------------------------------------------------------------------------
# non-spatial diagnostics
# --------------------------------------------------------------------------
def plot_reliability(y, p, n_bins: int = 10, ax=None, label: str = "PEGPN"):
    from .metrics import reliability_curve
    xs, ys, ns = reliability_curve(y, p, n_bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", label=label)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed landslide frequency")
    ax.set_title("Reliability (out-of-fold)")
    ax.legend(fontsize=8)
    return ax


def plot_roc_pr(results: Dict[str, object], y_key: str = "y", p_key: str = "p_cal",
                figsize=(11, 4.5)):
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for name, res in results.items():
        y = res.oof[y_key].to_numpy()
        p = res.oof[p_key].to_numpy()
        ok = np.isfinite(p)
        fpr, tpr, _ = roc_curve(y[ok], p[ok])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        pr, rc, _ = precision_recall_curve(y[ok], p[ok])
        axes[1].plot(rc, pr, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC (out-of-fold)"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig, axes


def plot_parameter_distributions(pred: Dict[str, np.ndarray], figsize=(13, 3.4)):
    keys = [("c", "cohesion $c$ [kPa]"), ("phi_deg", r"friction $\varphi$ [deg]"),
            ("R0", "$R_0$ [kPa]"), ("Fs", "$F_s$")]
    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    for ax, (k, lab) in zip(axes, keys):
        v = np.asarray(pred[k], dtype=float)
        v = v[np.isfinite(v)]
        ax.hist(v, bins=60, color="#4c72b0")
        if k == "Fs":
            ax.axvline(1.0, color="crimson", lw=1.2, label="$F_s=1$")
            ax.legend(fontsize=8)
        ax.set_xlabel(lab); ax.set_ylabel("slope units")
    fig.tight_layout()
    return fig, axes


def export_fields(gdf, fields: Dict[str, np.ndarray], path: str, layer: str = "pegpn"):
    """Write the predicted fields alongside the geometry for GIS use."""
    out = gdf[["geometry"]].copy()
    for k, v in fields.items():
        out[k] = np.asarray(v)
    driver = "GPKG" if str(path).lower().endswith(".gpkg") else None
    out.to_file(path, layer=layer, driver=driver) if driver else out.to_file(path)
    return path
