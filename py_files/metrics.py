import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import contextily as cx
import seaborn as sns
from .data import dataframe_to_dataset
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, brier_score_loss, precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds

def export_to_geopackage(gdf, layer_values, output_path):
    """Attach value columns to a GeoDataFrame and write to GeoPackage.

    Parameters
    ----------
    gdf : GeoDataFrame
        Slope-unit polygons (must have geometry and CRS).
    layer_values : dict
        Mapping of ``{column_name: 1D array}`` where each array is aligned
        with *gdf* rows (e.g. ``{"cohesion_kpa": cohesion, ...}``).
    output_path : str or Path
        Destination ``.gpkg`` file path.
    """
    out = gdf.copy()
    for col_name, arr in layer_values.items():
        out[col_name] = np.asarray(arr, dtype=np.float64)
    out.to_file(output_path, driver="GPKG")
    print(f"  Wrote {output_path}  ({len(out)} features, {list(layer_values.keys())})")


def rasterize_to_geotiff(gdf, values, output_path, pixel_size=30.0, nodata=-9999.0, layer_name=None):
    """Burn polygon values into a GeoTIFF raster.

    Parameters
    ----------
    gdf : GeoDataFrame
        Slope-unit polygons (must have a CRS set).
    values : array-like
        1-D array of values aligned with *gdf* rows.
    output_path : str or Path
        Destination .tif file path.
    pixel_size : float
        Output resolution in CRS units (default 30 m).
    nodata : float
        NoData fill value.
    layer_name : str, optional
        Band description written into the GeoTIFF metadata.
    """
    values = np.asarray(values, dtype=np.float64)

    # Filter out empty / null geometries
    mask = gdf.geometry.notnull() & ~gdf.geometry.is_empty
    gdf_valid = gdf.loc[mask]
    values_valid = values[mask.values]

    minx, miny, maxx, maxy = gdf_valid.total_bounds
    width = max(1, int(np.ceil((maxx - minx) / pixel_size)))
    height = max(1, int(np.ceil((maxy - miny) / pixel_size)))

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    shapes = list(zip(gdf_valid.geometry, values_valid))

    raster = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype="float64",
        all_touched=False,
    )

    profile = {
        "driver": "GTiff",
        "dtype": "float64",
        "width": width,
        "height": height,
        "count": 1,
        "crs": gdf_valid.crs,
        "transform": transform,
        "nodata": nodata,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(raster, 1)
        if layer_name:
            dst.set_band_description(1, layer_name)

    print(f"  Wrote {output_path}  ({width}x{height} px)")


def create_slope_unit_template(gdf, output_path, pixel_size=30.0, nodata=-1):
    """Create a slope-unit ID raster from polygons.

    Each pixel is assigned the row index (0-based) of the slope unit it
    falls within.  This raster serves as a reusable template so that all
    intermediate-output TIFs share the exact same grid, extent, and
    slope-unit boundaries.

    Parameters
    ----------
    gdf : GeoDataFrame
        Slope-unit polygons (must have a CRS set).
    output_path : str or Path
        Destination .tif file for the template raster.
    pixel_size : float
        Output resolution in CRS units (default 30 m).
    nodata : int
        NoData fill value (default -1).

    Returns
    -------
    str
        The *output_path* that was written, for convenience.
    """
    mask = gdf.geometry.notnull() & ~gdf.geometry.is_empty
    gdf_valid = gdf.loc[mask]

    minx, miny, maxx, maxy = gdf_valid.total_bounds
    width = max(1, int(np.ceil((maxx - minx) / pixel_size)))
    height = max(1, int(np.ceil((maxy - miny) / pixel_size)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Burn the positional index of each slope unit into the raster
    shapes = [(geom, idx) for idx, geom in enumerate(gdf_valid.geometry)]

    id_raster = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype="int32",
        all_touched=False,
    )

    profile = {
        "driver": "GTiff",
        "dtype": "int32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": gdf_valid.crs,
        "transform": transform,
        "nodata": nodata,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(id_raster, 1)
        dst.set_band_description(1, "Slope Unit ID")

    print(f"  Wrote slope-unit template {output_path}  ({width}x{height} px, "
          f"{len(gdf_valid)} units)")
    return output_path


def rasterize_from_template(template_path, values, output_path,
                            nodata=-9999.0, layer_name=None):
    """Create a value GeoTIFF by mapping slope-unit IDs to values.

    Reads the slope-unit ID template raster produced by
    ``create_slope_unit_template`` and replaces each ID with the
    corresponding value from *values*.

    Parameters
    ----------
    template_path : str or Path
        Path to the slope-unit ID template raster.
    values : array-like
        1-D array of values indexed by slope-unit ID.
    output_path : str or Path
        Destination .tif file path.
    nodata : float
        NoData fill value for pixels outside any slope unit.
    layer_name : str, optional
        Band description written into the GeoTIFF metadata.
    """
    values = np.asarray(values, dtype=np.float64)

    with rasterio.open(template_path) as src:
        id_raster = src.read(1)
        template_nodata = src.nodata
        profile = src.profile.copy()

    # Build output: map each slope-unit ID to its value
    out_raster = np.full(id_raster.shape, nodata, dtype=np.float64)
    valid_mask = id_raster != template_nodata
    ids = id_raster[valid_mask]
    out_raster[valid_mask] = values[ids]

    profile.update(dtype="float64", nodata=nodata)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_raster, 1)
        if layer_name:
            dst.set_band_description(1, layer_name)

    print(f"  Wrote {output_path}  ({profile['width']}x{profile['height']} px)")


## ---------------------------------------------------------------------------
#  Statistical analysis & validation utilities
# ---------------------------------------------------------------------------

def plot_calibration(y_true, y_pred_probs, n_bins=10, title="Calibration Plot"):
    """Reliability diagram: predicted probability vs observed frequency."""
    fraction_pos, mean_pred = calibration_curve(y_true, y_pred_probs, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_pred_probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_pred, fraction_pos, "s-", label=f"Model (Brier={brier:.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return brier


def plot_precision_recall(y_true, y_pred_probs, title="Precision-Recall Curve"):
    """Precision-Recall curve with AUPR."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    aupr = average_precision_score(y_true, y_pred_probs)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color="purple", label=f"AUPR={aupr:.3f}")
    baseline = y_true.sum() / len(y_true)
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Baseline={baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    return aupr


def success_rate_curve(y_true, y_pred_probs, title="Success Rate Curve"):
    """Cumulative % of observed landslides captured vs % area (sorted by descending susceptibility)."""
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs).flatten()

    order = np.argsort(-y_pred_probs)
    sorted_labels = y_true[order]

    cum_landslides = np.cumsum(sorted_labels)
    total_landslides = sorted_labels.sum()
    pct_landslides = cum_landslides / total_landslides if total_landslides > 0 else cum_landslides
    pct_area = np.arange(1, len(y_true) + 1) / len(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pct_area * 100, pct_landslides * 100, color="red", lw=2)
    ax.plot([0, 100], [0, 100], "k--", label="Random model")
    ax.set_xlabel("% of study area (sorted by susceptibility)")
    ax.set_ylabel("% of observed landslides captured")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Report key thresholds
    for pct in [10, 20, 30, 50]:
        idx = int(len(y_true) * pct / 100) - 1
        if idx >= 0:
            print(f"  Top {pct}% area captures {pct_landslides[idx]*100:.1f}% of landslides")

    return pct_area, pct_landslides


def landslide_density_by_class(y_true, y_pred_probs, gdf=None,
                               bins=None, labels=None):
    """Observed landslide count (and density if areas available) per susceptibility bin."""
    if bins is None:
        bins = [0, 0.125, 0.375, 0.625, 0.875, 1.0]
    if labels is None:
        labels = ["Very Low", "Low", "Moderate", "High", "Very High"]

    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs).flatten()
    bin_idx = np.digitize(y_pred_probs, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(labels) - 1)

    rows = []
    for i, label in enumerate(labels):
        mask = bin_idx == i
        n_units = mask.sum()
        n_ls = y_true[mask].sum() if n_units > 0 else 0
        area_km2 = None
        if gdf is not None and n_units > 0:
            area_km2 = gdf.loc[mask, "geometry"].area.sum() / 1e6
        rows.append({
            "class": label,
            "n_slope_units": int(n_units),
            "n_landslides": int(n_ls),
            "ls_ratio": n_ls / n_units if n_units > 0 else 0,
            "area_km2": area_km2,
        })
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    return result


def _default_class_labels(n):
    """Standard 5-class susceptibility labels, or generic C1..Cn otherwise."""
    if n == 5:
        return ["Very Low", "Low", "Moderate", "High", "Very High"]
    return [f"C{i + 1}" for i in range(n)]


def _quantile_area_edges(preds, weight, n_classes):
    """Area-weighted quantile (equal-area) class edges over predictions in [0, 1].

    Sorts units by susceptibility and places interior edges where the cumulative
    area share crosses k/n_classes, so each class holds ~1/n_classes of the total
    area. Returns a strictly increasing edge list anchored at 0.0 and 1.0.
    Duplicate edges (collapsed by a value gap in a bimodal distribution) are
    dropped, which may yield fewer than n_classes classes.
    """
    order = np.argsort(preds)
    sorted_preds = preds[order]
    cum_area = np.cumsum(weight[order]) / weight.sum()
    interior_q = np.linspace(0.0, 1.0, n_classes + 1)[1:-1]
    interior_edges = np.interp(interior_q, cum_area, sorted_preds)
    edges = np.unique(np.concatenate([[0.0], interior_edges, [1.0]]))
    return [float(e) for e in edges]


def frequency_ratio_table(gdf, predictions, label_col="landslide",
                          model_name=None, bins=None, labels=None,
                          area_based=True, scheme="fixed", n_classes=5):
    """Inventory-vs-susceptibility linkage table (reviewer "Table 3", §4.4).

    Spatially overlays the landslide inventory (``label_col`` ground truth) on
    the model's susceptibility classes and reports, per class:

    - ``pct_area``         : share of study area in this susceptibility class
    - ``n_landslides``     : number of inventory slope units in this class
    - ``pct_landslides``   : share of all inventory landslides in this class
    - ``freq_ratio``       : Frequency Ratio = pct_landslides / pct_area
                             (>1 = landslides over-represented; a good model
                             rises monotonically Very Low -> Very High)
    - ``cum_hit_rate``     : cumulative % of inventory captured, accumulating
                             from the highest class downward

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain ``geometry`` (for area) and ``label_col``. Use a projected
        CRS (e.g. EPSG:3857 / UTM) so areas are metric.
    predictions : array-like
        Per-unit predicted susceptibility in [0, 1], aligned to ``gdf`` rows.
    area_based : bool
        If True, ``pct_area`` / ``pct_landslides`` are weighted by polygon area
        (rigorous for variable-size slope units). If False, uses unit counts.
    scheme : {"fixed", "quantile"}
        Class edges used when ``bins`` is None. "fixed" = value breakpoints
        [0, .125, .375, .625, .875, 1]; "quantile" = area-weighted equal-area
        classes with data-driven edges. See docs/frequency_ratio_classing.md.
    n_classes : int
        Number of quantile classes when ``scheme="quantile"`` (default 5).

    Returns
    -------
    DataFrame
        One row per susceptibility class (ordered high -> low so ``cum_hit_rate``
        reads top-down), with an optional leading ``model`` column.
    """
    preds = np.asarray(predictions).flatten()
    y_true = gdf[label_col].to_numpy()
    if len(preds) != len(gdf):
        raise ValueError(
            f"predictions ({len(preds)}) and gdf ({len(gdf)}) length mismatch"
        )

    # Per-unit weight: polygon area (metric) or 1 (count-based).
    if area_based:
        weight = gdf.geometry.area.to_numpy()
    else:
        weight = np.ones(len(gdf))

    # Assign each unit to a susceptibility class. See docs/frequency_ratio_classing.md.
    #   - explicit `bins` / "fixed" / area-weighted "quantile": value-edge binning.
    #   - count-based "quantile": RANK-based equal-count classes (robust to ties /
    #     saturated outputs, where value-edge quantiles collapse).
    edges = None
    if bins is not None:
        edges = list(bins)
    elif scheme == "fixed":
        edges = [0, 0.125, 0.375, 0.625, 0.875, 1.0]
    elif scheme == "quantile" and area_based:
        edges = _quantile_area_edges(preds, weight, n_classes)
    elif scheme == "quantile":
        # rank ties broken by order -> exactly n_classes equal-count bins.
        ranks = pd.Series(preds).rank(method="first").to_numpy()
        bin_idx = pd.qcut(ranks, q=n_classes, labels=False).astype(int)
    else:
        raise ValueError(f"unknown scheme {scheme!r}; use 'fixed' or 'quantile'")

    if edges is not None:
        bin_idx = np.clip(np.digitize(preds, edges) - 1, 0, len(edges) - 2)

    n_cls = int(bin_idx.max()) + 1
    if labels is None:
        labels = _default_class_labels(n_cls)
    elif len(labels) != n_cls:
        print(f"  warning: classing produced {n_cls} populated class(es) but "
              f"{len(labels)} labels given (likely tie/saturation collapse); relabeling")
        labels = _default_class_labels(n_cls)

    total_w = weight.sum()
    total_ls_w = weight[y_true == 1].sum()

    rows = []
    for i, label in enumerate(labels):
        mask = bin_idx == i
        ls_mask = mask & (y_true == 1)
        class_w = weight[mask].sum()
        ls_w = weight[ls_mask].sum()
        pct_area = class_w / total_w if total_w > 0 else 0.0
        pct_ls = ls_w / total_ls_w if total_ls_w > 0 else 0.0
        # Observed susceptibility span of the class (honest under tie-split boundaries).
        lo = float(preds[mask].min()) if mask.any() else float("nan")
        hi = float(preds[mask].max()) if mask.any() else float("nan")
        rows.append({
            "class": label,
            "sus_range": f"[{lo:.3f}, {hi:.3f}]",
            "n_slope_units": int(mask.sum()),
            "area_km2": class_w / 1e6,
            "pct_area": pct_area,
            "n_landslides": int((ls_mask).sum()),
            "pct_landslides": pct_ls,
            "freq_ratio": pct_ls / pct_area if pct_area > 0 else np.nan,
        })

    result = pd.DataFrame(rows)
    # Order high -> low so cumulative hit-rate reads top-down.
    result = result.iloc[::-1].reset_index(drop=True)
    result["cum_pct_area"] = result["pct_area"].cumsum()
    result["cum_hit_rate"] = result["pct_landslides"].cumsum()

    if model_name is not None:
        result.insert(0, "model", model_name)

    result.attrs["bins"] = list(edges) if edges is not None else None
    result.attrs["scheme"] = scheme

    edge_desc = ("rank-tertiles" if edges is None
                 else "[" + ", ".join(f"{b:.4f}" for b in edges) + "]")
    print(f"[{model_name or 'model'}] scheme={scheme!r}  classes={n_cls}  edges: {edge_desc}")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(result.to_string(index=False))
    return result


def frequency_ratio_summary(model_results, high_label="High"):
    """Compact multi-model Table 3: one row per model from per-class FR tables.

    Pure reshape of `frequency_ratio_table()` outputs into the manuscript "Table 3"
    layout (inventory distribution across classes + hit rate and FR for the High
    class). No new metric logic.

    Parameters
    ----------
    model_results : dict[str, DataFrame]
        Maps model name -> the DataFrame returned by `frequency_ratio_table(...)`.
        Each frame must contain columns `class`, `pct_landslides`, `pct_area`,
        `freq_ratio`, `n_slope_units`.
    high_label : str
        The class label treated as "High" for the hit-rate / FR columns.

    Returns
    -------
    DataFrame
        One row per model: a `<class> (%)` column per susceptibility class (inventory
        distribution), plus `hit_rate_high`, `fr_high`, `n_high`, `n_total`.
        Percentages are 0-100. NaN in `fr_high` flags a collapsed/degenerate High
        class (e.g. quantile classing on saturated predictions).
    """
    rows = []
    for name, table in model_results.items():
        if table is None or high_label not in set(table["class"]):
            rows.append({"model": name, "hit_rate_high": np.nan, "fr_high": np.nan,
                         "n_high": 0, "n_total": int(table["n_slope_units"].sum())
                         if table is not None else 0})
            continue
        row = {"model": name}
        for _, r in table.iterrows():
            row[f"{r['class']} (%)"] = 100.0 * r["pct_landslides"]
        high = table[table["class"] == high_label].iloc[0]
        row["hit_rate_high"] = 100.0 * high["pct_landslides"]
        row["fr_high"] = high["freq_ratio"]
        row["n_high"] = int(high["n_slope_units"])
        row["n_total"] = int(table["n_slope_units"].sum())
        rows.append(row)

    summary = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(summary.to_string(index=False))
    return summary


def plot_intermediate_correlation(intermediates, method="spearman",
                                  title="Intermediate Parameter Correlation"):
    """Correlation heatmap of intermediate physics outputs and selected inputs.

    Parameters
    ----------
    intermediates : dict
        ``{name: 1-D array}`` — e.g. cohesion, ifi, fos, displacement, slope, pga.
    """
    corr_df = pd.DataFrame(intermediates).corr(method=method)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax)
    ax.set_title(f"{title} ({method})")
    plt.tight_layout()
    plt.show()
    return corr_df


def plot_geotech_by_soil_type(values, soil_labels, value_name="Cohesion (kPa)",
                              lit_ranges=None):
    """Boxplot of a geotechnical parameter grouped by soil type.

    Parameters
    ----------
    values : array-like
        1-D predicted values (e.g. cohesion).
    soil_labels : array-like
        Soil type name per sample (same length as *values*).
    lit_ranges : dict, optional
        ``{soil_name: (min, max)}`` from literature for overlay.
    """
    tmp = pd.DataFrame({"value": np.asarray(values).flatten(),
                        "soil": np.asarray(soil_labels)})
    order = tmp.groupby("soil")["value"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=tmp, x="soil", y="value", order=order, ax=ax)

    if lit_ranges:
        for i, soil in enumerate(order):
            if soil in lit_ranges:
                lo, hi = lit_ranges[soil]
                ax.hlines([lo, hi], i - 0.4, i + 0.4,
                          colors="red", linestyles="dashed", linewidth=1)

    ax.set_ylabel(value_name)
    ax.set_xlabel("Soil Type")
    ax.set_title(f"{value_name} by Soil Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_fold_stability(fold_values, param_name="Cohesion (kPa)"):
    """Overlapping KDE plots comparing a parameter across folds.

    Parameters
    ----------
    fold_values : list of arrays
        Each element is the 1-D prediction array from one fold model.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, vals in enumerate(fold_values):
        sns.kdeplot(vals.flatten(), ax=ax, label=f"Fold {i+1}", fill=True, alpha=0.2)
    ax.set_xlabel(param_name)
    ax.set_title(f"Cross-fold stability: {param_name}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def fold_ensemble_uncertainty(fold_predictions):
    """Compute mean and std across fold predictions.

    Parameters
    ----------
    fold_predictions : list of arrays
        Each element is the 1-D susceptibility array from one fold model.

    Returns
    -------
    mean_pred, std_pred : arrays
    """
    stacked = np.stack([p.flatten() for p in fold_predictions], axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


## ---------------------------------------------------------------------------
#  Incomplete-inventory validation: false-positive characterization
# ---------------------------------------------------------------------------

def classify_predictions(y_true, y_pred_probs, threshold=0.5):
    """Partition samples into TP, FP, TN, FN groups.

    Returns a 1-D array of category labels aligned with input arrays.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred_probs).flatten()
    cats = np.empty(len(y_true), dtype=object)
    cats[(y_true == 1) & (y_pred >= threshold)] = "TP"
    cats[(y_true == 0) & (y_pred >= threshold)] = "FP"
    cats[(y_true == 1) & (y_pred < threshold)]  = "FN"
    cats[(y_true == 0) & (y_pred < threshold)]  = "TN"
    return cats


def false_positive_analysis(gdf, y_pred_probs, feature_cols, threshold=0.5,
                            label_col="landslide"):
    """Compare feature distributions of TP vs FP vs TN.

    If false positives share geomorphological characteristics with true
    positives (and differ from true negatives), the model is likely
    identifying genuinely susceptible areas that are missing from an
    incomplete inventory.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain *label_col* and all *feature_cols*.
    y_pred_probs : array-like
        Predicted susceptibility (0-1).
    feature_cols : list of str
        Features to compare (e.g. Slope_mean, Elev_mean, Prc_mean, …).
    threshold : float
        Classification threshold.

    Returns
    -------
    summary : DataFrame
        Mean feature values per group (TP, FP, TN, FN).
    cats : array
        Per-sample category labels.
    """
    y_true = gdf[label_col].values
    cats = classify_predictions(y_true, y_pred_probs, threshold)

    tmp = gdf[feature_cols].copy()
    tmp["_group"] = cats
    summary = tmp.groupby("_group")[feature_cols].agg(["mean", "std"])

    # Print concise table
    means = tmp.groupby("_group")[feature_cols].mean()
    counts = tmp["_group"].value_counts()
    print("Sample counts per group:")
    print(counts.to_string())
    print("\nMean feature values by group:")
    print(means.round(4).to_string())
    return summary, cats


def plot_fp_vs_tp_distributions(gdf, cats, feature_cols, ncols=3):
    """KDE plots comparing TP, FP, TN distributions for key features.

    Overlapping TP and FP distributions (both distinct from TN) indicate
    the model is generalizing to genuinely susceptible unlabeled areas.
    """
    groups_to_plot = ["TP", "FP", "TN"]
    colors = {"TP": "red", "FP": "orange", "TN": "steelblue", "FN": "gray"}

    n = len(feature_cols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    axes_flat = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes_flat[i]
        for grp in groups_to_plot:
            mask = cats == grp
            if mask.sum() == 0:
                continue
            vals = gdf.loc[mask, col].dropna()
            if len(vals) > 1:
                sns.kdeplot(vals, ax=ax, label=grp, color=colors[grp],
                            fill=True, alpha=0.2)
        ax.set_title(col)
        ax.legend(fontsize=8)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions: TP vs FP vs TN", fontsize=14)
    fig.tight_layout()
    plt.show()


def fp_tp_statistical_tests(gdf, cats, feature_cols):
    """Mann-Whitney U tests comparing FP vs TP and FP vs TN for each feature.

    A non-significant FP-vs-TP test (high p-value) means false positives
    are statistically similar to true positives — evidence that the model
    is finding real susceptible areas missing from the inventory.

    A significant FP-vs-TN test (low p-value) means false positives are
    distinct from true negatives — they aren't random noise.

    Returns a DataFrame with test statistics and p-values.
    """
    from scipy.stats import mannwhitneyu

    rows = []
    for col in feature_cols:
        tp_vals = gdf.loc[cats == "TP", col].dropna()
        fp_vals = gdf.loc[cats == "FP", col].dropna()
        tn_vals = gdf.loc[cats == "TN", col].dropna()

        fp_tp_stat, fp_tp_p = (np.nan, np.nan)
        fp_tn_stat, fp_tn_p = (np.nan, np.nan)

        if len(fp_vals) > 0 and len(tp_vals) > 0:
            fp_tp_stat, fp_tp_p = mannwhitneyu(fp_vals, tp_vals, alternative="two-sided")
        if len(fp_vals) > 0 and len(tn_vals) > 0:
            fp_tn_stat, fp_tn_p = mannwhitneyu(fp_vals, tn_vals, alternative="two-sided")

        rows.append({
            "feature": col,
            "FP_vs_TP_U": fp_tp_stat,
            "FP_vs_TP_p": fp_tp_p,
            "FP_similar_to_TP": "Yes" if fp_tp_p > 0.05 else "No",
            "FP_vs_TN_U": fp_tn_stat,
            "FP_vs_TN_p": fp_tn_p,
            "FP_differs_from_TN": "Yes" if fp_tn_p < 0.05 else "No",
        })

    result = pd.DataFrame(rows)
    print("\nMann-Whitney U tests (FP characterization):")
    print(result.to_string(index=False))
    print("\nInterpretation:")
    print("  FP similar to TP (p>0.05)  = model finds areas like known landslides")
    print("  FP differs from TN (p<0.05) = false positives are NOT random noise")
    return result


def plot_fp_map(gdf, cats, title="Spatial Distribution of Prediction Groups"):
    """Map showing TP, FP, TN, FN locations."""
    color_map = {"TP": "red", "FP": "orange", "TN": "lightblue", "FN": "gray"}
    plot_order = ["TN", "FN", "FP", "TP"]

    fig, ax = plt.subplots(figsize=(10, 10))
    for grp in plot_order:
        mask = cats == grp
        if mask.sum() == 0:
            continue
        gdf.loc[mask].plot(ax=ax, color=color_map[grp], label=grp, alpha=0.6)

    ax.legend(title="Group", loc="upper right")
    ax.set_title(title)
    ax.set_axis_off()
    try:
        cx.add_basemap(ax, crs=gdf.crs.to_string(),
                       source=cx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"  basemap unavailable: {e}")
    plt.tight_layout()
    plt.show()


def plot_fp_susceptibility_histogram(y_pred_probs, cats,
                                     title="Susceptibility Distribution by Group"):
    """Histogram of predicted susceptibility split by TP/FP/TN/FN."""
    colors = {"TP": "red", "FP": "orange", "TN": "steelblue", "FN": "gray"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for grp in ["TN", "FP", "TP"]:
        mask = cats == grp
        if mask.sum() == 0:
            continue
        ax.hist(np.asarray(y_pred_probs).flatten()[mask], bins=50,
                alpha=0.4, label=grp, color=colors[grp])
    ax.set_xlabel("Predicted Susceptibility")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(preds, test_y):
    y_pred_classes = (preds > 0.5).astype("int32")
    cm = confusion_matrix(test_y, y_pred_classes)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
def find_best_threshold(y_true, y_pred_probs):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred_probs)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh, fpr, tpr


def plot_distribution(df, title, x_label, y_label, label):
    sns.histplot(df[label], bins=30, kde=True, color="red")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_predicted_observed_map(gdf, predicted_col, observed_col):
    fig, axs = plt.subplots(1, 2, dpi=300, figsize=(8, 7))
    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)

    cbar = fig.colorbar(sm, ax=axs[0])
    cbar.set_ticks([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    cbar.set_ticklabels(["0.0", "0.125", "0.375", "0.625", "0.875", "1.0"])

    gdf.plot(column=observed_col, cmap='RdYlGn_r', norm=norm, ax=axs[0])
    cx.add_basemap(axs[0], source=cx.providers.CartoDB.Positron)
    axs[0].set_title("Observed/Landslide Inventory")
    axs[0].set_axis_off()



    cbar = fig.colorbar(sm, ax=axs[1])
    cbar.set_ticks([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    cbar.set_ticklabels(["0.0", "0.125", "0.375", "0.625", "0.875", "1.0"])

    gdf.plot(column=predicted_col, cmap='RdYlGn_r', norm=norm, ax=axs[1],legend_kwds={"label": "Predicted Susceptibility Map"},)
    cx.add_basemap(axs[1], source=cx.providers.CartoDB.Positron)
    axs[1].set_title("Predicted Susceptibility")
    axs[1].set_axis_off()
    plt.tight_layout()
    plt.show()

def _draw_barangay_labels(
    ax,
    brgy_boundaries,
    name_col="ADM4_EN",
    fontsize=3,
    min_sep_frac=0.035,
):
    """Draw decluttered barangay labels clipped to the current axes extent.

    The raw shapefile labels every barangay at its representative point in the
    same colour with no collision handling, so they pile into an unreadable
    mass. This keeps only barangays whose label point falls inside the visible
    map, then greedily drops any label that would sit within ``min_sep_frac`` of
    the map diagonal from a label already placed (largest barangays win, since
    they anchor the most area). A white halo keeps text legible over the map.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes whose current x/y limits define the visible extent.
    brgy_boundaries : GeoDataFrame
        Barangay polygons containing ``name_col``. May optionally carry a
        ``label_point`` geometry column; otherwise interior points are derived.
    name_col : str
        Column holding the barangay name to render.
    fontsize : int
        Label font size in points.
    min_sep_frac : float
        Minimum spacing between placed labels as a fraction of the map diagonal.
    """
    if name_col not in brgy_boundaries.columns:
        raise KeyError(f"barangay name column '{name_col}' not found")

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    diagonal = np.hypot(x_max - x_min, y_max - y_min)
    min_sep = diagonal * min_sep_frac

    has_label_point = "label_point" in brgy_boundaries.columns

    # Collect (name, point, area) only for barangays visible in the extent.
    candidates = []
    for _, row in brgy_boundaries.iterrows():
        point = row["label_point"] if has_label_point else row.geometry.representative_point()
        if not (x_min <= point.x <= x_max and y_min <= point.y <= y_max):
            continue
        name = row[name_col]
        if not isinstance(name, str) or not name.strip():
            continue
        candidates.append((name, point.x, point.y, row.geometry.area))

    # Largest barangays are placed first so they win ties for scarce label space.
    candidates.sort(key=lambda c: c[3], reverse=True)

    placed = []
    for name, px, py, _area in candidates:
        if any(np.hypot(px - qx, py - qy) < min_sep for qx, qy in placed):
            continue
        placed.append((px, py))
        text = ax.text(
            px, py, name,
            fontsize=fontsize,
            color="black",
            ha="center",
            va="center",
            zorder=5,
        )
        text.set_path_effects([
            path_effects.withStroke(linewidth=1.6, foreground="white"),
        ])

    return len(placed), len(candidates)


def plot_susceptibility_map(
    gdf,
    predictions,
    label_name,
    title="PINN Susceptibility Map",
    figsize=(10, 9),
    dpi=300,
    zoom="auto",
    save_path=None,
    brgy_boundaries = None,
    boundary_label_col="ADM4_EN",
):
    """Plot a slope-unit susceptibility map over a basemap.

    Resolution levers:
    - ``dpi``: raster resolution of the rendered figure (display + saved file).
      300 is print quality; bump to 600 for very large maps.
    - ``zoom``: contextily basemap tile zoom level. ``"auto"`` lets contextily
      choose; pass an int (e.g. 11-13) for a sharper basemap. Higher = more
      tiles downloaded = slower.
    - ``save_path``: if given, writes a high-res file (PNG/PDF/SVG by extension)
      with the same ``dpi``. Use ``.pdf``/``.svg`` for fully vector output.

    Admin overlay:
    - ``brgy_boundaries``: optional GeoDataFrame of admin polygons drawn as an
      overlay with decluttered labels. Pass municipal polygons (dissolved to
      ADM3) with ``boundary_label_col="ADM3_EN"`` for a coarser, cleaner map.
    - ``boundary_label_col``: attribute in ``brgy_boundaries`` used for labels.
    """
    gdf = gdf.copy()
    gdf['predicted_susceptibility'] = predictions
    norm = mcolors.Normalize(vmin=0, vmax=1.0)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Thin edges keep slope-unit boundaries crisp instead of bleeding together.
    gdf.plot(
        column='predicted_susceptibility',
        cmap='RdYlGn_r',
        ax=ax,
        norm=norm,
        edgecolor='none',
        antialiased=True,
    )

    # Add single custom colorbar
    sm = plt.cm.ScalarMappable(cmap='plasma_r', norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    cbar.set_ticklabels(["0.0", "0.125", "0.375", "0.625", "0.875", "1.0"])

    ax.set_title(f"{title} - {label_name}")
    if brgy_boundaries is not None:
        # Draw boundaries first so the axes reach their final extent, then
        # declutter labels against that extent (interior points, largest-first
        # greedy spacing, white halo) instead of stamping every barangay.
        brgy_boundaries.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
        n_placed, n_visible = _draw_barangay_labels(
            ax, brgy_boundaries, name_col=boundary_label_col,
        )
        print(f"  [labels] placed {n_placed}/{n_visible} boundary labels in view")


    cx.add_basemap(
        ax,
        crs=gdf.crs.to_string(),
        source=cx.providers.CartoDB.Positron,
        zoom=zoom,
    )
    plt.tight_layout()

    if save_path is not None:
        # LZW keeps TIFF output lossless but far smaller than uncompressed.
        save_kwargs = {}
        if str(save_path).lower().endswith(('.tif', '.tiff')):
            save_kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', **save_kwargs)

    plt.show()

def bootstrap_geotech(df, model, columns,filepath, n_bootstrap=50, ):

    data = dataframe_to_dataset(df[columns], shuffle=False)
    geotech_model_cohesion = tf.keras.Model(inputs=model.input, outputs=(model.get_layer("cohesion_layer").output * 5))
    geotech_model_ifi = tf.keras.Model(inputs=model.input, outputs=(model.get_layer("internal_friction").output))
    for i in range(1, n_bootstrap + 1):
        cohesion = geotech_model_cohesion.predict(data)
        ifi = geotech_model_ifi.predict(data)
        np.save(f"{filepath}/cohesion_bootstrap_{i}.npy", cohesion)
        np.save(f"{filepath}/ifi_bootstrap_{i}.npy", ifi)



def roc_auc_score_multiclass(actual_class, pred_class, average='macro'):
    unique_class = set(actual_class)
    roc_auc_dict = {}
    
    print(f"Pred class: {pred_class}")
    for per_class in unique_class:
        y_true = (actual_class == per_class).astype(int)

        y_score = pred_class[:, per_class]
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
        auc = sklearn.metrics.auc(fpr, tpr)
        roc_auc_dict[per_class] = [fpr, tpr, auc]

    return roc_auc_dict


def plot_auc(y_true, y_pred_probs):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred_probs)
    auc = sklearn.metrics.auc(fpr, tpr)
    acc = round(sklearn.metrics.balanced_accuracy_score(y_true, y_pred_probs > 0.5), 2)

    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.plot(fpr, tpr, color="blue", label=f"(AUC={auc:.2f}, Accuracy={acc})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

def plot_auc_with_distribution(y_true, y_pred_probs, bins=20):
    # ROC curve
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred_probs)
    auc_score = sklearn.metrics.auc(fpr, tpr)
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Plot ROC curve on left y-axis
    
    ax1.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC={auc_score:.2f})")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title("ROC Curve & Prediction Distribution")
    ax1.legend(loc="lower right")
    
    # Create a second y-axis for the histogram
    ax2 = ax1.twinx()
    
    # Histogram / bar plot of predicted probabilities
    ax2.hist(y_pred_probs, bins=bins, color="orange", alpha=0.3, label="Predicted Probabilities")
    ax2.set_ylabel("Count", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    
    # Optional: add legend for histogram
    ax2.legend(loc="upper center")
    
    plt.show()


def plot_auc_with_boxplot(y_true, y_pred_probs):
    # ROC curve
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred_probs)
    auc_score = sklearn.metrics.auc(fpr, tpr)
    
    # Main figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC={auc_score:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve with Prediction Distribution")
    ax.legend(loc="lower right")
    
    # Inset axes for boxplot
    ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper left")  # adjust as needed
    ax_inset.boxplot(y_pred_probs, vert=True)
    ax_inset.set_title("Prediction Susceptibility", fontsize=10)
    ax_inset.set_ylabel("Predicted Prob", fontsize=8)
    ax_inset.tick_params(axis='x', labelbottom=False)  # hide x-axis ticks
    
    plt.show()

def plot_landslide_distribution(data):
    plt.bar(data.index, data.values, color=["skyblue", "salmon"])
    plt.xticks([0, 1], ["Non-Landslide (0)", "Landslide (1)"])
    plt.ylabel("Count")
    plt.title("Distribution of Landslide vs Non-Landslide")
    plt.show()


def calculate_distribution(df, column = 'predicted_susceptibility'):
    ranges = [[0, 0.125], [0.125, 0.375], [0.0375, 0.625], [0.625, 0.875], [0.875, 1.0]]
    
    range_values = {
        "0.125":0,
        "0.375":0,
        "0.625":0,
        "0.875":0,
        "1.0":0
    }
    for bin_range in ranges:
        count = df[(df[column] > bin_range[0]) & (df[column] < bin_range[1])].shape[0]
        range_values[str(bin_range[1])] = count
        
    return range_values

def compute_shap_values(
    model, df, feature_cols, n_background=200, n_explain=500,
    categorical_cols=None,
):
    """Compute SHAP values for a Keras model with dict-based string/numeric inputs.

    Two modes for string-valued categorical columns:

    1. ``categorical_cols=None`` (default, back-compat):
       Each string column is integer-encoded via ``pd.factorize`` and SHAP
       sees a single float feature per column. The Shapley value answers
       "how much does this row's category contribute vs the background's
       category distribution". Magnitudes are valid; beeswarm color (integer
       code on a continuous scale) is not.

    2. ``categorical_cols=['type', ...]``:
       Each listed column is one-hot expanded into ``{col}__{value}`` dummies
       before SHAP. The predict wrapper argmax-decodes each dummy block back
       to the original string before calling the model. Returns SHAP values
       in the *expanded* feature space; use ``aggregate_categorical_shap`` to
       collapse the dummies back to a single SHAP column per original
       categorical feature (preserving SHAP's additive property by summing
       signed dummy SHAP values).

    Returns ``(shap_values, explain_data, feature_names)`` where the shape
    of ``shap_values`` and ``feature_names`` reflects whichever mode was used.
    """
    import shap
    import pandas as pd

    if categorical_cols is None:
        return _compute_shap_integer_coded(
            model, df, feature_cols, n_background, n_explain,
        )

    return _compute_shap_one_hot(
        model, df, feature_cols, categorical_cols, n_background, n_explain,
    )


def _compute_shap_integer_coded(model, df, feature_cols, n_background, n_explain):
    """Original (back-compat) path: integer-encode each string column."""
    import shap
    import pandas as pd

    string_cols = df[feature_cols].select_dtypes(include='object').columns.tolist()
    category_maps = {}

    df_numeric = df[feature_cols].copy()
    for col in string_cols:
        codes, uniques = pd.factorize(df_numeric[col])
        df_numeric[col] = codes.astype(float)
        category_maps[col] = dict(enumerate(uniques))

    data_array = df_numeric.values.astype(np.float64)

    rng = np.random.default_rng(42)
    n_bg = min(n_background, len(data_array))
    n_ex = min(n_explain, len(data_array))
    bg_idx = rng.choice(len(data_array), size=n_bg, replace=False)
    ex_idx = rng.choice(len(data_array), size=n_ex, replace=False)
    background = data_array[bg_idx]
    explain = data_array[ex_idx]

    def predict_fn(X):
        input_dict = {}
        for i, col in enumerate(feature_cols):
            if col in string_cols:
                int_codes = np.round(X[:, i]).astype(int)
                int_codes = np.clip(int_codes, 0, max(category_maps[col].keys()))
                str_vals = np.array([category_maps[col].get(c, category_maps[col][0]) for c in int_codes])
                input_dict[col] = str_vals
            else:
                input_dict[col] = X[:, i].astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices(input_dict).batch(256)
        ds = ds.map(lambda d: {k: tf.expand_dims(v, -1) if tf.rank(v) == 1 else v for k, v in d.items()})
        preds = model.predict(ds, verbose=0)
        if isinstance(preds, dict):
            preds = preds.get("final_head", next(iter(preds.values())))
        elif isinstance(preds, (list, tuple)):
            preds = preds[0]
        return np.asarray(preds).flatten()

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain)
    return shap_values, explain, feature_cols


def _compute_shap_one_hot(
    model, df, feature_cols, categorical_cols, n_background, n_explain,
):
    """One-hot path: expand listed categoricals, argmax-decode in predict_fn.

    The model still consumes the original string column, so the predict wrapper
    argmax-decodes the dummy block on each SHAP perturbation before calling
    the model. Use ``aggregate_categorical_shap`` to collapse the per-dummy
    SHAP back to one column per original categorical (signed sum across
    dummies — preserves SHAP additivity).
    """
    import shap
    import pandas as pd

    cat_set = set(categorical_cols)
    missing = [c for c in categorical_cols if c not in feature_cols]
    if missing:
        raise ValueError(f"categorical_cols not in feature_cols: {missing}")

    numerical_cols = [c for c in feature_cols if c not in cat_set]

    # Build one-hot dummy blocks. Vocabulary is fixed by the data passed in
    # (validation_df), so SHAP's expanded feature names are deterministic.
    dummy_blocks = {}
    for cat in categorical_cols:
        vocab = sorted(df[cat].astype(str).unique())
        dummy_blocks[cat] = vocab

    expanded_cols = list(numerical_cols) + [
        f"{cat}__{v}" for cat in categorical_cols for v in dummy_blocks[cat]
    ]

    df_exp = df[numerical_cols].copy()
    for cat, vocab in dummy_blocks.items():
        col_str = df[cat].astype(str)
        for v in vocab:
            df_exp[f"{cat}__{v}"] = (col_str == v).astype(float)

    data_array = df_exp.values.astype(np.float64)

    rng = np.random.default_rng(42)
    n_bg = min(n_background, len(data_array))
    n_ex = min(n_explain, len(data_array))
    bg_idx = rng.choice(len(data_array), size=n_bg, replace=False)
    ex_idx = rng.choice(len(data_array), size=n_ex, replace=False)
    background = data_array[bg_idx]
    explain = data_array[ex_idx]

    num_n = len(numerical_cols)

    def predict_fn(X):
        input_dict = {}
        # Numerical pass-through
        for i, col in enumerate(numerical_cols):
            input_dict[col] = X[:, i].astype(np.float32)
        # Categorical argmax-decode each dummy block. Coalitions where SHAP
        # zeroed all dummies (rare) fall back to vocab[0]; coalitions with
        # multiple "ones" pick the highest value, which is the natural
        # argmax semantics.
        ptr = num_n
        for cat in categorical_cols:
            vocab = dummy_blocks[cat]
            k = len(vocab)
            block = X[:, ptr:ptr + k]
            idx = np.argmax(block, axis=1)
            str_vals = np.array([vocab[i] for i in idx])
            input_dict[cat] = str_vals
            ptr += k

        ds = tf.data.Dataset.from_tensor_slices(input_dict).batch(256)
        ds = ds.map(lambda d: {k: tf.expand_dims(v, -1) if tf.rank(v) == 1 else v for k, v in d.items()})
        preds = model.predict(ds, verbose=0)
        if isinstance(preds, dict):
            preds = preds.get("final_head", next(iter(preds.values())))
        elif isinstance(preds, (list, tuple)):
            preds = preds[0]
        return np.asarray(preds).flatten()

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain)
    return shap_values, explain, expanded_cols


def aggregate_categorical_shap(
    shap_values, shap_data, feature_names, categorical_cols,
):
    """Collapse one-hot dummy SHAP columns back to one column per categorical.

    The merged SHAP value for row ``r`` is ``sum_k shap_values[r, dummy_k]``
    -- the signed sum across the dummy block -- which preserves SHAP's
    additive property: the total contribution of the categorical to row r's
    prediction is exactly this merged value, and ``|merged|`` is its
    magnitude (directly comparable to mean ``|SHAP|`` of numerical features).

    For the merged ``shap_data`` matrix, the categorical entry is the argmax
    index across the dummy block (i.e. which category was "on" for that row).
    This gives the beeswarm a meaningful per-row color: dots colored by
    category index instead of a meaningless continuous value.

    Parameters
    ----------
    shap_values : ndarray (n_rows, n_expanded)
    shap_data   : ndarray (n_rows, n_expanded)
    feature_names : list[str] of length n_expanded, with dummies named
                    ``{cat}__{value}``.
    categorical_cols : list[str] of original categorical column names.

    Returns
    -------
    merged_shap_values, merged_shap_data, merged_feature_names
    """
    import numpy as np

    shap_values = np.asarray(shap_values)
    shap_data = np.asarray(shap_data)

    cat_to_indices = {c: [] for c in categorical_cols}
    non_cat_indices = []
    non_cat_names = []
    for i, name in enumerate(feature_names):
        matched_cat = None
        for cat in categorical_cols:
            if name.startswith(f"{cat}__"):
                matched_cat = cat
                break
        if matched_cat is not None:
            cat_to_indices[matched_cat].append(i)
        else:
            non_cat_indices.append(i)
            non_cat_names.append(name)

    for cat in categorical_cols:
        if not cat_to_indices[cat]:
            raise ValueError(
                f"No dummy columns found for categorical '{cat}'. "
                f"Expected names like '{cat}__<value>' in feature_names."
            )

    n_rows = shap_values.shape[0]
    n_merged = len(non_cat_names) + len(categorical_cols)
    merged_shap_values = np.zeros((n_rows, n_merged))
    merged_shap_data = np.zeros((n_rows, n_merged))
    merged_feature_names: list[str] = []

    for j, (i, name) in enumerate(zip(non_cat_indices, non_cat_names)):
        merged_shap_values[:, j] = shap_values[:, i]
        merged_shap_data[:, j] = shap_data[:, i]
        merged_feature_names.append(name)

    base = len(non_cat_names)
    for j, cat in enumerate(categorical_cols):
        dummy_idx = cat_to_indices[cat]
        merged_shap_values[:, base + j] = shap_values[:, dummy_idx].sum(axis=1)
        merged_shap_data[:, base + j] = np.argmax(shap_data[:, dummy_idx], axis=1)
        merged_feature_names.append(cat)

    return merged_shap_values, merged_shap_data, merged_feature_names


def plot_shap_summary(shap_values, feature_data, feature_names):
    """Plot SHAP beeswarm and bar summary plots."""
    import shap

    shap.summary_plot(shap_values, feature_data, feature_names=feature_names)
    shap.summary_plot(shap_values, feature_data, feature_names=feature_names, plot_type="bar")


class OrdinalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="ordinal_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.acc = tf.keras.metrics.Accuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_cls = tf.reduce_sum(y_true, axis=1)
        y_pred_cls = tf.reduce_sum(tf.cast(y_pred > 0.5, tf.float32), axis=1)
        self.acc.update_state(y_true_cls, y_pred_cls)

    def result(self):
        return self.acc.result()

    def reset_state(self):
        self.acc.reset_state()
