
import argparse
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
from tensorflow.keras.models import load_model

from py_files.data import preprocessing, dataframe_to_dataset
from py_files.helpers import add_soil_texture_index

# Custom layers (needed for deserialization even though they are
# registered via @register_keras_serializable)
from py_files.GallenModel import CriticalAcceleration, DisplacementIntermediate, FosLayer
from py_files.GallenModel_v1 import (
    NewmarkActivation,
    DisplacementLayerRainFall,
    WetnessLayer,
    ClipLayer,
    CohesionLayer,
    InternalFrictionLayer,
)
from py_files.GallenModel_v3 import (
    HydraulicConductivityLayerV3,
    HydraulicConductivityLayerCms,
    WetnessRatioLayer,
    WetnessLayerDimensional,
    WetnessLayerFixedRecharge,
)
from py_files.LandslideRainfall_v3 import LogitLayer
from py_files.Landslidev2_Old import DiceCrossEntropyLoss
from py_files.metrics import (
    export_to_geopackage,
    create_slope_unit_template, rasterize_from_template,
)

DEFAULT_DATA = "~/Documents/ml-prep/ML-PREP-2025/learn/data/SU_17_training_v3_contri.gpkg"
MODEL_SAVE_PATH = "outputs/obj2/wetness/v1"
DEFAULT_MODEL = f"{MODEL_SAVE_PATH}/fold-1-model-v3.keras"

# Layer names drift between model variants (e.g. the dimensional wetness path uses
# `hydraulic_conductivity_cms` where the older path used `hydraulic_conductivity_v3`).
# Each entry is tried in order and the first one present in the model wins, so one
# script keeps working across variants instead of dying in `get_layer`.
LAYER_ALIASES = {
    "fos": ("fos_layer",),
    "critical_acceleration": ("critical_acceleration",),
    "displacement": ("displacement_layer",),
    "cohesion": ("cohesion_layer",),
    "internal_friction": ("internal_friction",),
    "wetness": ("m_clip", "wetness_layer_dimensional", "wetness_ratio_layer", "wetness_layer"),
    "conductivity": ("hydraulic_conductivity_cms", "hydraulic_conductivity_v3"),
}

# Multi-head models (final_head + fos_layer + physics_prob) emit a list from
# `predict`; this is the head that carries the calibrated susceptibility.
SUSCEPTIBILITY_OUTPUT = "final_head"

LABEL_COL = "landslide"

COLUMNS_DROP = [
    "Landslide1", "descriptio", "sus_pinn_ground truth", "ds",
    "cohesion", "internal_friction", "sus_pinn_landslide",
    "confusion", "landslide_preds", "landslide_probability",
    "Lithology", "LITHO", "Geomorphology", "LITHODESC",
    "LITHO_2", "LITHODESC_2", "value",
]

BATCH_SIZE = 128

# --- Figure rendering ---
FIGURE_DPI = 150
MAP_FIGSIZE = (8, 7)
BOXPLOT_FIGSIZE = (14, 8)
BOXPLOT_NCOLS = 3
HIST_FIGSIZE = (7, 4.5)
HIST_BINS = 40
KPLOT_FIGSIZE = (9, 5)
# Colour scales for the non-susceptibility layers are clipped to these
# percentiles so a handful of extreme units don't flatten the whole map.
COLOR_CLIP_PERCENTILES = (2, 98)
FIGURE_SUBDIR = "figures"

# Both conductivity layers emit m/hr; 100 cm per m for the reported cm/h.
M_HR_TO_CM_H = 100.0

SOIL_NAMES = [
    "Sand", "Loamy Sand", "Sandy Loam", "Silt Loam", "Loam", "Silt",
    "Sandy Clay Loam", "Clay Loam", "Silty Clay Loam", "Sandy Clay",
    "Silty Clay", "Clay",
]


def load_and_preprocess(data_path: str):
    """Load GeoPackage, run preprocessing, add soil texture index.

    Returns the preprocessed GeoDataFrame, feature columns, and the
    *full* (unfiltered) GeoDataFrame for gap-free rasterization.
    """
    df_full = gpd.read_file(data_path)

    # Drop columns that exist in the dataframe (some may not be present)
    cols_to_drop = [c for c in COLUMNS_DROP if c in df_full.columns]
    df, columns, numeric_cols = preprocessing(df_full.copy(), columns_drop=cols_to_drop)
    df = add_soil_texture_index(df)

    feature_cols = columns + ["soil_texture_idx"]
    return df, feature_cols, df_full


def load_trained_model(model_path: str):
    """Load a saved .keras model with all custom objects."""
    custom_objects = {
        "NewmarkActivation": NewmarkActivation,
        "DisplacementLayerRainFall": DisplacementLayerRainFall,
        "WetnessLayer": WetnessLayer,
        "ClipLayer": ClipLayer,
        "CohesionLayer": CohesionLayer,
        "InternalFrictionLayer": InternalFrictionLayer,
        "CriticalAcceleration": CriticalAcceleration,
        "DisplacementIntermediate": DisplacementIntermediate,
        "FosLayer": FosLayer,
        "HydraulicConductivityLayerV3": HydraulicConductivityLayerV3,
        "HydraulicConductivityLayerCms": HydraulicConductivityLayerCms,
        "WetnessRatioLayer": WetnessRatioLayer,
        "WetnessLayerDimensional": WetnessLayerDimensional,
        "WetnessLayerFixedRecharge": WetnessLayerFixedRecharge,
        "LogitLayer": LogitLayer,
        "DiceCrossEntropyLoss": DiceCrossEntropyLoss,
    }
    return load_model(model_path, custom_objects=custom_objects)


def resolve_layer(model, key):
    """Return the model's layer for a logical role, trying each known alias.

    Raises with the model's actual layer names so a renamed layer is an obvious
    fix rather than a bare KeyError.
    """
    present = {layer.name for layer in model.layers}
    for candidate in LAYER_ALIASES[key]:
        if candidate in present:
            return model.get_layer(candidate)
    raise KeyError(
        f"No layer found for '{key}'. Tried {LAYER_ALIASES[key]}. "
        f"Model layers: {sorted(present)}"
    )


def extract_intermediate(model, dataset, key):
    """Build a sub-model on the layer filling ``key`` and predict its output."""
    layer = resolve_layer(model, key)
    sub = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
    return sub.predict(dataset)


def predict_susceptibility(model, dataset):
    """Predict, keeping the susceptibility head when the model is multi-output."""
    preds = model.predict(dataset)
    if isinstance(preds, dict):
        return np.asarray(preds[SUSCEPTIBILITY_OUTPUT]).flatten()
    if isinstance(preds, (list, tuple)):
        names = list(getattr(model, "output_names", None) or [])
        idx = names.index(SUSCEPTIBILITY_OUTPUT) if SUSCEPTIBILITY_OUTPUT in names else 0
        print(f"  Multi-output model {names or '(unnamed)'} — using index {idx}")
        return np.asarray(preds[idx]).flatten()
    return np.asarray(preds).flatten()


def learned_conductivity(model):
    """Per-texture learned Ksat in m/hr, computed by calling the K layer itself.

    Deliberately does NOT re-derive the sigmoid math: the two conductivity layers
    parameterize it differently (linear cm/h bounds vs log-uniform cm/s bounds), and
    both already emit m/hr, so invoking the layer keeps this correct for either.
    """
    k_layer = resolve_layer(model, "conductivity")
    soil_idx = tf.constant(np.arange(len(SOIL_NAMES)).reshape(-1, 1), dtype=tf.float32)
    return k_layer.name, np.asarray(k_layer(soil_idx)).flatten()


def build_layer_specs(results):
    """Map layer name -> (values, label, cmap, vmin, vmax) for every export.

    Shared by the GeoTIFF export and the figure export so the two never drift.
    ``vmin``/``vmax`` of ``None`` means "derive from the data percentiles".
    """
    return {
        "susceptibility": (results["susceptibility"], "Susceptibility", "plasma_r", 0.0, 1.0),
        "cohesion": (results["cohesion"], "Cohesion (kPa)", "viridis", None, None),
        "internal_friction": (
            results["internal_friction"], "Internal Friction Angle (rad)", "viridis", None, None,
        ),
        "wetness": (results["wetness"], "Wetness (m)", "Blues", None, None),
        "fos": (results["fos"], "Factor of Safety", "RdYlGn", None, None),
        "displacement": (results["displacement"], "Displacement (cm)", "magma_r", None, None),
    }


def _pyplot():
    """Import pyplot on a headless backend.

    Kept lazy (rather than a module-level ``matplotlib.use``) so importing this
    module from a notebook does not clobber an interactive backend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _color_limits(values, vmin, vmax):
    """Resolve explicit limits, falling back to clipped percentiles."""
    lo = np.nanpercentile(values, COLOR_CLIP_PERCENTILES[0]) if vmin is None else vmin
    hi = np.nanpercentile(values, COLOR_CLIP_PERCENTILES[1]) if vmax is None else vmax
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Degenerate (constant or all-NaN) layer — widen so Normalize is valid.
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
    return float(lo), float(hi)


def save_value_map(gdf, values, out_png, label, cmap, vmin=None, vmax=None, add_basemap=True):
    """Render a per-slope-unit choropleth of ``values`` and write it to PNG."""
    plt = _pyplot()
    import matplotlib.colors as mcolors

    lo, hi = _color_limits(values, vmin, vmax)
    norm = mcolors.Normalize(vmin=lo, vmax=hi)

    plot_gdf = gdf.copy()
    plot_gdf["_value"] = values

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE)
    plot_gdf.plot(column="_value", cmap=cmap, ax=ax, norm=norm, edgecolor="none", antialiased=True)

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cbar.set_label(label)
    ax.set_title(label)
    ax.set_axis_off()

    if add_basemap:
        try:  # best-effort: needs network access and a resolvable CRS
            import contextily as cx
            cx.add_basemap(ax, crs=plot_gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
        except Exception as exc:
            print(f"  [map] basemap skipped for {label}: {exc}")

    fig.tight_layout()
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_boxplot_panel(layer_specs, out_png):
    """One figure with a boxplot per predicted layer (spread at a glance)."""
    plt = _pyplot()

    names = list(layer_specs)
    nrows = int(np.ceil(len(names) / BOXPLOT_NCOLS))
    fig, axes = plt.subplots(nrows, BOXPLOT_NCOLS, figsize=BOXPLOT_FIGSIZE)
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        values, label = layer_specs[name][0], layer_specs[name][1]
        ax.boxplot(values, vert=True, showfliers=True, flierprops={"markersize": 2, "alpha": 0.3})
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])

    for ax in axes[len(names):]:
        ax.set_axis_off()

    fig.suptitle("PINN v3 predicted distributions")
    fig.tight_layout()
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_susceptibility_histogram(susceptibility, out_png):
    """Histogram of the susceptibility scores on the fixed [0, 1] scale."""
    plt = _pyplot()

    fig, ax = plt.subplots(figsize=HIST_FIGSIZE)
    ax.hist(susceptibility, bins=HIST_BINS, range=(0.0, 1.0), color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Susceptibility")
    ax.set_ylabel("Slope units")
    ax.set_title("Susceptibility distribution")
    fig.tight_layout()
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_learned_k_plot(k_m_hr, soil_counts, out_png, layer_name=""):
    """Bar chart of learned Ksat per USDA texture, on a log axis.

    Log scale is not cosmetic: the cm/s table spans ~5 orders of magnitude across
    textures, so a linear axis collapses everything below Sand to a flat line.
    Classes absent from this dataset are drawn faded — their learned value is
    whatever training left behind and carries no signal here.
    """
    plt = _pyplot()

    values = np.asarray(k_m_hr[: len(SOIL_NAMES)], dtype=float)
    present = [soil_counts.get(i, 0) > 0 for i in range(len(SOIL_NAMES))]
    colors = ["#4c72b0" if is_present else "#cccccc" for is_present in present]

    fig, ax = plt.subplots(figsize=KPLOT_FIGSIZE)
    ax.bar(range(len(SOIL_NAMES)), values, color=colors)
    ax.set_xticks(range(len(SOIL_NAMES)))
    ax.set_xticklabels(SOIL_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Learned Ksat (m/hr)")
    positive = values[values > 0]
    if positive.size:
        ax.set_yscale("log")
        ax.set_ylim(positive.min() / 5, positive.max() * 5)
    suffix = f" — {layer_name}" if layer_name else ""
    ax.set_title(f"Learned Ksat per USDA texture{suffix}\n(faded = absent from data)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_figures(gdf, layer_specs, k_m_hr, soil_counts, figure_dir, k_layer_name=""):
    """Write every PNG for this run into ``figure_dir``.

    A failure in one figure is reported and skipped rather than discarding the
    (expensive) inference results that produced it.
    """
    os.makedirs(figure_dir, exist_ok=True)

    for name, (values, label, cmap, vmin, vmax) in layer_specs.items():
        out_png = os.path.join(figure_dir, f"map_{name}.png")
        try:
            save_value_map(gdf, values, out_png, label, cmap, vmin=vmin, vmax=vmax)
            print(f"  wrote {out_png}")
        except Exception as exc:
            print(f"  [figures] FAILED map_{name}.png: {type(exc).__name__}: {exc}")

    extras = {
        "boxplots.png": lambda p: save_boxplot_panel(layer_specs, p),
        "susceptibility_histogram.png": lambda p: save_susceptibility_histogram(
            layer_specs["susceptibility"][0], p,
        ),
        "learned_k_per_soil.png": lambda p: save_learned_k_plot(
            k_m_hr, soil_counts, p, layer_name=k_layer_name,
        ),
    }
    for fname, render in extras.items():
        out_png = os.path.join(figure_dir, fname)
        try:
            render(out_png)
            print(f"  wrote {out_png}")
        except Exception as exc:
            print(f"  [figures] FAILED {fname}: {type(exc).__name__}: {exc}")


def match_columns(df, required):
    """Map each model input name to a dataframe column, tolerating case drift.

    GeoPackage round-trips sometimes lowercase field names — SU_19 carries
    ``contributingfactor_mean`` where SU_17 training used
    ``ContributingFactor_mean``. That is the same variable under a different
    spelling, so matching it is safe; ambiguous or absent names are reported
    instead of guessed.

    Returns ``({model_input: source_column}, [unmatched inputs])``.
    """
    by_lower = {}
    for col in df.columns:
        by_lower.setdefault(str(col).lower(), []).append(col)

    resolved, missing = {}, []
    for name in required:
        if name in df.columns:
            resolved[name] = name
            continue
        candidates = by_lower.get(name.lower(), [])
        if len(candidates) == 1:
            resolved[name] = candidates[0]
            print(f"  Column case fix: data '{candidates[0]}' -> model input '{name}'")
        else:
            missing.append(name)
    return resolved, missing


def report_pga_choice(df, source_col, slot, explicit):
    """Print the medians of every PGA-ish column so a wrong slot is visible.

    Cross-dataset PGA naming does not imply cross-dataset PGA meaning: SU_19's own
    ``PGA2_max`` is a weaker, different field than the ``PGA2_max`` the model trained
    on, and feeding it collapses low-shaking areas to ~0 susceptibility. Only the
    training median settles it and this script does not know it, so it prints the
    numbers and warns when the slot was filled by name-match rather than by choice.
    """
    pga_cols = [c for c in df.columns if "PGA" in str(c).upper()]
    if not pga_cols:
        return
    medians = ", ".join(f"{c}={float(df[c].median()):.4f}" for c in pga_cols)
    print(f"  PGA input '{slot}' <- data '{source_col}'. Medians in data: {medians}")
    if not explicit and len(pga_cols) > 1:
        print(f"  [PGA] {len(pga_cols)} PGA columns present and '{source_col}' was matched "
              "by name alone — confirm its median matches the TRAINING distribution "
              "(pass --pga-col to switch).")


def build_model_inputs(df, model, pga_col=None):
    """Select exactly the columns this model declares as inputs.

    ``pga_col`` copies a dataset PGA column into the model's PGA input slot,
    overriding any same-named column. Left unset, the model's own input name is
    used as-is — which is not always right across datasets, hence
    ``report_pga_choice``.
    """
    # Keras 3 dropped Functional.input_names; the keras tensors carry them.
    required = [t.name for t in model.inputs]
    resolved, missing = match_columns(df, required)

    pga_slots = [c for c in required if "PGA" in c.upper()]
    if pga_col:
        if pga_col not in df.columns:
            raise KeyError(f"--pga-col '{pga_col}' not in the data. Columns: {sorted(df.columns)}")
        if len(pga_slots) != 1:
            raise ValueError(f"Expected exactly one PGA model input to alias, got {pga_slots}")
        resolved[pga_slots[0]] = pga_col
        missing = [c for c in missing if c != pga_slots[0]]

    if missing:
        raise KeyError(
            f"Model inputs missing from the data: {missing}. "
            f"Model expects {required}. Data has {sorted(df.columns)}. "
            f"If a PGA input is missing, pass --pga-col <column> to alias it."
        )

    for slot in pga_slots:
        report_pga_choice(df, resolved[slot], slot, explicit=bool(pga_col))

    out = pd.DataFrame({name: df[resolved[name]].values for name in required}, index=df.index)
    # dataframe_to_dataset pops `landslide` as the label. Predictions ignore it,
    # but the key has to exist — so unlabeled inference data gets a dummy column.
    out[LABEL_COL] = df[LABEL_COL].values if LABEL_COL in df.columns else 0
    return out


def run(data_path: str, model_path: str, output_dir: str = None, pixel_size: float = 30.0,
        figure_dir: str = None, pga_col: str = None):
    print(f"Loading data from {data_path}")
    df, feature_cols, df_full = load_and_preprocess(data_path)

    print(f"  Samples after preprocessing: {len(df)}")
    print(f"  COLS : {feature_cols}")
    print(df['soil_texture_idx'].head())

    print(f"Loading model from {model_path}")
    model = load_trained_model(model_path)
    model.summary()

    feed_df = build_model_inputs(df, model, pga_col=pga_col)
    ds = dataframe_to_dataset(feed_df, shuffle=False, batch_size=BATCH_SIZE)

    susceptibility = predict_susceptibility(model, ds)

    fos = extract_intermediate(model, ds, "fos").flatten()
    print(f"  Factor of Safety     — mean: {fos.mean():.4f}, std: {fos.std():.4f}")

    ac_outputs = extract_intermediate(model, ds, "critical_acceleration")
    ac = ac_outputs[0].flatten()
    acpg = ac_outputs[1].flatten()
    print(f"  Critical Accel (ac)  — mean: {ac.mean():.4f}, std: {ac.std():.4f}")
    print(f"  ac / PGA (acpg)      — mean: {acpg.mean():.4f}, std: {acpg.std():.4f}")

    displacement = extract_intermediate(model, ds, "displacement").flatten()
    print(f"  Displacement (cm)    — mean: {displacement.mean():.4f}, std: {displacement.std():.4f}")

    cohesion = extract_intermediate(model, ds, "cohesion").flatten()
    print(f"  Cohesion (kPa)       — mean: {cohesion.mean():.4f}, std: {cohesion.std():.4f}")

    ifi = extract_intermediate(model, ds, "internal_friction").flatten()
    print(f"  Int. Friction (rad)  — mean: {ifi.mean():.4f}, std: {ifi.std():.4f}")

    wetness = extract_intermediate(model, ds, "wetness").flatten()
    print(f"  Wetness (m)          — mean: {wetness.mean():.4f}, std: {wetness.std():.4f}")

    k_layer_name, k_m_hr = learned_conductivity(model)
    print(f"\n  Learned Ksat per USDA soil texture (from '{k_layer_name}'):")
    for i, name in enumerate(SOIL_NAMES):
        print(f"    {name:20s}: {k_m_hr[i]:.6g} m/hr  ({k_m_hr[i] * M_HR_TO_CM_H:.4g} cm/h)")

    results = {
        "susceptibility": susceptibility,
        "fos": fos,
        "critical_acceleration": ac,
        "acpg": acpg,
        "displacement": displacement,
        "cohesion": cohesion,
        "internal_friction": ifi,
        "wetness": wetness,
        "k_m_hr": k_m_hr,
        "k_layer": k_layer_name,
    }
    layer_specs = build_layer_specs(results)

    if figure_dir:
        print(f"\nRendering figures to {figure_dir} …")
        soil_counts = df["soil_texture_idx"].astype(int).value_counts().to_dict()
        save_figures(
            df, layer_specs, k_m_hr, soil_counts, figure_dir, k_layer_name=k_layer_name,
        )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # --- Vector export (GeoPackage) ---
        gpkg_path = os.path.join(output_dir, "geotechnical_values.gpkg")
        print(f"\nExporting GeoPackage to {gpkg_path} …")
        export_to_geopackage(df, {
            "cohesion_kpa": cohesion,
            "internal_friction_rad": ifi,
            "wetness_m": wetness,
            "fos": fos,
            "displacement_cm": displacement,
            "susceptibility": susceptibility,
        }, gpkg_path)

        # --- Slope-unit raster export (GeoTIFFs) ---
        # Use the FULL (unfiltered) GeoDataFrame for the template so that
        # all slope-unit polygons are rasterized — no gaps from dropped rows.
        template_path = os.path.join(output_dir, "slope_unit_template.tif")
        print(f"\nCreating slope-unit template at {template_path} (pixel_size={pixel_size}) …")
        create_slope_unit_template(df_full, template_path, pixel_size=pixel_size)

        # Build a full-length value array: predicted values for kept rows,
        # nodata for rows dropped during preprocessing.
        nodata_val = -9999.0
        kept_indices = df.index.values  # original row indices that survived preprocessing

        print(f"Exporting slope-unit GeoTIFFs to {output_dir} …")
        for fname, (vals, desc, _cmap, _vmin, _vmax) in layer_specs.items():
            full_vals = np.full(len(df_full), nodata_val, dtype=np.float64)
            full_vals[kept_indices] = vals
            rasterize_from_template(
                template_path, full_vals,
                os.path.join(output_dir, f"{fname}.tif"),
                layer_name=desc,
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINN Landslide Model v3 inference")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to GeoPackage file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .keras model checkpoint")
    parser.add_argument("--output-dir", default=None, help="Directory for GeoTIFF exports (skip if omitted)")
    parser.add_argument("--pixel-size", type=float, default=30.0, help="GeoTIFF pixel size in CRS units (default: 30)")
    parser.add_argument(
        "--pga-col", default=None,
        help=(
            "Data column to feed into the model's PGA input when the names differ "
            "(e.g. --pga-col PGA1_max for a model whose input is PGA2_max). "
            "Verify the medians match the training distribution before using this."
        ),
    )
    parser.add_argument(
        "--figure-dir", default=None,
        help=(
            "Directory for PNG figures (maps, boxplots, histogram, learned-K chart). "
            f"Defaults to <output-dir>/{FIGURE_SUBDIR} when --output-dir is given; "
            "omit both to skip figures."
        ),
    )
    args = parser.parse_args()

    figure_dir = args.figure_dir
    if figure_dir is None and args.output_dir:
        figure_dir = os.path.join(args.output_dir, FIGURE_SUBDIR)

    results = run(
        args.data, args.model,
        output_dir=args.output_dir,
        pixel_size=args.pixel_size,
        figure_dir=figure_dir,
        pga_col=args.pga_col,
    )
