"""Render the PINN peak surface displacement as a filled SLOPE-UNIT (polygon) map.

Same predictions as predict_map.py, but drawn on the actual slope-unit polygons from
makilala_pinn_conditioning.gpkg instead of centroid points. The gpkg is row-aligned with
load_unit_table() (unit_id is NOT unique, so geometry is attached POSITIONALLY, never merged
on unit_id).

Run:  cd mlprep && venv/bin/python -m pinn_eil.predict_map_polygons

Output:
  pinn_eil/outputs/makilala_pinn_displacement_map_polygons.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import geopandas as gpd  # noqa: E402
import tensorflow as tf  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pinn_eil as P  # noqa: E402  (registers custom layers for model loading)
from pinn_eil import config as CFG  # noqa: E402
from pinn_eil.predict_map import peak_surface_displacement, CKPT_PATH  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
MAP_PNG = OUT_DIR / "makilala_pinn_displacement_map_polygons.png"
COND_GPKG = CFG.PINN_WAVE_DATA / "makilala_pinn_conditioning.gpkg"


def main() -> None:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"No checkpoint at {CKPT_PATH}; run run_training.py first.")
    consts = P.load_global_consts()
    model = tf.keras.models.load_model(CKPT_PATH)

    table = P.load_unit_table()                     # prediction rows (parquet order)
    gdf = gpd.read_file(COND_GPKG)                  # polygons (same order)

    # Geometry is attached POSITIONALLY (unit_id is not unique). Verify alignment.
    if len(gdf) != len(table) or not np.array_equal(
        gdf["unit_id"].to_numpy(), table["unit_id"].to_numpy()
    ):
        raise ValueError("gpkg and unit table are not row-aligned; cannot attach positionally")

    print(f"[predict] {len(table)} slope-unit polygons, z=H surface over [0,{consts.T}s]")
    peaks = peak_surface_displacement(model, consts, table)
    gdf = gdf.assign(peak_disp_mm=peaks * 1000.0).to_crs(epsg=4326)
    print(f"[predict] peak disp (mm): min={peaks.min()*1000:.1f} "
          f"med={np.median(peaks)*1000:.1f} max={peaks.max()*1000:.1f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    gdf.plot(column="peak_disp_mm", cmap="inferno", linewidth=0.0, ax=ax,
             legend=True, legend_kwds={"label": "Peak surface displacement [mm]",
                                       "shrink": 0.85})
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Makilala EIL PINN — predicted peak surface displacement per slope unit\n"
                 "(2019-12-15 M6.7 Cotabato SPECFEM base motion, scaled per unit by PGA)")
    fig.tight_layout()
    fig.savefig(MAP_PNG, dpi=150)
    print(f"[saved] {MAP_PNG}")


if __name__ == "__main__":
    main()
