"""Turn an `align_rasters_to_dem` grid GeoPackage into a model-ready dataset.

Three steps, applied chunk by chunk so a province-sized grid (~10M cells) never
has to fit in memory at once:

1. Rename columns to the training names (`py_files.column_mapping`, values unchanged).
2. Recode the numeric `Soil Type.tif` codes in `type` to the soil names the model's
   StringLookup was adapted on, using the mapping's `metadata.type_recode` table.
3. Add `landslide` from an inventory raster: 1 when any non-zero inventory pixel
   overlaps the cell, else 0. Cells outside the inventory raster's extent are
   still 0 — that area was never mapped, so restrict metrics to the extent.

`--clip-to-inventory` keeps only cells inside the inventory raster's extent (the
mapped area), so unmapped cells are dropped instead of labelled 0. Step 2 is
skipped when the mapping has no `type` column (e.g. a PGA + landslide subset).
`--min-slope` keeps only cells whose grid slope is at least that many degrees
(NaN slope is dropped); the slope column is used for filtering, not written.

    venv/bin/python -m scripts.finalize_grid_dataset \
        --grid datasets/nc_combined.gpkg \
        --grid-dem datasets/EIL_INPUTS/aligned_nc/Elevation.tif \
        --mapping datasets/nc_mapped.mapping.json \
        --inventory "~/Downloads/LSI_NorthCotabato_10m (1).tif" \
        --output datasets/nc_mapped.gpkg

    venv/bin/python -m scripts.finalize_grid_dataset \
        --grid datasets/nc_combined.gpkg \
        --grid-dem datasets/EIL_INPUTS/aligned_nc/Elevation.tif \
        --mapping datasets/nc_east_pga_landslide.mapping.json \
        --inventory "~/Downloads/LSI_NorthCotabato_10m (1).tif" \
        --output datasets/nc_east_pga_landslide.gpkg --clip-to-inventory --min-slope 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_files.column_mapping import apply_mapping, load_mapping  # noqa: E402

LABEL_COL = "landslide"
TYPE_COL = "type"
ROW_COL, COL_COL = "row", "col"
DEFAULT_CHUNK = 1_000_000
DEFAULT_SLOPE_COL = "slope_cot_prs92"
INVENTORY_NODATA = 0


def load_type_recode(mapping_path):
    """`{code: soil name}` from the mapping JSON's metadata."""
    with open(mapping_path) as f:
        recode = json.load(f).get("metadata", {}).get("type_recode")
    if not recode:
        raise ValueError(f"{mapping_path} has no metadata.type_recode table")
    return {int(float(code)): name for code, name in recode.items()}


def inventory_on_grid(inventory_path, grid_dem_path):
    """Inventory presence (0/1) resampled onto the grid with max: any overlap counts."""
    with rasterio.open(grid_dem_path) as grid:
        transform, shape, crs = grid.transform, grid.shape, grid.crs
    with rasterio.open(inventory_path) as inv:
        values = inv.read(1)
        present = ((values != INVENTORY_NODATA) & np.isfinite(values)).astype("uint8")
        on_grid = np.zeros(shape, "uint8")
        reproject(
            present, on_grid,
            src_transform=inv.transform, src_crs=inv.crs,
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.max,
        )
    return on_grid


def inventory_window(inventory_path, grid_dem_path):
    """`(row_min, row_max, col_min, col_max)` of grid cells whose centre lies in the inventory extent."""
    with rasterio.open(grid_dem_path) as grid:
        transform, shape, crs = grid.transform, grid.shape, grid.crs
    with rasterio.open(inventory_path) as inv:
        bounds = transform_bounds(inv.crs, crs, *inv.bounds)
    window = from_bounds(*bounds, transform=transform)
    row_min = max(int(np.ceil(window.row_off - 0.5)), 0)
    col_min = max(int(np.ceil(window.col_off - 0.5)), 0)
    row_max = min(int(np.floor(window.row_off + window.height - 0.5)), shape[0] - 1)
    col_max = min(int(np.floor(window.col_off + window.width - 0.5)), shape[1] - 1)
    if row_min > row_max or col_min > col_max:
        raise ValueError(f"{inventory_path} does not overlap the grid of {grid_dem_path}")
    return row_min, row_max, col_min, col_max


def window_bbox(window, grid_dem_path):
    """Projected bbox of a cell window, for the GeoPackage spatial-index prefilter."""
    row_min, row_max, col_min, col_max = window
    with rasterio.open(grid_dem_path) as grid:
        transform = grid.transform
    left, top = transform * (col_min, row_min)
    right, bottom = transform * (col_max + 1, row_max + 1)
    return min(left, right), min(top, bottom), max(left, right), max(top, bottom)


def in_window(gdf, window):
    row_min, row_max, col_min, col_max = window
    rows, cols = gdf[ROW_COL].to_numpy(), gdf[COL_COL].to_numpy()
    keep = (rows >= row_min) & (rows <= row_max) & (cols >= col_min) & (cols <= col_max)
    return gdf[keep]


def steep_enough(gdf, slope_col, min_slope):
    return gdf[gdf[slope_col].to_numpy() >= min_slope]


def recode_types(codes, recode):
    """Map numeric soil codes to names; unknown codes are an error, not a silent NaN."""
    rounded = codes.round().astype("Int64")
    unknown = sorted(set(rounded.dropna().unique()) - set(recode))
    if unknown:
        raise ValueError(f"Soil codes with no entry in type_recode: {unknown}")
    return rounded.map(recode).astype(object)


def finalize_chunk(gdf, mapping, recode, labels):
    out = apply_mapping(gdf, mapping, keep_extra=False)
    if recode is not None:
        out[TYPE_COL] = recode_types(gdf[mapping[TYPE_COL]], recode).values
    out[LABEL_COL] = labels[gdf[ROW_COL].to_numpy(), gdf[COL_COL].to_numpy()].astype("int64")
    return out


def run(grid_path, grid_dem_path, mapping_path, inventory_path, output_path, chunk_size,
        clip_to_inventory=False, min_slope=None, slope_col=DEFAULT_SLOPE_COL):
    mapping = load_mapping(mapping_path)
    recode = load_type_recode(mapping_path) if TYPE_COL in mapping else None
    labels = inventory_on_grid(inventory_path, grid_dem_path)
    columns = sorted({ROW_COL, COL_COL, *mapping.values(),
                      *([slope_col] if min_slope is not None else [])})
    window = inventory_window(inventory_path, grid_dem_path) if clip_to_inventory else None
    bbox = window_bbox(window, grid_dem_path) if window else None
    total = pyogrio.read_info(grid_path)["features"]
    print(f"{total:,} cells in {grid_path}; {int(labels.sum()):,} grid cells touch the inventory")
    if window:
        print(f"Clipping to inventory extent: rows {window[0]}-{window[1]}, cols {window[2]}-{window[3]}")
    if min_slope is not None:
        print(f"Keeping cells with {slope_col} >= {min_slope} degrees")

    Path(output_path).unlink(missing_ok=True)
    written, positives, start = 0, 0, 0
    while True:
        gdf = pyogrio.read_dataframe(grid_path, columns=columns, bbox=bbox,
                                     skip_features=start, max_features=chunk_size)
        if gdf.empty:
            break
        start += len(gdf)
        if window:
            gdf = in_window(gdf, window)
        if min_slope is not None:
            gdf = steep_enough(gdf, slope_col, min_slope)
        out = finalize_chunk(gdf, mapping, recode, labels)
        pyogrio.write_dataframe(out, output_path, driver="GPKG", append=written > 0)
        written += len(out)
        positives += int(out[LABEL_COL].sum())
        print(f"  {written:,} cells written ({start:,} read)")

    print(f"Wrote {written:,} cells to {output_path}: {positives:,} landslide=1")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", required=True, help="GeoPackage from align_rasters_to_dem")
    parser.add_argument("--grid-dem", required=True, help="The gridded Elevation.tif (defines row/col)")
    parser.add_argument("--mapping", required=True, help="Mapping JSON with metadata.type_recode")
    parser.add_argument("--inventory", required=True, help="Landslide inventory raster (0 = none)")
    parser.add_argument("--output", required=True, help="Output GeoPackage")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK, help="Cells per chunk")
    parser.add_argument("--clip-to-inventory", action="store_true",
                        help="Keep only cells inside the inventory raster's extent")
    parser.add_argument("--min-slope", type=float, default=None,
                        help="Keep only cells with slope >= this many degrees")
    parser.add_argument("--slope-column", default=DEFAULT_SLOPE_COL,
                        help="Grid column holding slope in degrees")
    args = parser.parse_args()
    run(args.grid, args.grid_dem, args.mapping, os.path.expanduser(args.inventory),
        args.output, args.chunk_size, args.clip_to_inventory, args.min_slope, args.slope_column)
