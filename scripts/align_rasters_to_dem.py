"""Resample every raster in a directory onto a regular grid snapped to the DEM, and combine them into one GeoPackage.

A grid-cell alternative to the plugin's slope-unit zonal statistics
(MLPREP_Plugin_v4/ml-prep-scripts): no slope units, no zonal step. The output
grid shares the DEM's CRS and origin, with square cells of `--cell-size` metres
(default 25 m, i.e. 25x fewer cells than a 5 m DEM). Each raster is warped
straight onto that grid, and the resampling plays the role of the zonal
statistic:

    finer than the cell (DEM, slope, NDVI ...)   average of the pixels in the cell (= per-cell mean)
    coarser than the cell (SoilGrids, PGA ...)   bilinear, as the plugin warps them
    categorical (soil type, LULC)                mode (majority), never interpolated
    distance rasters (Distance*)                 min

Only band 1 of each raster is used — the plugin does the same — so a 6-band
SoilGrids raster contributes its first (0-5 cm) layer and every raster becomes
exactly one column.

With `--roi`, the grid covers only the ROI's bounding box and cells whose centre
lies outside the ROI polygon(s) are left out — e.g. one municipality instead of
a whole province.

Output GeoPackage: one square polygon per grid cell (cells where the DEM is
nodata are dropped), so QGIS draws it like a raster. Columns: `row`/`col` on the
output grid, `elevation` from the DEM, then one column per raster named after
its file (`Soil Type.tif` -> `soil_type`). Nodata becomes NaN.

    venv/bin/python -m scripts.align_rasters_to_dem \
        --input-dir datasets/EIL_INPUTS \
        --dem datasets/EIL_INPUTS/Elevation.tif \
        --roi datasets/roi/makilala_boundary.gpkg \
        --gpkg datasets/combined.gpkg

Gridded rasters go to `--output-dir` (default `<input-dir>/aligned/`) with the
same filenames. `--cell-size 5` (the DEM's own resolution) reproduces a
pixel-for-pixel DEM grid.

Memory: rasters are warped and the GeoPackage built in strips of `--chunk-rows`
grid rows, so memory stays bounded whatever the extent. Parallelism
(`--workers`, default all cores): rasters are gridded concurrently; GeoPackage
strips are read in parallel but written by one process (SQLite has one writer).
"""

import argparse
import math
import os
import re
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform

RASTER_SUFFIXES = {".tif", ".tiff"}
DEFAULT_OUTPUT_SUBDIR = "aligned"
DEFAULT_GPKG_NAME = "combined.gpkg"
DEFAULT_LAYER = "grid"
DEFAULT_CELL_SIZE_M = 25.0
ELEVATION_COLUMN = "elevation"
FLOAT_NODATA = -9999.0
DEFAULT_CHUNK_ROWS = 256  # grid rows per strip; bounds memory for very large grids
MAX_PENDING_STRIPS = 8  # strips read ahead of the GeoPackage writer
WARP_MEM_LIMIT_MB = 512
VALUE_DTYPE = "float32"
# Mirrors statRules in the plugin's zonal.go, keyed by column_base_name().
CATEGORICAL_NAMES = {"type", "soil_type", "soiltype", "lulc"}
MIN_RULE_PREFIX = "distance"


def column_base_name(path: Path) -> str:
    """Filename stem as a safe column name: 'PGA 2019.tif' -> 'pga_2019'."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", path.stem).strip("_").lower() or "raster"
    return f"r_{name}" if name[0].isdigit() else name


def resampling_for(path: Path, cell_size: float) -> Resampling:
    """Per-raster rule: mode for classes, min for distances, else mean (fine) or bilinear (coarse)."""
    name = column_base_name(path)
    if name in CATEGORICAL_NAMES:
        return Resampling.mode
    if name.startswith(MIN_RULE_PREFIX):
        return Resampling.min
    with rasterio.open(path) as src:
        source_res = max(abs(src.res[0]), abs(src.res[1]))
    return Resampling.average if source_res < cell_size else Resampling.bilinear


def find_rasters(input_dir: Path, exclude: Path, output_dir: Path) -> list[Path]:
    """All .tif/.tiff files directly in input_dir, minus the DEM and any outputs."""
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in RASTER_SUFFIXES
        and p.resolve() != exclude.resolve()
        and p.resolve().parent != output_dir.resolve()
    )


def load_roi_shapes(roi_path: Path, crs) -> list[dict]:
    """ROI polygons reprojected to the DEM CRS, as GeoJSON-like dicts (picklable for workers)."""
    roi = gpd.read_file(roi_path, engine="pyogrio")
    if roi.empty:
        raise ValueError(f"ROI {roi_path} has no features")
    if roi.crs is None:
        raise ValueError(f"ROI {roi_path} has no CRS")
    geoms = roi.to_crs(crs).geometry
    geoms = geoms[geoms.notna() & ~geoms.is_empty]
    if geoms.empty:
        raise ValueError(f"ROI {roi_path} has only empty geometries")
    return [g.__geo_interface__ for g in geoms]


def read_grid(dem_path: Path, roi_path: Path | None, cell_size: float) -> dict:
    """Output grid: DEM CRS and origin (snapped to DEM pixels), square cells, cropped to the ROI if given."""
    with rasterio.open(dem_path) as dem:
        if dem.crs is None:
            raise ValueError(f"DEM {dem_path} has no CRS")
        shapes = load_roi_shapes(roi_path, dem.crs) if roi_path else None
        window = Window(0, 0, dem.width, dem.height)
        if shapes is not None:
            try:
                window = geometry_window(dem, shapes)
            except rasterio.errors.WindowError as exc:
                raise ValueError(f"ROI {roi_path} does not overlap the DEM") from exc
        window = Window(int(window.col_off), int(window.row_off), int(window.width), int(window.height))
        origin = window_transform(window, dem.transform)
        width = math.ceil(window.width * abs(dem.transform.a) / cell_size)
        height = math.ceil(window.height * abs(dem.transform.e) / cell_size)
        return {
            "crs": dem.crs,
            "transform": Affine(cell_size, 0.0, origin.c, 0.0, -cell_size, origin.f),
            "width": width,
            "height": height,
            "shapes": shapes,
        }


def resolve_nodata(src: rasterio.io.DatasetReader) -> float | int:
    """Source nodata if set; otherwise a sentinel that fits the dtype."""
    if src.nodata is not None:
        return src.nodata
    if np.issubdtype(np.dtype(src.dtypes[0]), np.floating):
        return FLOAT_NODATA
    return np.iinfo(np.dtype(src.dtypes[0])).max


def strip_windows(width: int, height: int, chunk_rows: int):
    """Full-width row strips covering a width x height grid."""
    for row_off in range(0, height, chunk_rows):
        yield Window(0, row_off, width, min(chunk_rows, height - row_off))


def grid_raster(src_path: Path, grid: dict, out_path: Path, resampling: Resampling, threads: int, chunk_rows: int) -> None:
    """Warp band 1 of one raster onto the grid strip by strip, so memory stays bounded."""
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"{src_path.name} has no CRS; cannot grid it")
        nodata = resolve_nodata(src)
        profile = {
            "driver": "GTiff", "crs": grid["crs"], "transform": grid["transform"],
            "width": grid["width"], "height": grid["height"],
            "count": 1, "dtype": src.dtypes[0], "nodata": nodata,
            "compress": "lzw", "tiled": True, "BIGTIFF": "IF_SAFER",
        }
        vrt_options = {
            "crs": grid["crs"], "transform": grid["transform"],
            "width": grid["width"], "height": grid["height"],
            "resampling": resampling, "src_nodata": nodata, "nodata": nodata,
            "warp_mem_limit": WARP_MEM_LIMIT_MB, "warp_extras": {"NUM_THREADS": threads},
        }
        with WarpedVRT(src, **vrt_options) as vrt, rasterio.open(out_path, "w", **profile) as dst:
            for window in strip_windows(grid["width"], grid["height"], chunk_rows):
                dst.write(vrt.read(1, window=window), 1, window=window)


def grid_all(
    jobs: list[tuple[Path, Path, Resampling]], grid: dict, workers: int, chunk_rows: int,
) -> list[str]:
    """Grid (source, output, resampling) jobs concurrently; returns names of failed sources."""
    processes = max(1, min(workers, len(jobs)))
    threads = max(1, workers // processes)
    with ProcessPoolExecutor(max_workers=processes) as pool:
        futures = [
            (src, out, rs, pool.submit(grid_raster, src, grid, out, rs, threads, chunk_rows))
            for src, out, rs in jobs
        ]
    failures = []
    for src, out, resampling, future in futures:
        exc = future.exception()
        if exc is None:
            print(f"  ok    {src.name} [{resampling.name}] -> {out}")
        else:
            failures.append(src.name)
            print(f"  FAIL  {src.name}: {exc}", file=sys.stderr)
    return failures


def unique_name(name: str, taken: set[str]) -> str:
    """Append _2, _3, ... until name no longer collides."""
    candidate, suffix = name, 2
    while candidate in taken:
        candidate, suffix = f"{name}_{suffix}", suffix + 1
    return candidate


def column_specs(gridded: list[Path]) -> list[tuple[Path, str]]:
    """(gridded raster, unique column name), one per raster."""
    specs, taken = [], {ELEVATION_COLUMN, "row", "col"}
    for path in gridded:
        name = unique_name(column_base_name(path), taken)
        taken = taken | {name}
        specs.append((path, name))
    return specs


def read_window(src: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    """Band 1 over a window as float32, nodata as NaN."""
    return src.read(1, window=window, masked=True).astype(VALUE_DTYPE).filled(np.nan)


def strip_keep_mask(elevation: np.ndarray, grid: dict, window: Window) -> np.ndarray:
    """Cells to keep in a strip: valid DEM and, with an ROI, centre inside its polygons."""
    keep = ~np.isnan(elevation)
    if grid["shapes"] is None:
        return keep
    inside = geometry_mask(
        grid["shapes"], out_shape=elevation.shape,
        transform=window_transform(window, grid["transform"]), invert=True,
    )
    return keep & inside


def read_strip(dem_grid: Path, grid: dict, specs: list[tuple[Path, str]], window: Window) -> dict | None:
    """Columns for the kept cells in one grid strip; None if nothing in it is kept."""
    with ExitStack() as stack:
        elevation = read_window(stack.enter_context(rasterio.open(dem_grid)), window)
        keep = strip_keep_mask(elevation, grid, window)
        if not keep.any():
            return None
        rows, cols = np.nonzero(keep)
        rows = rows + window.row_off
        values = {name: read_window(stack.enter_context(rasterio.open(path)), window)[keep] for path, name in specs}
    return {
        "row": rows.astype("int32"), "col": cols.astype("int32"),
        ELEVATION_COLUMN: elevation[keep], **values,
    }


def iter_strips(dem_grid: Path, grid: dict, specs: list, workers: int, chunk_rows: int):
    """Yield (last row, strip) in row order, read by worker processes with bounded read-ahead."""
    windows = deque(strip_windows(grid["width"], grid["height"], chunk_rows))
    max_pending = max(1, min(MAX_PENDING_STRIPS, workers))
    with ProcessPoolExecutor(max_workers=max_pending) as pool:
        pending = deque()
        while windows or pending:
            while windows and len(pending) < max_pending:
                window = windows.popleft()
                pending.append((window.row_off + window.height, pool.submit(read_strip, dem_grid, grid, specs, window)))
            row_end, future = pending.popleft()
            yield row_end, future.result()


def cell_polygons(rows: np.ndarray, cols: np.ndarray, transform: Affine) -> np.ndarray:
    """Square cell outlines for (row, col) indices on a north-up grid."""
    x0, y0 = transform * (cols, rows)
    x1, y1 = transform * (cols + 1, rows + 1)
    return shapely.box(x0, y1, x1, y0)


def build_combined_gpkg(
    dem_grid: Path, grid: dict, gridded: list[Path], gpkg_path: Path, layer: str, workers: int, chunk_rows: int,
) -> int:
    """Write one square polygon per kept cell with every raster as a column. Returns row count."""
    specs = column_specs(gridded)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gpkg_path.unlink(missing_ok=True)

    written = 0
    for row_end, strip in iter_strips(dem_grid, grid, specs, workers, chunk_rows):
        if strip is None:
            continue
        geometry = cell_polygons(strip["row"], strip["col"], grid["transform"])
        gdf = gpd.GeoDataFrame(pd.DataFrame(strip), geometry=geometry, crs=grid["crs"])
        gdf.to_file(gpkg_path, layer=layer, driver="GPKG", engine="pyogrio", append=written > 0)
        written += len(gdf)
        print(f"  rows {row_end:>6}/{grid['height']}  cells written {written:,}", flush=True)
    return written


def validate_args(args: argparse.Namespace, input_dir: Path, dem_path: Path, roi_path: Path | None) -> str | None:
    """Error message for invalid arguments, or None."""
    if not input_dir.is_dir():
        return f"--input-dir {input_dir} is not a directory"
    if not dem_path.is_file():
        return f"--dem {dem_path} does not exist"
    if roi_path is not None and not roi_path.exists():
        return f"--roi {roi_path} does not exist"
    if args.cell_size <= 0:
        return "--cell-size must be positive"
    if args.workers < 1:
        return "--workers must be at least 1"
    if args.chunk_rows < 1:
        return "--chunk-rows must be at least 1"
    return None


def main(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser()
    dem_path = Path(args.dem).expanduser()
    roi_path = Path(args.roi).expanduser() if args.roi else None
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else input_dir / DEFAULT_OUTPUT_SUBDIR
    gpkg_path = Path(args.gpkg).expanduser() if args.gpkg else output_dir / DEFAULT_GPKG_NAME

    error = validate_args(args, input_dir, dem_path, roi_path)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        grid = read_grid(dem_path, roi_path, args.cell_size)
    except (rasterio.errors.RasterioError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    rasters = find_rasters(input_dir, exclude=dem_path, output_dir=output_dir)
    if not rasters:
        print(f"No .tif/.tiff files found in {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    scope = f"cropped to ROI {roi_path.name}" if roi_path else "full DEM extent"
    print(f"Grid ({scope}): {grid['width']} x {grid['height']} cells of {args.cell_size:g} m, "
          f"{grid['crs']}; {args.workers} workers, {args.chunk_rows}-row chunks")

    dem_grid = output_dir / dem_path.name
    jobs = [(dem_path, dem_grid, Resampling.average)] + [
        (src, output_dir / src.name, resampling_for(src, args.cell_size)) for src in rasters
    ]
    failures = grid_all(jobs, grid, args.workers, args.chunk_rows)
    if dem_path.name in failures:
        print("Error: the DEM itself could not be gridded", file=sys.stderr)
        return 1
    gridded = [output_dir / src.name for src in rasters if src.name not in failures]
    print(f"Gridded {len(gridded)}/{len(rasters)} rasters (+ DEM) into {output_dir}")

    n_cells = build_combined_gpkg(dem_grid, grid, gridded, gpkg_path, args.layer, args.workers, args.chunk_rows)
    print(f"Wrote {n_cells:,} cells ({len(gridded)} rasters + elevation) to {gpkg_path} [layer '{args.layer}']")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid all rasters in a directory onto a DEM-snapped grid and combine them into one GeoPackage."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing the .tif rasters")
    parser.add_argument("--dem", required=True, help="DEM GeoTIFF that sets the grid's CRS, origin and extent")
    parser.add_argument("--roi", default=None, help="Optional ROI vector file (.gpkg/.shp/.geojson) to crop and mask to")
    parser.add_argument(
        "--cell-size", type=float, default=DEFAULT_CELL_SIZE_M,
        help=f"Grid cell size in DEM CRS units, metres (default: {DEFAULT_CELL_SIZE_M:g})",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Where to write gridded rasters (default: <input-dir>/{DEFAULT_OUTPUT_SUBDIR})",
    )
    parser.add_argument(
        "--gpkg", default=None,
        help=f"Combined GeoPackage path (default: <output-dir>/{DEFAULT_GPKG_NAME})",
    )
    parser.add_argument("--layer", default=DEFAULT_LAYER, help=f"GeoPackage layer name (default: {DEFAULT_LAYER})")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1,
        help="Parallel processes/threads to use (default: all cores)",
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
        help=f"Grid rows processed per chunk; lower uses less memory (default: {DEFAULT_CHUNK_ROWS})",
    )
    sys.exit(main(parser.parse_args()))
