#!/usr/bin/env python3
"""Refine slope-unit susceptibility into a cell-scale landslide source mask.

The calculation uses the infinite-slope Factor of Safety equation supplied for
Objective 2, but replaces slope-unit mean slope with a gridded slope raster.
The slope raster is the target grid; coarser Objective 2 rasters are aligned to
it during processing.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import logging
import math
import os
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Prefer the GDAL/PROJ data bundled with Rasterio. External GIS applications
# can leave incompatible overrides in the calling process.
for _gis_override in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_gis_override, None)

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.io import DatasetReader  # noqa: E402
from rasterio.vrt import WarpedVRT  # noqa: E402
from rasterio.windows import Window  # noqa: E402


LOGGER = logging.getLogger("fos_source_refinement")
DEFAULT_FLOAT_NODATA = -9999.0
DEFAULT_MASK_NODATA = 255
DEFAULT_FOS_THRESHOLD = 1.5
DEFAULT_BLOCK_ROWS = 256
WATER_UNIT_WEIGHT_KN_M3 = 9.81
SLIDING_BLOCK_THICKNESS_M = 3.33
TAN_SLOPE_EPSILON = 1.0e-12


@dataclass(frozen=True)
class SourceRefinementParams:
    susceptibility_raster: Path
    cohesion_raster: Path
    friction_angle_raster: Path
    saturation_ratio_raster: Path
    total_unit_weight_raster: Path
    slope_angle_raster: Path
    output_fos_raster: Path
    output_source_mask_raster: Path
    output_summary_json: Path
    total_unit_weight_band: int = 1
    total_unit_weight_scale_to_kn_m3: float = 1.0
    total_unit_weight_resampling: str = "bilinear"
    friction_angle_unit: str = "radians"
    fos_threshold: float = DEFAULT_FOS_THRESHOLD
    float_nodata: float = DEFAULT_FLOAT_NODATA
    mask_nodata: int = DEFAULT_MASK_NODATA
    block_rows: int = DEFAULT_BLOCK_ROWS
    overwrite: bool = False


@dataclass
class RunningStatistics:
    count: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    total: float = 0.0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
        if finite.size == 0:
            return
        self.count += int(finite.size)
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))
        self.total += float(finite.sum(dtype=np.float64))

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total / self.count,
        }


def calculate_factor_of_safety(
    cohesion_kpa: np.ndarray,
    friction_angle: np.ndarray,
    saturation_ratio: np.ndarray,
    total_unit_weight_kn_m3: np.ndarray,
    slope_angle_deg: np.ndarray,
    valid_mask: np.ndarray,
    *,
    friction_angle_unit: str = "radians",
) -> tuple[np.ndarray, np.ndarray]:
    """Return Factor of Safety and its final valid-cell mask.

    c' in kPa is numerically equivalent to kN/m2. Combined with soil unit
    weight in kN/m3 and block thickness in m, every term is dimensionless.
    Slope is supplied in degrees. Friction angle can be supplied in radians
    (the Objective 2 output convention) or degrees, as declared by config.
    """

    arrays = (
        cohesion_kpa,
        friction_angle,
        saturation_ratio,
        total_unit_weight_kn_m3,
        slope_angle_deg,
        valid_mask,
    )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("All FoS arrays and the validity mask must have identical shapes.")
    physical = (
        valid_mask
        & np.isfinite(cohesion_kpa)
        & np.isfinite(friction_angle)
        & np.isfinite(saturation_ratio)
        & np.isfinite(total_unit_weight_kn_m3)
        & np.isfinite(slope_angle_deg)
        & (cohesion_kpa >= 0.0)
        & (saturation_ratio >= 0.0)
        & (saturation_ratio <= 1.0)
        & (total_unit_weight_kn_m3 > 0.0)
        & (slope_angle_deg > 0.0)
        & (slope_angle_deg < 90.0)
    )

    slope_rad = np.deg2rad(slope_angle_deg)
    if friction_angle_unit == "radians":
        friction_rad = friction_angle
    elif friction_angle_unit == "degrees":
        friction_rad = np.deg2rad(friction_angle)
    else:
        raise ValueError("friction_angle_unit must be 'radians' or 'degrees'.")
    physical &= (friction_rad >= 0.0) & (friction_rad < (np.pi / 2.0))
    sin_slope = np.sin(slope_rad)
    tan_slope = np.tan(slope_rad)
    tan_friction = np.tan(friction_rad)
    physical &= (
        np.isfinite(sin_slope)
        & np.isfinite(tan_slope)
        & np.isfinite(tan_friction)
        & (sin_slope > 0.0)
        & (np.abs(tan_slope) > TAN_SLOPE_EPSILON)
    )

    fos = np.full(cohesion_kpa.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cohesion_term = cohesion_kpa[physical] / (
            SLIDING_BLOCK_THICKNESS_M
            * total_unit_weight_kn_m3[physical]
            * sin_slope[physical]
        )
        friction_term = tan_friction[physical] / tan_slope[physical]
        pore_pressure_term = (
            saturation_ratio[physical]
            * WATER_UNIT_WEIGHT_KN_M3
            * tan_friction[physical]
        ) / (total_unit_weight_kn_m3[physical] * tan_slope[physical])
        fos[physical] = cohesion_term + friction_term - pore_pressure_term

    final_valid = physical & np.isfinite(fos)
    fos[~final_valid] = np.nan
    return fos, final_valid


def refine_source_mask(
    susceptibility_valid: np.ndarray,
    terrain_valid: np.ndarray,
    fos: np.ndarray,
    fos_valid: np.ndarray,
    *,
    fos_threshold: float,
    mask_nodata: int = DEFAULT_MASK_NODATA,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build the refined source mask and return masks used for diagnostics.

    Every valid susceptibility pixel is a susceptible slope-unit cell. NoData
    in the susceptibility raster over valid slope terrain means non-susceptible
    and becomes 0. Cells outside valid slope terrain remain output NoData.
    Susceptible cells are 1 when FoS is at or below the threshold and 0 when
    FoS is above it. Susceptible cells whose FoS cannot be calculated remain
    NoData so missing inputs are not silently interpreted as stable terrain.
    """

    arrays = (susceptibility_valid, terrain_valid, fos, fos_valid)
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Susceptibility and FoS arrays must have identical shapes.")
    if not math.isfinite(fos_threshold):
        raise ValueError("fos_threshold must be finite.")
    if mask_nodata in (0, 1) or not 0 <= mask_nodata <= 255:
        raise ValueError("mask_nodata must be an integer from 2 through 255.")

    susceptible = terrain_valid & susceptibility_valid
    resolved_susceptible = susceptible & fos_valid
    retained = resolved_susceptible & (fos <= fos_threshold)
    excluded_stable = resolved_susceptible & (fos > fos_threshold)
    unresolved_susceptible = susceptible & ~fos_valid

    source = np.full(susceptibility_valid.shape, mask_nodata, dtype=np.uint8)
    source[terrain_valid & ~susceptible] = 0
    source[excluded_stable] = 0
    source[retained] = 1
    return source, {
        "susceptible": susceptible,
        "retained": retained,
        "excluded_stable": excluded_stable,
        "unresolved_susceptible": unresolved_susceptible,
    }


def _read_float_window(
    dataset: DatasetReader,
    window: Window,
    band: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    masked = dataset.read(band, window=window, masked=True)
    values = np.asarray(masked.filled(np.nan), dtype=np.float64)
    valid = ~np.ma.getmaskarray(masked) & np.isfinite(values)
    return values, valid


def _resampling_from_name(value: str) -> Resampling:
    methods = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
    }
    try:
        return methods[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported resampling method: {value}") from exc


def _aligned_vrt(
    stack: ExitStack,
    dataset: DatasetReader,
    reference: DatasetReader,
    *,
    resampling: str,
) -> WarpedVRT:
    """Return a float64 VRT aligned exactly to the slope reference grid."""

    if dataset.crs is None:
        raise ValueError(f"Input raster must declare a CRS: {dataset.name}")
    return stack.enter_context(
        WarpedVRT(
            dataset,
            crs=reference.crs,
            transform=reference.transform,
            width=reference.width,
            height=reference.height,
            resampling=_resampling_from_name(resampling),
            nodata=DEFAULT_FLOAT_NODATA,
            dtype="float64",
        )
    )


def _input_grid_summary(
    dataset: DatasetReader,
    *,
    band: int,
    resampling: str,
    scale_to_model_units: float,
) -> dict[str, Any]:
    return {
        "band": band,
        "band_count": dataset.count,
        "dtype": dataset.dtypes[band - 1],
        "nodata": dataset.nodatavals[band - 1],
        "crs": str(dataset.crs),
        "width": dataset.width,
        "height": dataset.height,
        "resolution": list(dataset.res),
        "transform": list(dataset.transform)[:6],
        "resampling_to_slope_grid": resampling,
        "scale_to_model_units": scale_to_model_units,
    }


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp{target.suffix}")


def _float_profile(reference: DatasetReader, nodata: float) -> dict[str, Any]:
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=nodata,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="if_safer",
    )
    return profile


def _mask_profile(reference: DatasetReader, nodata: int) -> dict[str, Any]:
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="if_safer",
    )
    profile.pop("predictor", None)
    return profile


def _validate_params(params: SourceRefinementParams) -> None:
    inputs = (
        params.susceptibility_raster,
        params.cohesion_raster,
        params.friction_angle_raster,
        params.saturation_ratio_raster,
        params.total_unit_weight_raster,
        params.slope_angle_raster,
    )
    outputs = (
        params.output_fos_raster,
        params.output_source_mask_raster,
        params.output_summary_json,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Input raster does not exist: {path}")
    resolved_inputs = {path.resolve() for path in inputs}
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("FoS, source-mask, and summary output paths must be distinct.")
    if resolved_inputs.intersection(resolved_outputs):
        raise ValueError("An output path must not overwrite an input raster.")
    for path in outputs:
        if path.exists() and not params.overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Set overwrite=true to replace it."
            )
    if params.total_unit_weight_band <= 0:
        raise ValueError("total_unit_weight_band must be a positive integer.")
    if (
        not math.isfinite(params.total_unit_weight_scale_to_kn_m3)
        or params.total_unit_weight_scale_to_kn_m3 <= 0.0
    ):
        raise ValueError("total_unit_weight_scale_to_kn_m3 must be greater than 0.")
    _resampling_from_name(params.total_unit_weight_resampling)
    if params.friction_angle_unit not in {"radians", "degrees"}:
        raise ValueError("friction_angle_unit must be 'radians' or 'degrees'.")
    if not math.isfinite(params.fos_threshold):
        raise ValueError("fos_threshold must be finite.")
    if not math.isfinite(params.float_nodata):
        raise ValueError("float_nodata must be finite.")
    if params.mask_nodata in (0, 1) or not 0 <= params.mask_nodata <= 255:
        raise ValueError("mask_nodata must be an integer from 2 through 255.")
    if params.block_rows <= 0:
        raise ValueError("block_rows must be greater than 0.")


def run_source_refinement(params: SourceRefinementParams) -> dict[str, Any]:
    """Calculate cell-scale FoS and atomically publish the refined source mask."""

    _validate_params(params)
    outputs = (
        params.output_fos_raster,
        params.output_source_mask_raster,
        params.output_summary_json,
    )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    temp_fos, temp_mask, temp_summary = (_temporary_path(path) for path in outputs)
    temp_paths = (temp_fos, temp_mask, temp_summary)

    counts = {
        "total_cells": 0,
        "susceptibility_nodata_cells": 0,
        "susceptible_cells": 0,
        "non_susceptible_cells": 0,
        "fos_valid_cells": 0,
        "fos_nodata_cells": 0,
        "retained_source_cells": 0,
        "stable_cells_excluded_from_susceptible_units": 0,
        "unresolved_cells_in_susceptible_units": 0,
        "cells_exactly_at_fos_threshold": 0,
        "negative_cohesion_cells": 0,
        "friction_out_of_range_cells": 0,
        "saturation_ratio_out_of_range_cells": 0,
        "nonpositive_total_unit_weight_cells": 0,
        "slope_out_of_range_cells": 0,
    }
    input_nodata = {
        "susceptibility": 0,
        "cohesion": 0,
        "friction_angle": 0,
        "saturation_ratio": 0,
        "total_unit_weight": 0,
        "slope_angle": 0,
    }
    fos_stats = RunningStatistics()

    try:
        with ExitStack() as stack:
            slope_src = stack.enter_context(rasterio.open(params.slope_angle_raster))
            susceptibility_src = stack.enter_context(
                rasterio.open(params.susceptibility_raster)
            )
            cohesion_src = stack.enter_context(rasterio.open(params.cohesion_raster))
            friction_src = stack.enter_context(
                rasterio.open(params.friction_angle_raster)
            )
            saturation_src = stack.enter_context(
                rasterio.open(params.saturation_ratio_raster)
            )
            total_unit_weight_src = stack.enter_context(
                rasterio.open(params.total_unit_weight_raster)
            )
            if slope_src.count != 1:
                raise ValueError("Slope raster must contain exactly one band.")
            if slope_src.crs is None:
                raise ValueError("Slope raster must declare a CRS.")
            for dataset, label in (
                (susceptibility_src, "Susceptibility raster"),
                (cohesion_src, "Cohesion raster"),
                (friction_src, "Friction-angle raster"),
                (saturation_src, "Saturation-ratio raster"),
            ):
                if dataset.count != 1:
                    raise ValueError(f"{label} must contain exactly one band.")
                if dataset.crs is None:
                    raise ValueError(f"{label} must declare a CRS.")
            if not 1 <= params.total_unit_weight_band <= total_unit_weight_src.count:
                raise ValueError(
                    "total_unit_weight_band is outside the raster's available "
                    f"band range 1..{total_unit_weight_src.count}."
                )
            if total_unit_weight_src.crs is None:
                raise ValueError("Total-unit-weight raster must declare a CRS.")

            input_grids = {
                "susceptibility": _input_grid_summary(
                    susceptibility_src,
                    band=1,
                    resampling="nearest",
                    scale_to_model_units=1.0,
                ),
                "cohesion": _input_grid_summary(
                    cohesion_src,
                    band=1,
                    resampling="nearest",
                    scale_to_model_units=1.0,
                ),
                "friction_angle": _input_grid_summary(
                    friction_src,
                    band=1,
                    resampling="nearest",
                    scale_to_model_units=1.0,
                ),
                "saturation_ratio": _input_grid_summary(
                    saturation_src,
                    band=1,
                    resampling="nearest",
                    scale_to_model_units=1.0,
                ),
                "total_unit_weight": _input_grid_summary(
                    total_unit_weight_src,
                    band=params.total_unit_weight_band,
                    resampling=params.total_unit_weight_resampling,
                    scale_to_model_units=params.total_unit_weight_scale_to_kn_m3,
                ),
                "slope_angle": _input_grid_summary(
                    slope_src,
                    band=1,
                    resampling="none (target grid)",
                    scale_to_model_units=1.0,
                ),
            }

            susceptibility_grid = _aligned_vrt(
                stack, susceptibility_src, slope_src, resampling="nearest"
            )
            cohesion_grid = _aligned_vrt(
                stack, cohesion_src, slope_src, resampling="nearest"
            )
            friction_grid = _aligned_vrt(
                stack, friction_src, slope_src, resampling="nearest"
            )
            saturation_grid = _aligned_vrt(
                stack, saturation_src, slope_src, resampling="nearest"
            )
            total_unit_weight_grid = _aligned_vrt(
                stack,
                total_unit_weight_src,
                slope_src,
                resampling=params.total_unit_weight_resampling,
            )

            with (
                rasterio.open(
                    temp_fos,
                    "w",
                    **_float_profile(slope_src, params.float_nodata),
                ) as fos_dst,
                rasterio.open(
                    temp_mask,
                    "w",
                    **_mask_profile(slope_src, params.mask_nodata),
                ) as mask_dst,
            ):
                fos_dst.set_band_description(1, "Cell-scale Factor of Safety")
                fos_dst.update_tags(
                    MODEL="infinite_slope_factor_of_safety",
                    EQUATION=(
                        "c/(t*gamma_t*sin(theta)) + tan(phi)/tan(theta) "
                        "- (m*gamma_w*tan(phi))/(gamma_t*tan(theta))"
                    ),
                    COHESION_UNIT="kPa",
                    SLOPE_ANGLE_UNIT="degrees",
                    FRICTION_ANGLE_UNIT=params.friction_angle_unit,
                    TOTAL_UNIT_WEIGHT_UNIT="kN/m3",
                    TOTAL_UNIT_WEIGHT_SOURCE_BAND=str(params.total_unit_weight_band),
                    TOTAL_UNIT_WEIGHT_SCALE_TO_KN_M3=str(
                        params.total_unit_weight_scale_to_kn_m3
                    ),
                    WATER_UNIT_WEIGHT_KN_M3=str(WATER_UNIT_WEIGHT_KN_M3),
                    SLIDING_BLOCK_THICKNESS_M=str(SLIDING_BLOCK_THICKNESS_M),
                    FOS_THRESHOLD=str(params.fos_threshold),
                )
                mask_dst.set_band_description(1, "FoS-refined landslide source mask")
                mask_dst.update_tags(
                    VALUE_0="non-source or FoS above threshold",
                    VALUE_1="susceptible slope-unit cell with FoS at or below threshold",
                    NODATA_MEANING="outside valid slope terrain or unresolved FoS in susceptible unit",
                    FOS_THRESHOLD=str(params.fos_threshold),
                )

                for row in range(0, slope_src.height, params.block_rows):
                    height = min(params.block_rows, slope_src.height - row)
                    window = Window(0, row, slope_src.width, height)
                    _susceptibility, susceptibility_valid = _read_float_window(
                        susceptibility_grid, window
                    )
                    cohesion, cohesion_valid = _read_float_window(cohesion_grid, window)
                    friction, friction_valid = _read_float_window(friction_grid, window)
                    saturation, saturation_valid = _read_float_window(saturation_grid, window)
                    total_unit_weight_raw, total_unit_weight_valid = _read_float_window(
                        total_unit_weight_grid,
                        window,
                        band=params.total_unit_weight_band,
                    )
                    total_unit_weight = (
                        total_unit_weight_raw * params.total_unit_weight_scale_to_kn_m3
                    )
                    slope, slope_valid = _read_float_window(slope_src, window)

                    window_cells = int(height * slope_src.width)
                    counts["total_cells"] += window_cells
                    for name, valid in (
                        ("susceptibility", susceptibility_valid),
                        ("cohesion", cohesion_valid),
                        ("friction_angle", friction_valid),
                        ("saturation_ratio", saturation_valid),
                        ("total_unit_weight", total_unit_weight_valid),
                        ("slope_angle", slope_valid),
                    ):
                        input_nodata[name] += window_cells - int(np.count_nonzero(valid))

                    counts["negative_cohesion_cells"] += int(
                        np.count_nonzero(cohesion_valid & (cohesion < 0.0))
                    )
                    friction_limit = (
                        np.pi / 2.0
                        if params.friction_angle_unit == "radians"
                        else 90.0
                    )
                    counts["friction_out_of_range_cells"] += int(
                        np.count_nonzero(
                            friction_valid
                            & ((friction < 0.0) | (friction >= friction_limit))
                        )
                    )
                    counts["saturation_ratio_out_of_range_cells"] += int(
                        np.count_nonzero(
                            saturation_valid & ((saturation < 0.0) | (saturation > 1.0))
                        )
                    )
                    counts["nonpositive_total_unit_weight_cells"] += int(
                        np.count_nonzero(
                            total_unit_weight_valid & (total_unit_weight <= 0.0)
                        )
                    )
                    counts["slope_out_of_range_cells"] += int(
                        np.count_nonzero(slope_valid & ((slope <= 0.0) | (slope >= 90.0)))
                    )

                    combined_valid = (
                        susceptibility_valid
                        & cohesion_valid
                        & friction_valid
                        & saturation_valid
                        & total_unit_weight_valid
                        & slope_valid
                    )
                    fos, fos_valid = calculate_factor_of_safety(
                        cohesion,
                        friction,
                        saturation,
                        total_unit_weight,
                        slope,
                        combined_valid,
                        friction_angle_unit=params.friction_angle_unit,
                    )
                    source_mask, diagnostic_masks = refine_source_mask(
                        susceptibility_valid,
                        slope_valid,
                        fos,
                        fos_valid,
                        fos_threshold=params.fos_threshold,
                        mask_nodata=params.mask_nodata,
                    )

                    fos_output = np.where(
                        fos_valid, fos, params.float_nodata
                    ).astype(np.float32)
                    fos_dst.write(fos_output, 1, window=window)
                    mask_dst.write(source_mask, 1, window=window)
                    fos_stats.update(fos[fos_valid])

                    susceptible = diagnostic_masks["susceptible"]
                    counts["susceptibility_nodata_cells"] += int(
                        np.count_nonzero(~susceptibility_valid)
                    )
                    counts["susceptible_cells"] += int(np.count_nonzero(susceptible))
                    counts["non_susceptible_cells"] += int(
                        np.count_nonzero(slope_valid & ~susceptibility_valid)
                    )
                    counts["fos_valid_cells"] += int(np.count_nonzero(fos_valid))
                    counts["fos_nodata_cells"] += window_cells - int(np.count_nonzero(fos_valid))
                    counts["retained_source_cells"] += int(
                        np.count_nonzero(diagnostic_masks["retained"])
                    )
                    counts["stable_cells_excluded_from_susceptible_units"] += int(
                        np.count_nonzero(diagnostic_masks["excluded_stable"])
                    )
                    counts["unresolved_cells_in_susceptible_units"] += int(
                        np.count_nonzero(diagnostic_masks["unresolved_susceptible"])
                    )
                    counts["cells_exactly_at_fos_threshold"] += int(
                        np.count_nonzero(
                            susceptible & fos_valid & (fos == params.fos_threshold)
                        )
                    )

            susceptible_count = counts["susceptible_cells"]
            resolved_susceptible = (
                counts["retained_source_cells"]
                + counts["stable_cells_excluded_from_susceptible_units"]
            )
            warnings: list[str] = []
            if counts["unresolved_cells_in_susceptible_units"]:
                warnings.append(
                    "Some susceptible cells have unresolved FoS and remain NoData in the source mask."
                )
            if any(
                counts[key]
                for key in (
                    "negative_cohesion_cells",
                    "friction_out_of_range_cells",
                    "saturation_ratio_out_of_range_cells",
                    "nonpositive_total_unit_weight_cells",
                    "slope_out_of_range_cells",
                )
            ):
                warnings.append(
                    "Physically invalid parameter cells were written as FoS NoData; review the summary counts."
                )

            summary: dict[str, Any] = {
                "step": "fos_source_refinement",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "method": {
                    "equation": (
                        "FS = c'/(t*gamma_t*sin(theta)) + tan(phi')/tan(theta) "
                        "- (m*gamma_w*tan(phi'))/(gamma_t*tan(theta))"
                    ),
                    "slope_basis": "unaggregated gridded slope angle",
                    "source_rule": (
                        "valid susceptibility pixel AND valid FoS <= threshold"
                    ),
                    "susceptibility_rule": (
                        "valid pixels are susceptible; susceptibility NoData over "
                        "valid slope terrain is non-susceptible"
                    ),
                    "stable_exclusion_rule": "FoS > threshold",
                    "threshold_inclusive_for_source": True,
                    "target_grid": "slope_angle_raster",
                    "resampling_performed": True,
                    "processing_mode": "windowed",
                    "temporary_output_commit": True,
                },
                "parameters": {
                    "friction_angle_unit": params.friction_angle_unit,
                    "total_unit_weight_band": params.total_unit_weight_band,
                    "total_unit_weight_scale_to_kn_m3": (
                        params.total_unit_weight_scale_to_kn_m3
                    ),
                    "total_unit_weight_resampling": params.total_unit_weight_resampling,
                    "water_unit_weight_kn_m3": WATER_UNIT_WEIGHT_KN_M3,
                    "sliding_block_thickness_m": SLIDING_BLOCK_THICKNESS_M,
                    "fos_threshold": params.fos_threshold,
                    "float_nodata": params.float_nodata,
                    "mask_nodata": params.mask_nodata,
                    "block_rows": params.block_rows,
                },
                "inputs": {
                    "susceptibility_raster": str(params.susceptibility_raster.resolve()),
                    "cohesion_raster": str(params.cohesion_raster.resolve()),
                    "friction_angle_raster": str(params.friction_angle_raster.resolve()),
                    "saturation_ratio_raster": str(params.saturation_ratio_raster.resolve()),
                    "total_unit_weight_raster": str(
                        params.total_unit_weight_raster.resolve()
                    ),
                    "slope_angle_raster": str(params.slope_angle_raster.resolve()),
                },
                "outputs": {
                    "factor_of_safety_raster": str(params.output_fos_raster.resolve()),
                    "refined_source_mask_raster": str(
                        params.output_source_mask_raster.resolve()
                    ),
                    "summary_json": str(params.output_summary_json.resolve()),
                },
                "grid": {
                    "height": slope_src.height,
                    "width": slope_src.width,
                    "crs": str(slope_src.crs),
                    "transform": list(slope_src.transform)[:6],
                    "resolution": list(slope_src.res),
                    "bounds": list(slope_src.bounds),
                },
                "input_grids": input_grids,
                "factor_of_safety_statistics": fos_stats.as_dict(),
                "counts": counts,
                "input_nodata_cells": input_nodata,
                "percentages": {
                    "retained_of_all_susceptible": (
                        100.0 * counts["retained_source_cells"] / susceptible_count
                        if susceptible_count
                        else None
                    ),
                    "excluded_stable_of_resolved_susceptible": (
                        100.0
                        * counts["stable_cells_excluded_from_susceptible_units"]
                        / resolved_susceptible
                        if resolved_susceptible
                        else None
                    ),
                    "unresolved_of_all_susceptible": (
                        100.0
                        * counts["unresolved_cells_in_susceptible_units"]
                        / susceptible_count
                        if susceptible_count
                        else None
                    ),
                },
                "warnings": warnings,
            }

        temp_summary.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_fos, params.output_fos_raster)
        os.replace(temp_mask, params.output_source_mask_raster)
        os.replace(temp_summary, params.output_summary_json)
        return summary
    finally:
        for path in temp_paths:
            if path.exists():
                path.unlink()


def _resolve_config_path(config_dir: Path, value: Any, field_name: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Missing required config value: {field_name}")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def load_config(config_path: Path) -> SourceRefinementParams:
    resolved = config_path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {resolved}")
    section = payload.get("fos_source_refinement", payload)
    if not isinstance(section, dict):
        raise ValueError("fos_source_refinement config section must be a JSON object.")

    required = {
        "susceptibility_raster",
        "cohesion_raster",
        "friction_angle_raster",
        "saturation_ratio_raster",
        "total_unit_weight_raster",
        "slope_angle_raster",
        "output_fos_raster",
        "output_source_mask_raster",
        "output_summary_json",
    }
    optional = {
        "fos_threshold",
        "friction_angle_unit",
        "total_unit_weight_band",
        "total_unit_weight_scale_to_kn_m3",
        "total_unit_weight_resampling",
        "float_nodata",
        "mask_nodata",
        "block_rows",
        "overwrite",
    }
    missing = sorted(required - set(section))
    unknown = sorted(set(section) - required - optional)
    if missing:
        raise ValueError("Missing required config keys: " + ", ".join(missing))
    if unknown:
        raise ValueError("Unknown config keys: " + ", ".join(unknown))
    config_dir = resolved.parent
    return SourceRefinementParams(
        susceptibility_raster=_resolve_config_path(
            config_dir, section["susceptibility_raster"], "susceptibility_raster"
        ),
        cohesion_raster=_resolve_config_path(
            config_dir, section["cohesion_raster"], "cohesion_raster"
        ),
        friction_angle_raster=_resolve_config_path(
            config_dir, section["friction_angle_raster"], "friction_angle_raster"
        ),
        saturation_ratio_raster=_resolve_config_path(
            config_dir, section["saturation_ratio_raster"], "saturation_ratio_raster"
        ),
        total_unit_weight_raster=_resolve_config_path(
            config_dir,
            section["total_unit_weight_raster"],
            "total_unit_weight_raster",
        ),
        slope_angle_raster=_resolve_config_path(
            config_dir, section["slope_angle_raster"], "slope_angle_raster"
        ),
        output_fos_raster=_resolve_config_path(
            config_dir, section["output_fos_raster"], "output_fos_raster"
        ),
        output_source_mask_raster=_resolve_config_path(
            config_dir,
            section["output_source_mask_raster"],
            "output_source_mask_raster",
        ),
        output_summary_json=_resolve_config_path(
            config_dir, section["output_summary_json"], "output_summary_json"
        ),
        total_unit_weight_band=int(section.get("total_unit_weight_band", 1)),
        total_unit_weight_scale_to_kn_m3=float(
            section.get("total_unit_weight_scale_to_kn_m3", 1.0)
        ),
        total_unit_weight_resampling=str(
            section.get("total_unit_weight_resampling", "bilinear")
        ).strip().lower(),
        friction_angle_unit=str(section.get("friction_angle_unit", "radians"))
        .strip()
        .lower(),
        fos_threshold=float(section.get("fos_threshold", DEFAULT_FOS_THRESHOLD)),
        float_nodata=float(section.get("float_nodata", DEFAULT_FLOAT_NODATA)),
        mask_nodata=int(section.get("mask_nodata", DEFAULT_MASK_NODATA)),
        block_rows=int(section.get("block_rows", DEFAULT_BLOCK_ROWS)),
        overwrite=bool(section.get("overwrite", False)),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate cell-scale Factor of Safety and refine an Objective 2 "
            "slope-unit susceptibility raster into a source mask."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to the JSON config.")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the config's overwrite setting.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
    args = parse_args(argv)
    try:
        params = load_config(args.config)
        if args.overwrite is not None:
            params = replace(params, overwrite=args.overwrite)
        summary = run_source_refinement(params)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Factor of Safety raster: %s", summary["outputs"]["factor_of_safety_raster"])
    LOGGER.info("Refined source mask: %s", summary["outputs"]["refined_source_mask_raster"])
    LOGGER.info("Summary: %s", summary["outputs"]["summary_json"])
    for warning in summary["warnings"]:
        LOGGER.warning("%s", warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
