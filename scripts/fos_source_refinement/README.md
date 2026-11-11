# FoS Source-Area Refinement

## Purpose

Objective 2 identifies potential landslide initiation at slope-unit scale. A
susceptible classification therefore applies to an entire slope unit even when
stable and unstable terrain coexist inside it. This preparation step converts
that regional classification into a cell-scale source mask for the
D-Infinity runout model.

The regional decision is preserved: cells outside susceptible slope units are
never added as sources. In the supplied Objective 2 susceptibility raster,
every valid pixel belongs to a susceptible slope unit and NoData over valid
terrain means non-susceptible. Inside susceptible units, local Factor of Safety
(FoS) is recalculated with the gridded slope angle. Cells with `FoS > 1.5` are
excluded; cells with valid `FoS <= 1.5` are retained as effective source cells.

## Equation

The step evaluates the same infinite-slope formulation used in Objective 2:

```text
FS = c' / (t gamma_t sin(theta))
     + tan(phi') / tan(theta)
     - (m gamma_w tan(phi')) / (gamma_t tan(theta))
```

where:

- `phi'`: effective friction angle, radians in the supplied Objective 2 raster;
- `c'`: effective cohesion, kPa, equivalent to `kN/m2`;
- `theta`: unaggregated grid-cell slope angle, degrees;
- `gamma_t`: total soil unit weight, `kN/m3`;
- `gamma_w`: water unit weight, `kN/m3`;
- `t`: sliding-block thickness normal to the slope, m; and
- `m`: submerged or saturated soil proportion, dimensionless from 0 to 1.

Unlike the reference SINMAP script, this step does not calculate `m` from
specific catchment area and recharge. The Objective 2 `m` result is supplied
directly so the only intended model change is replacing slope-unit mean slope
with cell-scale slope.

## Required Inputs

The slope raster is the target grid. The 30 m Objective 2 slope-unit rasters
are aligned to it with nearest-neighbour resampling. The coarser bulk-density
raster is aligned with bilinear resampling by default. Inputs must have valid
CRS metadata, but they do not need matching dimensions, resolution, transform,
or extent.

- `susceptibility_raster`: Objective 2 slope-unit susceptibility result. Its
  numeric score is not thresholded by this step: every valid pixel is treated
  as susceptible, while NoData over valid slope terrain is non-susceptible.
- `cohesion_raster`: Objective 2 effective cohesion `c'` in kPa.
- `friction_angle_raster`: Objective 2 effective friction angle `phi'`. The
  supplied `Makilala_IFA.tif` is in radians, selected with
  `friction_angle_unit: "radians"`.
- `saturation_ratio_raster`: Objective 2 `m`, dimensionless in `[0, 1]`.
- `total_unit_weight_raster`: raster used to obtain total soil unit weight
  `gamma_t` in `kN/m3`. The supplied `Makilala_BulkDensity.tif` is a six-band
  bulk-density raster. Band 1 stores `g/cm3 x 100`, so the default config uses
  band 1 and scale `0.0981` (`0.01 x 9.81`) to produce `kN/m3`.
- `slope_angle_raster`: unaggregated DEM-derived slope in degrees; this defines
  the output CRS, extent, resolution, transform, width, and height.

The alignment methods are intentional: nearest neighbour preserves each
slope-unit value, whereas bilinear interpolation is appropriate for the
continuous bulk-density surface. No permanent aligned intermediate rasters are
created.

The model fixes water unit weight `gamma_w` at `9.81 kN/m3` and sliding-block
thickness `t` at `3.33 m`. They are recorded in each run summary and cannot be
overridden by the config.

The default config is populated with the supplied Makilala inputs. If a true
unit-weight raster is supplied later, set
`total_unit_weight_scale_to_kn_m3` to `1.0` and select its correct band.

## Outputs

- `output_fos_raster`: Float32 cell-scale FoS; NoData where the equation cannot
  be evaluated from valid, physically admissible inputs.
- `output_source_mask_raster`: UInt8 source mask:
  - `1`: valid susceptibility pixel with valid `FoS <= threshold`;
  - `0`: susceptibility NoData over valid slope terrain, or susceptible cell
    with `FoS > threshold`;
  - `255` by default: outside valid slope terrain or unresolved FoS inside a
    susceptible unit.
- `output_summary_json`: complete inputs, constants, grid metadata, FoS
  statistics, retained/excluded/unresolved counts, and warnings.

The threshold comparison is inclusive. A cell with FoS exactly `1.5` remains a
source; only values strictly greater than `1.5` are filtered out.

## Run

Edit a copy of `config.default.json`, then run:

```powershell
python steps/data_preparation/fos_source_refinement/fos_source_refinement.py `
  --config path/to/config.json
```

On Windows, the included launcher uses `MLDFI_PYTHON`, `.venv`,
`.venv-local`, or `python`, in that order:

```powershell
steps\data_preparation\fos_source_refinement\run_fos_source_refinement.bat `
  --config path\to\config.json
```

Outputs are written to temporary sibling files and published only after the
complete raster calculation and summary generation succeed. Existing outputs
are protected unless `overwrite` is set to `true` or `--overwrite` is passed.

## Quality Controls

The step:

- aligns all non-slope inputs onto the slope grid in memory;
- requires single-band susceptibility, cohesion, friction, and wetness inputs;
- supports an explicitly selected band for the unit-weight source raster;
- records every source grid, selected band, scale, and resampling method;
- treats cohesion below zero, nonpositive total unit weight, angles outside
  their physical range, `m` outside `[0, 1]`, and slope outside `(0, 90)`
  degrees as unresolved FoS;
- records all input NoData and invalid-parameter counts; and
- leaves unresolved susceptible cells as source-mask NoData instead of
  silently classifying them as stable.
