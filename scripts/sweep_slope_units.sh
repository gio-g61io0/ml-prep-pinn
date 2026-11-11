#!/usr/bin/env bash
# Grid-search r.slopeunits over (areamin, cvmin) and export each candidate as a GeoPackage.
#
# Also exports cos/sin(aspect) rasters, which scripts/score_slope_units.py needs.
# Run it from a normal shell; it launches GRASS itself:
#
#   ./scripts/sweep_slope_units.sh /path/to/dem.tif [out_dir]
#   SU_RES=20 ./scripts/sweep_slope_units.sh /path/to/dem.tif   # resample DEM to 20 m first
#
# On first run it creates a GRASS project in the DEM's CRS ($GRASS_PROJECT) and installs
# the r.slopeunits add-on if missing. See docs/slope_unit_validation.md.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <dem.tif> [out_dir]" >&2
  exit 1
fi
if [[ ! -f "$1" ]]; then
  echo "DEM not found: $1" >&2
  exit 1
fi
DEM_TIF="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT="${2:-su_candidates}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

GRASS_PROJECT="${GRASS_PROJECT:-$HOME/grassdata/su_sweep}"
SU_RES="${SU_RES:-}"   # optional working resolution (m); empty = native DEM resolution

# Not inside GRASS yet: create the project from the DEM's CRS if needed, then re-run inside it
if [[ -z "${GISBASE:-}" ]]; then
  command -v grass >/dev/null || { echo "grass not found on PATH" >&2; exit 1; }
  if [[ ! -d "$GRASS_PROJECT/PERMANENT" ]]; then
    echo "creating GRASS project $GRASS_PROJECT from $DEM_TIF"
    grass -c "$DEM_TIF" "$GRASS_PROJECT" -e
  fi
  # Install before entering the session: GRASS only puts add-ons on PATH at startup
  if ! grass "$GRASS_PROJECT/PERMANENT" --exec g.extension -a 2>/dev/null | grep -q "^r.slopeunits"; then
    echo "installing r.slopeunits add-on"
    grass "$GRASS_PROJECT/PERMANENT" --exec g.extension extension=r.slopeunits
  fi
  exec grass "$GRASS_PROJECT/PERMANENT" --exec \
    env SU_RES="$SU_RES" bash "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" "$DEM_TIF" "$OUT"
fi

AREAMIN_VALUES=(25000 50000 100000 200000)   # m²
CVMIN_VALUES=(0.1 0.2 0.3 0.4 0.5)
THRESH_FACTOR=8                              # initial thresh = THRESH_FACTOR * areamin
REDUCTION_FACTOR=2
MAX_ITERATION=30

# Some GRASS builds (e.g. 8.6dev on macOS) ship without the sqlite driver; dbf always works
db.connect driver=dbf database='$GISDBASE/$LOCATION_NAME/$MAPSET/dbf/'

r.in.gdal input="$DEM_TIF" output=dem_native --overwrite
if [[ -n "$SU_RES" ]]; then
  g.region raster=dem_native res="$SU_RES" -a
  r.resamp.stats input=dem_native output=dem method=average --overwrite
else
  g.region raster=dem_native
  g.copy raster=dem_native,dem --overwrite
fi
g.region raster=dem

# Aspect is circular, so the scorer averages its sin/cos components
r.slope.aspect elevation=dem aspect=aspect --overwrite
r.mapcalc "asp_cos = cos(aspect)" --overwrite
r.mapcalc "asp_sin = sin(aspect)" --overwrite
r.out.gdal -c input=asp_cos output="$OUT/asp_cos.tif" --overwrite
r.out.gdal -c input=asp_sin output="$OUT/asp_sin.tif" --overwrite

for A in "${AREAMIN_VALUES[@]}"; do
  for C in "${CVMIN_VALUES[@]}"; do
    TAG="a${A}_c${C/./}"
    echo "=== candidate $TAG ==="
    r.slopeunits.create demmap=dem slumap="su_$TAG" \
      thresh=$((A * THRESH_FACTOR)) areamin="$A" cvmin="$C" \
      rf="$REDUCTION_FACTOR" maxiteration="$MAX_ITERATION" --overwrite
    r.slopeunits.clean -m demmap=dem slumap="su_$TAG" slumapclean="suc_$TAG" \
      cleansize=$((A / 2)) --overwrite
    r.to.vect input="suc_$TAG" output="v_$TAG" type=area --overwrite
    v.out.ogr input="v_$TAG" output="$OUT/su_$TAG.gpkg" format=GPKG --overwrite
  done
done

echo "Done. Candidates in $OUT/ — next: venv/bin/python -m scripts.score_slope_units --scars <scars.gpkg>"
