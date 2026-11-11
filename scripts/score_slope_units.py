"""Score r.slopeunits candidates so the slope-unit (SU) delineation can be chosen objectively.

Reads the candidates written by `scripts/sweep_slope_units.sh` and computes, per candidate:

    V         area-weighted mean circular variance of aspect inside each SU (low = homogeneous)
    I         Moran's I of mean aspect between neighbouring SUs (low = boundaries on real breaks)
    F         V_norm + I_norm, normalized across candidates (Alvioli et al., 2020) -> lowest wins
    scar_fit  mean fraction of each scar's area inside its dominant SU (1 = scars never split)

    venv/bin/python -m scripts.score_slope_units \
        --candidates su_candidates \
        --scars datasets/<scar_inventory>.gpkg

Writes `<candidates>/su_scores.csv` sorted by F. Needs `pip install rasterstats libpysal esda`.
See docs/slope_unit_validation.md for how to read the results.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import Queen
from rasterstats import zonal_stats

CANDIDATE_GLOB = "su_*.gpkg"
SCORES_FILENAME = "su_scores.csv"


def aspect_stats(su: gpd.GeoDataFrame, candidates_dir: Path) -> pd.DataFrame:
    """Per-SU mean cos/sin of aspect and the resulting circular variance."""
    cos_mean = [z["mean"] for z in zonal_stats(su, str(candidates_dir / "asp_cos.tif"), stats="mean")]
    sin_mean = [z["mean"] for z in zonal_stats(su, str(candidates_dir / "asp_sin.tif"), stats="mean")]
    df = pd.DataFrame({"c": cos_mean, "s": sin_mean}).fillna(0.0)
    resultant = np.hypot(df["c"], df["s"])  # mean resultant length R
    return df.assign(circ_var=1.0 - resultant)  # circular variance = 1 - R


def scar_fit(su: gpd.GeoDataFrame, scars: gpd.GeoDataFrame) -> float:
    """Mean fraction of each scar's area inside its dominant SU (1 = never split)."""
    pieces = gpd.overlay(scars.reset_index(names="scar_id"), su, how="intersection")
    if pieces.empty:
        return float("nan")
    pieces = pieces.assign(a=pieces.area)
    dominant = pieces.groupby("scar_id")["a"].max()
    total = pieces.groupby("scar_id")["a"].sum()
    return float((dominant / total).mean())


def score_candidate(path: Path, candidates_dir: Path, scars: gpd.GeoDataFrame) -> dict:
    su = gpd.read_file(path).reset_index(drop=True)
    if su.crs is None or not su.crs.is_projected:
        raise ValueError(f"{path.name}: needs a projected CRS (areas in m²), got {su.crs}")
    stats = aspect_stats(su, candidates_dir)
    area = su.area
    weights = Queen.from_dataframe(su, use_index=False)
    weights.transform = "r"
    return {
        "candidate": path.name,
        "n_units": len(su),
        "median_area_m2": float(area.median()),
        "V": float(np.average(stats["circ_var"], weights=area)),
        "I": float(np.mean([Moran(stats[k].values, weights).I for k in ("c", "s")])),
        "scar_fit": scar_fit(su, scars.to_crs(su.crs)),
    }


def min_max(x: pd.Series) -> pd.Series:
    span = x.max() - x.min()
    return x * 0.0 if span == 0 else (x - x.min()) / span


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", type=Path, default=Path("su_candidates"),
                        help="directory written by sweep_slope_units.sh")
    parser.add_argument("--scars", type=Path, required=True,
                        help="landslide scar polygons (scar zone only, not deposition)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.scars, args.candidates / "asp_cos.tif", args.candidates / "asp_sin.tif"):
        if not required.exists():
            sys.exit(f"missing input: {required}")
    paths = sorted(args.candidates.glob(CANDIDATE_GLOB))
    if not paths:
        sys.exit(f"no {CANDIDATE_GLOB} files in {args.candidates}")

    scars = gpd.read_file(args.scars)
    rows = []
    for path in paths:
        print(f"scoring {path.name} ...", flush=True)
        rows.append(score_candidate(path, args.candidates, scars))

    df = pd.DataFrame(rows)
    df = df.assign(F=min_max(df["V"]) + min_max(df["I"])).sort_values("F")
    out_path = args.candidates / SCORES_FILENAME
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
