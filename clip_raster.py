"""Clip raster values below a threshold.

Usage:
    python clip_raster.py <input.tif> <output.tif> [--min 1.0]

Values below --min are set to --min; values >= --min are retained. NoData
pixels are preserved.
"""

import argparse

import numpy as np
import rasterio


def clip_raster(input_path: str, output_path: str, min_value: float = 1.0) -> None:
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        data = src.read()
        nodata = src.nodata

        if nodata is not None:
            mask = data == nodata
            clipped = np.where(data < min_value, min_value, data)
            clipped[mask] = nodata
        else:
            clipped = np.where(data < min_value, min_value, data)

        clipped = clipped.astype(profile['dtype'])

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(clipped)

    print(f"Clipped {input_path} -> {output_path}")
    print(f"  min_value = {min_value}")
    print(f"  shape     = {clipped.shape}")
    print(f"  data min  = {clipped.min()}, max = {clipped.max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clip raster values below a threshold.")
    parser.add_argument("input", help="Path to input GeoTIFF")
    parser.add_argument("output", help="Path to output GeoTIFF")
    parser.add_argument("--min", type=float, default=1.0,
                        help="Minimum value; anything below is clipped up to this (default: 1.0)")
    args = parser.parse_args()

    clip_raster(args.input, args.output, args.min)


if __name__ == "__main__":
    main()
