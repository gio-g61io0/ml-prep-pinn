"""Resample a raster to a target pixel size.

Usage:
    python resample_raster.py <input.tif> <output.tif> --pixel-size 30
    python resample_raster.py <input.tif> <output.tif> --pixel-size 30 --resampling bilinear

Pixel size is in the raster's native CRS units (meters for projected CRS,
degrees for geographic CRS).
"""

import argparse

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "max": Resampling.max,
    "min": Resampling.min,
    "med": Resampling.med,
    "q1": Resampling.q1,
    "q3": Resampling.q3,
    "sum": Resampling.sum,
    "rms": Resampling.rms,
}


def resample_raster(
    input_path: str,
    output_path: str,
    pixel_size: float,
    resampling: str = "bilinear",
) -> None:
    method = RESAMPLING_METHODS[resampling]

    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            src.crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=pixel_size,
        )

        # Estimate output size to decide whether BIGTIFF is needed.
        # GeoTIFF 4GB limit => enable BIGTIFF when estimated bytes > 3.5GB.
        bytes_per_pixel = np.dtype(src.dtypes[0]).itemsize
        estimated_bytes = width * height * src.count * bytes_per_pixel
        needs_bigtiff = estimated_bytes > 3.5 * 1024**3

        profile = src.profile.copy()
        profile.update({
            "transform": transform,
            "width": width,
            "height": height,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
        })
        if needs_bigtiff:
            profile["BIGTIFF"] = "YES"

        print(f"Resampling {input_path} -> {output_path}")
        print(f"  source: {src.width}x{src.height} @ {src.res[0]:.4f} x {src.res[1]:.4f}")
        print(f"  target: {width}x{height} @ {pixel_size} (method={resampling})")
        print(f"  estimated size: {estimated_bytes / 1024**3:.2f} GB"
              f"{' (BIGTIFF enabled)' if needs_bigtiff else ''}")

        if width * height > src.width * src.height * 10:
            upscale_factor = (width * height) / (src.width * src.height)
            print(f"  WARNING: output has {upscale_factor:.0f}x more pixels than source. "
                  f"Resampling cannot add detail beyond the source resolution.")

        with rasterio.open(output_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=method,
                )

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample a raster to a target pixel size.")
    parser.add_argument("input", help="Path to input GeoTIFF")
    parser.add_argument("output", help="Path to output GeoTIFF")
    parser.add_argument("--pixel-size", type=float, required=True,
                        help="Target pixel size in CRS units (e.g. 30 for 30m)")
    parser.add_argument("--resampling", default="bilinear",
                        choices=list(RESAMPLING_METHODS.keys()),
                        help="Resampling method (default: bilinear). "
                             "Use 'nearest' for categorical data.")
    args = parser.parse_args()

    resample_raster(args.input, args.output, args.pixel_size, args.resampling)


if __name__ == "__main__":
    main()
