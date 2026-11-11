"""Standalone gradient-flow check for WetnessRatioLayer (no training needed).

    m = clip( r_t * area' / sin(slope), 0, 1 ),   area' = log1p(A) if use_log_area

tf.clip_by_value passes gradient 1 inside [0, 1] and exactly 0 outside, so any
pixel whose unclipped m is > 1 sends NO gradient back to r_t (the recharge head).

Two checks:
  1. Synthetic: compares autodiff dm/dr_t with the analytic area'/sin(slope)
     and confirms clipped samples get zero gradient.
  2. Real data (--data): with the actual Slope_mean / ContributingFactor_mean,
     sweeps r_t over its sigmoid range (0, 1) and reports what fraction of
     pixels are saturated (gradient-dead) at each r_t.

Physics columns are excluded from log/clip transforms in training (see
feature_manifests/*_transforms*.json), so raw gpkg values are what the layer sees.

Usage:
    venv/bin/python scripts/check_wetness_ratio_grad.py
    venv/bin/python scripts/check_wetness_ratio_grad.py --data <path.gpkg>
    venv/bin/python scripts/check_wetness_ratio_grad.py --data <path.gpkg> --no-log-area
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from py_files.GallenModel_v3 import WetnessRatioLayer  # noqa: E402

DEG2RAD = np.pi / 180.0
SIN_FLOOR = 1e-6
SLOPE_COL = "Slope_mean"
AREA_COL = "ContributingFactor_mean"
R_T_SWEEP = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99)


def layer_grads(layer, r_t, area, slope):
    """Return (m, dm/dr_t, dm/dA, dm/dslope) per sample via autodiff."""
    tensors = [tf.constant(v, dtype=tf.float32) for v in (r_t, area, slope)]
    with tf.GradientTape() as tape:
        tape.watch(tensors)
        m = layer(tensors)
        total = tf.reduce_sum(m)  # sum -> per-sample grads (samples independent)
    grads = tape.gradient(total, tensors)
    return (m.numpy(),) + tuple(None if g is None else g.numpy() for g in grads)


def unclipped_m(r_t, area, slope, use_log_area):
    area_p = np.log1p(area) if use_log_area else area
    return r_t * area_p / np.maximum(np.sin(slope * DEG2RAD), SIN_FLOOR)


def synthetic_check(use_log_area):
    print("=== 1. Synthetic check ===")
    layer = WetnessRatioLayer(use_log_area=use_log_area)
    rng = np.random.default_rng(0)
    n = 1000
    r_t = rng.uniform(0.0, 1.0, (n, 1))
    area = 10 ** rng.uniform(0, 5, (n, 1))      # 1 .. 1e5 m^2
    slope = rng.uniform(1.0, 60.0, (n, 1))      # degrees

    m, g_r, g_a, g_s = layer_grads(layer, r_t, area, slope)
    for name, g in (("r_t", g_r), ("contributing_area", g_a), ("slope", g_s)):
        print(f"  dm/d{name:18s}: {'None (DISCONNECTED)' if g is None else 'connected'}")
    if g_r is None:
        return

    raw = unclipped_m(r_t, area, slope, use_log_area)
    inside = (raw >= 0.0) & (raw <= 1.0)
    area_p = np.log1p(area) if use_log_area else area
    expected = np.where(inside, area_p / np.maximum(np.sin(slope * DEG2RAD), SIN_FLOOR), 0.0)
    max_err = np.max(np.abs(g_r - expected) / np.maximum(np.abs(expected), 1.0))
    print(f"  autodiff vs analytic dm/dr_t max rel err: {max_err:.2e}"
          f"  -> {'PASS' if max_err < 1e-4 else 'FAIL'}")
    dead_ok = np.all(g_r[~inside] == 0.0)
    print(f"  clipped samples ({(~inside).mean():.1%}) have zero grad: "
          f"{'PASS' if dead_ok else 'FAIL'}")


def load_columns(path):
    import geopandas as gpd

    df = gpd.read_file(path, ignore_geometry=True)
    missing = [c for c in (SLOPE_COL, AREA_COL) if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns {missing} not in {path}. Available: {list(df.columns)}")
    df = df[[SLOPE_COL, AREA_COL]].dropna()
    return df[AREA_COL].to_numpy()[:, None], df[SLOPE_COL].to_numpy()[:, None]


def real_data_check(path, use_log_area):
    print(f"\n=== 2. Real data: {path} ===")
    area, slope = load_columns(path)
    layer = WetnessRatioLayer(use_log_area=use_log_area)
    scale = unclipped_m(1.0, area, slope, use_log_area)  # m when r_t = 1
    print(f"  n={len(area)}  area'/sin(slope): "
          f"p1={np.percentile(scale, 1):.3g}  median={np.median(scale):.3g}  "
          f"p99={np.percentile(scale, 99):.3g}")
    print(f"  r_t at which the median pixel saturates (m=1): {1 / np.median(scale):.4f}")
    print(f"\n  {'r_t':>6} | {'saturated (grad=0)':>18} | {'mean |dm/dr_t|':>14}")
    for r in R_T_SWEEP:
        r_t = np.full_like(area, r)
        _, g_r, _, _ = layer_grads(layer, r_t, area, slope)
        dead = np.mean(g_r == 0.0)
        print(f"  {r:6.3f} | {dead:18.1%} | {np.mean(np.abs(g_r)):14.3e}")
    print("\n  r_t comes from a sigmoid; at init it sits near 0.5. If most pixels are"
          "\n  saturated there, the recharge head gets almost no gradient from m.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="gpkg with Slope_mean and ContributingFactor_mean")
    parser.add_argument("--no-log-area", action="store_true", default=True,
                        help="use raw contributing area (use_log_area=False)")
    args = parser.parse_args()
    use_log_area = not args.no_log_area

    print(f"use_log_area={use_log_area}\n")
    synthetic_check(use_log_area)
    if args.data:
        real_data_check(args.data, use_log_area)


if __name__ == "__main__":
    main()
