"""Predict per-unit surface displacement with the trained PINN and plot it as a map.

For every Makilala slope unit, query the trained network at the surface (z = H) over a
time grid, take the peak |u| (the displacement-hazard metric), and plot it at the unit's
(lon, lat).

Run:  cd mlprep && venv/bin/python -m pinn_eil.predict_map

Outputs:
  pinn_eil/outputs/makilala_pinn_predictions.csv       (unit_id, lon, lat, peak_disp_m)
  pinn_eil/outputs/makilala_pinn_displacement_map.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pinn_eil as P  # noqa: E402  (registers custom layers for model loading)

HERE = Path(__file__).resolve().parent
CKPT_PATH = HERE / "checkpoints" / "pinn_makilala.keras"
OUT_DIR = HERE / "outputs"
PRED_CSV = OUT_DIR / "makilala_pinn_predictions.csv"
MAP_PNG = OUT_DIR / "makilala_pinn_displacement_map.png"

N_T = 240          # time samples per unit for the peak
UNIT_CHUNK = 512   # units per evaluation batch


def peak_surface_displacement(model, consts, table: pd.DataFrame) -> np.ndarray:
    """Peak |u(z=H, t)| over [0, T] for each unit (physical metres)."""
    t_grid = np.linspace(0.0, consts.T, N_T, dtype=np.float32)
    n = len(table)
    peaks = np.empty(n, dtype=np.float32)

    rho = table["rho"].to_numpy(np.float32)
    k = table["k"].to_numpy(np.float32)
    eps_y = table["eps_y"].to_numpy(np.float32)
    H = table["H"].to_numpy(np.float32)
    pga = table["pga"].to_numpy(np.float32)

    for s in range(0, n, UNIT_CHUNK):
        e = min(s + UNIT_CHUNK, n)
        m = e - s
        # (m, N_T) grid -> flatten to (m*N_T, 1)
        zc = np.repeat(H[s:e], N_T).reshape(-1, 1)                 # z = H (surface)
        tc = np.tile(t_grid, m).reshape(-1, 1)
        rep = lambda a: np.repeat(a[s:e], N_T).reshape(-1, 1)      # noqa: E731
        x = np.concatenate([zc, tc, rep(rho), rep(k), rep(eps_y), rep(H), rep(pga)], axis=1)
        out = model(tf.constant(x, tf.float32), training=False)
        u = consts.u_scale * out["u"].numpy().reshape(m, N_T)      # physical displacement
        peaks[s:e] = np.max(np.abs(u), axis=1)
    return peaks


def main() -> None:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"No checkpoint at {CKPT_PATH}; run run_training.py first.")
    consts = P.load_global_consts()
    model = tf.keras.models.load_model(CKPT_PATH)

    table = P.load_unit_table()  # already includes lon, lat (positional, no unit_id merge)

    print(f"[predict] {len(table)} units, z=H surface, {N_T} time samples over [0,{consts.T}s]")
    peaks = peak_surface_displacement(model, consts, table)
    table["peak_disp_m"] = peaks
    print(f"[predict] peak disp (m): min={peaks.min():.3e} "
          f"med={np.median(peaks):.3e} max={peaks.max():.3e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table[["unit_id", "lon", "lat", "peak_disp_m"]].to_csv(PRED_CSV, index=False)

    # --- map ---
    fig, ax = plt.subplots(figsize=(9, 8))
    disp_mm = table["peak_disp_m"].to_numpy() * 1000.0  # metres -> mm
    sc = ax.scatter(table["lon"], table["lat"], c=disp_mm, s=6, cmap="inferno",
                    linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("Peak surface displacement [mm]")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Makilala EIL PINN — predicted peak surface displacement\n"
                 "(driven by 2019-12-15 M6.7 Cotabato SPECFEM base motion, scaled per unit by PGA)")
    fig.tight_layout()
    fig.savefig(MAP_PNG, dpi=150)
    print(f"[saved] {PRED_CSV}")
    print(f"[saved] {MAP_PNG}")


if __name__ == "__main__":
    main()
