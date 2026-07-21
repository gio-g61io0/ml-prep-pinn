"""Train the amortized Makilala EIL PINN with a time-window curriculum + save checkpoint.

Run:  cd mlprep && venv/bin/python -m pinn_eil.run_training [epochs_scale]

Curriculum: train on growing time windows (5s -> 10s -> 20s -> full T) so the network
learns the initial pulse first, then extends through the ringdown. Saves the trained
model and the loss history.

Outputs:
  pinn_eil/checkpoints/pinn_makilala.keras
  pinn_eil/outputs/training_history.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pinn_eil as P  # noqa: E402
from pinn_eil import train as T  # noqa: E402

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
OUT_DIR = HERE / "outputs"
CKPT_PATH = CKPT_DIR / "pinn_makilala.keras"
HIST_PATH = OUT_DIR / "training_history.csv"


def main() -> None:
    # optional CLI multiplier to scale epochs (e.g. 0.25 for a quick run)
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

    ranges = P.load_norm_ranges()
    consts = P.load_global_consts()
    scales = P.load_scales(consts)
    sampler = P.UnitSampler(consts.T, seed=0)
    model = P.build_pinn(ranges, consts.T)

    print(f"[setup] units={sampler._n_units} params={model.count_params()} "
          f"u_scale={consts.u_scale:.4e} T={consts.T}")

    stages = [
        (5.0, int(1200 * scale)),
        (10.0, int(1200 * scale)),
        (20.0, int(1200 * scale)),
        (consts.T, int(1800 * scale)),
    ]
    history = T.train_curriculum(
        model, consts, scales, sampler, stages=stages,
        n_col=4000, n_bc=1000, n_ic=1000, resample_every=100, log_every=200,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(CKPT_PATH)
    pd.DataFrame(history).to_csv(HIST_PATH, index=False)
    print(f"\n[saved] {CKPT_PATH}")
    print(f"[saved] {HIST_PATH}")
    print(f"[final] loss={history[-1]['loss']:.4e}")


if __name__ == "__main__":
    main()
