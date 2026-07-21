"""Sampling of real Makilala units + collocation/boundary/initial points.

Each training point is anchored to a REAL slope unit's parameters (drawn from the
joined conditioning + Bouc-Wen table), so training uses the Makilala data directly.
For each drawn unit we sample coordinates within that unit's domain:
  * collocation: z in [0, H_unit], t in [0, T]
  * boundary:    t in [0, T]        (z = 0 base / z = H handled in the residual)
  * initial:     z in [0, H_unit]   (t = 0 handled in the residual)

Returns raw physical batches (dicts of (N, 1) float32 arrays) with keys
z (or t), rho, k, eps_y, H, pga -- ready for the residual functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_unit_table

_COND = ("rho", "k", "eps_y", "H", "pga")


class UnitSampler:
    """Draws real units (with replacement) and coordinates within each unit's domain."""

    def __init__(self, T: float, table: pd.DataFrame | None = None, seed: int = 0):
        self.T = float(T)
        self.table = load_unit_table() if table is None else table
        self.rng = np.random.default_rng(seed)
        self._arr = {c: self.table[c].to_numpy(dtype=np.float64) for c in _COND}
        self._n_units = len(self.table)

    def _draw_units(self, n: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self._n_units, size=n)
        return {c: self._arr[c][idx].reshape(-1, 1).astype(np.float32) for c in _COND}

    def collocation(self, n: int, t_max: float | None = None) -> dict[str, np.ndarray]:
        cond = self._draw_units(n)
        H = cond["H"]
        tm = self.T if t_max is None else float(t_max)
        z = (self.rng.random((n, 1)) * H).astype(np.float32)
        t = (self.rng.random((n, 1)) * tm).astype(np.float32)
        return {"z": z, "t": t, **cond}

    def boundary(self, n: int, t_max: float | None = None) -> dict[str, np.ndarray]:
        cond = self._draw_units(n)
        tm = self.T if t_max is None else float(t_max)
        t = (self.rng.random((n, 1)) * tm).astype(np.float32)
        return {"t": t, **cond}

    def initial(self, n: int) -> dict[str, np.ndarray]:
        cond = self._draw_units(n)
        H = cond["H"]
        z = (self.rng.random((n, 1)) * H).astype(np.float32)
        return {"z": z, **cond}
