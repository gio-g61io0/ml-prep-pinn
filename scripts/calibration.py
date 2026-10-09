"""
Probability calibration fitted on inner-validation predictions only.

The calibrator never sees the outer held-out block, so calibrated out-of-fold
probabilities remain leakage-free.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        if method not in {"none", "platt", "isotonic"}:
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self._model = None
        self.fitted = False

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y).astype(int)
        if self.method == "none" or len(np.unique(y)) < 2:
            self.method = "none"
            self.fitted = True
            return self
        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            z = np.log(p / (1 - p)).reshape(-1, 1)
            self._model = LogisticRegression(C=1e6, solver="lbfgs").fit(z, y)
        else:
            from sklearn.isotonic import IsotonicRegression
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        self.fitted = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "none" or self._model is None:
            return p
        if self.method == "platt":
            z = np.log(p / (1 - p)).reshape(-1, 1)
            return self._model.predict_proba(z)[:, 1]
        return np.clip(self._model.predict(p), 1e-6, 1 - 1e-6)
