"""
Fold-safe preprocessing.

Every learned transformation - robust scaling, category vocabularies, rare-class
pooling and imputation - is fitted on outer-training rows only and then applied
unchanged to the held-out block. Index 0 of every vocabulary is permanently
reserved for rare-or-unseen categories, so an unseen lithology in a held-out
barangay is representable rather than fatal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CategoryEncoder:
    min_count: int = 30
    vocab: Dict[str, int] = field(default_factory=dict)
    n_classes: int = 1                   # includes the reserved bucket 0
    pooled: List[str] = field(default_factory=list)
    unseen_at_transform: int = 0

    def fit(self, values: np.ndarray) -> "CategoryEncoder":
        vals, counts = np.unique(np.asarray(values, dtype=object).astype(str), return_counts=True)
        keep = [v for v, c in zip(vals, counts) if c >= self.min_count]
        self.pooled = [v for v, c in zip(vals, counts) if c < self.min_count]
        self.vocab = {v: i + 1 for i, v in enumerate(sorted(keep))}
        self.n_classes = len(self.vocab) + 1
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=object).astype(str)
        out = np.zeros(len(arr), dtype=np.int32)
        unseen = 0
        for i, v in enumerate(arr):
            idx = self.vocab.get(v, 0)
            out[i] = idx
            if idx == 0:
                unseen += 1
        self.unseen_at_transform = unseen
        return out


@dataclass
class RobustScaler:
    center_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    median_impute_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RobustScaler":
        X = np.asarray(X, dtype=np.float64)
        self.median_impute_ = np.nanmedian(X, axis=0)
        self.median_impute_ = np.where(np.isfinite(self.median_impute_),
                                       self.median_impute_, 0.0)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        iqr = np.where(np.isfinite(iqr) & (iqr > 1e-12), iqr, 1.0)
        self.center_ = self.median_impute_.copy()
        self.scale_ = iqr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.median_impute_, np.where(bad)[1])
        return (X - self.center_) / self.scale_


@dataclass
class FoldPreprocessor:
    """Bundle of all learned transformations for a single outer-training fold."""

    min_count: int = 30
    scaler: RobustScaler = field(default_factory=RobustScaler)
    litho: CategoryEncoder = field(default_factory=CategoryEncoder)
    texture: CategoryEncoder = field(default_factory=CategoryEncoder)

    def fit(self, X_num: np.ndarray, litho: np.ndarray, texture: np.ndarray) -> "FoldPreprocessor":
        self.scaler.fit(X_num)
        self.litho.min_count = self.min_count
        self.texture.min_count = self.min_count
        self.litho.fit(litho)
        self.texture.fit(texture)
        return self

    def transform(self, X_num: np.ndarray, litho: np.ndarray,
                  texture: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.scaler.transform(X_num).astype(np.float32),
                self.litho.transform(litho),
                self.texture.transform(texture))

    @property
    def report(self) -> Dict[str, object]:
        return {
            "n_litho_classes": self.litho.n_classes,
            "n_texture_classes": self.texture.n_classes,
            "pooled_litho": self.litho.pooled,
            "pooled_texture": self.texture.pooled,
            "unseen_litho_rows": self.litho.unseen_at_transform,
            "unseen_texture_rows": self.texture.unseen_at_transform,
        }
