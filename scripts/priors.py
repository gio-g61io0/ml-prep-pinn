"""
Soft geotechnical priors.

These are *priors*, not observations. They express broad, literature-consistent
plausibility intervals for effective cohesion and friction angle by USDA soil
texture class (mapped onto coarse USCS behaviour). Sensitivity of results to
both the interval limits and the penalty weight must be reported.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# texture class -> (c_low, c_high) kPa, (phi_low, phi_high) degrees
DEFAULT_TEXTURE_BOUNDS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "sand":            ((0.0, 5.0),  (30.0, 42.0)),
    "loamy sand":      ((0.0, 8.0),  (29.0, 40.0)),
    "sandy loam":      ((1.0, 12.0), (28.0, 38.0)),
    "loam":            ((2.0, 18.0), (25.0, 36.0)),
    "silt loam":       ((2.0, 22.0), (23.0, 34.0)),
    "silt":            ((2.0, 22.0), (22.0, 33.0)),
    "sandy clay loam": ((3.0, 25.0), (24.0, 34.0)),
    "clay loam":       ((5.0, 32.0), (20.0, 31.0)),
    "silty clay loam": ((5.0, 35.0), (18.0, 30.0)),
    "sandy clay":      ((5.0, 35.0), (20.0, 32.0)),
    "silty clay":      ((8.0, 45.0), (15.0, 28.0)),
    "clay":            ((8.0, 50.0), (14.0, 27.0)),
}

# used when a texture label cannot be matched
FALLBACK_BOUNDS: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 50.0), (14.0, 42.0))


def _norm(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("-", " ").replace("_", " ").split())


def bounds_from_texture(
    texture_labels: np.ndarray,
    table: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    widen: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Per-row prior bounds arrays.

    ``widen`` > 1 inflates every interval about its midpoint, which is the knob
    used for the prior-sensitivity analysis.
    """
    table = table or DEFAULT_TEXTURE_BOUNDS
    n = len(texture_labels)
    cL = np.empty(n); cU = np.empty(n); pL = np.empty(n); pU = np.empty(n)
    matched = np.zeros(n, dtype=bool)

    for i, raw in enumerate(texture_labels):
        key = _norm(raw)
        entry = table.get(key)
        if entry is None:  # try a loose containment match
            for k, v in table.items():
                if k in key or key in k:
                    entry = v
                    break
        if entry is None:
            entry = FALLBACK_BOUNDS
        else:
            matched[i] = True
        (c0, c1), (p0, p1) = entry
        cm, pm = 0.5 * (c0 + c1), 0.5 * (p0 + p1)
        cL[i], cU[i] = cm - widen * (cm - c0), cm + widen * (c1 - cm)
        pL[i], pU[i] = pm - widen * (pm - p0), pm + widen * (p1 - pm)

    return {"c_low": np.maximum(cL, 0.0), "c_high": cU,
            "phi_low": np.maximum(pL, 1.0), "phi_high": np.minimum(pU, 50.0),
            "matched": matched}


def bounds_to_R0_range(bounds: Dict[str, np.ndarray], sigma_ref: float) -> Tuple[float, float]:
    """Reference resistance range implied by the c and phi priors.

    R0 = c + sigma_ref tan(phi), so the priors themselves suggest sensible
    R0_min / R0_max instead of arbitrary constants.
    """
    lo = float(np.percentile(bounds["c_low"] + sigma_ref * np.tan(np.radians(bounds["phi_low"])), 1))
    hi = float(np.percentile(bounds["c_high"] + sigma_ref * np.tan(np.radians(bounds["phi_high"])), 99))
    return max(lo * 0.5, 1.0), hi * 1.5
