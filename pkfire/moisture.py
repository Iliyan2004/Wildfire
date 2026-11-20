"""Hourly dead fuel moisture: equilibrium moisture (Simard 1968) with time-lag relaxation (1, 10, 100 h)
and the rain-wetting term of WRF-SFIRE (see docs/background/accuracy_improvements.md): per hour with rain r > 0.05 mm/h the
moisture approaches saturation by (1 - exp(-(r - 0.05)/8)) / 14 of the gap. Saturation is capped at 60 %
(WRF uses 250 %, far above the range any spread model is valid for)."""
from __future__ import annotations

import numpy as np

from .fuels import rothermel

LAGS_H = (1.0, 10.0, 100.0)
M_SAT = 0.60


def dead_moisture(temp_c, rh, rain_mm_h):
    """Arrays of equal length n (hourly, including any conditioning period). Returns (3, n) fractions."""
    n = len(temp_c)
    emc = np.array([rothermel.equilibrium_moisture(float(t), float(min(max(r, 1.0), 100.0))) for t, r in zip(temp_c, rh)])
    out = np.zeros((3, n))
    m = np.full(3, emc[0])
    for h in range(n):
        for k, lag in enumerate(LAGS_H):
            m[k] += (emc[h] - m[k]) * (1.0 - np.exp(-1.0 / lag))
            r = rain_mm_h[h]
            if r > 0.05:
                m[k] += (M_SAT - m[k]) * (1.0 - np.exp(-(r - 0.05) / 8.0)) / 14.0
            m[k] = min(m[k], M_SAT)
        out[:, h] = m
    return np.maximum(out, 0.02)


def ignition_probability(temp_c, fine_moisture_frac):
    """Schroeder (1969) probability that a firebrand landing on fine dead fuel ignites it, per hour (arrays).
    temp_c in degrees C, fine_moisture_frac the 1 h dead fuel moisture as a fraction."""
    t = np.asarray(temp_c, float)
    m = np.asarray(fine_moisture_frac, float) * 100.0
    q = 144.5 - 0.266 * t + 0.00058 * t * t - 0.01 * t * m + 18.54 * (1 - np.exp(-0.151 * m)) + 6.4 * m
    x = np.maximum((400.0 - q) / 10.0, 0.0)
    return np.minimum(0.000048 * x ** 4.3 / 50.0, 1.0)
