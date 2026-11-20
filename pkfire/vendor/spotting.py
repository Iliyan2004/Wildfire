"""Ember spotting from wind-driven surface fires.

Maximum spotting distance follows Albini (1983), "Potential spotting distance from wind-driven
surface fires", USDA For. Serv. Gen. Tech. Rep. INT-309 (as implemented in the Canadian Forest
Service fbp97spot.c reference code). Ignition of a landed ember uses Schroeder's (1969)
probability of ignition.
"""
import numpy as np

# Albini (1983) Table 4 coefficients for f(U) = A * U^B, by the closest fuel type to each model.
ALBINI_COEF = {
    1: (129.0, -1.238),   # FM1 short grass      -> grass
    2: (129.0, -1.238),   # FM2 grass understory  -> grass (surface fire carried by grass)
    6: (83.9, -1.04),     # FM6 dormant brush     -> mature chaparral
    9: (69.6, -0.818),    # FM9 pine litter        -> medium conifer slash
    102: (129.0, -1.238),  # GR2 dry grass         -> grass
    122: (83.9, -1.04),    # GS2 grass-shrub       -> mature chaparral
    161: (129.0, -1.238),  # TU1 timber-grass-shrub -> grass (surface fire carried by grass)
    188: (69.6, -0.818),   # TL8 long-needle litter -> medium conifer slash
}
MAX_SPOT_M = 2000.0


def byram_intensity_kw_m(ir_btu_ft2_min, ros_ft_min, sigma):
    """Fireline intensity (kW/m) from Rothermel reaction intensity, spread rate and residence time."""
    residence_min = 384.0 / sigma                      # Anderson (1969)
    i_btu_ft_s = ir_btu_ft2_min * ros_ft_min * residence_min / 60.0
    return i_btu_ft_s * 3.4613


def max_spot_distance_m(hfi_kw_m, u10_ms, coef):
    """Albini (1983) maximum spotting distance (m) from a wind-driven surface fire."""
    A, B = coef
    hfi = np.asarray(hfi_kw_m, float)
    u = np.maximum(np.asarray(u10_ms, float), 0.1)
    E = hfi * A * u ** B                               # thermal strength
    H = 0.173 * np.sqrt(E)                             # maximum firebrand lofting height, m
    h = np.power(np.maximum(H, 1e-6), 0.337) - 1.22    # Chase (1981) characteristic height
    ok = (h > 0.1) & (H > h * 1.01)
    h = np.where(ok, h, 1.0)
    Hs = np.where(ok, H, 2.0)
    u_h = u * (np.maximum(h, 6.0) / 10.0) ** (1 / 7)   # wind at height h (1/7 power law)
    u_H = u * (Hs / 10.0) ** (1 / 7)                   # wind at lofting height
    descent = 1.3 * 3.6 * u_h * np.sqrt(h) * (0.362 + np.sqrt(Hs / h) * 0.5 * np.log(Hs / h))
    lofting = np.where(h < 10, 2.78 * u_H * np.sqrt(Hs), u_h * np.sqrt(Hs) * (2.13 + np.log(Hs / h)))
    return np.where(ok, np.clip(descent + lofting, 0, MAX_SPOT_M), 0.0)


def ignition_probability(temp_c, fine_moisture_pct):
    """Schroeder (1969) probability that a firebrand ignites fine dead fuel (0-1)."""
    t, m = temp_c, fine_moisture_pct
    q_ig = (144.5 - 0.266 * t + 0.00058 * t * t - 0.01 * t * m
            + 18.54 * (1 - np.exp(-0.151 * m)) + 6.4 * m)
    x = max((400.0 - q_ig) / 10.0, 0.0)
    return float(min(0.000048 * x ** 4.3 / 50.0, 1.0))
