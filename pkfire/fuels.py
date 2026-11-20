"""Fuel models and hourly spread-rate tables.

Surface-fire physics is the project's own Albini/Scott & Burgan implementation (pkfire/vendor/rothermel.py,
verified against pyretechnics). This module adds the Pakistan crosswalk and builds, for every fuel and
every hour of the weather series, the no-wind/no-slope rate R0 and reaction intensity, plus the static
wind/slope coefficients the spread kernels need.
"""
from __future__ import annotations

import os

import numpy as np

from .vendor import rothermel                       # Rothermel / Albini surface fire (pkfire/vendor/rothermel.py)

# Anderson (1982) models not present in rothermel.py (parameters from the original publication)
_extra = {
    3: rothermel._anderson("FM3 tall grass / reeds", (3.0, 0, 0, 0, 0), 1500, 2.5, 0.25),
    5: rothermel._anderson("FM5 brush", (1.0, 0.5, 0, 0, 2.0), 2000, 2.0, 0.20),
}
MODELS = {**rothermel.FUEL_MODELS, **_extra}

# id -> (wind adjustment factor 10 m wind -> midflame, can produce firebrands)
# WAF: Andrews (2012) unsheltered 0.36-0.44 for grass/shrub, sheltered 0.2-0.3 under canopy.
TRAITS = {1: (0.40, False), 2: (0.36, False), 3: (0.44, False), 5: (0.35, False), 6: (0.35, False),
          9: (0.25, True), 102: (0.40, False), 122: (0.36, False), 161: (0.28, True), 188: (0.22, True)}

# Pakistan crosswalk targets (see technical reference ch. 6)
F_GRASS, F_GRASSSHRUB, F_REEDS, F_SHRUB = 102, 122, 3, 5
F_TIMBER_UNDERSTORY, F_CONIFER_LITTER, F_STUBBLE = 161, 188, 102

# Albini (1983) firebrand lofting coefficients f(U) = A * U^B per spotting fuel (Albini 1983, INT-309, Table 4, nearest fuel type).
# Only fuels that shed burning bark, cones and twigs can start spot fires; grass and crop fires are given none.
ALBINI = {9: (69.6, -0.818), 161: (129.0, -1.238), 188: (69.6, -0.818), 5: (83.9, -1.04), 6: (83.9, -1.04), 122: (83.9, -1.04)}

# Crown fuel (canopy base height m, canopy bulk density kg/m3) by surface fuel. ASSUMED stand defaults, not measured for Pakistan:
# long-needle pine forest (chir, blue pine, deodar) under TL8 / FM9; the broadleaf and scrub fuels carry no crown fire in this model.
CROWN = {188: (3.0, 0.10), 9: (3.0, 0.10)}      # CBH 3 m is an effective height including shrub and sapling ladder fuel: a 5 m canopy over bare litter never ignites in this model
FMC_PCT = 100.0                  # foliar moisture content, percent (Van Wagner default)

ORDER = sorted(MODELS)                       # table row k+1 <-> ORDER[k]; row 0 = non-burnable


def row_of(fuel_id: int) -> int:
    return ORDER.index(fuel_id) + 1


N_ROWS = len(ORDER) + 1


def greenness_to_live_moisture(ndvi_mean: float):
    """NDVI 0.15 (cured) .. 0.60 (green) -> live herbaceous / woody moisture (fractions)."""
    g = float(np.clip((ndvi_mean - 0.15) / 0.45, 0.0, 1.0))
    return 0.30 + 0.90 * g, 0.70 + 0.60 * g       # herb: 30% (cured) .. 120% (green), the S&B curing range


def crown_cells(grid, tables, fuel_rows):
    """Per-cell crown fire initiation intensity (kW/m) and canopy bulk density (kg/m3), flat float32. The values come from the grid when the scene
    carries measured canopy (grid['crown_ci0'], grid['crown_cbd'], see canopy_crown), otherwise from the per-fuel assumed stand defaults."""
    if "crown_ci0" in grid:
        return np.ascontiguousarray(grid["crown_ci0"], np.float32).ravel(), np.ascontiguousarray(grid["crown_cbd"], np.float32).ravel()
    r = np.asarray(fuel_rows).ravel()
    return tables["crown_i0"][r].astype(np.float32), tables["crown_cbd"][r].astype(np.float32)


def canopy_crown(fuel_rows, mean_h, p90_h, cbh_frac=0.35, cbd_max=0.15, min_h=4.0):
    """Crown fuel per cell from measured canopy height (Meta/WRI map reduced to 30 m: mean and 90th-percentile height of the 1 m pixels), in the same
    pine fuels that carry crown fire by default. ASSUMED relations, not fitted to Pakistani stands:
      stand top height H = p90 height;  canopy base height = cbh_frac * H  (live crown ratio about 0.65);
      canopy bulk density = cbd_max * fill, fill = mean / p90 (share of the 30 m cell filled by canopy, 0-1).
    Cells under min_h metres have no crown fire. Returns (crown_i0, crown_cbd) arrays shaped like fuel_rows."""
    pine_rows = np.array([row_of(f) for f in CROWN], int)
    pine = np.isin(fuel_rows, pine_rows) & (p90_h >= min_h)
    H = np.where(pine, p90_h, 0.0).astype(np.float32)
    cbh = np.maximum(cbh_frac * H, 1.0)
    fill = np.clip(mean_h / np.maximum(p90_h, 1.0), 0.0, 1.0)
    cbd = np.where(pine, cbd_max * fill, 0.0).astype(np.float32)
    i0 = ((0.010 * cbh * (460.0 + 25.9 * FMC_PCT)) ** 1.5).astype(np.float32)
    return np.where(pine, i0, 0.0).astype(np.float32), cbd


def build_tables(m_dead, m_herb, m_wood):
    """m_dead: (3, nh) dead 1/10/100 h moisture fractions. Returns kernel tables (float32)."""
    nh = m_dead.shape[1]
    r0 = np.zeros((nh, N_ROWS), np.float32)
    ir = np.zeros((nh, N_ROWS), np.float32)
    B = np.zeros(N_ROWS, np.float32); C = np.zeros(N_ROWS, np.float32); E = np.zeros(N_ROWS, np.float32)
    ratio = np.ones(N_ROWS, np.float32); slope_k = np.zeros(N_ROWS, np.float32)
    waf = np.full(N_ROWS, 0.3, np.float32); tau = np.ones(N_ROWS, np.float32)
    spot = np.zeros(N_ROWS, np.uint8); sigma = np.ones(N_ROWS, np.float32)
    spot_a = np.zeros(N_ROWS, np.float32); spot_b = np.zeros(N_ROWS, np.float32)
    crown_i0 = np.zeros(N_ROWS, np.float32); crown_cbd = np.zeros(N_ROWS, np.float32)
    M = np.vstack([m_dead, np.full((1, nh), m_herb), np.full((1, nh), m_wood)])
    for fid in ORDER:
        fm, k = MODELS[fid], row_of(fid)
        p = rothermel.surface_fire(fm, M)
        ok = np.isfinite(p["r0"]) & (p["r0"] > 0) & (p["ir"] > 0)
        r0[:, k] = np.where(ok, p["r0"] * 0.3048, 0.0)            # ft/min -> m/min
        ir[:, k] = np.where(ok, p["ir"], 0.0)
        B[k] = np.atleast_1d(p["B"])[0]; C[k] = np.atleast_1d(p["C"])[0]; E[k] = np.atleast_1d(p["E"])[0]
        beta = np.atleast_1d(p["beta"])[0]
        ratio[k] = np.atleast_1d(p["ratio"])[0]
        slope_k[k] = 5.275 * beta ** -0.3
        sg = float(np.atleast_1d(p["sigma"])[0]); sigma[k] = sg
        tau[k] = 384.0 / sg                                       # residence time, min (Anderson 1969)
        waf[k], can_spot = TRAITS[fid]
        spot[k] = 1 if can_spot else 0
        if fid in CROWN:
            cbh, cbd = CROWN[fid]
            crown_i0[k] = (0.010 * cbh * (460.0 + 25.9 * FMC_PCT)) ** 1.5        # Van Wagner (1977) initiation intensity, kW/m
            crown_cbd[k] = cbd
        if fid in ALBINI:
            spot_a[k], spot_b[k] = ALBINI[fid]
    return dict(r0=r0, ir=ir, B=B, C=C, E=E, ratio=ratio, slope_k=slope_k, waf=waf, tau=tau, spot=spot, sigma=sigma, spot_a=spot_a, spot_b=spot_b, crown_i0=crown_i0, crown_cbd=crown_cbd)
