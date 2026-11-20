"""Rothermel (1972) surface fire spread with Albini (1976) multi-class fuels.

Imperial units internally (lb, ft, min, BTU), as in the original equations.
Six size classes, in order: dead 1 h, dead 10 h, dead 100 h, dead herbaceous, live herbaceous,
live woody. Net fuel load uses Albini's size-class weights (g_ij), as in BehavePlus.
Scott & Burgan (2005) dynamic models transfer live herbaceous load to dead herbaceous as the
herbaceous fuel cures (fully cured at 30 % live herbaceous moisture, fully green at 120 %).
"""
import numpy as np

T2LB = 0.0459137      # tons/acre -> lb/ft^2
RHO_P = 32.0          # particle density, lb/ft^3
HEAT = 8000.0         # heat content, BTU/lb
S_T, S_E = 0.0555, 0.010   # total / effective mineral content
DEAD = np.array([True, True, True, True, False, False])
LIVE = ~DEAD


def _anderson(name, w_tons, sav1, depth, mx):
    """Anderson (1982) static model; w_tons = (1 h, 10 h, 100 h, live herb, live woody) in tons/acre."""
    w = np.array([w_tons[0], w_tons[1], w_tons[2], 0.0, w_tons[3], w_tons[4]]) * T2LB
    return dict(name=name, w=w, sav=np.array([sav1, 109, 30, 1500, 1500, 1500.0]), depth=depth, mx=mx,
                dynamic=False, system="Anderson 13")


def _scott_burgan(name, row):
    """Scott & Burgan (2005) model from the standard parameter row:
    (depth ft, Mx %, heat kBTU/lb, w 1h, 10h, 100h, live herb, live woody [lb/ft^2], SAV 1h, 10h, 100h, herb, woody)."""
    d, mx, h, w1, w10, w100, wh, ww, s1, s10, s100, sh, sw = row
    assert h == 8.0
    return dict(name=name, w=np.array([w1, w10, w100, 0.0, wh, ww]),
                sav=np.array([s1, s10, s100, sh or 1500.0, sh or 1500.0, sw or 1500.0]),
                depth=d, mx=mx / 100, dynamic=wh > 0, system="Scott & Burgan 40")


FUEL_MODELS = {
    # Anderson (1982), kept for comparison runs.
    1: _anderson("FM1 short grass", (0.74, 0, 0, 0, 0), 3500, 1.0, 0.12),
    2: _anderson("FM2 timber, grass understory", (2.0, 1.0, 0.5, 0.5, 0), 3000, 1.0, 0.15),
    6: _anderson("FM6 dormant brush", (1.5, 2.5, 2.0, 0, 0), 1750, 2.5, 0.25),
    9: _anderson("FM9 long-needle pine litter", (2.92, 0.41, 0.15, 0, 0), 2500, 0.2, 0.25),
    # Scott & Burgan (2005), RMRS-GTR-153, Table 7 parameters.
    102: _scott_burgan("GR2 low load, dry climate grass",
                       (1.0, 15.0, 8.0, 0.0046, 0.0000, 0.0000, 0.0459, 0.0000, 2000.0, 109.0, 30.0, 1800.0, 0.0)),
    122: _scott_burgan("GS2 moderate load, dry climate grass-shrub",
                       (1.5, 15.0, 8.0, 0.0230, 0.0230, 0.0000, 0.0275, 0.0459, 2000.0, 109.0, 30.0, 1800.0, 1800.0)),
    161: _scott_burgan("TU1 low load, dry climate timber-grass-shrub",
                       (0.6, 20.0, 8.0, 0.0092, 0.0413, 0.0689, 0.0092, 0.0413, 2000.0, 109.0, 30.0, 1800.0, 1600.0)),
    188: _scott_burgan("TL8 long-needle litter",
                       (0.3, 35.0, 8.0, 0.2663, 0.0643, 0.0505, 0.0000, 0.0000, 1800.0, 109.0, 30.0, 0.0, 0.0)),
}


def _damping(m, mx):
    r = np.clip(m / mx, 0, 1)
    return 1 - 2.59 * r + 5.11 * r**2 - 3.52 * r**3


def _size_class(sav):
    """Albini (1976) SAV bins used for the g_ij weights."""
    return np.digitize(sav, [16, 48, 96, 192, 1200])


def _g_weights(f, sav, group):
    """g_ij: sum of f_ik over the particles k in the same SAV bin and the same dead/live group."""
    bins = _size_class(sav)
    g = np.zeros_like(f)
    for j in np.flatnonzero(group):
        same = group & (bins == bins[j])
        g[j] = f[same].sum(0)
    return g


def surface_fire(fm, M):
    """No-wind, no-slope spread for one fuel model.

    M: moisture fractions, shape (5, n): dead 1 h, 10 h, 100 h, live herbaceous, live woody.
    Dead herbaceous moisture = 1 h moisture. Returns R0 (ft/min, shape n) and the coefficients
    needed for wind/slope factors.
    """
    M = np.asarray(M, float)
    n = M.shape[1]
    M6 = np.vstack([M[0], M[1], M[2], M[0], M[3], M[4]])
    w0 = np.repeat(fm["w"][:, None], n, axis=1)
    if fm["dynamic"]:
        green = np.clip(M[3] / 0.9 - 1 / 3, 0, 1)
        herb = fm["w"][4]
        w0[3], w0[4] = herb * (1 - green), herb * green
    sav = fm["sav"][:, None]

    a = sav * w0 / RHO_P
    a_dead, a_live = a[DEAD].sum(0), a[LIVE].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.zeros_like(a)
        f[DEAD] = np.where(a_dead > 0, a[DEAD] / a_dead, 0)
        f[LIVE] = np.where(a_live > 0, a[LIVE] / a_live, 0)
    f_dead, f_live = a_dead / (a_dead + a_live), a_live / (a_dead + a_live)
    has_live = a_live > 0

    sigma = f_dead * (f[DEAD] * sav[DEAD]).sum(0) + f_live * (f[LIVE] * sav[LIVE]).sum(0)
    g = _g_weights(f, fm["sav"], DEAD) + _g_weights(f, fm["sav"], LIVE)
    wn = w0 * (1 - S_T)
    wn_dead, wn_live = (g[DEAD] * wn[DEAD]).sum(0), (g[LIVE] * wn[LIVE]).sum(0)

    rho_b = w0.sum(0) / fm["depth"]
    beta = rho_b / RHO_P
    ratio = beta / (3.348 * sigma**-0.8189)
    gamma_max = sigma**1.5 / (495 + 0.0594 * sigma**1.5)
    A = 133 * sigma**-0.7913
    gamma = gamma_max * ratio**A * np.exp(A * (1 - ratio))
    eta_s = min(0.174 * S_E**-0.19, 1.0)

    m_dead = (f[DEAD] * M6[DEAD]).sum(0)
    ir = wn_dead * _damping(m_dead, fm["mx"])
    if has_live.any():
        m_live = (f[LIVE] * M6[LIVE]).sum(0)
        wd = w0[DEAD] * np.exp(-138 / sav[DEAD])
        wl = w0[LIVE] * np.exp(-500 / sav[LIVE])
        with np.errstate(invalid="ignore", divide="ignore"):
            m_fine_dead = (wd * M6[DEAD]).sum(0) / wd.sum(0)
            mx_live = np.maximum(2.9 * wd.sum(0) / wl.sum(0) * (1 - m_fine_dead / fm["mx"]) - 0.226, fm["mx"])
        ir = ir + np.where(has_live, wn_live * _damping(m_live, np.where(has_live, mx_live, 1.0)), 0)
    ir = gamma * HEAT * eta_s * ir                     # reaction intensity, BTU/ft^2/min

    xi = np.exp((0.792 + 0.681 * sigma**0.5) * (beta + 0.1)) / (192 + 0.2595 * sigma)
    q_ig = 250 + 1116 * M6
    eps = np.exp(-138 / sav)
    sink = f_dead * (f[DEAD] * eps[DEAD] * q_ig[DEAD]).sum(0) + f_live * (f[LIVE] * eps[LIVE] * q_ig[LIVE]).sum(0)
    r0 = ir * xi / (rho_b * sink)

    return dict(r0=r0, ir=ir, beta=beta, ratio=ratio, sigma=sigma,
                C=7.47 * np.exp(-0.133 * sigma**0.55), B=0.02526 * sigma**0.54,
                E=0.715 * np.exp(-3.59e-4 * sigma))


def head_fire(p, wind_ftmin, wind_heading, slope_deg, upslope_deg):
    """Combine wind and slope as vectors (Rothermel / fireLib).

    wind_ftmin: midflame wind speed; wind_heading: direction wind blows TOWARD (deg).
    Returns max spread rate (ft/min), its direction (deg), and ellipse eccentricity.
    """
    phi_w = p["C"] * np.asarray(wind_ftmin, float) ** p["B"] * p["ratio"] ** -p["E"]
    phi_s = 5.275 * p["beta"] ** -0.3 * np.tan(np.radians(slope_deg)) ** 2
    x = phi_w * np.sin(np.radians(wind_heading)) + phi_s * np.sin(np.radians(upslope_deg))
    y = phi_w * np.cos(np.radians(wind_heading)) + phi_s * np.cos(np.radians(upslope_deg))
    phi_e = np.hypot(x, y)
    direction = np.degrees(np.arctan2(x, y)) % 360

    # Effective wind speed (ft/min) implied by the combined wind + slope factor. Rothermel's wind
    # limit applies to this effective wind: U_eff <= 0.9 * I_R.
    u_eff = (phi_e * p["ratio"] ** p["E"] / p["C"]) ** (1 / p["B"])
    u_lim = 0.9 * p["ir"]
    limited = u_eff > u_lim
    u_eff = np.where(limited, u_lim, u_eff)
    phi_e = np.where(limited, p["C"] * u_lim ** p["B"] * p["ratio"] ** -p["E"], phi_e)
    r_max = p["r0"] * (1 + phi_e)

    # Anderson (1983) length-to-width ratio; wind in m/s (0.2566 per m/s = 0.1147 per mph).
    u_ms = u_eff * 0.3048 / 60
    lw = np.minimum(0.936 * np.exp(0.2566 * u_ms) + 0.461 * np.exp(-0.1548 * u_ms) - 0.397, 8.0)
    ecc = np.sqrt(np.maximum(lw**2 - 1, 0)) / lw
    return r_max, direction, ecc


def equilibrium_moisture(temp_c, rh):
    """Simard (1968) equilibrium moisture content, fraction."""
    t = temp_c * 9 / 5 + 32
    if rh < 10:
        emc = 0.03229 + 0.281073 * rh - 0.000578 * rh * t
    elif rh < 50:
        emc = 2.22749 + 0.160107 * rh - 0.01478 * t
    else:
        emc = 21.0606 + 0.005565 * rh**2 - 0.00035 * rh * t - 0.483199 * rh
    return emc / 100
