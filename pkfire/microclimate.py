"""Terrain and canopy effects on dead fuel moisture (slope, aspect, cast shadow, elevation, canopy shade).

The weather arrives for one point. Dead fine fuel does not sit at that air temperature and humidity: a sunlit south slope is hotter and drier, a
shaded north slope or a forest floor stays cooler and moister, and higher ground is cooler with (at constant vapour pressure) a higher relative humidity.
For each cell this module estimates the difference in equilibrium moisture (Simard 1968) from a reference cell (flat, open, at the weather point):

  1. solar position (NOAA low-precision formulae) for sampled daylight hours of the run;
  2. beam exposure of the cell: cosine of the angle between the sun and the slope normal, zero when the slope faces away or a ridge shades it
     (horizon angle in the sun's azimuth from a 3 km search);
  3. fuel surface temperature  Tf = Ta + DT0 * exposure * (1 - canopy shade), with DT0 = 20 K for full sun on bare fuel (published surface fuel
     excess temperatures are 10-25 K; this value is an assumption, not a fit, and clear sky is assumed because cloud cover is not available);
  4. air temperature at the cell's elevation (6.5 K/km lapse) and the same vapour pressure lowered with the 2 K/km dew-point lapse, so relative
     humidity at the fuel is  100 * e / es(Tf);
  5. delta = EMC(Tf, RH_fuel) - EMC(reference), averaged over the sampled daylight hours.

The delta is converted to a spread-rate multiplier per fuel model from the Rothermel no-wind rate at moisture raised or lowered by that delta
(1 h fuel by delta, 10 h by 0.6 delta, 100 h by 0.3 delta because thicker fuel responds more slowly). The result is applied to the continuity field.
It is a first-order correction of the spread rate at the dryness of the fire's busiest hours, not a time-resolved fuel-moisture model.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from . import fuels as fm
from . import moisture

DT0 = 20.0                      # K of fuel surface heating above air in full sun (assumption)
LAPSE_T = 6.5e-3                # K per m
LAPSE_TD = 2.0e-3               # K per m
CANOPY_SHADE = {2: 0.7, 4: 0.3}  # ESRI land cover: trees, flooded vegetation
DELTAS = np.array([-0.06, -0.03, 0.0, 0.03, 0.06])
LAG_WEIGHT = np.array([1.0, 0.6, 0.3])
MAX_SAMPLE_HOURS = 10


def sun_position(lat, lon, t):
    """Solar elevation and azimuth (radians; azimuth clockwise from north) for a naive UTC datetime."""
    doy = t.timetuple().tm_yday
    frac = t.hour + t.minute / 60.0
    g = 2 * math.pi / 365.0 * (doy - 1 + (frac - 12) / 24.0)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    ha = math.radians((frac * 60 + eqt + 4 * lon) / 4.0 - 180.0)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    el = math.pi / 2 - math.acos(max(-1.0, min(1.0, cz)))
    az = math.atan2(math.sin(ha), math.cos(ha) * math.sin(la) - math.tan(decl) * math.cos(la)) + math.pi
    return el, az % (2 * math.pi)


def _es(t_c):
    """Saturation vapour pressure (hPa), Magnus."""
    return 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))


def _emc(t_c, rh):
    """Simard (1968) equilibrium moisture, fraction (same formulas as pkfire/vendor/rothermel.equilibrium_moisture, vectorised)."""
    t = t_c * 9 / 5 + 32
    rh = np.clip(rh, 1.0, 100.0)
    lo = 0.03229 + 0.281073 * rh - 0.000578 * rh * t
    mid = 2.22749 + 0.160107 * rh - 0.01478 * t
    hi = 21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * t - 0.483199 * rh
    return np.where(rh < 10, lo, np.where(rh < 50, mid, hi)) / 100.0


def horizon_tangent(dem, dx, n_az=8, max_m=3000.0, coarse=4):
    """(n_az, ny, nx) tangent of the horizon elevation angle toward azimuth k * 360 / n_az, from a coarsened DEM."""
    ny, nx = dem.shape
    cd = ndimage.zoom(dem, 1.0 / coarse, order=1)
    cdx = dx * coarse
    ny2, nx2 = cd.shape
    steps = max(int(max_m / cdx), 1)
    pad = steps + 1
    P = np.pad(cd, pad, mode="edge")
    out = np.zeros((n_az, ny, nx), np.float32)
    for k in range(n_az):
        az = 2 * math.pi * k / n_az
        dcol, drow = math.sin(az), -math.cos(az)
        best = np.full(cd.shape, -1.0, np.float32)
        for s in range(1, steps + 1):
            dr, dc = int(round(s * drow)), int(round(s * dcol))
            sh = P[pad + dr: pad + dr + ny2, pad + dc: pad + dc + nx2]
            best = np.maximum(best, (sh - cd) / (s * cdx))
        out[k] = ndimage.zoom(best, (ny / ny2, nx / nx2), order=1)[:ny, :nx]
    return out


def _centre_lonlat(scene):
    from pyproj import Transformer
    ny, nx = scene["ny"], scene["nx"]
    x, y = scene["transform"] * (nx / 2, ny / 2)
    lon, lat = Transformer.from_crs(scene["crs"], 4326, always_xy=True).transform(x, y)
    return lat, lon


def moisture_delta(scene, w, start, hours, lead):
    """Per-cell mean difference (fraction) of dead fine fuel equilibrium moisture from the flat open reference, over sampled daylight hours.
    Cached on the scene. Returns (delta array, number of daylight hours sampled)."""
    key = ("_mdelta", start.isoformat(), int(hours), int(lead))
    if key in scene:
        return scene[key]
    dem, dx = scene["dem"], scene["dx"]
    lat, lon = _centre_lonlat(scene)
    ny, nx = dem.shape
    z0 = float(dem[ny // 2, nx // 2])
    hs = []
    for h in range(int(hours) + 1):
        el, az = sun_position(lat, lon, start + __import__("datetime").timedelta(hours=h))
        if el > math.radians(8.0):
            hs.append((h, el, az))
    if not hs:
        scene[key] = (np.zeros(dem.shape, np.float32), 0)
        return scene[key]
    if len(hs) > MAX_SAMPLE_HOURS:
        hs = [hs[int(i)] for i in np.linspace(0, len(hs) - 1, MAX_SAMPLE_HOURS)]
    if "_hor" not in scene:
        scene["_hor"] = horizon_tangent(dem, dx)
    hor = scene["_hor"]
    # slope normal: tan(slope) = scene['slope'], upslope azimuth = scene['upaz']; the slope faces opposite to upslope
    sl = np.arctan(scene["slope"])
    facing = scene["upaz"] + math.pi
    shade = np.zeros(dem.shape, np.float32)
    for lc, f in CANOPY_SHADE.items():
        shade[scene["lc"] == lc] = f
    dz = (z0 - dem).astype(np.float32)
    acc = np.zeros(dem.shape, np.float64)
    for h, el, az in hs:
        i = min(lead + h, len(w["temp"]) - 1)
        ta, rh = float(w["temp"][i]), float(min(max(w["rh"][i], 1.0), 100.0))
        td = 243.12 * math.log(rh / 100.0 * math.exp(17.62 * ta / (243.12 + ta))) / (17.62 - math.log(rh / 100.0 * math.exp(17.62 * ta / (243.12 + ta))))
        cos_i = math.sin(el) * np.cos(sl) + math.cos(el) * np.sin(sl) * np.cos(az - facing)
        k = int(round(az / (2 * math.pi) * hor.shape[0])) % hor.shape[0]
        lit = (cos_i > 0) & (hor[k] < math.tan(el))
        expo = np.where(lit, cos_i, 0.0)
        tf = ta + LAPSE_T * dz + DT0 * expo * (1.0 - shade)
        e_cell = _es(td + LAPSE_TD * dz)
        rh_f = 100.0 * e_cell / _es(tf)
        tf_ref = ta + DT0 * math.sin(el)
        rh_ref = 100.0 * _es(td) / _es(tf_ref)
        acc += _emc(tf, rh_f) - float(_emc(np.float64(tf_ref), np.float64(rh_ref)))
    delta = np.clip(acc / len(hs), -0.06, 0.06).astype(np.float32)
    scene[key] = (delta, len(hs))
    return scene[key]


def rate_multiplier(scene, w, start, hours, lead, strength=1.0):
    """Per-cell spread-rate multiplier (float32) for the moisture difference above, or None when it is zero everywhere."""
    delta, n = moisture_delta(scene, w, start, hours, lead)
    if n == 0 or strength <= 0:
        return None
    md = moisture.dead_moisture(np.asarray(w["temp"], float), np.asarray(w["rh"], float), np.asarray(w["precip"], float))
    seg = md[:, lead: lead + int(hours) + 1]
    ref = seg.mean(axis=1)
    burnable = scene["fuel"] != 0
    ndvi = float(scene["ndvi"][burnable].mean()) if burnable.any() else 0.25
    herb, wood = fm.greenness_to_live_moisture(ndvi)
    r = []
    for d in DELTAS:
        m = np.clip(ref + LAG_WEIGHT * d, 0.02, 0.6).reshape(3, 1)
        r.append(fm.build_tables(m, herb, wood)["r0"][0])
    r = np.array(r)                                                     # (n_delta, rows)
    base = r[2]
    ratio = np.where(base > 1e-6, r / np.maximum(base, 1e-6), 1.0)      # (n_delta, rows)
    out = np.ones(scene["fuel"].shape, np.float32)
    d = np.clip(delta * strength, DELTAS[0], DELTAS[-1])
    for row in np.unique(scene["fuel"][burnable]):
        m = scene["fuel"] == row
        out[m] = np.interp(d[m], DELTAS, ratio[:, row])
    return out


def channel_strength(scene, w, start, hours, lead):
    """Stability index (0-1) of this run for terrain channelling of the wind: strongest at night with light wind (stable air),
    weaker by day (mixed air) and falling as 1 / (1 + (u / 8 m/s)^2) because strong synoptic wind overrides the terrain. A parameterisation, not a model of the boundary layer."""
    import datetime
    lat, lon = _centre_lonlat(scene)
    vals = []
    for h in range(int(hours) + 1):
        el, _ = sun_position(lat, lon, start + datetime.timedelta(hours=h))
        u = float(w["wind"][min(lead + h, len(w["wind"]) - 1)])
        vals.append((1.0 if el <= 0 else 0.6) / (1.0 + (u / 8.0) ** 2))
    return float(np.mean(vals))
