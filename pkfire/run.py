"""One PAKFIRE run: ignition -> local scene -> weather -> spread (GPU first) -> exports for the viewer.

Outputs in data/runs/<id>/:
  meta.json      bounds, stats, weather, events, spread axis, camera track, ensemble summary
  arrival.bin    uint16 little-endian, arrival time in 1/8 min (65535 = never), north-up EPSG:4326 grid
  intensity.bin  uint8 fireline intensity (kW/m / 40, capped), same grid
  prob.bin       uint8 burn probability in percent (only when an ensemble was requested), same grid
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import time
import uuid

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds
from shapely.geometry import LineString, shape
from shapely.ops import transform as shp_transform
from scipy import ndimage

from . import fuels as fm
import json as _json
from . import adjust, moisture, solver, terrain as fd
from . import weather as wx

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "runs")
T_SCALE = 8.0
NEVER = 65535
MAX_RUN_H = 120             # longest run: arrival times are stored as uint16 in 1/8 min (136 h at most)
LEAD_IN_H = 48               # fuel moisture conditioning before the start
LC_NAMES = {1: "water", 2: "trees", 4: "flooded vegetation", 5: "crops", 7: "built area", 8: "bare ground",
            9: "snow/ice", 11: "rangeland"}


def load_calibration(name):
    """The saved calibration profile (pkfire/calibration.json) or None for the uncalibrated physics."""
    if name in (None, "", "none", "off"):
        return None
    return _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")))


GROUP_IDS = dict(grass=(1, 3, 102), shrub=(2, 5, 6, 122), forest=(9, 161, 188))


class RunError(ValueError):
    pass


def _snap(scene, r, c, max_cells):
    ok = scene["fuel"] != 0
    if ok[r, c]:
        return r, c
    r0, r1 = max(r - max_cells, 0), min(r + max_cells + 1, ok.shape[0])
    c0, c1 = max(c - max_cells, 0), min(c + max_cells + 1, ok.shape[1])
    sub = ok[r0:r1, c0:c1]
    if not sub.any():
        return None
    rs, cs = np.nonzero(sub)
    k = int(np.argmin((rs + r0 - r) ** 2 + (cs + c0 - c) ** 2))
    return int(rs[k] + r0), int(cs[k] + c0)


def _apply_interventions(scene, breaks, treatments):
    """Fuel breaks (lines) and treatments (polygons) written into the fuel and continuity grids.

    A break is at least three cells wide: the solver also travels along knight moves (two cells in one direction),
    which would otherwise hop over a one-cell line. Treatment mode "remove" takes fuel out completely;
    mode "factor" multiplies the continuity (and so the spread rate) by ``factor``."""
    ny, nx, dx = scene["ny"], scene["nx"], scene["dx"]
    to_utm = Transformer.from_crs(4326, scene["crs"], always_xy=True).transform
    zero, factor = [], []
    for b in breaks or []:
        g = shp_transform(to_utm, LineString(b["coords"])).buffer(max(float(b.get("width_m", 30.0)), 3 * dx) / 2)
        zero.append((g, 1))
    for t in treatments or []:
        g = shp_transform(to_utm, shape(t["geometry"]))
        if t.get("mode", "remove") == "remove":
            zero.append((g, 1))
        else:
            factor.append((g, float(t.get("factor", 0.3))))
    info = dict(break_cells=0, treated_cells=0)
    if zero:
        m = rasterize(zero, out_shape=(ny, nx), transform=scene["transform"], fill=0, dtype=np.uint8).astype(bool)
        info["break_cells"] = int((m & (scene["fuel"] != 0)).sum())
        scene["fuel"][m] = 0
    for g, f in factor:
        m = rasterize([(g, 1)], out_shape=(ny, nx), transform=scene["transform"], fill=0, dtype=np.uint8).astype(bool)
        scene["cont"][m] *= f
        info["treated_cells"] += int((m & (scene["fuel"] != 0)).sum())
    info["break_ha"] = round(info["break_cells"] * dx * dx / 1e4, 1)
    info["treated_ha"] = round(info["treated_cells"] * dx * dx / 1e4, 1)
    return info


def _weather(lat, lon, start, hours, manual):
    """Hourly weather from LEAD_IN_H before the start to the end. Returns dict + conditioning flag."""
    n = LEAD_IN_H + hours + 2
    if manual:
        w = wx.manual(hours + LEAD_IN_H, manual["speed_ms"], manual["from_deg"], manual.get("temp_c", 32.0), manual.get("rh", 25.0))
        w["conditioning"] = "none (manual weather held constant)"
        return w
    try:
        w = wx.hourly(lat, lon, start - dt.timedelta(hours=LEAD_IN_H), hours + LEAD_IN_H)
        w["conditioning"] = f"{LEAD_IN_H} h of observed/forecast weather before the start"
        return w
    except Exception:
        pass
    try:                                                # forecast-only source: hold the first hour for the lead-in
        w = wx.hourly(lat, lon, start, hours)
    except Exception as e:
        raise RunError(f"Weather service unavailable ({e}); supply a manual wind and re-run.")
    for k in ("temp", "rh", "wind", "wdir"):
        w[k] = np.concatenate([np.full(LEAD_IN_H, w[k][0]), w[k]])
    w["precip"] = np.concatenate([np.zeros(LEAD_IN_H), w["precip"]])
    w["conditioning"] = "none (no weather history from this source; first hour held, no antecedent rain)"
    return w


def _tables(w, ndvi_mean, perturb=None):
    temp, rh, rain = w["temp"], w["rh"], w["precip"]
    md = moisture.dead_moisture(temp, rh, rain)
    if perturb:
        md = np.clip(md * perturb["moist"], 0.02, 0.6)
    tb = fm.build_tables(md, *fm.greenness_to_live_moisture(ndvi_mean))
    tb["spot_ign"] = moisture.ignition_probability(temp, md[0]).astype(np.float32)
    tb["emc_pct"] = (md[0] * 100.0).astype(np.float32)
    return tb, md


ENS_WIDTHS = dict(wind=0.20, wdir=12.0, moist=0.15, ndvi=0.03, rate=0.0)   # standard deviations: wind speed (fraction), wind direction (deg), dead moisture (fraction), NDVI, log of the spread-rate multiplier


def ensemble_inputs(w, ndvi_mean, moist_cal, n, seed, widths=None):
    """Perturbed (tables, wind speeds, wind directions) for n ensemble members. The widths are scaled by validation (tools/ensemble_widths.py)."""
    wd = {**ENS_WIDTHS, **(widths or {})}
    rng = np.random.default_rng(seed)
    tbs, wss, wds = [], [], []
    for _ in range(int(n)):
        wss.append(np.clip(w["wind"] * rng.normal(1.0, wd["wind"]), 0, None))
        wds.append((w["wdir"] + rng.normal(0, wd["wdir"])) % 360)
        pert = dict(moist=moist_cal * float(np.clip(rng.normal(1.0, wd["moist"]), 0.6, 1.5)))
        tb = _tables(w, ndvi_mean + rng.normal(0, wd["ndvi"]), pert)[0]
        if wd["rate"] > 0:                                              # spread-rate model error: lognormal multiplier with median 1
            tb["r0"] = (tb["r0"] * float(np.exp(rng.normal(0, wd["rate"])))).astype(np.float32)
        tbs.append(tb)
    return tbs, wss, wds


def _lead_points(arr, ign_rc, dx, t_end, step=15.0):
    """Head-of-fire position per time step, in grid (col,row) units, from the main fire only.

    Spot fires that landed away from the main body are excluded (connected component of the burned cells, closing gaps of
    up to two cells, that contains the ignition), and the path is smoothed with a running median so that it follows the
    advance of the front instead of the scatter of single cells."""
    burnt = np.isfinite(arr)
    lab, _ = ndimage.label(ndimage.binary_dilation(burnt, structure=np.ones((3, 3)), iterations=2))
    ids = {int(lab[r, c]) for r, c in ign_rc if 0 <= r < lab.shape[0] and 0 <= c < lab.shape[1] and lab[r, c] > 0}
    main = burnt & np.isin(lab, list(ids)) if ids else burnt
    rr, cc = np.nonzero(main)
    ta = arr[main]
    o = np.argsort(ta)
    rr, cc, ta = rr[o], cc[o], ta[o]
    ir = np.mean([p[0] for p in ign_rc]); ic = np.mean([p[1] for p in ign_rc])
    pts = []
    for t in np.arange(step, t_end + step, step):
        lo, hi = np.searchsorted(ta, t - step), np.searchsorted(ta, t, side="right")
        if hi - lo < 1:
            continue
        ar, ac = rr[lo:hi], cc[lo:hi]
        d = np.hypot(ar - ir, ac - ic)
        lead = d >= np.quantile(d, 0.8)
        pts.append([float(t), float(ac[lead].mean()), float(ar[lead].mean()), float(max(ac.max() - ac.min(), ar.max() - ar.min()) * dx)])
    if len(pts) >= 5:                                   # running median over 5 steps for the position, never going back in time
        a = np.array(pts)
        for k in (1, 2):
            sm = a[:, k].copy()
            for i in range(len(a)):
                sm[i] = np.median(a[max(i - 2, 0):i + 3, k])
            a[:, k] = sm
        pts = a.tolist()
    return [tuple(p) for p in pts], (rr, cc, ta)


def _camera(pts, scene, ign_rc, t_end):
    tr = scene["transform"]
    to_ll = Transformer.from_crs(scene["crs"], 4326, always_xy=True)
    ir = np.mean([p[0] for p in ign_rc]); ic = np.mean([p[1] for p in ign_rc])
    cen = np.array([ic, ir], float); prev = cen.copy()
    zoom, bearing, head = 13.0, 0.0, None
    times = np.array([p[0] for p in pts]) if pts else np.array([])
    track = []
    for t in np.arange(0.0, t_end + 4.0, 4.0):
        if len(times):
            k = min(int(np.searchsorted(times, t)), len(times) - 1)
            _, cx, cy, ext = pts[k]
            tgt, width_m = np.array([cx, cy]), max(1800.0, ext * 2.6)
        else:
            tgt, width_m = cen, 1800.0
        cen = cen + (tgt - cen) * 0.22
        mv = cen - prev
        if np.hypot(*mv) > 0.02:
            a = (math.degrees(math.atan2(mv[0], -mv[1])) + 360) % 360
            head = a if head is None else head + ((a - head + 180) % 360 - 180) * 0.08
        prev = cen.copy()
        if head is not None:
            bearing += ((head - bearing + 180) % 360 - 180) * 0.06
        x, y = tr * (cen[0] + 0.5, cen[1] + 0.5)
        lon, lat = to_ll.transform(x, y)
        want = math.log2(78271.5 * math.cos(math.radians(lat)) / (width_m / 1000.0))
        zoom += (min(max(want, 9.6), 15.4) - zoom) * 0.12
        track.append([round(float(t), 2), round(lon, 5), round(lat, 5), round(zoom, 2), round(bearing % 360, 1)])
    return track


def _export_grid(scene, ny, nx, cell_m):
    l, b, r_, t_ = rasterio.transform.array_bounds(ny, nx, scene["transform"])
    W, S, E, N = transform_bounds(scene["crs"], 4326, l, b, r_, t_, densify_pts=21)
    clat = (S + N) / 2
    px_lat = cell_m / 111320.0
    px_lon = cell_m / (111320.0 * math.cos(math.radians(clat)))
    ow = min(int(round((E - W) / px_lon)), 2400)
    oh = min(int(round((N - S) / px_lat)), 2400)
    return (W, S, E, N), (ow, oh), from_origin(W, N, (E - W) / ow, (N - S) / oh)


def _warp(a, scene, otr, ow, oh, dtype, nodata):
    out = np.full((oh, ow), nodata, dtype)
    reproject(a, out, src_transform=scene["transform"], src_crs=scene["crs"], dst_transform=otr,
              dst_crs=CRS.from_epsg(4326), src_nodata=nodata, dst_nodata=nodata, resampling=Resampling.nearest)
    return out


def run(lat, lon, start=None, hours=24, half_km=40.0, cell_m=50.0, spotting=True, wind=None,
        extra_ignitions=None, seed=1, ensemble=0, backend="auto", progress=lambda *_: None, out_id=None,
        breaks=None, treatments=None, label=None, cancel=lambda: False, calibration="default", detections=None):
    t_all = time.time()
    if not (23.5 <= lat <= 37.2 and 60.8 <= lon <= 77.9):
        raise RunError("Ignition is outside Pakistan's data coverage (lat 23.5-37.2, lon 60.8-77.9).")
    start = start or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    if isinstance(start, str):
        start = dt.datetime.fromisoformat(start.replace("Z", ""))
    hours = int(min(max(hours, 1), MAX_RUN_H))
    cell_m = float(cell_m)
    while 2 * half_km * 1000 / cell_m > 2400:
        cell_m *= 1.25

    def check():
        if cancel():
            raise RunError("cancelled")

    progress("terrain", "Loading terrain, land cover and vegetation…")
    t0 = time.time()
    scene = fd.build_scene(lat, lon, half_km, cell_m, start.date())
    t_scene = time.time() - t0
    ny, nx = scene["ny"], scene["nx"]
    to_utm = Transformer.from_crs(4326, scene["crs"], always_xy=True)
    ign = []
    for la, lo in [(lat, lon)] + [tuple(p) for p in (extra_ignitions or [])]:
        x, y = to_utm.transform(lo, la)
        c, r = ~scene["transform"] * (x, y)
        r, c = int(r), int(c)
        if not (0 <= r < ny and 0 <= c < nx):
            continue
        s = _snap(scene, r, c, int(600 / cell_m) + 2)
        if s is None:
            cls = LC_NAMES.get(int(scene["lc"][r, c]), "non-fuel")
            raise RunError(f"No burnable fuel within 600 m of the ignition (land cover there: {cls}). "
                           "Pick a point on vegetation, crops or forest.")
        ign.append((s[0], s[1], 0.0))
    seen = {(g[0], g[1]) for g in ign}
    n_assim = 0
    for dlon, dlat, dmin in (detections or [])[:600]:                       # satellite detections seeded at the time they were seen
        x, y = to_utm.transform(dlon, dlat)
        c, r = ~scene["transform"] * (x, y)
        r, c = int(r), int(c)
        if 0 <= r < ny and 0 <= c < nx:
            s2 = _snap(scene, r, c, int(300 / cell_m) + 1)
            if s2 is not None and s2 not in seen:
                seen.add(s2); ign.append((s2[0], s2[1], max(float(dmin), 0.0))); n_assim += 1
    if not ign:
        raise RunError("Ignition falls outside the simulation window.")
    check()
    mods = _apply_interventions(scene, breaks, treatments) if (breaks or treatments) else None
    if mods:
        ign = [g for g in ign if scene["fuel"][g[0], g[1]] != 0]
        if not ign:
            raise RunError("The ignition lies inside a fuel break or treated area; move the ignition or the intervention.")

    progress("weather", "Fetching hourly fire weather…")
    w = _weather(lat, lon, start, hours, wind)
    prof = load_calibration(calibration)
    adj = adjust.apply(scene, w, (prof or {}).get("params"), hours, LEAD_IN_H, user_wind=(w["source"] == "manual"), start=start)
    w = adj["w"]
    moist_cal = adj["moist"]
    burnable = scene["fuel"] != 0
    ndvi_mean = float(scene["ndvi"][burnable].mean()) if burnable.any() else 0.25
    tables, md = _tables(w, ndvi_mean, dict(moist=moist_cal) if moist_cal != 1.0 else None)
    grid = dict(ny=ny, nx=nx, dx=cell_m, slope=scene["slope"], upaz=scene["upaz"], cont=adj["cont"], wmod=adj["wmod"], wrot=adj["wrot"])
    if "crown_ci0" in adj:
        grid.update(crown_ci0=adj["crown_ci0"], crown_cbd=adj["crown_cbd"])
    spot_rate = 2e-4 if spotting else 0.0

    check()
    progress("spread", "Simulating fire spread…")
    arr, ros, fli, spots, info = solver.simulate(grid, scene["fuel"], tables, w["wind"], w["wdir"], ign, hours,
                                                 spot_rate=spot_rate, seed=seed, hour0=LEAD_IN_H, backend=backend, decay_min=adj["decay_min"])
    arr = np.where(np.isfinite(arr) & (arr <= hours * 60), arr, np.inf).astype(np.float32)
    burnt = np.isfinite(arr)
    nb = int(burnt.sum())
    if nb == 0:
        raise RunError("The fire did not spread (fuel too wet or sparse at these conditions).")

    # --- ensemble (same solver, perturbed wind / moisture / ignition time), burn probability ---------------------
    prob = None
    ens_info = None
    if ensemble and ensemble > 1:
        progress("ensemble", f"Running {ensemble}-member ensemble…")
        t1 = time.time()
        tbs, wss, wds = ensemble_inputs(w, ndvi_mean, moist_cal, ensemble, seed + 7, (prof or {}).get("ensemble_widths"))
        arrs, ens_solve = solver.simulate_batch(grid, scene["fuel"], tbs, wss, wds, ign, hours, spot_rate=spot_rate, seed=seed + 100, hour0=LEAD_IN_H,
                                                backend=backend, decay_min=adj["decay_min"])
        mem = np.isfinite(arrs) & (arrs <= hours * 60)
        counts = mem.sum(0).astype(np.float32)
        areas = [float(m_.sum() * cell_m * cell_m / 1e4) for m_ in mem]
        prob = counts / ensemble
        ens_info = dict(members=int(ensemble), area_ha_p10=round(float(np.percentile(areas, 10)), 1),
                        area_ha_median=round(float(np.median(areas)), 1), area_ha_p90=round(float(np.percentile(areas, 90)), 1),
                        wind_speed_sd_pct=round(100 * {**ENS_WIDTHS, **((prof or {}).get('ensemble_widths') or {})}['wind']), wind_dir_sd_deg={**ENS_WIDTHS, **((prof or {}).get('ensemble_widths') or {})}['wdir'],
                        dead_moisture_sd_pct=round(100 * {**ENS_WIDTHS, **((prof or {}).get('ensemble_widths') or {})}['moist']), ndvi_sd={**ENS_WIDTHS, **((prof or {}).get('ensemble_widths') or {})}['ndvi'], spread_rate_sd_log={**ENS_WIDTHS, **((prof or {}).get('ensemble_widths') or {})}['rate'],
                        seconds=round(time.time() - t1, 1), solver=ens_solve.get("backend"), solve_s=ens_solve.get("solve_s"))

    check()
    progress("export", "Preparing the animation…")
    run_id = out_id or (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4])
    d = os.path.join(RUNS, run_id)
    os.makedirs(d, exist_ok=True)
    bounds, (ow, oh), otr = _export_grid(scene, ny, nx, cell_m)
    a8 = np.where(burnt, np.clip(arr * T_SCALE, 0, NEVER - 1), NEVER).astype(np.uint16)
    _warp(a8, scene, otr, ow, oh, np.uint16, NEVER).astype("<u2").tofile(os.path.join(d, "arrival.bin"))
    _warp(np.clip(fli / 40.0, 0, 255).astype(np.uint8), scene, otr, ow, oh, np.uint8, 0).tofile(os.path.join(d, "intensity.bin"))
    dzdy, dzdx = np.gradient(scene["dem"], cell_m)                                  # hillshade (315 deg, 45 deg sun) for the time-lapse background
    sl_ = np.arctan(np.hypot(dzdx, dzdy)); asp = np.arctan2(-dzdx, dzdy)
    hs = np.clip(np.cos(np.radians(45)) * np.cos(sl_) + np.sin(np.radians(45)) * np.sin(sl_) * np.cos(np.radians(315) - asp), 0, 1)
    _warp((hs * 254 + 1).astype(np.uint8), scene, otr, ow, oh, np.uint8, 0).tofile(os.path.join(d, "hillshade.bin"))
    if prob is not None:
        _warp(np.round(prob * 100).astype(np.uint8), scene, otr, ow, oh, np.uint8, 0).tofile(os.path.join(d, "prob.bin"))

    t_end = float(arr[burnt].max())
    ign_rc = [(p[0], p[1]) for p in ign]
    pts, (rr, cc, ta) = _lead_points(arr, ign_rc, cell_m, t_end)
    track = _camera(pts, scene, ign_rc, t_end)
    to_ll = Transformer.from_crs(scene["crs"], 4326, always_xy=True)

    def ll(col, row):
        x, y = scene["transform"] * (col + 0.5, row + 0.5)
        lo, la = to_ll.transform(x, y)
        return round(lo, 5), round(la, 5)

    # spread axis with elevation profile
    axis = []
    dem = scene["dem"]
    cum = 0.0
    prev_pt = None
    ign_c, ign_r = np.mean([p[1] for p in ign]), np.mean([p[0] for p in ign])
    path = [(0.0, ign_c, ign_r)] + [(t, cx, cy) for t, cx, cy, _ in pts]
    for t, cx, cy in path:
        r_i, c_i = int(min(max(cy, 0), ny - 1)), int(min(max(cx, 0), nx - 1))
        if prev_pt is not None:
            cum += math.hypot(cx - prev_pt[0], cy - prev_pt[1]) * cell_m
        prev_pt = (cx, cy)
        lo, la = ll(cx, cy)
        axis.append([round(t, 1), lo, la, round(float(dem[r_i, c_i]), 1), round(cum / 1000.0, 3)])
    step = max(len(axis) // 80, 1)
    axis = axis[::step] + ([axis[-1]] if (len(axis) - 1) % step else [])

    # stats and series (every 30 min)
    cell_ha = cell_m * cell_m / 1e4
    mins = np.arange(30, t_end + 30, 30)
    area_series = [[float(tm), round(float((arr <= tm).sum() * cell_ha), 1)] for tm in mins]
    ros_series, fl_series = [], []
    for tm in mins:
        sel = burnt & (arr > tm - 30) & (arr <= tm)
        mr = float(ros[sel].max()) if sel.any() else 0.0
        mi = float(fli[sel].max()) if sel.any() else 0.0
        ros_series.append([float(tm), round(mr, 2)])
        fl_series.append([float(tm), round(0.0775 * mi ** 0.46, 2) if mi > 0 else 0.0])   # Byram flame length, m
    lc = scene["lc"]
    by_class = {LC_NAMES.get(int(k), str(int(k))): round(float(((lc == k) & burnt).sum() * cell_ha), 1)
                for k in np.unique(lc[burnt])}
    built = lc == fd.LC_BUILT
    near = ndimage.distance_transform_edt(~burnt) * cell_m < 1000.0
    by_fuel = {}
    for fid in fm.ORDER:
        n_ = int(((scene["fuel"] == fm.row_of(fid)) & burnt).sum())
        if n_:
            by_fuel[fm.MODELS[fid]["name"]] = round(n_ * cell_ha, 1)

    # key events
    events = [dict(t=0.0, kind="ignition", label="Ignition", lon=ll(ign_c, ign_r)[0], lat=ll(ign_c, ign_r)[1])]
    if len(spots):
        k = int(np.argmin(spots[:, 2]))
        j = int(spots[k, 1]); r_, c_ = divmod(j, nx)
        events.append(dict(t=round(float(spots[k, 2]), 1), kind="spot", label="First spot fire", lon=ll(c_, r_)[0], lat=ll(c_, r_)[1]))
    if len(area_series) > 2:
        inc = np.diff([0.0] + [a for _, a in area_series])
        k = int(np.argmax(inc))
        sel = burnt & (arr > mins[k] - 30) & (arr <= mins[k])
        if sel.any():
            r_, c_ = np.unravel_index(np.argmax(np.where(sel, ros, 0)), ros.shape)
            events.append(dict(t=float(mins[k] - 15), kind="growth", label="Fastest growth", lon=ll(c_, r_)[0], lat=ll(c_, r_)[1]))
    edge = np.zeros_like(burnt); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    if (burnt & edge).any():
        te = float(arr[burnt & edge].min())
        r_, c_ = np.unravel_index(np.argmin(np.where(burnt & edge, arr, np.inf)), arr.shape)
        events.append(dict(t=round(te, 1), kind="edge", label="Reaches model window edge", lon=ll(c_, r_)[0], lat=ll(c_, r_)[1]))
    bdil = ndimage.binary_dilation(built, iterations=max(int(300 / cell_m), 1))
    if (bdil & burnt).any():
        te = float(arr[bdil & burnt].min())
        r_, c_ = np.unravel_index(np.argmin(np.where(bdil & burnt, arr, np.inf)), arr.shape)
        events.append(dict(t=round(te, 1), kind="settlement", label="Within 300 m of a settlement", lon=ll(c_, r_)[0], lat=ll(c_, r_)[1]))
    events.sort(key=lambda e: e["t"])

    spot_json = []
    for s in spots:
        r_, c_ = divmod(int(s[1]), nx)
        lo, la = ll(c_, r_)
        spot_json.append([lo, la, round(float(s[2]), 1)])
    ign_ll = [list(ll(p[1], p[0])) for p in ign]
    if n_assim:
        warnings_extra = [f"Assimilated {n_assim} satellite detections as ignitions at the times they were seen: the run starts from the observed state, so the early part of the fire is observed, not simulated."]
    else:
        warnings_extra = []
    if prof:
        r_ = prof["results"]
        warnings = [f"Rates are calibrated to 2024-2025 satellite fire events ({prof['name']}): burned-area bias is reduced (median simulated/mapped area 7.0 to {r_['test_area_ratio']['calibrated']} on held-back 2026 fires) and the model's agreement with satellite footprints on held-back fires is only slightly above an equal-area circle (IoU {r_['test_firms_iou']['calibrated']} against {r_['test_firms_iou']['circle']}). Treat the result as a scenario."]
    else:
        warnings = ["Uncalibrated physics: on satellite-mapped fires the model burned about 5 to 7 times the mapped area; use the calibrated setting unless you are testing the physics."]
    warnings += warnings_extra
    if (burnt & edge).any():
        warnings.append("The fire reached the edge of the simulated window and is truncated there; choose a larger area.")
    if str(w.get("conditioning", "")).startswith("none"):
        warnings.append("Fuel moisture was not conditioned on prior weather (" + w["conditioning"] + ").")
    if w["source"] == "manual":
        warnings.append("Manual weather: one constant wind, temperature and humidity, no rain, for the whole run.")
    if info.get("gpu_error"):
        warnings.append("GPU solver failed and the CPU solver was used: " + info["gpu_error"])
    tree_ha = sum(v for k, v in by_fuel.items() if "litter" in k or "timber" in k.lower())
    crown_ha = float(((tables["crown_i0"][scene["fuel"]] > 0) & burnt & (fli >= tables["crown_i0"][scene["fuel"]])).sum() * cell_m * cell_m / 1e4)
    if crown_ha > 0:
        warnings.append(f"{crown_ha:.0f} ha burned as crown fire. Canopy base height (3 m effective) and bulk density (0.10 kg/m3) of pine forest are assumed stand defaults, not measured, and crown fire has not been validated.")
    elif tree_ha > 1:
        warnings.append(f"{tree_ha:.0f} ha of forest burned as surface fire. Crown fire is modelled only in pine fuels from assumed canopy values (no crown fire was triggered here); spread in forest may be underestimated.")
    if mods:
        warnings.append(f"Interventions: {mods['break_ha']} ha of fuel break and {mods['treated_ha']} ha of treated fuel applied.")
    wh = hours + 1
    sl = slice(LEAD_IN_H, LEAD_IN_H + wh)
    meta = dict(
        id=run_id, bounds=list(bounds), size=[ow, oh], t_scale=T_SCALE, never=NEVER, hours=hours,
        t_end_min=round(t_end, 1), start=start.isoformat(), ignitions=ign_ll, spots=spot_json, camera=track,
        axis=axis, events=events, cell_m=cell_m, half_km=half_km, spotting=bool(spotting),
        calibration=(prof or {}).get("name"), has_prob=prob is not None, ensemble=ens_info, warnings=warnings, interventions=mods, label=label,
        params=dict(assimilated=int(n_assim), lat=lat, lon=lon, hours=hours, half_km=half_km, cell_m=cell_m, spotting=bool(spotting), ensemble=int(ensemble or 0),
                    manual_weather=bool(wind), backend=backend, seed=seed),
        weather=dict(source=w["source"], conditioning=w.get("conditioning", ""),
                     temp=[round(float(x), 1) for x in w["temp"][sl]], rh=[round(float(x), 0) for x in w["rh"][sl]],
                     wind=[round(float(x), 1) for x in w["wind"][sl]], wdir=[round(float(x), 0) for x in w["wdir"][sl]],
                     rain=[round(float(x), 2) for x in w["precip"][sl]],
                     dead1h=[round(float(x) * 100, 1) for x in md[0, sl]]),
        stats=dict(burned_ha=round(nb * cell_ha, 1), max_head_ros_m_min=round(float(ros.max()), 1),
                   max_intensity_kw_m=round(float(fli.max()), 0), crown_fire_ha=round(crown_ha, 1), spot_fires=len(spot_json),
                   area_series=area_series, ros_series=ros_series, flame_series=fl_series, by_landcover_ha=by_class,
                   by_fuel_ha=by_fuel, builtup_burned_ha=round(float((built & burnt).sum() * cell_ha), 1),
                   builtup_within_1km_ha=round(float((built & near & ~burnt).sum() * cell_ha), 1),
                   ndvi_file=scene["ndvi_file"], ndvi_mean=round(ndvi_mean, 3),
                   reached_edge=bool((burnt & edge).any()), backend=info.get("backend"),
                   solve_s=info.get("solve_s"), iterations=info.get("iterations"), gpu_error=info.get("gpu_error"),
                   scene_s=round(t_scene, 1), compute_s=round(time.time() - t_all, 1),
                   axis_length_km=axis[-1][4] if axis else 0.0),
    )
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    progress("done", "Done")
    return meta
