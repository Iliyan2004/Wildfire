"""Validation of PAKFIRE against NASA VIIRS satellite detections.

Events: detections are linked in space and time (1.5 km, 2 days) into clusters; clusters that look like a spreading vegetation fire
(enough distinct pixels, 18 h or more, 1.5-40 km across, mostly on burnable land) become events. Each event is simulated from the
detections of its first overpass with weather observed at the ignition, and scored against the later detections.

What the satellite gives is coarse: 375 m pixels seen at overpass times (about two per day for S-NPP), so the "observed" footprint
(union of detected pixels) is smaller than the true burn scar and misses the quiet parts of the fire. Scores are proxies, useful to
compare model versions and baselines, not absolute accuracy.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import pickle

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import box

from . import adjust, burned, fuels as fm
from . import run as runmod, solver, terrain

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(HERE, "data", "validation")
FIRMS = os.path.join(HERE, "data", "external", "firms_pak_2024_2026.parquet")
YEARLY = os.path.join(HERE, "data", "external", "firms_yearly")
BURNABLE_LC = (2, 4, 5, 8, 11)          # trees, flooded vegetation, crops, bare ground, rangeland
GROUPS = dict(grass=(1, 3, 102), shrub=(2, 5, 6, 122), forest=(9, 161, 188))
SPLIT_DATE = "2026-01-01"                # development before, held-back test from this date
CELL_M = 60.0
MAX_HOURS = 72


# ------------------------------------------------------------------------------------------------- events
def _lc_at(lons, lats):
    out = np.zeros(len(lons), np.uint8)
    with rasterio.open(terrain.LULC_TIF) as d:
        order = np.argsort(lats)
        vals = np.array([v[0] for v in d.sample(list(zip(np.asarray(lons)[order], np.asarray(lats)[order])))], np.uint8)
    out[order] = vals
    return out


def load_detections(full=False):
    """VIIRS S-NPP detections: 2024-2026 from the Area API download, plus (full=True) the yearly NASA archives 2012-2023."""
    df = pd.read_parquet(FIRMS)
    if full:
        import glob
        parts = [pd.read_csv(f, dtype={"version": str}) for f in sorted(glob.glob(os.path.join(YEARLY, "viirs-snpp_*_Pakistan.csv")))]
        parts = [p for p in parts if p.acq_date.min() < "2024-01-01"]
        old = pd.concat(parts, ignore_index=True)
        old = old[old.acq_date < "2024-01-01"]
        df = pd.concat([old, df], ignore_index=True)
    return df.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"])


def build_events(min_det=12, min_pixels=8, min_hours=18.0, min_km=1.5, max_km=40.0, min_burnable=0.6, max_built=0.25, force=False, full=False):
    p = os.path.join(VAL, "events_v2.json" if full else "events.json")
    if os.path.exists(p) and not force:
        return json.load(open(p))
    os.makedirs(VAL, exist_ok=True)
    df = load_detections(full)
    if "type" in df:                                                     # SP archive marks static/other sources; keep vegetation fire
        df = df[df["type"].isna() | (df["type"] == 0)]
    df = df.reset_index(drop=True)
    hh = (df.acq_time // 100) + (df.acq_time % 100) / 60.0
    df["t_h"] = (pd.to_datetime(df.acq_date) - pd.Timestamp("2000-01-01")).dt.days * 24.0 + hh
    kx = 111.32 * np.cos(np.radians(df.latitude.values))
    X = np.c_[df.longitude.values * kx / 1.5, df.latitude.values * 110.57 / 1.5, df.t_h.values / 48.0]
    pairs = cKDTree(X).query_pairs(1.0, p=np.inf, output_type="ndarray")
    n = len(df)
    _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)), directed=False)
    df["cl"] = lab
    sz = df.groupby("cl").size()
    cand = df[df.cl.isin(sz[sz >= min_det].index)].copy()
    cand["lc"] = _lc_at(cand.longitude.values, cand.latitude.values)
    events = []
    for cl, g in cand.groupby("cl"):
        px = g.groupby([(g.latitude / 0.0035).round(), (g.longitude / 0.0035).round()]).ngroups
        dur = g.t_h.max() - g.t_h.min()
        ext = max((g.longitude.max() - g.longitude.min()) * 111.32 * math.cos(math.radians(g.latitude.mean())), (g.latitude.max() - g.latitude.min()) * 110.57)
        burn = float(g.lc.isin(BURNABLE_LC).mean()); built = float((g.lc == 7).mean())
        if px < min_pixels or dur < min_hours or not (min_km <= ext <= max_km) or burn < min_burnable or built > max_built:
            continue
        t0 = g.t_h.min()
        g = g[g.t_h <= t0 + MAX_HOURS]
        first = g[g.t_h <= t0 + 3.0].drop_duplicates(["latitude", "longitude"]).head(25)
        mode_lc = int(g.lc.mode().iloc[0])
        events.append(dict(
            id=(f"F{int(cl):06d}" if full else f"E{int(cl):06d}"), t0_h=float(t0), t0_utc=(pd.Timestamp("2000-01-01") + pd.Timedelta(hours=float(t0))).strftime("%Y-%m-%dT%H:%M"),
            n_det=int(len(g)), n_pix=int(px), hours=float(g.t_h.max() - t0), extent_km=round(ext, 2), burnable=round(burn, 2), built=round(built, 2),
            lc_mode=mode_lc, frp_sum=round(float(g.frp.sum()), 1), lat=float(first.latitude.mean()), lon=float(first.longitude.mean()),
            split="test" if (pd.Timestamp("2000-01-01") + pd.Timedelta(hours=float(t0))) >= pd.Timestamp("2024-01-01" if full else SPLIT_DATE) else "dev",
            ignitions=[[round(float(r.longitude), 5), round(float(r.latitude), 5)] for r in first.itertuples()],
            dets=[[round(float(r.longitude), 5), round(float(r.latitude), 5), round(float(r.t_h - t0), 2), float(r.scan), float(r.track), float(r.frp)] for r in g.itertuples()]))
    events.sort(key=lambda e: e["t0_h"])
    json.dump(events, open(p, "w"))
    return events


# ------------------------------------------------------------------------------------------------- preparation (scene + weather, cached)
def prepare(ev, force=False):
    d = os.path.join(VAL, "cache")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, ev["id"] + ".pkl")
    if os.path.exists(p) and not force:
        return pickle.load(open(p, "rb"))
    half = float(np.clip(ev["extent_km"] * 0.8 + 6.0, 8.0, 40.0))
    start = dt.datetime.fromisoformat(ev["t0_utc"]).replace(minute=0)
    hours = int(min(math.ceil(ev["hours"]) + 2, MAX_HOURS + 2))
    scene = terrain.build_scene(ev["lat"], ev["lon"], half, CELL_M, start.date())
    w = runmod._weather(ev["lat"], ev["lon"], start, hours, None)
    adjust.prepare_scene_extras(scene)                                      # road barrier raster (one database query)
    prep = dict(scene=scene, w=w, start=start, hours=hours, half=half)
    pickle.dump(prep, open(p, "wb"))
    return prep


# ------------------------------------------------------------------------------------------------- simulation with calibration hooks
DEFAULT_CALIB = dict(adjust.DEFAULTS)


def _setup(ev, prep, calib=None, assimilate_h=None):
    sc = prep["scene"]
    sc.setdefault("_wf_cache", os.path.join(VAL, "wind", ev["id"]))
    fuel = sc["fuel"]
    hours = min(int(math.ceil(ev["hours"])) + 1, MAX_HOURS)
    adj = adjust.apply(sc, prep["w"], calib, hours, runmod.LEAD_IN_H, start=prep["start"])
    w = adj["w"]
    burnable = fuel != 0
    ndvi = float(sc["ndvi"][burnable].mean()) if burnable.any() else 0.25
    ny, nx = sc["ny"], sc["nx"]
    to_utm = Transformer.from_crs(4326, sc["crs"], always_xy=True)
    ign = []
    for lon, lat in ev["ignitions"]:
        x, y = to_utm.transform(lon, lat)
        col, row = ~sc["transform"] * (x, y)
        r, cc = int(row), int(col)
        if 0 <= r < ny and 0 <= cc < nx:
            s = runmod._snap(sc, r, cc, int(600 / CELL_M) + 2)
            if s is not None:
                ign.append((s[0], s[1], 0.0))
    if assimilate_h is not None:                                           # seed every detection seen up to assimilate_h, at the time it was seen
        seen = {(a, b) for a, b, _ in ign}
        for lon, lat, t, *_ in ev["dets"]:
            if t > assimilate_h:
                continue
            x, y = to_utm.transform(lon, lat)
            col, row = ~sc["transform"] * (x, y)
            r, cc = int(row), int(col)
            if 0 <= r < ny and 0 <= cc < nx:
                s2 = runmod._snap(sc, r, cc, int(300 / CELL_M) + 1)
                if s2 is not None and s2 not in seen:
                    seen.add(s2); ign.append((s2[0], s2[1], max(t, 0.0) * 60.0))
    grid = dict(ny=ny, nx=nx, dx=CELL_M, slope=sc["slope"], upaz=sc["upaz"], cont=adj["cont"], wmod=adj["wmod"], wrot=adj["wrot"])
    if "crown_ci0" in adj:
        grid.update(crown_ci0=adj["crown_ci0"], crown_cbd=adj["crown_cbd"])
    return dict(sc=sc, fuel=fuel, hours=hours, adj=adj, w=w, ndvi=ndvi, ign=ign, grid=grid)


def simulate(ev, prep, calib=None, spotting=True, seed=1, backend="cpu", assimilate_h=None):
    S = _setup(ev, prep, calib, assimilate_h)
    if not S["ign"]:
        return None
    adj, w, hours = S["adj"], S["w"], S["hours"]
    tables, _ = runmod._tables(w, S["ndvi"], dict(moist=adj["moist"]) if adj["moist"] != 1.0 else None)
    arr, ros, fli, spots, info = solver.simulate(S["grid"], S["fuel"], tables, w["wind"], w["wdir"], S["ign"], hours, spot_rate=2e-4 if spotting else 0.0,
                                                 seed=seed, hour0=runmod.LEAD_IN_H, backend=backend, decay_min=adj["decay_min"])
    arr = np.where(np.isfinite(arr) & (arr <= hours * 60), arr, np.inf)
    return dict(arr=arr, ign=S["ign"], hours=hours, fli=fli, ros=ros, n_spots=int(len(spots)))


def simulate_ensemble(ev, prep, calib=None, members=24, widths=None, seed=1, backend="auto"):
    """Ensemble of burned masks (members, ny, nx) with the perturbation widths of run.ENS_WIDTHS scaled by `widths`; None if no ignition."""
    S = _setup(ev, prep, calib)
    if not S["ign"]:
        return None
    adj, w, hours = S["adj"], S["w"], S["hours"]
    tbs, wss, wds = runmod.ensemble_inputs(w, S["ndvi"], adj["moist"], members, seed + 7, widths)
    arrs, _ = solver.simulate_batch(S["grid"], S["fuel"], tbs, wss, wds, S["ign"], hours, spot_rate=2e-4, seed=seed + 100, hour0=runmod.LEAD_IN_H,
                                    backend=backend, decay_min=adj["decay_min"])
    return np.isfinite(arrs) & (arrs <= hours * 60)


# ------------------------------------------------------------------------------------------------- scoring
def _obs_mask(ev, sc):
    to_utm = Transformer.from_crs(4326, sc["crs"], always_xy=True)
    shapes = []
    for lon, lat, t, scan, track, frp in ev["dets"]:
        x, y = to_utm.transform(lon, lat)
        hx, hy = scan * 500.0, track * 500.0                                # half the pixel size, metres
        shapes.append((box(x - hx, y - hy, x + hx, y + hy), 1))
    return rasterize(shapes, out_shape=(sc["ny"], sc["nx"]), transform=sc["transform"], fill=0, dtype=np.uint8).astype(bool)


def _circle(sc, centre_rc, area_cells):
    r = math.sqrt(max(area_cells, 1) / math.pi)
    yy, xx = np.ogrid[:sc["ny"], :sc["nx"]]
    return (yy - centre_rc[0]) ** 2 + (xx - centre_rc[1]) ** 2 <= r * r


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def score(ev, prep, sim):
    """Metrics for one simulated event. All areas in hectares; time errors in hours."""
    sc = prep["scene"]
    if sim is None:
        return dict(id=ev["id"], ok=False, why="ignition not on burnable fuel")
    ha = CELL_M ** 2 / 1e4
    burnt = np.isfinite(sim["arr"])
    obs = _obs_mask(ev, sc)
    obs_buf = ndimage.binary_dilation(obs, iterations=max(int(round(375 / CELL_M)), 1))      # one pixel of tolerance for the sensor footprint
    ign_rc = np.mean([[r, c] for r, c, _ in sim["ign"]], axis=0)
    sim_cells = int(burnt.sum())
    circ_same = _circle(sc, ign_rc, sim_cells)                                                  # same area as the simulation, centred on the ignition
    circ_oracle = _circle(sc, ign_rc, int(obs.sum()))                                           # the observed area, centred on the ignition
    # timing at the detected pixels the simulation reached
    to_utm = Transformer.from_crs(4326, sc["crs"], always_xy=True)
    errs, hit = [], 0
    for lon, lat, t, *_ in ev["dets"]:
        x, y = to_utm.transform(lon, lat)
        col, row = ~sc["transform"] * (x, y)
        r, c = int(row), int(col)
        if not (0 <= r < sc["ny"] and 0 <= c < sc["nx"]):
            continue
        win = sim["arr"][max(r - 7, 0):r + 8, max(c - 7, 0):c + 8]           # +-420 m
        if np.isfinite(win).any():
            hit += 1
            if t > 3.0:                                                  # detections of the first overpass are the ignition seeds: zero error by construction
                errs.append((float(np.nanmin(np.where(np.isfinite(win), win, np.nan))) / 60.0) - t)
    nd = max(len(ev["dets"]), 1)
    rec = lambda m: float((m & obs).sum() / max(obs.sum(), 1))
    out = dict(id=ev["id"], ok=True, split=ev["split"], lc_mode=ev["lc_mode"], hours=ev["hours"], n_det=ev["n_det"], extent_km=ev["extent_km"],
               obs_ha=round(float(obs.sum() * ha), 1), sim_ha=round(float(sim_cells * ha), 1),
               area_ratio=round(float(sim_cells / max(obs.sum(), 1)), 3),
               iou=round(_iou(burnt, obs_buf), 4), iou_circle=round(_iou(circ_same, obs_buf), 4), iou_circle_oracle=round(_iou(circ_oracle, obs_buf), 4),
               det_capture=round(hit / nd, 3), det_capture_circle=0.0,
               timing_median_h=round(float(np.median(errs)), 2) if errs else None, timing_mae_h=round(float(np.mean(np.abs(errs))), 2) if errs else None)
    # detection capture of the baseline circle, with the same tolerance
    cap = 0
    for lon, lat, *_ in ev["dets"]:
        x, y = to_utm.transform(lon, lat)
        col, row = ~sc["transform"] * (x, y)
        r, c = int(row), int(col)
        if 0 <= r < sc["ny"] and 0 <= c < sc["nx"] and circ_same[max(r - 7, 0):r + 8, max(c - 7, 0):c + 8].any():
            cap += 1
    out["det_capture_circle"] = round(cap / nd, 3)
    return score_mcd(ev, prep, sim, out)


def mcd_truth(ev, prep):
    """MODIS burn mask near the event: pixels burned within the event window (from the day before the first detection to two days after the
    last) and within 1.5 km of a detection, so unrelated fires in the same scene are ignored. Cached on the prepared event."""
    if "_mcd" in prep:
        return prep["_mcd"]
    sc = prep["scene"]
    days = burned.burned_days_on(sc)
    d0 = (dt.datetime.fromisoformat(ev["t0_utc"]) - dt.datetime(2000, 1, 1)).total_seconds() / 86400.0
    lo, hi = int(math.floor(d0)) - 1, int(math.floor(d0 + ev["hours"] / 24.0)) + 2
    win = (days >= lo) & (days <= hi) & (days > 0)
    near = ndimage.binary_dilation(_obs_mask(ev, sc), iterations=int(round(1500 / CELL_M)))
    mask = win & near
    prep["_mcd"] = (mask, days)
    return prep["_mcd"]


def score_mcd(ev, prep, sim, out):
    """Adds MODIS-based metrics to the score dict: extent, area ratio and burn-day timing against MCD64A1."""
    sc = prep["scene"]
    mask, days = mcd_truth(ev, prep)
    ha = CELL_M ** 2 / 1e4
    out["mcd_ha"] = round(float(mask.sum() * ha), 1)
    out["mcd_mapped"] = bool(mask.sum() * ha >= 50.0)
    if not out["mcd_mapped"] or sim is None:
        return out
    burnt = np.isfinite(sim["arr"])
    tol = ndimage.binary_dilation(mask, iterations=int(round(500 / CELL_M)))               # one MODIS pixel of tolerance
    ign_rc = np.mean([[r, c] for r, c, _ in sim["ign"]], axis=0)
    circ_same = _circle(sc, ign_rc, int(burnt.sum()))
    circ_or = _circle(sc, ign_rc, int(mask.sum()))
    out.update(mcd_iou=round(_iou(burnt, tol), 4), mcd_iou_circle=round(_iou(circ_same, tol), 4), mcd_iou_circle_oracle=round(_iou(circ_or, tol), 4),
               mcd_area_ratio=round(float(burnt.sum() / max(mask.sum(), 1)), 3),
               mcd_recall=round(float((burnt & tol & mask).sum() / max(mask.sum(), 1)), 3),
               mcd_precision=round(float((burnt & tol).sum() / max(burnt.sum(), 1)), 3))
    d0 = (dt.datetime.fromisoformat(ev["t0_utc"]) - dt.datetime(2000, 1, 1)).total_seconds() / 86400.0
    both = burnt & mask
    if both.sum() >= 5:
        sim_day = d0 + sim["arr"][both] / 1440.0
        out["mcd_timing_median_d"] = round(float(np.median(sim_day - days[both])), 2)
    return out


def summarize(rows, by="split"):
    ok = [r for r in rows if r.get("ok")]
    df = pd.DataFrame(ok)
    if df.empty:
        return {}
    res = {}
    for key, g in [("all", df)] + [(k, g) for k, g in df.groupby(by)]:
        res[key] = dict(n=int(len(g)), iou_median=round(float(g.iou.median()), 3), iou_mean=round(float(g.iou.mean()), 3),
                        iou_circle_median=round(float(g.iou_circle.median()), 3), iou_circle_oracle_median=round(float(g.iou_circle_oracle.median()), 3),
                        frac_beats_circle=round(float((g.iou > g.iou_circle).mean()), 3), area_ratio_median=round(float(g.area_ratio.median()), 2),
                        capture_median=round(float(g.det_capture.median()), 3), capture_circle_median=round(float(g.det_capture_circle.median()), 3),
                        timing_bias_h=round(float(g.timing_median_h.median()), 2) if g.timing_median_h.notna().any() else None)
    return res


def summarize_mcd(rows, by="lc_mode"):
    df = pd.DataFrame([r for r in rows if r.get("ok") and r.get("mcd_mapped")])
    if df.empty:
        return {}
    names = {5: "crops", 11: "rangeland", 2: "trees", 8: "bare"}
    res = {}
    for key, g in [("all mapped", df)] + [(names.get(k, k), g) for k, g in df.groupby(by)]:
        res[key] = dict(n=int(len(g)), mcd_ha_median=round(float(g.mcd_ha.median()), 0), iou=round(float(g.mcd_iou.median()), 3), circle=round(float(g.mcd_iou_circle.median()), 3),
                        oracle_circle=round(float(g.mcd_iou_circle_oracle.median()), 3), beats_circle=round(float((g.mcd_iou > g.mcd_iou_circle).mean()), 2),
                        area_ratio=round(float(g.mcd_area_ratio.median()), 2), recall=round(float(g.mcd_recall.median()), 2), precision=round(float(g.mcd_precision.median()), 2),
                        timing_d=round(float(g.mcd_timing_median_d.median()), 2) if g.get("mcd_timing_median_d") is not None and g.mcd_timing_median_d.notna().any() else None)
    return res
