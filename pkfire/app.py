"""
PAKFIRE: Pakistan wildfire spread platform, a sister of SAILAB (flood) and LANDLAB (landslide).

    ~/flood/venv/bin/python -m pkfire.app        ->  http://<host>:8020/

Same layout as SAILAB: a 2D map over an animated Mapbox 3D pane, right-hand panels (area, discovery, context,
monitoring) and tier tabs (instant footprint, spread run, ensemble, mitigation). Compute order: GPU, RAM, CPU.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import tempfile
import threading
import time
import traceback
import uuid
import zipfile

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file

from . import exposure, geo, instant, kernel_gpu, overlay, report
from . import run as runmod
from . import weather as wx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(HERE, "web", "pk")
PORT = int(os.environ.get("FIRE_PORT", "8020"))
MODEL_NAME = "PAKFIRE · Wildfire Modelling"

app = Flask(__name__, static_folder=WEB, static_url_path="/static")
JOBS: dict = {}
RUN_LOCK = threading.Lock()                  # one simulation at a time keeps a shared server calm
STAGE_FRAC = dict(queued=0.02, terrain=0.12, weather=0.3, spread=0.5, ensemble=0.72, export=0.9, done=1.0)


def jerr(msg, code=400, **kw):
    return jsonify(error=msg, **kw), code


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    """Landing page (forest fire scene); its button opens the application at /app."""
    resp = Response(open(os.path.join(WEB, "landing.html"), encoding="utf-8").read(), mimetype="text/html")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/instant")
def instant_page():
    """PAKFIRE Instant: place a fire, pick a weather scenario, see the footprint in about a second (the full model stays at /app)."""
    resp = Response(open(os.path.join(WEB, "instant.html"), encoding="utf-8").read().replace("__MAPBOX_TOKEN__", geo.MAPBOX_TOKEN), mimetype="text/html")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/app")
def main_app():
    return open(os.path.join(WEB, "app.html"), encoding="utf-8").read().replace("__MAPBOX_TOKEN__", geo.MAPBOX_TOKEN)


@app.route("/runs/<rid>/mapbox3d")
def mapbox3d(rid):
    _rd(rid)
    page = open(os.path.join(WEB, "viewer", "fire3d.html"), encoding="utf-8").read()
    return page.replace("__MAPBOX_TOKEN__", geo.MAPBOX_TOKEN).replace("__RUN_ID__", rid)


@app.route("/api/hw")
def api_hw():
    txt = "CPU"
    try:
        if kernel_gpu.available():
            import cupy as cp
            name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            txt = f"GPU first · {name} · CUDA solver, RAM, CPU fallback"
    except Exception:
        pass
    return jsonify(text=txt, gpu=kernel_gpu.available(), cores=os.cpu_count())


@app.route("/api/status")
def api_status():
    return jsonify(gpu=kernel_gpu.available(), order="GPU, then RAM, then CPU", firms=bool(geo.FIRMS_KEY), running=RUN_LOCK.locked())


@app.route("/api/queue")
def api_queue():
    """Jobs that are waiting for or holding the solver (header queue chip)."""
    now = time.time()
    jobs = [dict(id=k, state=j["state"], stage=j.get("stage", ""), age_s=int(now - j["t0"])) for k, j in list(JOBS.items()) if j["state"] in ("queued", "running")]
    return jsonify(jobs=sorted(jobs, key=lambda x: -x["age_s"]), solver_busy=RUN_LOCK.locked())


@app.route("/places/search")
def places_search():
    """District names matching a typed text (header search box); no satellite statistics, so it answers at once."""
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify(results=[])
    rows = [dict(idx=int(r.idx), name=r.name, province=r.province) for r in geo.districts().itertuples() if q in r.name.lower() or q in r.province.lower()]
    rows.sort(key=lambda x: (not x["name"].lower().startswith(q), x["name"]))
    return jsonify(results=rows[:6])


@app.route("/api/layers")
def api_layers():
    return jsonify(ramps=overlay.RAMPS)


# ---------------------------------------------------------------- area
def _geom_info(g):
    from shapely.geometry import shape
    from pyproj import Geod
    s = shape(g)
    area = abs(Geod(ellps="WGS84").geometry_area_perimeter(s)[0]) / 1e6 if s.geom_type in ("Polygon", "MultiPolygon") else 0.0
    c = s.centroid
    minx, miny, maxx, maxy = s.bounds
    import math
    wkm = (maxx - minx) * 111.32 * math.cos(math.radians(c.y))
    hkm = (maxy - miny) * 110.57
    half = max(wkm, hkm) / 2
    return s, area, c, half


@app.route("/area/polygon", methods=["POST"])
def area_polygon():
    g = (request.get_json(force=True) or {}).get("geometry")
    if not g:
        return jerr("no geometry")
    try:
        s, area, c, half = _geom_info(g)
    except Exception as e:
        return jerr(f"invalid geometry: {e}")
    if not (60.8 <= c.x <= 77.9 and 23.5 <= c.y <= 37.2):
        return jerr("The area is outside Pakistan's data coverage.")
    d = geo.district_at(c.x, c.y)
    return jsonify(area_km2=round(area, 1), centre=[c.x, c.y], district=d, half_km=round(min(max(half * 1.05, 5), 100), 1),
                   sim_possible=half <= 100,
                   note=None if half <= 100 else "The area is larger than the 200 km window; the run will use a 200 km window around its centre.")


@app.route("/area/upload", methods=["POST"])
def area_upload():
    f = request.files.get("file")
    if not f:
        return jerr("no file")
    import geopandas as gpd
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, f.filename.replace("/", "_"))
        f.save(p)
        try:
            g = gpd.read_file(f"zip://{p}") if p.lower().endswith(".zip") else gpd.read_file(p)
        except Exception as e:
            return jerr(f"could not read the file: {e}")
    g = g.to_crs(4326) if g.crs else g.set_crs(4326)
    pts = [[round(x.x, 5), round(x.y, 5)] for x in g.geometry if x.geom_type == "Point"]
    polys = [x for x in g.geometry if x.geom_type in ("Polygon", "MultiPolygon")]
    out = dict(points=pts)
    if polys:
        from shapely.ops import unary_union
        out["geometry"] = unary_union(polys).convex_hull.__geo_interface__ if len(polys) > 1 else polys[0].__geo_interface__
    if not pts and not polys:
        return jerr("The file has no point or polygon features.")
    return jsonify(out)


# ---------------------------------------------------------------- discovery
@app.route("/districts/search")
def districts_search():
    metric = request.args.get("metric", "detections")
    q = request.args.get("q", "").lower()
    n = int(request.args.get("n", 15))
    days = int(request.args.get("days", 3))
    try:
        st = geo.district_stats(days)
    except Exception as e:
        return jerr(f"FIRMS unavailable: {e}", 502)
    g = geo.districts()
    rows = []
    for r in g.itertuples():
        if q and q not in r.name.lower() and q not in r.province.lower():
            continue
        s = st.get(int(r.idx), dict(n=0, frp=0.0))
        rows.append(dict(idx=int(r.idx), name=r.name, province=r.province, detections=s["n"], frp=round(s["frp"], 1)))
    rows.sort(key=lambda x: -(x["frp"] if metric == "frp" else x["detections"]))
    for x in rows:
        x["metric"] = x["frp"] if metric == "frp" else x["detections"]
    return jsonify(results=rows[:n], days=days, metric=metric)


@app.route("/districts/<int:i>")
def district_geom(i):
    g = geo.districts()
    if not (0 <= i < len(g)):
        abort(404)
    r = g.iloc[i]
    return jsonify(name=r["name"], province=r["province"], geometry=r.geometry.__geo_interface__)


@app.route("/api/districts")
def api_districts():
    try:
        return jsonify(geo.districts_geojson(int(request.args.get("days", 3))))
    except Exception as e:
        return jerr(f"FIRMS unavailable: {e}", 502)


@app.route("/fire-hotspots")
def fire_hotspots():
    days, n = int(request.args.get("days", 2)), int(request.args.get("n", 12))
    try:
        pts = geo.firms(days)
    except Exception as e:
        return jerr(f"FIRMS unavailable: {e}", 502)
    key = request.args.get("metric", "frp")
    pts = sorted(pts, key=lambda p: -p["frp"] if key == "frp" else p["when"], reverse=False)
    if key != "frp":
        pts = pts[::-1]
    return jsonify(results=pts[:n], total=len(pts), days=days)


@app.route("/api/firms")
def api_firms():
    days = int(request.args.get("days", 2))
    try:
        pts = geo.firms(days)
    except Exception as e:
        return jerr(f"FIRMS unavailable: {e}", 502, features=[])
    return jsonify(type="FeatureCollection", days=days,
                   features=[dict(type="Feature", geometry=dict(type="Point", coordinates=[p["lon"], p["lat"]]),
                                  properties=dict(frp=p["frp"], conf=p["conf"], when=p["when"], district=p["district"])) for p in pts])


@app.route("/api/weather")
def api_weather():
    try:
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
        w = wx.hourly(lat, lon, now, 6)
        return jsonify(temp=round(float(w["temp"][0]), 1), rh=round(float(w["rh"][0])), wind=round(float(w["wind"][0]), 1),
                       wdir=round(float(w["wdir"][0])), source=w["source"])
    except Exception as e:
        return jerr(f"{type(e).__name__}: {e}", 502)


def danger_class(w):
    """Simple fire-weather class from the driest 1 h fuel moisture, peak wind and recent rain over the next 72 h.
    Not a standard index (no KBDI/FWI): thresholds follow Rothermel's grass moisture of extinction (12 %) and typical
    wind effects, and are judgement, not calibration."""
    m, wind, rain = w["dead1h_min_72h_pct"], w["wind_peak_72h"], w["rain_prior_48h_mm"]
    score = (3 if m < 5 else 2 if m < 8 else 1 if m < 12 else 0) + (2 if wind >= 10 else 1 if wind >= 5 else 0) - (2 if rain >= 10 else 1 if rain >= 2 else 0)
    cls = ["low", "moderate", "high", "very high", "extreme"][max(0, min(score - 1, 4))] if score > 0 else "low"
    return dict(cls=cls, score=score, basis=f"driest 1 h fuel moisture {m} %, peak wind {wind} m/s, rain in the last 48 h {rain} mm")


@app.route("/area/context", methods=["POST"])
def area_context():
    """Weather, fuel moisture and fuel mix around a point (the 'context' panel)."""
    b = request.get_json(force=True)
    lat, lon = float(b["lat"]), float(b["lon"])
    half = min(max(float(b.get("half_km", 25)), 5), 60)
    out = {}
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    try:
        w = runmod._weather(lat, lon, now, 72, None)
        L = runmod.LEAD_IN_H
        tb, md = runmod._tables(w, 0.3)
        f = slice(L, L + 72)
        out["weather"] = dict(source=w["source"], temp_now=round(float(w["temp"][L]), 1), rh_now=round(float(w["rh"][L])),
                              wind_now=round(float(w["wind"][L]), 1), wdir_now=round(float(w["wdir"][L])),
                              wind_peak_72h=round(float(np.max(w["wind"][f])), 1), rh_min_72h=round(float(np.min(w["rh"][f]))),
                              temp_max_72h=round(float(np.max(w["temp"][f])), 1), rain_72h_mm=round(float(np.sum(w["precip"][f])), 1),
                              rain_prior_48h_mm=round(float(np.sum(w["precip"][:L])), 1), dead1h_now_pct=round(float(md[0, L]) * 100, 1),
                              dead1h_min_72h_pct=round(float(np.min(md[0, f])) * 100, 1), conditioning=w.get("conditioning"))
        out["danger"] = danger_class(out["weather"])
    except Exception as e:
        out["weather_error"] = str(e)
    try:
        sc = instant._scene(lat, lon, half, 100.0, now.date())
        lc, fuel = sc["lc"], sc["fuel"]
        tot = lc.size
        names = runmod.LC_NAMES
        out["landcover_pct"] = {names.get(int(k), str(int(k))): round(float((lc == k).sum()) / tot * 100, 1) for k in np.unique(lc)}
        out["burnable_pct"] = round(float((fuel != 0).sum()) / tot * 100, 1)
        out["ndvi_mean"] = round(float(sc["ndvi"][fuel != 0].mean()), 3) if (fuel != 0).any() else None
        out["elevation_m"] = [round(float(np.percentile(sc["dem"], 5))), round(float(np.percentile(sc["dem"], 95)))]
        out["ndvi_file"] = sc["ndvi_file"]
    except Exception as e:
        out["scene_error"] = str(e)
    d = geo.district_at(lon, lat)
    out["district"] = d
    return jsonify(out)


@app.route("/area/weather-series", methods=["POST"])
def area_weather_series():
    """Hourly weather the run would use (start..start+hours) with the dead-fuel moisture it implies."""
    b = request.get_json(force=True)
    try:
        start = dt.datetime.fromisoformat(b["start"]) if b.get("start") else dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
        hours = min(max(int(b.get("hours", 24)), 1), runmod.MAX_RUN_H)
        manual = b.get("wind")
        if manual:
            manual = dict(speed_ms=float(manual["speed_ms"]), from_deg=float(manual["from_deg"]), temp_c=float(manual.get("temp_c", 32)), rh=float(manual.get("rh", 25)))
        w = runmod._weather(float(b["lat"]), float(b["lon"]), start, hours, manual)
        L = runmod.LEAD_IN_H
        _, md = runmod._tables(w, 0.3)
        sl = slice(L, L + hours + 1)
        r = lambda a, d=1: [round(float(x), d) for x in a[sl]]
        return jsonify(source=w["source"], conditioning=w.get("conditioning"), wind=r(w["wind"]), wdir=r(w["wdir"], 0), temp=r(w["temp"]),
                       rh=r(w["rh"], 0), rain=r(w["precip"], 2), dead1h=[round(float(x) * 100, 1) for x in md[0, sl]])
    except runmod.RunError as e:
        return jerr(str(e))
    except Exception as e:
        return jerr(f"{type(e).__name__}: {e}", 502)


@app.route("/area/instant-footprint", methods=["POST"])
def area_instant():
    b = request.get_json(force=True)
    wind = b.get("wind")
    if wind:
        wind = dict(speed_ms=float(wind["speed_ms"]), from_deg=float(wind["from_deg"]), temp_c=float(wind.get("temp_c", 32)), rh=float(wind.get("rh", 25)))
    try:
        return jsonify(instant.footprint(float(b["lat"]), float(b["lon"]), hours=int(b.get("hours", 6)), wind=wind))
    except runmod.RunError as e:
        return jerr(str(e))


# ---------------------------------------------------------------- jobs
def _params_from(b, geometry):
    p = b.get("params", {})
    ign = b.get("ignition")
    if not ign and geometry:
        c = _geom_info(geometry)[2]
        ign = [c.x, c.y]
    if not ign:
        raise runmod.RunError("Place an ignition point first (click the map or use the point tool).")
    half = float(p.get("half_km", 40))
    if geometry:
        half = min(max(_geom_info(geometry)[3] * 1.05, 5), 100)
    kw = dict(lat=float(ign[1]), lon=float(ign[0]), hours=int(p.get("hours", 24)), half_km=min(max(half, 5), 100),
              cell_m=min(max(float(p.get("cell_m", 50 if half <= 50 else (75 if half <= 70 else 100))), 20), 250),
              spotting=bool(p.get("spotting", True)), seed=int(p.get("seed", 1)),
              ensemble=min(max(int(p.get("ensemble", 0)), 0), 60),
              backend=p.get("backend", "auto") if p.get("backend", "auto") in ("auto", "gpu", "cpu") else "auto",
              extra_ignitions=[tuple(reversed(x)) for x in b.get("extra_ignitions", [])] or None,
              calibration=p.get("calibration", "default") if p.get("calibration", "default") in ("default", "none") else "default")
    if p.get("start"):
        kw["start"] = p["start"]
    if p.get("assimilate"):                                    # satellite detections near the ignition become the starting state
        pts = geo.firms(3)
        import math as _m
        near = []
        for q in pts:
            d = _m.hypot((q["lon"] - kw["lon"]) * 111.32 * _m.cos(_m.radians(kw["lat"])), (q["lat"] - kw["lat"]) * 110.57)
            if d <= float(p.get("assim_radius_km", 6)):
                hh, mm = int(q["when"][11:13]), int(q["when"][13:15])
                near.append((q["lon"], q["lat"], dt.datetime.fromisoformat(q["when"][:10]) + dt.timedelta(hours=hh, minutes=mm)))
        if not near:
            raise runmod.RunError("No satellite detections within the radius in the last 3 days to assimilate.")
        t0 = min(x[2] for x in near).replace(minute=0)
        kw["start"] = t0.isoformat()
        kw["detections"] = [(x[0], x[1], (x[2] - t0).total_seconds() / 60.0) for x in near]
    w = p.get("wind")
    if w:
        kw["wind"] = dict(speed_ms=float(w["speed_ms"]), from_deg=float(w["from_deg"]), temp_c=float(w.get("temp_c", 32)), rh=float(w.get("rh", 25)))
    return kw


def _summary(meta):
    s = meta["stats"]
    return dict(burned_ha=s["burned_ha"], max_head_ros_m_min=s["max_head_ros_m_min"], max_intensity_kw_m=s["max_intensity_kw_m"],
                spot_fires=s["spot_fires"], crown_fire_ha=s.get("crown_fire_ha", 0.0), builtup_within_1km_ha=s["builtup_within_1km_ha"], builtup_burned_ha=s["builtup_burned_ha"],
                backend=s["backend"], solve_s=s["solve_s"], compute_s=s["compute_s"], hours=meta["hours"], t_end_min=meta["t_end_min"],
                ensemble=meta.get("ensemble"), by_landcover_ha=s["by_landcover_ha"], by_fuel_ha=s["by_fuel_ha"], reached_edge=s["reached_edge"],
                axis_length_km=s["axis_length_km"], events=meta["events"], interventions=meta.get("interventions"))


def _result(meta):
    return dict(run_id=meta["id"], outcome="burned", summary=_summary(meta), warnings=meta.get("warnings", []),
                layers=overlay.layers(meta["id"]), has_scene=True)


def _start_job(fn):
    jid = uuid.uuid4().hex[:10]
    job = dict(state="queued", stage="queued", frac=STAGE_FRAC["queued"], t0=time.time(), elapsed_s=0, cancel=False)
    JOBS[jid] = job

    def progress(stage, msg=""):
        job.update(stage=msg or stage, frac=job.get("base", 0) + STAGE_FRAC.get(stage, job["frac"]) * job.get("span", 1.0), elapsed_s=time.time() - job["t0"])

    def worker():
        try:
            with RUN_LOCK:
                job.update(state="running", stage="Starting…")
                res = fn(progress, lambda: job["cancel"], job)
            job.update(state="done", frac=1.0, result=res, elapsed_s=time.time() - job["t0"])
        except runmod.RunError as e:
            job.update(state="cancelled" if str(e) == "cancelled" else "error", error=str(e), elapsed_s=time.time() - job["t0"])
        except Exception as e:
            traceback.print_exc()
            job.update(state="error", error=f"{type(e).__name__}: {e}", elapsed_s=time.time() - job["t0"])

    threading.Thread(target=worker, daemon=True).start()
    return jid


def _launch(b, ensemble=None):
    try:
        kw = _params_from(b, b.get("geometry"))
    except (runmod.RunError, KeyError, ValueError, TypeError) as e:
        return jerr(str(e))
    if ensemble is not None:
        kw["ensemble"] = ensemble
    iv = b.get("interventions") or {}

    def fn(progress, cancel, job):
        return _result(runmod.run(progress=progress, cancel=cancel, breaks=iv.get("breaks"), treatments=iv.get("treatments"), **kw))
    return jsonify(job=_start_job(fn))


@app.route("/run", methods=["POST"])
def run_route():
    return _launch(request.get_json(force=True))


@app.route("/area/ensemble", methods=["POST"])
def ensemble_route():
    b = request.get_json(force=True)
    n = int((b.get("params") or {}).get("ensemble", 24)) or 24
    return _launch(b, ensemble=min(max(n, 2), 60))


@app.route("/area/mitigation", methods=["POST"])
def mitigation_route():
    b = request.get_json(force=True)
    iv = b.get("interventions") or {}
    if not (iv.get("breaks") or iv.get("treatments")):
        return jerr("Draw a fuel break line or a treatment polygon first.")
    try:
        kw = _params_from(b, b.get("geometry"))
    except (runmod.RunError, KeyError, ValueError, TypeError) as e:
        return jerr(str(e))
    kw["ensemble"] = 0

    def fn(progress, cancel, job):
        job.update(base=0.0, span=0.5)
        base = runmod.run(progress=progress, cancel=cancel, label="baseline", **kw)
        job.update(base=0.5, span=0.5)
        mit = runmod.run(progress=progress, cancel=cancel, label="mitigated", breaks=iv.get("breaks"), treatments=iv.get("treatments"), **kw)
        sb, sm = base["stats"], mit["stats"]
        keys = [("burned_ha", "Burned area (ha)"), ("builtup_within_1km_ha", "Built-up ≤ 1 km (ha)"), ("max_head_ros_m_min", "Peak spread (m/min)"),
                ("spot_fires", "Spot fires")]
        red = {k: round(sb[k] - sm[k], 2) for k, _ in keys}
        saved = sb["burned_ha"] - sm["burned_ha"]
        return dict(baseline=dict(run_id=base["id"], **{k: sb[k] for k, _ in keys}), mitigated=dict(run_id=mit["id"], **{k: sm[k] for k, _ in keys}),
                    reduction=red, labels=dict(keys), interventions=mit.get("interventions"),
                    note=f"The interventions changed the burned area by {saved:+.0f} ha ({(saved / sb['burned_ha'] * 100 if sb['burned_ha'] else 0):+.0f}%) "
                         "under identical weather and random seed. Fuel breaks are applied as at least 3 cells wide and are crossed only by embers.")
    return jsonify(job=_start_job(fn))


@app.route("/runs/<jid>/progress.json")
def progress_json(jid):
    j = JOBS.get(jid)
    if not j:
        abort(404)
    j["elapsed_s"] = time.time() - j["t0"] if j["state"] in ("queued", "running") else j["elapsed_s"]
    return jsonify({k: v for k, v in j.items() if k not in ("cancel",)})


@app.route("/runs/<jid>/cancel", methods=["POST"])
def cancel_job(jid):
    if jid in JOBS:
        JOBS[jid]["cancel"] = True
    return jsonify(ok=True)


# ---------------------------------------------------------------- run files
def _rd(rid):
    if not rid.replace("-", "").isalnum():
        abort(404)
    d = os.path.join(runmod.RUNS, rid)
    if not os.path.isfile(os.path.join(d, "meta.json")):
        abort(404)
    return d


@app.route("/runs")
def runs_list():
    out = []
    if os.path.isdir(runmod.RUNS):
        for rid in sorted(os.listdir(runmod.RUNS), reverse=True)[:60]:
            p = os.path.join(runmod.RUNS, rid, "meta.json")
            if not os.path.exists(p):
                continue
            try:
                m = json.load(open(p))
                out.append(dict(run_id=rid, t=m["start"], hours=m["hours"], ignition=m["ignitions"][0], burned_ha=m["stats"]["burned_ha"],
                                label=m.get("label"), ensemble=bool(m.get("ensemble")), backend=m["stats"]["backend"], outcome="burned"))
            except Exception:
                continue
    return jsonify(out)


@app.route("/runs/<rid>/summary.json")
def run_summary(rid):
    d = _rd(rid)
    m = json.load(open(os.path.join(d, "meta.json")))
    return jsonify(dict(run_id=rid, outcome="burned", summary=_summary(m), warnings=m.get("warnings", []), layers=overlay.layers(rid), has_scene=True, meta=m))


@app.route("/api/runs/<rid>/meta.json")
def api_meta(rid):
    return send_file(os.path.join(_rd(rid), "meta.json"))


@app.route("/api/runs/<rid>/<name>.bin")
def api_bin(rid, name):
    if name not in ("arrival", "intensity", "prob"):
        abort(404)
    path = os.path.join(_rd(rid), name + ".bin")
    if not os.path.exists(path):
        abort(404)
    raw = open(path, "rb").read()
    if "gzip" in request.headers.get("Accept-Encoding", ""):
        return Response(gzip.compress(raw, 3), headers={"Content-Encoding": "gzip", "Content-Type": "application/octet-stream"})
    return Response(raw, mimetype="application/octet-stream")


@app.route("/runs/<rid>/overlay/<layer>.png")
def run_overlay(rid, layer):
    _rd(rid)
    try:
        png, coords, legend = overlay.render(rid, layer)
    except KeyError:
        abort(404)
    return Response(png, mimetype="image/png", headers={"X-Coordinates": json.dumps(coords), "X-Legend": json.dumps(legend),
                                                        "Access-Control-Expose-Headers": "X-Coordinates, X-Legend"})


@app.route("/runs/<rid>/exposure.json")
def run_exposure(rid):
    _rd(rid)
    try:
        return jsonify(exposure.cached(rid))
    except Exception as e:
        traceback.print_exc()
        return jerr(f"{type(e).__name__}: {e}", 500)


@app.route("/runs/<rid>/evacuation.json")
def run_evacuation(rid):
    _rd(rid)
    from . import evac
    try:
        return jsonify(evac.cached(rid, float(request.args.get("lead", 60))))
    except Exception as e:
        traceback.print_exc()
        return jerr(f"{type(e).__name__}: {e}", 500)


@app.route("/data-status")
def data_status():
    """Every dataset the model reads, whether it exists, and its size (like SAILAB's /data-status)."""
    from . import terrain as T
    rows = [("SRTM elevation", T.DEM_TIF, True), ("ESRI land cover 2025", T.LULC_TIF, True), ("MODIS NDVI folder", T.NDVI_DIR, True),
            ("District boundaries", geo.DISTRICTS, True), ("WorldPop 2020 1 km", exposure.POP, False)] + \
           [(f"Facilities: {k}", v, False) for k, v in exposure.LAYERS.items()] + \
           [("FIRMS key", os.path.join(HERE, ".secrets", "firms_map_key"), False), ("Mapbox token", os.path.join(HERE, ".secrets", "mapbox_token"), True)]
    out = []
    for n, p, req in rows:
        ok = os.path.exists(p)
        out.append(dict(name=n, path=p if "secrets" not in p else ".secrets/…", exists=ok, required=req,
                        size_mb=round(os.path.getsize(p) / 1e6, 1) if ok and os.path.isfile(p) else None))
    nd = len([f for f in os.listdir(T.NDVI_DIR) if f.endswith(".tif")]) if os.path.isdir(T.NDVI_DIR) else 0
    return jsonify(items=out, ndvi_composites=nd, gpu=kernel_gpu.available(), missing_required=[x["name"] for x in out if x["required"] and not x["exists"]])


@app.route("/runs/<rid>/perimeters.geojson")
def run_perimeters(rid):
    _rd(rid)
    step = float(request.args.get("step_min", 60))
    gj = overlay.isochrones(rid, min(max(step, 5), 360))
    return Response(json.dumps(gj), mimetype="application/geo+json", headers={"Content-Disposition": f"attachment; filename=pakfire_{rid}_perimeters.geojson"})


@app.route("/runs/<rid>", methods=["DELETE"])
def run_delete(rid):
    import shutil
    d = _rd(rid)
    if not os.path.realpath(d).startswith(os.path.realpath(runmod.RUNS) + os.sep):
        abort(404)
    shutil.rmtree(d)
    return jsonify(ok=True)


@app.route("/runs/<rid>/timelapse.mp4")
def run_timelapse(rid):
    from . import timelapse
    d = _rd(rid)
    sec = int(min(max(float(request.args.get("seconds", 24)), 6), 90))
    path = os.path.join(d, f"timelapse_{sec}s.mp4")
    if not os.path.exists(path):
        try:
            timelapse.render(rid, seconds=sec, out_path=path)
        except Exception as e:
            traceback.print_exc()
            return jerr(f"{type(e).__name__}: {e}", 500)
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.route("/runs/<rid>/report.pdf")
def run_report(rid):
    _rd(rid)
    return Response(report.run_pdf(rid), mimetype="application/pdf", headers={"Content-Disposition": f"inline; filename=pakfire_{rid}.pdf"})


@app.route("/batch-report.pdf")
def batch_report():
    ids = [i for i in request.args.get("ids", "").split(",") if i][:30]
    if not ids:
        return jerr("no run ids")
    return Response(report.batch_pdf(ids), mimetype="application/pdf", headers={"Content-Disposition": "inline; filename=pakfire_batch.pdf"})


@app.route("/compare-runs", methods=["POST"])
def compare_runs():
    b = request.get_json(force=True)
    try:
        ma, aa, _, _ = overlay.load(b["a"])
        mb, ab, _, _ = overlay.load(b["b"])
    except Exception as e:
        return jerr(f"cannot read the runs: {e}")
    ba = aa < ma["never"]
    bb = ab < mb["never"]
    # map A's pixels into B's grid by geographic position
    (Wa, Sa, Ea, Na), (Wb, Sb, Eb, Nb) = ma["bounds"], mb["bounds"]
    h, w = ba.shape
    lon = Wa + (np.arange(w) + 0.5) / w * (Ea - Wa)
    lat = Na - (np.arange(h) + 0.5) / h * (Na - Sa)
    hb, wb = bb.shape
    ci = np.floor((lon - Wb) / (Eb - Wb) * wb).astype(int)
    ri = np.floor((Nb - lat) / (Nb - Sb) * hb).astype(int)
    okc, okr = (ci >= 0) & (ci < wb), (ri >= 0) & (ri < hb)
    inb = np.zeros_like(ba)
    inb[np.ix_(okr, okc)] = bb[np.ix_(ri[okr], ci[okc])]
    ha = ma["cell_m"] ** 2 / 1e4
    both, only_a = int((ba & inb).sum()), int((ba & ~inb).sum())
    return jsonify(area_a_ha=round(float(ba.sum() * ha), 1), area_b_ha=round(float(bb.sum() * mb["cell_m"] ** 2 / 1e4), 1),
                   both_ha=round(both * ha, 1), only_a_ha=round(only_a * ha, 1), only_b_ha=round(max(float(bb.sum() * mb["cell_m"] ** 2 / 1e4) - both * ha, 0.0), 1),
                   spots_a=ma["stats"]["spot_fires"], spots_b=mb["stats"]["spot_fires"],
                   ros_a=ma["stats"]["max_head_ros_m_min"], ros_b=mb["stats"]["max_head_ros_m_min"],
                   note="Overlap is measured on the export grid of run A; runs from different windows overlap only where they cover the same ground.")


@app.route("/runs/<rid>/validate-observed", methods=["POST"])
def validate_observed(rid):
    _rd(rid)
    f = request.files.get("file")
    if not f:
        return jerr("no file")
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    meta, arr, _, _ = overlay.load(rid)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, f.filename.replace("/", "_"))
        f.save(p)
        try:
            g = gpd.read_file(f"zip://{p}") if p.lower().endswith(".zip") else gpd.read_file(p)
        except Exception as e:
            return jerr(f"could not read the perimeter: {e}")
    g = g.to_crs(4326) if g.crs else g.set_crs(4326)
    h, w = arr.shape
    W, S, E, N = meta["bounds"]
    obs = rasterize([(x, 1) for x in g.geometry if x is not None], out_shape=(h, w), transform=from_bounds(W, S, E, N, w, h), fill=0, dtype=np.uint8).astype(bool)
    sim = arr < meta["never"]
    inter, uni = int((obs & sim).sum()), int((obs | sim).sum())
    if not obs.any():
        return jerr("The observed perimeter does not overlap this run's window.")
    ha = meta["cell_m"] ** 2 / 1e4
    cy = lambda m: np.array(np.nonzero(m)).mean(1)
    co, cs = cy(obs), cy(sim) if sim.any() else cy(obs)
    off = float(np.hypot((co[0] - cs[0]) * (N - S) / h * 110.57, (co[1] - cs[1]) * (E - W) / w * 111.32 * np.cos(np.radians((N + S) / 2))))
    return jsonify(IoU=inter / uni if uni else 0, Dice=2 * inter / (obs.sum() + sim.sum()), omission=1 - inter / obs.sum(),
                   commission=(1 - inter / sim.sum()) if sim.any() else 1, observed_ha=round(obs.sum() * ha, 1), simulated_ha=round(sim.sum() * ha, 1),
                   area_error_pct=round((sim.sum() / obs.sum() - 1) * 100, 1), centroid_offset_km=round(off, 2),
                   note="Overlap of the final burned footprint with the supplied perimeter. Timing is not scored.")


# ---------------------------------------------------------------- monitoring
WATCH_FILE = os.path.join(HERE, "data", "monitor.json")
WATCHES: list = []


def _save_watches():
    os.makedirs(os.path.dirname(WATCH_FILE), exist_ok=True)
    json.dump(WATCHES, open(WATCH_FILE, "w"))


if os.path.exists(WATCH_FILE):
    try:
        WATCHES.extend(json.load(open(WATCH_FILE)))
    except Exception:
        pass


@app.route("/monitor/watch", methods=["POST"])
def monitor_watch():
    b = request.get_json(force=True)
    ctr = b.get("centre")
    if not ctr and b.get("geometry"):
        c = _geom_info(b["geometry"])[2]
        ctr = [c.x, c.y]
    if not ctr:
        return jerr("choose an area or ignition first")
    WATCHES.append(dict(id=uuid.uuid4().hex[:8], name=b.get("name") or "area", centre=ctr, radius_km=float(b.get("radius_km", 15)),
                        every_h=float(b.get("every_h", 3)), webhook=b.get("webhook", ""), seen=[], last=None, last_check=0, alerts=[]))
    _save_watches()
    return jsonify(ok=True)


@app.route("/monitor/list")
def monitor_list():
    return jsonify([{k: v for k, v in w.items() if k != "seen"} for w in WATCHES])


@app.route("/monitor/delete", methods=["POST"])
def monitor_delete():
    i = (request.get_json(force=True) or {}).get("id")
    WATCHES[:] = [w for w in WATCHES if w["id"] != i]
    _save_watches()
    return jsonify(ok=True)


def _watch_loop():
    import math
    import requests
    while True:
        time.sleep(60)
        for w in list(WATCHES):
            if time.time() - w["last_check"] < w["every_h"] * 3600:
                continue
            w["last_check"] = time.time()
            try:
                pts = geo.firms(1)
            except Exception as e:
                w["last"] = dict(error=str(e), t=dt.datetime.now(dt.timezone.utc).isoformat())
                continue
            near = []
            for p in pts:
                d = math.hypot((p["lon"] - w["centre"][0]) * 111.32 * math.cos(math.radians(w["centre"][1])), (p["lat"] - w["centre"][1]) * 110.57)
                key = f"{p['lat']:.3f},{p['lon']:.3f},{p['when']}"
                if d <= w["radius_km"] and key not in w["seen"]:
                    near.append((p, key))
            w["last"] = dict(t=dt.datetime.now(dt.timezone.utc).isoformat(), new_detections=len(near))
            if near:
                p, key = max(near, key=lambda x: x[0]["frp"])
                w["seen"] += [k for _, k in near]
                try:
                    with RUN_LOCK:
                        meta = runmod.run(p["lat"], p["lon"], hours=24, half_km=40)
                    alert = dict(t=w["last"]["t"], run_id=meta["id"], burned_ha=meta["stats"]["burned_ha"], detection=p)
                    w["alerts"] = (w["alerts"] + [alert])[-5:]
                    if w["webhook"]:
                        requests.post(w["webhook"], json=dict(text=f"PAKFIRE: new satellite fire detection near {w['name']} (FRP {p['frp']} MW). "
                                                                   f"24 h scenario: {meta['stats']['burned_ha']} ha, run {meta['id']}"), timeout=10)
                except Exception as e:
                    w["last"]["error"] = str(e)
            _save_watches()


threading.Thread(target=_watch_loop, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
