"""Districts, FIRMS detections and district statistics shared by the discovery panels."""
from __future__ import annotations

import csv
import datetime as dt
import io
import os
import threading
import time

import requests

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISTRICTS = os.environ.get(
    "FIRE_DISTRICTS",
    "/mnt/external-ssd/DGX SHARED/4. Vector Data/Pakistan Boundary/Pakistan, Tehsil and District Boundaries/District_Boundary.shp")
BOUNDARY = DISTRICTS.replace("District_Boundary", "Pakistan_Boundary")
PAK_BBOX = "60.8,23.5,77.9,37.2"


def _secret(name, env):
    if os.environ.get(env):
        return os.environ[env].strip()
    p = os.path.join(HERE, ".secrets", name)
    return open(p).read().strip() if os.path.exists(p) else ""


FIRMS_KEY = _secret("firms_map_key", "FIRMS_MAP_KEY")
MAPBOX_TOKEN = _secret("mapbox_token", "MAPBOX_TOKEN")

_lock = threading.Lock()
_cache = {}


def districts():
    """GeoDataFrame of the 164 districts (EPSG:4326) with a spatial index."""
    with _lock:
        if "d" not in _cache:
            import geopandas as gpd
            g = gpd.read_file(DISTRICTS).to_crs(4326)[["name", "province", "division", "geometry"]].reset_index(drop=True)
            g["idx"] = g.index
            _cache["d"] = g
            _cache["d_simple"] = g.copy()
            _cache["d_simple"]["geometry"] = g.geometry.simplify(0.01, preserve_topology=True)
        return _cache["d"]


def district_at(lon, lat):
    from shapely.geometry import Point
    g = districts()
    hit = g[g.contains(Point(lon, lat))]
    if len(hit):
        r = hit.iloc[0]
        return dict(idx=int(r.idx), name=r["name"], province=r["province"])
    return None


def firms(days=2):
    """VIIRS S-NPP near-real-time detections inside Pakistan, last ``days`` days (cached 30 min)."""
    days = min(max(int(days), 1), 5)
    c = _cache.get(("f", days))
    if c and time.time() - c[0] < 1800:
        return c[1]
    if not FIRMS_KEY:
        raise RuntimeError("FIRMS key not configured")
    r = requests.get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_KEY}/VIIRS_SNPP_NRT/{PAK_BBOX}/{days}", timeout=40)
    r.raise_for_status()
    import geopandas as gpd
    rows = []
    for row in csv.DictReader(io.StringIO(r.text)):
        try:
            rows.append(dict(lat=float(row["latitude"]), lon=float(row["longitude"]), frp=float(row.get("frp") or 0),
                             conf=row.get("confidence"), when=f"{row['acq_date']} {int(row['acq_time']):04d}Z",
                             daynight=row.get("daynight")))
        except (KeyError, ValueError):
            continue
    if rows:
        pts = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([x["lon"] for x in rows], [x["lat"] for x in rows]), crs=4326)
        j = gpd.sjoin(pts, districts()[["name", "province", "idx", "geometry"]], how="inner", predicate="within")
        j = j.drop_duplicates(subset=["lat", "lon", "when"])
        rows = [dict(lon=float(x.lon), lat=float(x.lat), frp=float(x.frp), conf=x.conf, when=x.when, daynight=x.daynight,
                     district=x.name, province=x.province, didx=int(x.idx)) for x in j.itertuples()]
    _cache[("f", days)] = (time.time(), rows)
    return rows


def district_stats(days=3):
    """Per-district detection count and summed fire radiative power for the last ``days`` days."""
    pts = firms(days)
    out = {}
    for p in pts:
        d = out.setdefault(p["didx"], dict(n=0, frp=0.0))
        d["n"] += 1
        d["frp"] += p["frp"]
    return out


def districts_geojson(days=3):
    import json
    districts()
    st = district_stats(days)
    feats = json.loads(_cache["d_simple"].to_json())["features"]
    for f in feats:
        s = st.get(f["properties"]["idx"], dict(n=0, frp=0.0))
        f["properties"].update(detections=s["n"], frp=round(s["frp"], 1))
    return dict(type="FeatureCollection", features=feats, days=days)
