"""
Hourly fire weather at the ignition from Open-Meteo (forecast API for recent and
upcoming dates, archive API for older ones). Responses are cached on disk because
the shared server's IP has a small daily call budget.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

import numpy as np
import requests

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "weather")
VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation"


def _fetch(lat, lon, start: dt.date, end: dt.date):
    today = dt.date.today()
    if start >= today - dt.timedelta(days=80) and end <= today + dt.timedelta(days=15):
        url = "https://api.open-meteo.com/v1/forecast"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
    params = dict(latitude=round(lat, 3), longitude=round(lon, 3), hourly=VARS, wind_speed_unit="ms",
                  timezone="UTC", start_date=start.isoformat(), end_date=end.isoformat())
    key = hashlib.md5(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        return json.load(open(path)), "cache"
    r = requests.get(url, params=params, timeout=(6, 20))
    r.raise_for_status()
    js = r.json()
    os.makedirs(CACHE, exist_ok=True)
    json.dump(js, open(path, "w"))
    return js, "open-meteo"


def _met_norway(lat, lon):
    """MET Norway locationforecast: free fallback for the next ~9 days (hourly, then 3-6 hourly)."""
    key = hashlib.md5(f"met{round(lat,2)}{round(lon,2)}{dt.date.today()}{dt.datetime.now(dt.timezone.utc).hour//3}".encode()).hexdigest()
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        js = json.load(open(path))
    else:
        r = requests.get("https://api.met.no/weatherapi/locationforecast/2.0/compact",
                         params=dict(lat=round(lat, 3), lon=round(lon, 3)),
                         headers={"User-Agent": "pakistan-fire-sim/0.1 research"}, timeout=20)
        r.raise_for_status()
        js = r.json()
        os.makedirs(CACHE, exist_ok=True)
        json.dump(js, open(path, "w"))
    ts = js["properties"]["timeseries"]
    t = [dt.datetime.fromisoformat(x["time"].replace("Z", "")) for x in ts]
    d = [x["data"]["instant"]["details"] for x in ts]
    sec = [(x["data"].get("next_1_hours") or {}).get("details", {}).get("precipitation_amount", 0.0) for x in ts]
    return t, dict(temperature_2m=[x["air_temperature"] for x in d], relative_humidity_2m=[x["relative_humidity"] for x in d],
                   wind_speed_10m=[x["wind_speed"] for x in d], wind_direction_10m=[x["wind_from_direction"] for x in d],
                   precipitation=sec)


def _resample(times, series, first, n):
    """Interpolate irregular series to n hourly values from ``first`` (wind direction interpolated as a vector)."""
    x = np.array([(t - first).total_seconds() / 3600.0 for t in times])
    grid = np.arange(n, dtype=float)
    out = {}
    for k, v in series.items():
        v = np.array([np.nan if q is None else q for q in v], float)
        ok = ~np.isnan(v)
        if k == "wind_direction_10m":
            rad = np.deg2rad(v[ok])
            u = np.interp(grid, x[ok], np.sin(rad)); w = np.interp(grid, x[ok], np.cos(rad))
            out[k] = (np.rad2deg(np.arctan2(u, w)) + 360) % 360
        else:
            out[k] = np.interp(grid, x[ok], v[ok]) if ok.any() else np.zeros(n)
    return out


def hourly(lat, lon, start: dt.datetime, hours: int):
    """Arrays covering [start, start+hours] hourly (UTC). Returns dict with provenance."""
    first = start.replace(minute=0, second=0, microsecond=0)
    n = hours + 2
    last = first + dt.timedelta(hours=n)
    try:
        js, src = _fetch(lat, lon, first.date(), last.date())
        h = js["hourly"]
        times = [dt.datetime.fromisoformat(t) for t in h["time"]]
        series = {k: h[k] for k in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation")}
    except Exception as e_om:
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        if not (first >= now - dt.timedelta(hours=2) and last <= now + dt.timedelta(days=8)):
            raise RuntimeError(f"Open-Meteo unreachable ({type(e_om).__name__}) and MET Norway only covers the next 8 days") from e_om
        times, series = _met_norway(lat, lon)
        src = "met.no"
    r = _resample(times, series, first, n)
    return dict(temp=r["temperature_2m"], rh=np.clip(r["relative_humidity_2m"], 1, 100), wind=np.maximum(r["wind_speed_10m"], 0),
                wdir=r["wind_direction_10m"], precip=np.maximum(r["precipitation"], 0), source=src, start=first.isoformat())


def manual(hours: int, wind_ms: float, wind_from_deg: float, temp_c: float, rh: float):
    n = hours + 2
    return dict(temp=np.full(n, temp_c), rh=np.full(n, rh), wind=np.full(n, wind_ms),
                wdir=np.full(n, wind_from_deg), precip=np.zeros(n), source="manual", start="")
