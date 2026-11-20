"""Tier 1: instant footprint. A small real simulation (the same engine, +-12 km window, CPU) contoured at 1, 3 and 6 h."""
from __future__ import annotations

import datetime as dt
import math
from collections import OrderedDict

import numpy as np
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage import measure

from . import run as runmod, solver, terrain

_scenes: "OrderedDict[tuple, dict]" = OrderedDict()


def _scene(lat, lon, half_km, cell_m, date):
    key = (round(lat, 3), round(lon, 3), half_km, cell_m, date.isoformat())
    if key not in _scenes:
        _scenes[key] = terrain.build_scene(lat, lon, half_km, cell_m, date)
        while len(_scenes) > 4:
            _scenes.popitem(last=False)
    return _scenes[key]


def footprint(lat, lon, hours=6, wind=None, half_km=12.0, cell_m=50.0):
    if not (23.5 <= lat <= 37.2 and 60.8 <= lon <= 77.9):
        raise runmod.RunError("Ignition is outside Pakistan's data coverage.")
    hours = int(min(max(hours, 1), 12))
    start = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    sc = _scene(lat, lon, half_km, cell_m, start.date())
    ny, nx = sc["ny"], sc["nx"]
    x, y = Transformer.from_crs(4326, sc["crs"], always_xy=True).transform(lon, lat)
    c, r = ~sc["transform"] * (x, y)
    s = runmod._snap(sc, int(r), int(c), int(600 / cell_m) + 2)
    if s is None:
        raise runmod.RunError("No burnable fuel within 600 m of this point.")
    w = runmod._weather(lat, lon, start, hours, wind)
    burnable = sc["fuel"] != 0
    ndvi = float(sc["ndvi"][burnable].mean())
    tables, md = runmod._tables(w, ndvi)
    grid = dict(ny=ny, nx=nx, dx=cell_m, slope=sc["slope"], upaz=sc["upaz"], cont=sc["cont"])
    arr, ros, fli, _, info = solver.simulate(grid, sc["fuel"], tables, w["wind"], w["wdir"], [(s[0], s[1], 0.0)], hours,
                                             spot_rate=0.0, hour0=runmod.LEAD_IN_H, backend="cpu")
    to_ll = Transformer.from_crs(sc["crs"], 4326, always_xy=True)
    big = np.where(np.isfinite(arr), arr, 1e9)
    feats = []
    for h in sorted({1, 3, 6} & set(range(1, hours + 1)) or {hours}):
        lvl = h * 60.0
        if not (big <= lvl).any():
            continue
        polys = []
        m = np.pad(big, 1, constant_values=1e9)
        for ring in measure.find_contours(m, lvl):
            if len(ring) < 4:
                continue
            pts = [to_ll.transform(*(sc["transform"] * (cc - 1 + 0.5, rr - 1 + 0.5))) for rr, cc in ring]
            p = Polygon(pts).buffer(0)
            if p.area > 0:
                polys.append(p)
        if not polys:
            continue
        geom = unary_union(polys).simplify(0.0003)
        area_ha = float((big <= lvl).sum() * cell_m * cell_m / 1e4)
        feats.append(dict(type="Feature", geometry=geom.__geo_interface__, properties=dict(t_h=h, area_ha=round(area_ha, 1))))
    i0, j0 = s
    wx0 = runmod.LEAD_IN_H
    return dict(type="FeatureCollection", features=feats,
                ignition=[lon, lat], head_ros_m_min=round(float(ros[i0, j0]), 2), fuel_row=int(sc["fuel"][i0, j0]),
                weather=dict(wind=round(float(w["wind"][wx0]), 1), wdir=round(float(w["wdir"][wx0])), temp=round(float(w["temp"][wx0]), 1),
                             rh=round(float(w["rh"][wx0])), dead1h=round(float(md[0, wx0]) * 100, 1), source=w["source"]),
                note="Instant footprint: a real 12 km simulation with the weather of the current hour held constant; use Tier 2 for the full run.")
