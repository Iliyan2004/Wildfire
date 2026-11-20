"""Road corridors as a partial barrier to fire spread, from the national OSM roads in the SimCity PostGIS (read-only role).

A road (bare, often paved, 6-25 m wide) slows or stops a surface fire without a fuel break's width; at 50-100 m cells it is represented as a
multiplier below 1 on the spread rate of the cells it crosses. The database is optional: if it cannot be reached the barrier is one everywhere
and the scene records that. Connection URL: environment FIRE_ROADS_DB, else SIMCITY_DB_URL_RO from Smart City/.env (never printed or stored).
"""
from __future__ import annotations

import os
import re

import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from shapely import wkb
from shapely.ops import transform as shp_transform

ENV = os.path.join(os.environ.get("FIRE_SHARE", "/mnt/external-ssd/DGX SHARED"), "Iliyan", "Smart City", ".env")
# spread-rate multiplier of a cell crossed by a road of this OSM class (before the calibrated exponent)
CLASS_FACTOR = {"motorway": 0.05, "motorway_link": 0.1, "trunk": 0.08, "trunk_link": 0.12, "primary": 0.12, "primary_link": 0.2,
                "secondary": 0.25, "secondary_link": 0.35, "tertiary": 0.5, "tertiary_link": 0.6}


def _url():
    if os.environ.get("FIRE_ROADS_DB"):
        return os.environ["FIRE_ROADS_DB"]
    try:
        m = re.search(r"^SIMCITY_DB_URL_RO=(.+)$", open(ENV).read(), re.M)
        return m.group(1).strip().strip("\"'") if m else None
    except OSError:
        return None


# minor roads, farm tracks and lanes: field boundaries in cropland (spread-rate multiplier of a cell crossed by one)
MINOR_FACTOR = {"residential": 0.7, "unclassified": 0.6, "service": 0.75, "track": 0.55, "living_street": 0.8}


def road_factor(scene):
    """Arrays (major, minor) of spread-rate multipliers (1 = no road) and a flag that the database answered. Minor roads and tracks stand for field boundaries."""
    ny, nx = scene["ny"], scene["nx"]
    out = np.ones((ny, nx), np.float32)
    minor = np.ones((ny, nx), np.float32)
    url = _url()
    if not url:
        return out, minor, False
    try:
        import psycopg2
        from rasterio.transform import array_bounds
        l, b, r, t = array_bounds(ny, nx, scene["transform"])
        to_ll = Transformer.from_crs(scene["crs"], 4326, always_xy=True)
        W, S = to_ll.transform(l, b); E, N = to_ll.transform(r, t)
        con = psycopg2.connect(url, connect_timeout=8)
        con.set_session(readonly=True, autocommit=True)
        cur = con.cursor()
        cur.execute("select highway, ST_AsBinary(geom) from pk.roads where geom && ST_MakeEnvelope(%s,%s,%s,%s,4326) and highway = any(%s)",
                    (min(W, E) - 0.01, min(S, N) - 0.01, max(W, E) + 0.01, max(S, N) + 0.01, list(CLASS_FACTOR) + list(MINOR_FACTOR)))
        rows = cur.fetchall()
        con.close()
    except Exception:
        return out, minor, False
    to_utm = Transformer.from_crs(4326, scene["crs"], always_xy=True).transform
    for cls in sorted(CLASS_FACTOR, key=lambda k: -CLASS_FACTOR[k]):          # weakest first, so the strongest class wins where they cross
        shapes = [(shp_transform(to_utm, wkb.loads(bytes(g))), 1) for h, g in rows if h == cls]
        if not shapes:
            continue
        m = rasterize(shapes, out_shape=(ny, nx), transform=scene["transform"], fill=0, dtype=np.uint8, all_touched=True).astype(bool)
        out[m] = np.minimum(out[m], CLASS_FACTOR[cls])
    for cls in sorted(MINOR_FACTOR, key=lambda k: -MINOR_FACTOR[k]):
        shapes = [(shp_transform(to_utm, wkb.loads(bytes(g))), 1) for h, g in rows if h == cls]
        if not shapes:
            continue
        m = rasterize(shapes, out_shape=(ny, nx), transform=scene["transform"], fill=0, dtype=np.uint8, all_touched=True).astype(bool)
        minor[m] = np.minimum(minor[m], MINOR_FACTOR[cls])
    return out, minor, True
