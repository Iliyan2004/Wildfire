"""Evacuation and fire-service access for a saved run, over the OSM road network in the SimCity PostGIS (read-only role).

Two analyses, both on the same road graph:

1. Evacuation. The most populated 1 km WorldPop cells that the fire burns or comes within 1.5 km of become origins (snapped to the nearest road
   vertex within 2 km). Each leaves ``lead`` minutes before the model's fire reaches its location (but not before the first fire within 1.5 km if the
   cell is not burned) and drives to the nearest "safe" vertex: one the fire does not reach and that lies 3 km or more from the burn, or one on the
   edge of the window. Travel is time-dependent: a road section is unusable if the fire reaches it before the traveller has passed. For each origin
   the latest departure that still escapes is found by bisection, giving the warning time that is actually needed.
2. Access. Drive time from the fire stations (data/external/fire_stations.geojson or the file named in FIRE_STATIONS) to the ignition, with the
   area burned at the moment the first unit would arrive.

Road speeds are free-flow guesses by OSM class (no congestion, no panic, no evacuation behaviour), so times are lower bounds. Nothing here is for
operational use.
"""
from __future__ import annotations

import heapq
import json
import math
import os

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import exposure, overlay, roads, run as runmod

SPEED = {"motorway": 80, "motorway_link": 50, "trunk": 65, "trunk_link": 45, "primary": 55, "primary_link": 40, "secondary": 45, "secondary_link": 35,
         "tertiary": 38, "tertiary_link": 30, "unclassified": 28, "residential": 22, "service": 18, "living_street": 12}
UNPAVED = ("unpaved", "dirt", "gravel", "ground", "sand", "mud", "earth", "grass", "compacted", "fine_gravel")
SAFE_KM = 3.0
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POP = exposure.POP
STATIONS = os.environ.get("FIRE_STATIONS", os.path.join(HERE, "data", "external", "fire_stations.geojson"))


def load_graph(bounds):
    """Road graph in the box (lon/lat): vertices (lon, lat), CSR travel-time matrix (minutes), edge arrays."""
    import psycopg2
    from shapely import wkb
    W, S, E, N = bounds
    con = psycopg2.connect(roads._url(), connect_timeout=8)
    con.set_session(readonly=True, autocommit=True)
    cur = con.cursor()
    cur.execute("select highway, surface, oneway, ST_AsBinary(geom) from pk.roads where geom && ST_MakeEnvelope(%s,%s,%s,%s,4326) and highway = any(%s)",
                (W, S, E, N, list(SPEED)))
    rows = cur.fetchall()
    con.close()
    vid, coords = {}, []
    eu, ev, et, el = [], [], [], []
    for hw, surf, oneway, g in rows:
        v = SPEED[hw] * (0.6 if (surf or "") in UNPAVED else 1.0)
        geom = wkb.loads(bytes(g))
        for line in (geom.geoms if hasattr(geom, "geoms") else [geom]):
            pts = list(line.coords)
            prev = None
            for x, y in pts:
                k = (round(x, 5), round(y, 5))
                i = vid.get(k)
                if i is None:
                    i = vid[k] = len(coords); coords.append(k)
                if prev is not None and prev != i:
                    a, b = coords[prev], coords[i]
                    d = math.hypot((b[0] - a[0]) * 111.32 * math.cos(math.radians(a[1])), (b[1] - a[1]) * 110.57)         # km
                    eu.append(prev); ev.append(i); et.append(d / v * 60.0); el.append(d)
                prev = i
    n = len(coords)
    eu, ev, et, el = (np.array(a) for a in (eu, ev, et, el))
    # undirected (one-way streets are rare in this layer and are treated as two-way: optimistic)
    U = np.concatenate([eu, ev]); V = np.concatenate([ev, eu]); T = np.concatenate([et, et]); L = np.concatenate([el, el])
    return np.array(coords), U, V, T, L, n


def _arrival_at(coords, meta, arr):
    W, S, E, N = meta["bounds"]
    h, w = arr.shape
    c = np.clip(((coords[:, 0] - W) / (E - W) * w).astype(int), 0, w - 1)
    r = np.clip(((N - coords[:, 1]) / (N - S) * h).astype(int), 0, h - 1)
    inside = (coords[:, 0] >= W) & (coords[:, 0] <= E) & (coords[:, 1] >= S) & (coords[:, 1] <= N)
    a = arr[r, c].astype(np.float64)
    a = np.where(a >= meta["never"], np.inf, a / meta["t_scale"])
    return np.where(inside, a, np.inf), inside


def _escape(adj, t_arr, safe, start, t_dep, limit=360.0):
    """Time-dependent Dijkstra from ``start`` leaving at t_dep (min). A vertex is usable if the fire reaches it after the traveller does.
    Returns (arrival time at the first safe vertex or None, path of vertices, minimum margin in minutes between the traveller and the fire)."""
    if t_arr[start] <= t_dep:
        return None, [], -1.0
    best = {start: t_dep}
    prev = {}
    pq = [(t_dep, start)]
    while pq:
        t, u = heapq.heappop(pq)
        if t > best.get(u, 1e18) + 1e-9:
            continue
        if safe[u]:
            path = [u]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            path.reverse()
            margin = min(t_arr[x] - best[x] for x in path)
            return t, path, margin
        if t - t_dep > limit:
            break
        for v, dt in adj[u]:
            tv = t + dt
            if t_arr[v] <= tv:                                      # the fire is there before us
                continue
            if tv < best.get(v, 1e18):
                best[v] = tv
                prev[v] = u
                heapq.heappush(pq, (tv, v))
    return None, [], -1.0


def analyse(run_id, lead_min=60.0, max_origins=30):
    meta, arr, inten, _ = overlay.load(run_id)
    never, ts = meta["never"], meta["t_scale"]
    burnt = arr < never
    W, S, E, N = meta["bounds"]
    h, w = arr.shape
    px_m = (E - W) * 111320.0 * math.cos(math.radians((N + S) / 2)) / w
    out = dict(run_id=run_id, lead_min=lead_min, assumptions="Free-flow road speeds by OSM class (motorway 80 km/h to living street 12, unpaved x0.6); no congestion, no vehicle limits, one-way streets treated as two-way; "
               "people leave lead_min before the fire reaches them; a road section is lost when the fire reaches it before the traveller. Lower bounds on real times.")
    if not roads._url():
        return dict(out, error="the road database is not configured")
    pad = 0.05
    try:
        coords, U, V, T, L, n = load_graph((W - pad, S - pad, E + pad, N + pad))
    except Exception as e:
        return dict(out, error=f"road database unavailable: {type(e).__name__}")
    if n < 50:
        return dict(out, error="no road network in this window")
    adjl = [[] for _ in range(n)]
    for u, v, t in zip(U.tolist(), V.tolist(), T.tolist()):
        adjl[u].append((v, t))
    t_arr, inside = _arrival_at(coords, meta, arr)
    dist_km = ndimage.distance_transform_edt(~burnt) * px_m / 1000.0
    c = np.clip(((coords[:, 0] - W) / (E - W) * w).astype(int), 0, w - 1)
    r = np.clip(((N - coords[:, 1]) / (N - S) * h).astype(int), 0, h - 1)
    d_v = np.where(inside, dist_km[r, c], 0.0)
    safe = ((~np.isfinite(t_arr)) & (d_v >= SAFE_KM)) | (~inside)
    tree = cKDTree(coords * np.array([111.32 * math.cos(math.radians((N + S) / 2)), 110.57]))
    km = np.array([111.32 * math.cos(math.radians((N + S) / 2)), 110.57])
    out["graph"] = dict(vertices=int(n), edges=int(len(U) // 2), km=round(float(L.sum() / 2), 0))

    # ---- origins: populated 1 km cells burned or within 1.5 km of the burn
    import rasterio
    from rasterio.windows import from_bounds
    origins = []
    try:
        with rasterio.open(POP) as d:
            win = from_bounds(W, S, E, N, d.transform).round_offsets().round_lengths()
            pop = d.read(1, window=win, masked=True).filled(0).astype(np.float64)
            tr = rasterio.windows.transform(win, d.transform)
        pop[pop < 0] = 0
        ign0 = meta["ignitions"][0]
        census = exposure.pbs_district(ign0[0], ign0[1])
        if census and census["scale_used"]:
            pop *= census["scale"]                                  # WorldPop 2020 scaled to the PBS 2023 district count
            out["population_scaled_to"] = f"PBS 2023 census, {census['district']} (x{census['scale']})"
        rr, cc = np.nonzero(pop > 5)
        lon = tr.c + (cc + 0.5) * tr.a
        lat = tr.f + (rr + 0.5) * tr.e
        ec = np.clip(((lon - W) / (E - W) * w).astype(int), 0, w - 1)
        er = np.clip(((N - lat) / (N - S) * h).astype(int), 0, h - 1)
        near = dist_km[er, ec] <= 1.5
        _, idx = ndimage.distance_transform_edt(~burnt, return_indices=True)
        for k in np.nonzero(near)[0]:
            a_here = arr[er[k], ec[k]]
            a_near = arr[idx[0][er[k], ec[k]], idx[1][er[k], ec[k]]]
            t_fire = (a_here if a_here < never else a_near) / ts
            origins.append((float(pop[rr[k], cc[k]]), float(lon[k]), float(lat[k]), float(t_fire), bool(a_here < never)))
        origins.sort(reverse=True)
        origins = origins[:max_origins]
        out["population_threatened"] = round(float(pop[rr[near], cc[near]].sum()))
    except Exception as e:
        out["population_error"] = f"{type(e).__name__}: {e}"
    res = []
    for people, lon0, lat0, t_fire, burned_here in origins:
        dd, vi = tree.query(np.array([lon0, lat0]) * km)
        if dd > 2.0:
            res.append(dict(people=round(people), lon=round(lon0, 4), lat=round(lat0, 4), fire_reaches_min=round(t_fire, 0), status="no road within 2 km", burned=burned_here))
            continue
        t_dep = max(t_fire - lead_min, 0.0)
        # the road vertex itself can be reached by the fire earlier than the cell centre: leave before that
        t_dep = max(min(t_dep, t_arr[vi] - 1.0), 0.0)
        te, path, margin = _escape(adjl, t_arr, safe, vi, t_dep)
        row = dict(people=round(people), lon=round(lon0, 4), lat=round(lat0, 4), fire_reaches_min=round(t_fire, 0), burned=burned_here, depart_min=round(t_dep, 0))
        if te is None:
            # cut off even when leaving at t_dep: how early would one have to leave?
            lo, hi = 0.0, t_dep
            te0, p0, m0 = _escape(adjl, t_arr, safe, vi, 0.0)
            row["status"] = "cut off" if te0 is None else "escapes only if leaving earlier"
            if te0 is not None:
                for _ in range(7):
                    mid = (lo + hi) / 2
                    if _escape(adjl, t_arr, safe, vi, mid)[0] is not None:
                        lo = mid
                    else:
                        hi = mid
                row["latest_departure_min"] = round(lo, 0)
                row["warning_needed_min"] = round(max(t_fire - lo, 0.0), 0)
                row["route"] = [[round(float(coords[x][0]), 5), round(float(coords[x][1]), 5)] for x in p0[:: max(len(p0) // 60, 1)]]
        else:
            lo, hi = t_dep, max(t_fire, t_dep + 1)
            for _ in range(7):
                mid = (lo + hi) / 2
                if _escape(adjl, t_arr, safe, vi, mid)[0] is not None:
                    lo = mid
                else:
                    hi = mid
            row.update(status="escapes", travel_min=round(te - t_dep, 1), margin_min=round(float(margin), 1) if math.isfinite(margin) else None, latest_departure_min=round(lo, 0),
                       warning_needed_min=round(max(t_fire - lo, 0.0), 0), route_km=round(float(sum(L[(U == a) & (V == b)][:1].sum() for a, b in zip(path[:-1], path[1:]))), 1),
                       route=[[round(float(coords[x][0]), 5), round(float(coords[x][1]), 5)] for x in path[:: max(len(path) // 60, 1)]])
        res.append(row)
    out["origins"] = res
    if res:
        ok = [x for x in res if x["status"] == "escapes"]
        out["summary"] = dict(origins=len(res), people=sum(x["people"] for x in res), escape=len(ok), cut_off=sum(1 for x in res if x["status"] == "cut off"),
                              people_cut_off=sum(x["people"] for x in res if x["status"] == "cut off"),
                              needing_more_warning=sum(1 for x in res if x["status"].startswith("escapes only")),
                              median_travel_min=round(float(np.median([x["travel_min"] for x in ok])), 1) if ok else None,
                              median_warning_needed_min=round(float(np.median([x["warning_needed_min"] for x in ok])), 0) if ok else None)

    # ---- fire-service access
    try:
        stations = json.load(open(STATIONS))["features"]
        ign = np.array(meta["ignitions"][0])
        dist, vi0 = tree.query(ign * km)
        if dist < 3.0:
            M = csr_matrix((T, (U, V)), shape=(n, n))
            from_ign = dijkstra(M, directed=False, indices=int(vi0))
            cand = []
            for f in stations:
                lo_, la_ = f["geometry"]["coordinates"]
                if W - pad <= lo_ <= E + pad and S - pad <= la_ <= N + pad:
                    dd, vs = tree.query(np.array([lo_, la_]) * km)
                    if dd < 2.0 and np.isfinite(from_ign[vs]):
                        cand.append((float(from_ign[vs]), f["properties"].get("name") or "fire station", lo_, la_))
            cand.sort()
            area = meta["stats"]["area_series"]
            def area_at(t):
                a = 0.0
                for tm, ha in area:
                    if tm <= t:
                        a = ha
                return a
            out["access"] = dict(
                ignition_to_road_km=round(float(dist), 2), stations_in_window=len(cand),
                nearest=[dict(name=nm, drive_min=round(t, 1), lon=round(lo_, 4), lat=round(la_, 4), fire_ha_at_arrival=round(area_at(t), 1)) for t, nm, lo_, la_ in cand[:5]],
                note="Station positions come from OpenStreetMap (sparse in Pakistan); supply an official list in FIRE_STATIONS. Drive time excludes turn-out and dispatch time and fire-fighting capability.")
        else:
            out["access"] = dict(error=f"the ignition is {dist:.1f} km from the nearest mapped road")
    except Exception as e:
        out["access"] = dict(error=f"{type(e).__name__}: {e}")
    return out


def cached(run_id, lead_min=60.0):
    p = os.path.join(runmod.RUNS, run_id, f"evacuation_{int(lead_min)}.json")
    if os.path.exists(p):
        return json.load(open(p))
    r = analyse(run_id, lead_min)
    if "error" not in r:                                         # do not keep a transient failure (database down) as the answer
        json.dump(r, open(p, "w"))
    return r
