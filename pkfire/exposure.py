"""Exposure of people and facilities to a saved run: population (WorldPop 2020, 1 km), hospitals, schools, grid stations
and power lines. For each facility the time at which the fire reaches it (or its distance from the burn scar)."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import from_bounds
from scipy import ndimage

from . import overlay, run as runmod

SHARE = os.environ.get("FIRE_SHARE", "/mnt/external-ssd/DGX SHARED")
POP = f"{SHARE}/Iliyan/WorldPop/pak_ppp_2020_1km_UNadj.tif"
V = f"{SHARE}/4. Vector Data"
LAYERS = dict(hospitals=f"{V}/Hospital/Hospitals.shp", schools=f"{V}/Schools/Schools.shp",
              grid_stations=f"{V}/Electricity Transmission Line/Grid_stations.shp",
              power_lines=f"{V}/Electricity Transmission Line/Electricity_transmission_line.shp")
NEAR_KM = 1.0
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBS = os.path.join(HERE, "data", "external", "pbs_districts.json")
INFORMAL = os.path.join(HERE, "data", "external", "informal", "informal_settlements_2025.gpkg")
_pbs = None


def pbs_district(lon, lat):
    """PBS 2023 census record of the district containing a point (None where PBS has no table: Azad Kashmir, Gilgit-Baltistan) with the factor that
    scales WorldPop 2020 to the census count. The factor is used only where it is plausible (0.7-1.5): for a few districts the PBS table does not
    match the boundary polygon (Lahore, Quetta, Karachi sub-districts), and then WorldPop is left unscaled."""
    global _pbs
    from . import geo
    if _pbs is None:
        try:
            _pbs = json.load(open(PBS))["districts"]
        except OSError:
            _pbs = {}
    try:
        d = geo.district_at(lon, lat)
    except Exception:
        return None
    name = (d or {}).get("name")
    rec = _pbs.get(name)
    if not rec:
        return None
    ratio = rec.get("census_over_worldpop")
    ok = bool(ratio and 0.7 <= ratio <= 1.5)
    return dict(district=name, scale=ratio if ok else 1.0, scale_used=ok, **{k: rec[k] for k in ("pop_2023", "households", "pakka_pct", "semi_pakka_pct", "kacha_pct", "firewood_cooking_pct", "census_over_worldpop")})


def _informal(meta, burnt, near):
    """Hectares of mapped informal settlement (2025, Karachi and Lahore only) burned and within NEAR_KM of the burn."""
    W, S, E, N = meta["bounds"]
    if not os.path.exists(INFORMAL):
        return dict(available=False, note="No informal-settlement map installed.")
    import geopandas as gpd
    from rasterio.features import rasterize
    g = gpd.read_file(INFORMAL, bbox=(W, S, E, N))
    if len(g) == 0:
        return dict(available=False, note="No mapped informal settlements here. The only open map found covers Karachi and Lahore (VHR-GSIS 2025).")
    h, w = burnt.shape
    m = rasterize([(x, 1) for x in g.geometry], out_shape=(h, w), transform=from_bounds(W, S, E, N, w, h), dtype="uint8").astype(bool)
    ha = (meta["cell_m"] ** 2) / 1e4
    return dict(available=True, polygons=int(len(g)), in_window_ha=round(float(m.sum() * ha)), burned_ha=round(float((m & burnt).sum() * ha), 1),
                within_1km_ha=round(float((m & near & ~burnt).sum() * ha), 1), note="Informal settlements as mapped from satellite imagery in 2025 (VHR-GSIS, CC BY 4.0); Karachi and Lahore only.")


def _population(meta, burnt, near):
    """People in the burned area and within NEAR_KM of it, from WorldPop persons per 1 km pixel spread to the export grid."""
    W, S, E, N = meta["bounds"]
    h, w = burnt.shape
    tr = from_bounds(W, S, E, N, w, h)
    dens = np.full((h, w), np.nan, np.float32)
    with rasterio.open(POP) as d:
        reproject(rasterio.band(d, 1), dens, dst_transform=tr, dst_crs="EPSG:4326", src_nodata=d.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
        src_px_km2 = (d.res[0] * 111.32 * math.cos(math.radians((N + S) / 2))) * (d.res[1] * 110.57)
    px_km2 = (meta["cell_m"] / 1000.0) ** 2
    people = np.nan_to_num(dens, nan=0.0) / src_px_km2 * px_km2          # persons per export pixel
    ok = bool(np.isfinite(dens).any())
    res = dict(available=ok, in_burn=round(float(people[burnt].sum())), within_1km=round(float(people[near & ~burnt].sum())),
               note="WorldPop 2020 at 1 km spread evenly over each pixel; for small fires the figure is an order of magnitude only.")
    ign = (meta.get("ignitions") or [[(W + E) / 2, (S + N) / 2]])[0]
    census = pbs_district(ign[0], ign[1])
    if census:
        f = census["scale"]
        res["census"] = dict(census, in_burn_scaled=round(res["in_burn"] * f), within_1km_scaled=round(res["within_1km"] * f),
                             note="PBS 2023 census district totals. WorldPop 2020 is scaled by census / WorldPop for the district" + ("" if census["scale_used"] else " (not applied here: the PBS table does not match the district polygon)") +
                             ". Kacha = mud, thatch or timber construction; semi-pakka = mixed.")
    return res


def compute(run_id):
    meta, arr, inten, _ = overlay.load(run_id)
    never = meta["never"]
    burnt = arr < never
    W, S, E, N = meta["bounds"]
    h, w = arr.shape
    px_m = meta["cell_m"]
    near = ndimage.distance_transform_edt(~burnt) * px_m <= NEAR_KM * 1000
    out = dict(run_id=run_id, near_km=NEAR_KM, population=_population(meta, burnt, near), facilities={})
    try:
        out["informal"] = _informal(meta, burnt, near)
    except Exception as e:
        out["informal"] = dict(available=False, note=f"{type(e).__name__}: {e}")
    import geopandas as gpd
    dist_km = ndimage.distance_transform_edt(~burnt) * px_m / 1000.0
    for name in ("hospitals", "schools", "grid_stations"):
        try:
            g = gpd.read_file(LAYERS[name], bbox=(W, S, E, N))
        except Exception as e:
            out["facilities"][name] = dict(error=str(e))
            continue
        items = []
        for r in g.itertuples():
            pt = r.geometry
            if pt is None or pt.is_empty:
                continue
            c = int((pt.x - W) / (E - W) * w)
            rr = int((N - pt.y) / (N - S) * h)
            if not (0 <= c < w and 0 <= rr < h):
                continue
            a = int(arr[rr, c])
            nm = getattr(r, "Name", None) or getattr(r, "NAME", None) or name[:-1]
            extra = {}
            if name == "hospitals":
                extra["beds"] = getattr(r, "Beds", None)
            if name == "schools":
                extra["enrollment"] = getattr(r, "Enrollment", None)
            status = "burned" if a < never else ("threatened" if dist_km[rr, c] <= NEAR_KM else "clear")
            items.append(dict(name=str(nm), lon=round(pt.x, 5), lat=round(pt.y, 5), status=status,
                              arrival_min=round(a / meta["t_scale"], 1) if a < never else None,
                              distance_km=round(float(dist_km[rr, c]), 2), **{k: (None if v != v else v) for k, v in extra.items()}))
        order = {"burned": 0, "threatened": 1, "clear": 2}
        items.sort(key=lambda x: (order[x["status"]], x["arrival_min"] if x["arrival_min"] is not None else 1e9, x["distance_km"]))
        out["facilities"][name] = dict(total_in_window=len(items), burned=sum(i["status"] == "burned" for i in items),
                                       threatened=sum(i["status"] == "threatened" for i in items), items=[i for i in items if i["status"] != "clear"][:15])
    try:
        g = gpd.read_file(LAYERS["power_lines"], bbox=(W, S, E, N))
        from shapely.geometry import box
        g = g.clip(box(W, S, E, N))
        km_burn = km_near = km_all = 0.0
        for geom in g.geometry:
            for ln in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                if ln.geom_type != "LineString" or ln.length == 0:
                    continue
                n = max(int(ln.length * 111.32 * 1000 / 100), 2)
                seg_km = ln.length * 111.0 / n
                for k in range(n):
                    p = ln.interpolate(k / n, normalized=True)
                    c, rr = int((p.x - W) / (E - W) * w), int((N - p.y) / (N - S) * h)
                    if 0 <= c < w and 0 <= rr < h:
                        km_all += seg_km
                        if burnt[rr, c]:
                            km_burn += seg_km
                        elif dist_km[rr, c] <= NEAR_KM:
                            km_near += seg_km
        out["facilities"]["power_lines"] = dict(km_in_window=round(km_all, 1), km_burned=round(km_burn, 1), km_within_1km=round(km_near, 1))
    except Exception as e:
        out["facilities"]["power_lines"] = dict(error=str(e))
    out["note"] = ("Facility positions come from national shapefiles whose completeness and accuracy are unknown. 'Arrival' is the model's time at the "
                   "facility's 50-100 m cell; an unburned facility is 'threatened' when within 1 km of the burn scar at the end of the run. Evacuation "
                   "times are in the separate evacuation panel; these figures are not for operational use.")
    return out


def cached(run_id):
    p = os.path.join(runmod.RUNS, run_id, "exposure.json")
    if os.path.exists(p):
        return json.load(open(p))
    r = compute(run_id)
    json.dump(r, open(p, "w"))
    return r
