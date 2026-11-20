"""Applies a calibration profile (and the other run-time adjustments) to a prepared scene and its weather.

Used by both the application (run.py) and the validation (validate.py) so they cannot drift apart. Keys of the parameter dict:
  grass, shrub, forest   spread-rate multipliers per fuel group          wind   multiplier on forecast/archive 10 m wind
  road_minor             exponent on the minor-road / farm-track barrier (field boundaries; 0 = ignored)
  moist                  multiplier on dead fuel moisture. wind and moist were fitted together on model weather, so neither is applied to weather the user types in                road   exponent on the road-barrier multiplier (0 = roads ignored)
  tmoist                 strength 0-1.5 of the terrain/shading correction of dead fuel moisture (microclimate.py)
  chan                   strength 0-1.5 of stability-driven channelling: the terrain wind is solved with a smaller alpha (flow goes around and along terrain
                         instead of over it) at night and in light wind (microclimate.channel_strength)
  wfield  strength 0-1.5 of the mass-conserving terrain wind (windfield.py)    decay_h  fade of spread rate with fire age, hours (0 = off)
  suppress               suppression proxy 0-1: the spread rate is cut by this fraction at a road or settlement and recovers with distance (2 km scale)
"""
from __future__ import annotations

import math

import numpy as np

from . import fuels as fm
from . import microclimate, roads, terrain, windfield

GROUP_IDS = dict(grass=(1, 3, 102), shrub=(2, 5, 6, 122), forest=(9, 161, 188))
DEFAULTS = dict(grass=1.0, shrub=1.0, forest=1.0, wind=1.0, moist=1.0, road=0.0, road_minor=0.0, tmoist=0.0, chan=0.0, canopy=0.0, wfield=0.0, decay_h=0.0, suppress=0.0)


def dominant_from_deg(wdir, wspeed):
    a = np.radians(np.asarray(wdir, float)); w = np.asarray(wspeed, float) + 1e-6
    return float((math.degrees(math.atan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) + 360.0) % 360.0)


NEUTRAL_ALPHA = 0.35


def chan_alpha(scene, w, start, hours, lead, chan):
    """Sherman alpha for the terrain wind: 0.35 (neutral, tuned to Jackson-Hunt) lowered to as little as 0.35 * 0.4 for stable air (night, light wind)."""
    if not chan or start is None:
        return NEUTRAL_ALPHA
    s = min(chan * microclimate.channel_strength(scene, w, start, hours, lead), 1.0)
    return round(NEUTRAL_ALPHA * (1.0 - 0.6 * s), 2)


def terrain_wind(scene, from_deg, alpha=NEUTRAL_ALPHA):
    """Mass-conserving wind ratio and rotation for one wind direction, cached on the scene (and on disk when scene['_wf_cache'] names a path)."""
    import os
    from . import windfield
    cache = scene.setdefault("_wf", {})
    key = (float(from_deg), float(alpha))
    if key not in cache:
        path = scene.get("_wf_cache")
        if not path:                                          # repeat runs of the same place and wind direction reuse the solution
            import hashlib
            h = hashlib.md5(np.ascontiguousarray(scene["dem"][::5, ::5]).tobytes() + repr((scene["dem"].shape, scene["dx"])).encode()).hexdigest()[:16]
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "windfield", h)
        f = (f"{path}_{int(key[0])}.npz" if alpha == NEUTRAL_ALPHA else f"{path}_{int(key[0])}_a{int(round(alpha * 100))}.npz") if path else None
        if f and os.path.exists(f):
            z = np.load(f); cache[key] = (z["r"], z["a"])
        else:
            r, ang, _ = windfield.solve(scene["dem"], scene["dx"], key[0], windfield.z0_from_lc(scene["lc"]), alpha=alpha)
            cache[key] = (r, ang)
            if f:
                os.makedirs(os.path.dirname(f), exist_ok=True)
                np.savez_compressed(f, r=r, a=ang)
    return cache[key]


def prepare_scene_extras(scene):
    """Things worth computing once per scene (roads need a database round trip). Stored on the scene dict."""
    if "roadf" not in scene:
        scene["roadf"], scene["roadm"], scene["roads_ok"] = roads.road_factor(scene)
    return scene


def apply(scene, w, params, hours, lead, user_wind=False, start=None):
    """Returns dict(cont, wmod, w, moist, decay_min) for the solver; the scene is not modified."""
    p = {**DEFAULTS, **(params or {})}
    cont = scene["cont"].copy()
    fuel = scene["fuel"]
    for g, ids in GROUP_IDS.items():
        if p[g] != 1.0:
            for fid in ids:
                cont[fuel == fm.row_of(fid)] *= p[g]
    if p["road"] > 0 or p["road_minor"] > 0:
        prepare_scene_extras(scene)
        if p["road"] > 0:
            cont *= scene["roadf"] ** p["road"]
        if p["road_minor"] > 0:
            cont *= scene["roadm"] ** p["road_minor"]
    if p["suppress"] > 0:
        prepare_scene_extras(scene)
        if "_access" not in scene:
            from scipy import ndimage
            src = (scene["roadf"] < 1.0) | (scene["lc"] == 7)
            scene["_access"] = np.exp(-(ndimage.distance_transform_edt(~src) * scene["dx"]) / 2000.0).astype(np.float32) if src.any() else np.zeros(fuel.shape, np.float32)
        cont *= (1.0 - p["suppress"] * scene["_access"])
    w2 = dict(w)
    if not user_wind and p["wind"] != 1.0:
        w2["wind"] = np.asarray(w["wind"]) * p["wind"]
    wmod = np.ones(fuel.shape, np.float32)
    wrot = np.zeros(fuel.shape, np.float32)
    if p["wfield"] > 0:
        seg = slice(lead, lead + hours + 1)
        d = round(dominant_from_deg(np.asarray(w["wdir"])[seg], np.asarray(w["wind"])[seg]) / 22.5) * 22.5 % 360
        ratio, rot = terrain_wind(scene, d, chan_alpha(scene, w, start, hours, lead, p["chan"]))
        wmod = (wmod * ratio ** p["wfield"]).astype(np.float32)
        wrot = (-np.radians(rot) * p["wfield"]).astype(np.float32)            # solver rotation is anticlockwise; azimuth runs clockwise
    if p["tmoist"] > 0 and start is not None:
        mult = microclimate.rate_multiplier(scene, w, start, hours, lead, p["tmoist"])
        if mult is not None:
            cont = (cont * mult).astype(np.float32)
    out = dict(cont=cont, wmod=wmod, wrot=wrot, w=w2, moist=(1.0 if user_wind else p["moist"]), decay_min=p["decay_h"] * 60.0)
    if p["canopy"] > 0:                                   # measured canopy height sets crown base height and bulk density cell by cell
        cm, c90 = terrain.canopy_for_scene(scene)
        out["crown_ci0"], out["crown_cbd"] = fm.canopy_crown(fuel, cm, c90)
    return out
