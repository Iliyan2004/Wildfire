"""2D map overlays and static map images for a saved run (PNG in the north-up EPSG:4326 export grid)."""
from __future__ import annotations

import io
import json
import os

import numpy as np
from PIL import Image

from . import run as runmod

RAMPS = {
    "arrival": dict(label="Time of arrival", unit="h", stops=[[0, [255, 244, 160]], [0.2, [255, 190, 60]], [0.5, [240, 100, 30]], [0.8, [170, 30, 20]], [1, [60, 15, 20]]]),
    "intensity": dict(label="Fireline intensity", unit="kW/m", stops=[[0, [255, 235, 140]], [0.25, [255, 160, 40]], [0.5, [235, 80, 25]], [0.75, [170, 25, 30]], [1, [90, 10, 60]]]),
    "prob": dict(label="Burn probability", unit="%", stops=[[0, [255, 240, 150]], [0.3, [255, 190, 70]], [0.6, [240, 100, 40]], [1, [150, 15, 25]]]),
}


def _color(frac, stops):
    xs = [s[0] for s in stops]
    cols = np.array([s[1] for s in stops], float)
    return np.stack([np.interp(frac, xs, cols[:, k]) for k in range(3)], -1).astype(np.uint8)


def load(run_id):
    d = os.path.join(runmod.RUNS, run_id)
    meta = json.load(open(os.path.join(d, "meta.json")))
    w, h = meta["size"]
    arr = np.fromfile(os.path.join(d, "arrival.bin"), "<u2").reshape(h, w)
    inten = np.fromfile(os.path.join(d, "intensity.bin"), np.uint8).reshape(h, w)
    pp = os.path.join(d, "prob.bin")
    prob = np.fromfile(pp, np.uint8).reshape(h, w) if os.path.exists(pp) else None
    return meta, arr, inten, prob


def layers(run_id):
    meta, _, _, prob = load(run_id)
    out = ["arrival", "intensity"]
    if prob is not None:
        out.append("prob")
    return out


def render(run_id, layer):
    """Returns (png_bytes, coordinates, legend dict)."""
    meta, arr, inten, prob = load(run_id)
    never = meta["never"]
    burnt = arr < never
    rgba = np.zeros(arr.shape + (4,), np.uint8)
    if layer == "arrival":
        t_end = max(meta["t_end_min"], 1.0)
        rgba[..., :3] = _color(np.clip(arr / meta["t_scale"] / t_end, 0, 1), RAMPS["arrival"]["stops"])
        rgba[..., 3] = np.where(burnt, 225, 0)
        legend = dict(label="Time of arrival", unit="h", min=0, max=round(t_end / 60, 1), ramp="arrival")
    elif layer == "intensity":
        v = inten.astype(float) * 40.0
        rgba[..., :3] = _color(np.clip(v / 6000.0, 0, 1), RAMPS["intensity"]["stops"])
        rgba[..., 3] = np.where(burnt, 225, 0)
        legend = dict(label="Max fireline intensity", unit="kW/m", min=0, max=6000, ramp="intensity")
    elif layer == "prob" and prob is not None:
        rgba[..., :3] = _color(prob / 100.0, RAMPS["prob"]["stops"])
        rgba[..., 3] = np.where(prob > 0, (80 + prob * 1.5).clip(0, 230), 0).astype(np.uint8)
        legend = dict(label="Burn probability", unit="%", min=0, max=100, ramp="prob")
    else:
        raise KeyError(layer)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    W, S, E, N = meta["bounds"]
    return buf.getvalue(), [[W, N], [E, N], [E, S], [W, S]], legend


def static_map_png(run_id, max_px=900):
    """Dark background + burn scar coloured by arrival time, for the PDF report."""
    meta, arr, inten, prob = load(run_id)
    never = meta["never"]
    burnt = arr < never
    img = np.zeros(arr.shape + (3,), np.uint8)
    img[:] = (14, 20, 26)
    t_end = max(meta["t_end_min"], 1.0)
    img[burnt] = _color(np.clip(arr[burnt] / meta["t_scale"] / t_end, 0, 1), RAMPS["arrival"]["stops"])
    im = Image.fromarray(img, "RGB")
    s = max_px / max(im.size)
    if s < 1:
        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def isochrones(run_id, step_min=60.0):
    """Spread contours (one polygon feature per time step, nested) for GIS export, in EPSG:4326."""
    from shapely.geometry import Polygon, mapping
    from shapely.ops import unary_union
    from skimage import measure
    meta, arr, _, _ = load(run_id)
    never = meta["never"]
    t = np.where(arr < never, arr / meta["t_scale"], 1e9)
    W, S, E, N = meta["bounds"]
    h, w = t.shape
    step = max(step_min, meta["t_end_min"] / 24.0) if step_min <= 0 else step_min
    feats = []
    for lvl in np.arange(step, meta["t_end_min"] + step, step):
        lvl = min(lvl, meta["t_end_min"])
        if not (t <= lvl).any():
            continue
        padded = np.pad(t, 1, constant_values=1e9)
        polys = []
        for ring in measure.find_contours(padded, lvl):
            if len(ring) < 4:
                continue
            pts = [(W + (c - 1 + 0.5) / w * (E - W), N - (r - 1 + 0.5) / h * (N - S)) for r, c in ring]
            p = Polygon(pts).buffer(0)
            if p.area > 0:
                polys.append(p)
        if not polys:
            continue
        g = unary_union(polys)
        feats.append(dict(type="Feature", geometry=mapping(g.simplify(0.0002)),
                          properties=dict(t_min=round(float(lvl), 1), t_h=round(float(lvl) / 60, 2), area_ha=round(float((t <= lvl).sum() * meta["cell_m"] ** 2 / 1e4), 1))))
    return dict(type="FeatureCollection", features=feats, crs="EPSG:4326", run_id=run_id, start_utc=meta["start"])
