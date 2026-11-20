"""Forest-fire validation: the scored events of a results file regrouped by how much forest the fire is in, forest type and elevation.
VAL_SET=v2 python tools/forest_validation.py [results_name]   (default v2cal). Writes data/validation/forest_validation.json.

An event counts as a forest fire when 40 % or more of the burnable cells within 3 km of its ignition are trees (ESRI land cover class 2). Metrics are
the same as the main validation: IoU of the simulated burn with the VIIRS footprint and with the MODIS burned area, against the equal-area circle."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkfire import validate as v

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_validation as rv


def describe(ev):
    prep = v.prepare(ev)
    sc = prep["scene"]
    ny, nx = sc["ny"], sc["nx"]
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, sc["crs"], always_xy=True).transform(ev["lon"], ev["lat"])
    c, r = ~sc["transform"] * (x, y)
    r, c = int(r), int(c)
    k = int(3000 / v.CELL_M)
    sl = (slice(max(r - k, 0), r + k + 1), slice(max(c - k, 0), c + k + 1))
    burn = sc["fuel"][sl] != 0
    trees = (sc["lc"][sl] == 2) & burn
    forest_frac = float(trees.sum() / max(burn.sum(), 1))
    needle = float(((sc["ftype"][sl] == 1) & trees).sum() / max(trees.sum(), 1))
    z = float(sc["dem"][min(max(r, 0), ny - 1), min(max(c, 0), nx - 1)])
    return dict(forest_frac=forest_frac, needle_share=needle, elev_m=z)


def summarise(rows):
    if not rows:
        return None
    f = lambda k: [r[k] for r in rows if r.get(k) is not None]
    m = [r for r in rows if r.get("mcd_mapped")]
    return dict(n=len(rows), iou=round(float(np.median(f("iou"))), 3), circle=round(float(np.median(f("iou_circle"))), 3),
                beats_circle=round(float(np.mean([r["iou"] > r["iou_circle"] for r in rows])), 3), area_ratio=round(float(np.median(f("area_ratio"))), 2),
                timing_bias_h=round(float(np.median(f("timing_median_h"))), 1) if f("timing_median_h") else None,
                n_modis=len(m), modis_iou=round(float(np.median([r["mcd_iou"] for r in m])), 3) if m else None,
                modis_circle=round(float(np.median([r["mcd_iou_circle"] for r in m])), 3) if m and "mcd_iou_circle" in m[0] else None,
                modis_area_ratio=round(float(np.median([r["mcd_area_ratio"] for r in m])), 2) if m else None)


def main(name="v2cal"):
    res = {r["id"]: r for r in json.load(open(os.path.join(v.VAL, f"results_{name}.json"))) if r.get("ok")}
    evs = [e for e in rv.selected() if e["id"] in res]
    rows = []
    for i, e in enumerate(evs):
        d = describe(e)
        if d["forest_frac"] >= 0.4:
            rows.append({**res[e["id"]], **d})
        if i % 200 == 0:
            print(i, len(evs), len(rows), flush=True)
    out = dict(results=name, definition="forest fire: >= 40 % of the burnable cells within 3 km of the ignition are trees")
    groups = {"all forest fires": rows, "test 2024-2026": [r for r in rows if r["split"] == "test"], "dev 2012-2023": [r for r in rows if r["split"] == "dev"],
              "needleleaf (>= 60 % of trees)": [r for r in rows if r["needle_share"] >= 0.6], "broadleaf or unknown (< 60 % needleleaf)": [r for r in rows if r["needle_share"] < 0.6],
              "below 1000 m": [r for r in rows if r["elev_m"] < 1000], "1000-2000 m": [r for r in rows if 1000 <= r["elev_m"] < 2000], "above 2000 m": [r for r in rows if r["elev_m"] >= 2000]}
    out["groups"] = {k: summarise(g) for k, g in groups.items()}
    json.dump(out, open(os.path.join(v.VAL, "forest_validation.json"), "w"), indent=1)
    for k, s in out["groups"].items():
        print(f"{k:42s}", s)


if __name__ == "__main__":
    main(*sys.argv[1:2])
