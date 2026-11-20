"""Chooses the width of the ensemble perturbations from the validation events.   VAL_SET=v2 python tools/ensemble_widths.py [n_events] [members]

The ensemble perturbs wind speed (sd 20 %), wind direction (12 deg), dead fuel moisture (15 %) and NDVI (0.03). Those widths were assumed. Here
all four are multiplied by a common scale s, ensembles are run for events mapped by MODIS (50 ha or more) under the shipped calibration, and for each s:
  brier      mean squared difference between the burn probability and the MODIS burn mask, over the cells within 3 km of the observed or any
             simulated burn (everything farther is a trivial zero)
  coverage   share of events whose MODIS area falls inside the 10-90 % range of the member areas (a reliable 80 % range gives 0.80)
  area_pit   mean position of the MODIS area in the sorted member areas (0 = all members larger, 1 = all smaller; 0.5 is unbiased)
s is chosen on the development events (before 2024) and reported on the test events (2024 onward)."""
import json
import os
import random
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkfire import run as runmod, validate as v

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_validation as rv

SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def one(ev, calib, members, scale, rate=0.0):
    prep = v.prepare(ev)
    widths = {k: x * scale for k, x in runmod.ENS_WIDTHS.items() if k != 'rate'}
    widths['rate'] = rate
    mem = v.simulate_ensemble(ev, prep, calib, members, widths, seed=3)
    if mem is None:
        return None
    mask, _ = v.mcd_truth(ev, prep)
    truth = ndimage.binary_dilation(mask, iterations=0) if False else mask
    p = mem.mean(0)
    region = ndimage.binary_dilation(truth | (p > 0), iterations=int(round(3000 / v.CELL_M)))
    brier = float(((p - truth) ** 2)[region].mean())
    areas = np.sort(mem.reshape(len(mem), -1).sum(1))
    t = float(truth.sum())
    lo, hi = np.percentile(areas, [10, 90])
    pit = float((areas < t).mean() + 0.5 * (areas == t).mean())
    return brier, bool(lo <= t <= hi), pit


def stage2(n=60, members=16):
    """Second stage: with the wind, direction, moisture and NDVI widths at scale 1 and 2, add a log-normal spread-rate error of width `rate`."""
    calib = {**v.DEFAULT_CALIB, **json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pkfire", "calibration.json")))["params"]}
    evs = rv.selected()
    res = json.load(open(os.path.join(v.VAL, "results_v2cal.json")))
    ok = {r["id"] for r in res if r.get("ok") and r.get("mcd_mapped")}
    pick = {"dev": [], "test": []}
    for e in evs:
        if e["id"] in ok:
            pick[e["split"]].append(e)
    rnd = random.Random(11)
    for k in pick:
        rnd.shuffle(pick[k]); pick[k] = pick[k][:n]
    out = {}
    for split, evl in pick.items():
        for s in (1.0, 2.0):
            for rate in (0.0, 0.3, 0.6, 0.9):
                rows = [r for r in (one(e, calib, members, s, rate) for e in evl) if r]
                out[f"{split}_{s}_{rate}"] = dict(n=len(rows), brier=round(float(np.mean([r[0] for r in rows])), 5), coverage=round(float(np.mean([r[1] for r in rows])), 3),
                                                  area_pit=round(float(np.mean([r[2] for r in rows])), 3))
                print(split, "scale", s, "rate", rate, out[f"{split}_{s}_{rate}"], flush=True)
    json.dump(out, open(os.path.join(v.VAL, "ensemble_widths2.json"), "w"), indent=1)


def main(n=100, members=16):
    calib = {**v.DEFAULT_CALIB, **json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pkfire", "calibration.json")))["params"]}
    evs = rv.selected()
    res = json.load(open(os.path.join(v.VAL, "results_v2cal.json")))
    ok = {r["id"] for r in res if r.get("ok") and r.get("mcd_mapped")}
    pick = {"dev": [], "test": []}
    for e in evs:
        if e["id"] in ok:
            pick[e["split"]].append(e)
    rnd = random.Random(11)
    for k in pick:
        rnd.shuffle(pick[k]); pick[k] = pick[k][:n]
    out = {}
    for split, evl in pick.items():
        for s in SCALES:
            rows = [r for r in (one(e, calib, members, s) for e in evl) if r]
            out[f"{split}_{s}"] = dict(n=len(rows), brier=round(float(np.mean([r[0] for r in rows])), 5), coverage=round(float(np.mean([r[1] for r in rows])), 3),
                                       area_pit=round(float(np.mean([r[2] for r in rows])), 3))
            print(split, "scale", s, out[f"{split}_{s}"], flush=True)
    dev = {s: out[f"dev_{s}"]["brier"] for s in SCALES}
    best = min(dev, key=dev.get)
    out["best_scale"] = best
    json.dump(out, open(os.path.join(v.VAL, "ensemble_widths.json"), "w"), indent=1)
    print("best scale on dev:", best, "->", {k: round(x * best, 3) for k, x in runmod.ENS_WIDTHS.items()})


if __name__ == "__main__":
    if sys.argv[1:2] == ["stage2"]:
        stage2(*(int(a) for a in sys.argv[2:4]))
    else:
        main(*(int(a) for a in sys.argv[1:3]))
