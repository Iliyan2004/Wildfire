"""Bounded calibration on the development events only (2024-2025).  python tools/calibrate.py [name] [workers]
Coordinate descent over per-fuel-group rate multipliers (grass, shrub, forest), a wind multiplier and a dead-fuel-moisture multiplier,
maximising the mean IoU against the MODIS burned area of the development events mapped at >= 50 ha. The 2026 test events are never read here."""
import json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pkfire import validate as v
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_validation as rv

EV = {}
GRID = dict(grass=[0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.4, 2.0], shrub=[0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.4, 2.0],
            forest=[0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0], wind=[0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25], moist=[0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.15],
            road=[0.0, 0.5, 1.0, 1.5], road_minor=[0.0, 0.5, 1.0, 1.5, 2.5], wfield=[0.0, 0.5, 1.0, 1.5], decay_h=[0.0, 96.0, 48.0, 24.0, 12.0, 6.0], suppress=[0.0, 0.2, 0.4, 0.6, 0.8], tmoist=[0.0, 0.5, 1.0, 1.5], chan=[0.0, 0.5, 1.0, 1.5])
LAMBDA_AREA = float(os.environ.get("LAMBDA_AREA", 0.04))      # IoU points lost per factor 2 of area error (log2 of simulated/mapped area)
LAMBDA_TIME = float(os.environ.get("LAMBDA_TIME", 0.003))     # IoU points lost per hour of median timing error against the detection times
BOUNDS = dict(grass=(0.15, 3.0), shrub=(0.15, 3.0), forest=(0.5, 16.0), wind=(0.6, 1.5), moist=(0.7, 1.3))


def _one(args):
    eid, calib = args
    e = EV[eid]
    prep = v.prepare(e)
    sim = v.simulate(e, prep, calib)
    r = v.score(e, prep, sim)
    ar = r.get("mcd_area_ratio") or 1.0
    t = r.get("timing_median_h")
    j = r.get("mcd_iou", 0.0) - LAMBDA_AREA * abs(np.log2(max(ar, 0.05))) - LAMBDA_TIME * (abs(t) if t is not None else 0.0)
    return eid, j, ar, r.get("mcd_iou", 0.0), t


def evaluate(pool, ids, calib):
    res = pool.map(_one, [(i, calib) for i in ids], chunksize=4)
    j = np.array([r[1] for r in res]); ar = np.array([r[2] for r in res], float); iou = np.array([r[3] for r in res]); tm = np.array([r[4] if r[4] is not None else np.nan for r in res], float)
    return float(j.mean()), float(np.nanmedian(ar)), float(iou.mean()), float(np.nanmedian(tm))


def main(name="cal", workers=8):
    global EV
    EV = {e["id"]: e for e in rv.selected()}
    base = json.load(open(os.path.join(v.VAL, os.environ.get("BASE", "baseline") and f"results_{os.environ.get('BASE', 'baseline')}.json")))  # only used to list the development events mapped by MODIS
    ids = [r["id"] for r in base if r.get("ok") and r.get("mcd_mapped") and r.get("split") == "dev"]
    import random
    cap = int(os.environ.get("CAL_MAX", 100000))
    if len(ids) > cap:
        ids = random.Random(5).sample(ids, cap)
    print(len(ids), "development events mapped by MODIS", flush=True)
    calib = dict(v.DEFAULT_CALIB)
    if os.environ.get("START"): calib.update(json.load(open(os.path.join(v.VAL, os.environ["START"]))))
    with Pool(workers) as pool:
        best, ar, iou0, tm0 = evaluate(pool, ids, calib); print("start J", round(best, 3), "iou", round(iou0, 3), "area ratio", round(ar, 2), "timing h", round(tm0, 1), flush=True)
        for sweep in range(int(os.environ.get("SWEEPS", 2))):
            for key in (os.environ["KEYS"].split(",") if os.environ.get("KEYS") else ("grass", "shrub", "forest", "wind", "moist", "road", "road_minor", "wfield", "decay_h", "suppress", "tmoist", "chan")):
                scores = {}
                for val in GRID[key]:
                    c = dict(calib); c[key] = val
                    scores[val] = evaluate(pool, ids, c)
                bv = max(scores, key=lambda k: scores[k][0])
                if scores[bv][0] > best + 1e-4:
                    calib[key], best = bv, scores[bv][0]
                print(f"sweep {sweep} {key}: " + " ".join(f"{k}:{s[0]:.3f}" for k, s in scores.items()) + f" -> {calib[key]} (J {best:.3f}, iou {scores[bv][2]:.3f}, area ratio {scores[bv][1]:.2f}, timing {scores[bv][3]:.1f} h)", flush=True)
                json.dump(calib, open(os.path.join(v.VAL, f"calib_{name}.json"), "w"))
    print("final", calib, "dev mean objective", best)

if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["cal"]), *(int(a) for a in sys.argv[2:3]))
