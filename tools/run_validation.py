"""Validation driver.  python tools/run_validation.py prepare [workers]   build scene + observed weather for every selected event (cached)
                      python tools/run_validation.py score [name]        simulate with default or saved calibration and write results_<name>.json"""
import json, os, random, sys, time, traceback
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkfire import validate as v


def selected():
    """Event set. VAL_SET=v2 uses the 2012-2026 events (development before 2024, test from 2024); default is the 2024-2026 set of 527."""
    if os.environ.get("VAL_SET") == "v2":
        ev = v.build_events(full=True)
        rnd = random.Random(11)
        dev = [e for e in ev if e["split"] == "dev"]
        test = [e for e in ev if e["split"] == "test"]
        pick = lambda xs, k: rnd.sample(xs, min(k, len(xs)))
        keep = []
        for lc, cap in ((2, 500), (11, 800), (5, 300), (8, 10)):
            keep += pick([e for e in dev if e["lc_mode"] == lc], cap)
        keep += [e for e in test if e["lc_mode"] != 5] + pick([e for e in test if e["lc_mode"] == 5], 120)
        return sorted(keep, key=lambda e: e["t0_h"])
    ev = v.build_events()
    rnd = random.Random(7)
    crops = [e for e in ev if e["lc_mode"] == 5]
    keep = [e for e in ev if e["lc_mode"] != 5] + rnd.sample(crops, min(120, len(crops)))
    return sorted(keep, key=lambda e: e["t0_h"])


def _prep(e):
    try:
        v.prepare(e, force=bool(os.environ.get("FORCE"))); return e["id"], None
    except Exception as ex:
        return e["id"], f"{type(ex).__name__}: {str(ex)[:120]}"


def cmd_prepare(workers=4):
    evs = selected(); t = time.time(); fails = {}
    with Pool(workers) as p:
        for k, (i, err) in enumerate(p.imap_unordered(_prep, evs), 1):
            if err: fails[i] = err
            if k % 25 == 0: print(f"{k}/{len(evs)} prepared, {len(fails)} failed, {time.time()-t:.0f}s", flush=True)
    json.dump(fails, open(os.path.join(v.VAL, "prepare_failures.json"), "w"), indent=1)
    print("done", len(evs) - len(fails), "ok", len(fails), "failed", flush=True)


def cmd_score(name="baseline"):
    calib = None
    cp = os.path.join(v.VAL, f"calib_{name}.json")
    if os.path.exists(cp): calib = json.load(open(cp)); print("calibration", calib)
    rows = []
    for e in selected():
        p = os.path.join(v.VAL, "cache", e["id"] + ".pkl")
        if not os.path.exists(p): continue
        try:
            prep = v.prepare(e)
            rows.append(v.score(e, prep, v.simulate(e, prep, calib)))
        except Exception as ex:
            rows.append(dict(id=e["id"], ok=False, why=f"{type(ex).__name__}: {str(ex)[:100]}"))
    json.dump(rows, open(os.path.join(v.VAL, f"results_{name}.json"), "w"))
    print(len(rows), "scored,", sum(1 for r in rows if r.get("ok")), "ok")
    for k, s in v.summarize(rows).items(): print(k, s)
    print('--- against MODIS burned area (events with >= 50 ha mapped) ---')
    for k, sm in v.summarize_mcd(rows).items(): print(k, sm)
    import pandas as pd
    df = pd.DataFrame([r for r in rows if r.get("ok")])
    for lc, g in df.groupby("lc_mode"):
        print({5: "crops", 11: "rangeland", 2: "trees", 8: "bare"}.get(lc, lc), len(g), "iou", round(g.iou.median(), 3), "circle", round(g.iou_circle.median(), 3), "oracle circle", round(g.iou_circle_oracle.median(), 3), "area ratio", round(g.area_ratio.median(), 2), "capture", round(g.det_capture.median(), 3))

if __name__ == "__main__":
    c = sys.argv[1]
    cmd_prepare(int(sys.argv[2]) if len(sys.argv) > 2 else 4) if c == "prepare" else cmd_score(*(sys.argv[2:3]))
