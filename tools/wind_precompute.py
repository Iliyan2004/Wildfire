"""Computes and caches the mass-conserving terrain wind ratio/rotation for every validation scene (dominant wind direction of its run),
on the GPU, one scene at a time.   python tools/wind_precompute.py"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pkfire import adjust, validate as v
import run_validation as rv
t0 = time.time(); evs = rv.selected(); fails = 0
for k, e in enumerate(evs, 1):
    try:
        prep = v.prepare(e); sc = prep["scene"]
        sc["_wf_cache"] = os.path.join(v.VAL, "wind", e["id"])
        hours = min(int(math.ceil(e["hours"])) + 1, v.MAX_HOURS)
        seg = slice(48, 48 + hours + 1)
        d = round(adjust.dominant_from_deg(np.asarray(prep["w"]["wdir"])[seg], np.asarray(prep["w"]["wind"])[seg]) / 22.5) * 22.5 % 360
        for chan in [float(c) for c in os.environ.get('CHANS', '0').split(',')]:
            adjust.terrain_wind(sc, d, adjust.chan_alpha(sc, prep['w'], prep['start'], hours, 48, chan))
    except Exception as ex:
        fails += 1; print("FAIL", e["id"], type(ex).__name__, str(ex)[:100], flush=True)
    if k % 25 == 0: print(f"{k}/{len(evs)} {time.time()-t0:.0f}s, {fails} failed", flush=True)
print("done", flush=True)
