"""Does assimilating satellite detections help?  Seeds a run with every detection of the first CUT hours (at the time seen) and scores it on
the detections AFTER the cut, against the same run seeded only from the first overpass.   python tools/assim_test.py [calib_name] [cut_h]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pkfire import validate as v
import run_validation as rv
name = sys.argv[1] if len(sys.argv) > 1 else "cal"; cut = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
cal = json.load(open(os.path.join(v.VAL, f"calib_{name}.json")))
rows = []
for e in rv.selected():
    later = [d for d in e["dets"] if d[2] > cut]
    if len(later) < 3 or e["hours"] < cut + 6:
        continue
    prep = v.prepare(e)
    ev2 = dict(e, dets=later)                                              # score only what the model could not have seen
    rec = dict(id=e["id"], split=e["split"], lc=e["lc_mode"], n_later=len(later))
    for tag, kw in (("plain", {}), ("assim", dict(assimilate_h=cut))):
        sim = v.simulate(e, prep, cal, **kw)
        sc = v.score(ev2, prep, sim)
        if not sc.get("ok"): rec = None; break
        rec.update({f"{tag}_{k}": sc.get(k) for k in ("iou", "iou_circle", "det_capture", "timing_median_h", "sim_ha", "mcd_iou", "mcd_area_ratio", "mcd_mapped")})
    if rec: rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(v.VAL, f"assim_{name}.csv"), index=False)
print(len(df), "events with >=3 detections after", cut, "h")
for sp in ("dev", "test", "all"):
    g = df if sp == "all" else df[df.split == sp]
    print(f"{sp:5s} n={len(g):3d} | later-detection IoU plain {g.plain_iou.median():.3f} assim {g.assim_iou.median():.3f} circle {g.plain_iou_circle.median():.3f} | capture plain {g.plain_det_capture.median():.2f} assim {g.assim_det_capture.median():.2f} | timing(h) plain {g.plain_timing_median_h.median():.1f} assim {g.assim_timing_median_h.median():.1f} | assim beats plain {(g.assim_iou > g.plain_iou).mean():.2f}, beats circle {(g.assim_iou > g.assim_iou_circle).mean():.2f}")
m = df[(df.plain_mcd_mapped == True)]
print(f"MODIS final extent (n={len(m)}): IoU plain {m.plain_mcd_iou.median():.3f} assim {m.assim_mcd_iou.median():.3f}; area ratio plain {m.plain_mcd_area_ratio.median():.2f} assim {m.assim_mcd_area_ratio.median():.2f}")
