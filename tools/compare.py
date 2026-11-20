"""Compare validation result sets.  python tools/compare.py baseline cal spot"""
import json, sys, os
import pandas as pd
V = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "validation")
names = {5: "crops", 11: "rangeland", 2: "trees"}
sets = {n: pd.DataFrame([x for x in json.load(open(os.path.join(V, f"results_{n}.json"))) if x.get("ok")]) for n in sys.argv[1:]}
print(f"{'set':10s} {'split':5s} {'n':>4s} | MODIS IoU  circle  beats  area  recall prec | FIRMS IoU  circle  beats")
for n, df in sets.items():
    for sp in ("dev", "test"):
        g = df[(df.split == sp) & (df.mcd_mapped == True)]; f = df[df.split == sp]
        print(f"{n:10s} {sp:5s} {len(g):4d} | {g.mcd_iou.median():8.3f} {g.mcd_iou_circle.median():7.3f} {(g.mcd_iou > g.mcd_iou_circle).mean():6.2f} {g.mcd_area_ratio.median():5.2f} {g.mcd_recall.median():6.2f} {g.mcd_precision.median():5.2f} | {f.iou.median():8.3f} {f.iou_circle.median():7.3f} {(f.iou > f.iou_circle).mean():6.2f}")
last = sets[sys.argv[-1]]
g = last[(last.mcd_mapped == True)]
print("by land cover (MODIS-mapped, last set):")
for lc, h in g.groupby("lc_mode"): print(" ", names.get(lc, lc), len(h), "IoU", round(h.mcd_iou.median(), 3), "circle", round(h.mcd_iou_circle.median(), 3), "area", round(h.mcd_area_ratio.median(), 2), "recall", round(h.mcd_recall.median(), 2))
