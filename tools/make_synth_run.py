"""Writes a synthetic run (analytic elliptical spread, real file format) for testing routes and the viewer without
running the simulator. Usage: python tools/make_synth_run.py [run_id]. Delete data/runs/<id> afterwards."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rid = sys.argv[1] if len(sys.argv) > 1 else "00000000-000000-test"
d = os.path.join(HERE, "data", "runs", rid)
os.makedirs(d, exist_ok=True)
w = h = 400
W, S, E, N = 73.00, 33.60, 73.40, 33.90
lon0, lat0 = 73.12, 33.72
dx_m = (E - W) * 111320 * np.cos(np.radians(33.75)) / w
X = (np.arange(w) + .5) / w * (E - W) + W
Y = N - (np.arange(h) + .5) / h * (N - S)
xx = (X[None, :] - lon0) * 111320 * np.cos(np.radians(33.75))
yy = (Y[:, None] - lat0) * 110570
# head to the north-east, 40 m/min head rate, ellipse with e = .85 (fire at the rear focus)
az = np.radians(45); e = .85; ros_h = 40.0
r = np.hypot(xx, yy); ang = np.arctan2(xx, yy); th = ang - az
rate = ros_h * (1 - e) / (1 - e * np.cos(th))
t = r / np.maximum(rate, 1e-3)                         # minutes
t = t * (1 + 0.05 * np.sin(xx / 150) * np.cos(yy / 170))
t_end = 8 * 60.0
burnt = t <= t_end
a8 = np.where(burnt, np.clip(t * 8, 0, 65534), 65535).astype("<u2")
a8.tofile(os.path.join(d, "arrival.bin"))
inten = np.where(burnt, np.clip((300 + 4000 * np.exp(-np.abs(t - t.max() * 0) / 400) * (rate / ros_h)) / 40, 0, 255), 0).astype(np.uint8)
inten.tofile(os.path.join(d, "intensity.bin"))
prob = np.where(burnt, 100, np.clip(100 * np.exp(-(t - t_end) / 120), 0, 99)).astype(np.uint8)
prob.tofile(os.path.join(d, "prob.bin"))
ha = dx_m ** 2 / 1e4
mins = np.arange(30, t_end + 30, 30)
area = [[float(m), round(float((t <= m).sum() * ha), 1)] for m in mins]
ros = [[float(m), round(float(ros_h * (0.6 + 0.4 * np.sin(m / 90))), 2)] for m in mins]
fl = [[float(m), round(float(0.0775 * (500 + 20 * m) ** 0.46), 2)] for m in mins]
nh = 9
cam = []
for tm in np.arange(0, t_end + 4, 4.0):
    k = min(tm / t_end, 1)
    cam.append([float(tm), round(lon0 + 0.1 * k, 5), round(lat0 + 0.09 * k, 5), round(13.4 - 1.6 * k, 2), round(45 * k, 1)])
axis = [[float(tm), round(lon0 + 0.1 * tm / t_end, 5), round(lat0 + 0.09 * tm / t_end, 5), round(900 + 400 * tm / t_end, 1), round(14 * tm / t_end, 3)] for tm in np.linspace(0, t_end, 30)]
meta = dict(id=rid, bounds=[W, S, E, N], size=[w, h], t_scale=8.0, never=65535, hours=8, t_end_min=t_end, start="2026-06-05T01:00:00",
            ignitions=[[lon0, lat0]], spots=[[lon0 + .06, lat0 + .05, 200.0], [lon0 + .08, lat0 + .07, 300.0]], camera=cam, axis=axis,
            events=[dict(t=0.0, kind="ignition", label="Ignition", lon=lon0, lat=lat0), dict(t=200.0, kind="spot", label="First spot fire", lon=lon0 + .06, lat=lat0 + .05),
                    dict(t=300.0, kind="growth", label="Fastest growth", lon=lon0 + .07, lat=lat0 + .06)],
            cell_m=round(dx_m, 1), half_km=20, spotting=True, has_prob=True,
            ensemble=dict(members=24, area_ha_p10=1000.0, area_ha_median=1500.0, area_ha_p90=2100.0, wind_speed_sd_pct=20, wind_dir_sd_deg=12, dead_moisture_sd_pct=15, ndvi_sd=0.03, seconds=1.0),
            warnings=["SYNTHETIC test run: analytic ellipse, not produced by the simulator."], interventions=None, label="synthetic",
            params=dict(lat=lat0, lon=lon0, hours=8),
            weather=dict(source="synthetic", conditioning="none", temp=[34.0] * nh, rh=[20.0] * nh, wind=[7.0] * nh, wdir=[225.0] * nh, rain=[0.0] * nh, dead1h=[4.1] * nh),
            stats=dict(burned_ha=round(float(burnt.sum() * ha), 1), max_head_ros_m_min=ros_h, max_intensity_kw_m=4300.0, spot_fires=2, area_series=area, ros_series=ros,
                       flame_series=fl, by_landcover_ha={"trees": 400.0, "rangeland": 900.0}, by_fuel_ha={"GR2 low load, dry climate grass": 900.0, "TL8 long-needle litter": 400.0},
                       builtup_burned_ha=0.0, builtup_within_1km_ha=35.0, ndvi_file="synthetic.tif", ndvi_mean=0.3, reached_edge=False, backend="cpu", solve_s=0.01,
                       iterations=None, gpu_error=None, scene_s=0.0, compute_s=0.1, axis_length_km=14.0))
json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
print("wrote", d, meta["stats"]["burned_ha"], "ha")
