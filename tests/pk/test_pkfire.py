"""PAKFIRE engine tests: physics behaviour, moisture, crosswalk and GPU == CPU."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pkfire import fuels, kernel_cpu, kernel_gpu, moisture

LEAD = 48


def tables(rh=20.0, temp=32.0, hours=8, ndvi=0.3):
    n = LEAD + hours + 2
    md = moisture.dead_moisture(np.full(n, temp), np.full(n, rh), np.zeros(n))
    tb = fuels.build_tables(md, *fuels.greenness_to_live_moisture(ndvi))
    tb["spot_ign"] = moisture.ignition_probability(np.full(n, temp), md[0]).astype(np.float32)
    tb["emc_pct"] = (md[0] * 100.0).astype(np.float32)
    return tb


def grid(n=201, dx=30.0, slope=0.0, up_az=0.0, fuel_id=102):
    g = dict(ny=n, nx=n, dx=dx, slope=np.full((n, n), slope, np.float32), upaz=np.full((n, n), up_az, np.float32),
             cont=np.ones((n, n), np.float32))
    return g, np.full((n, n), fuels.row_of(fuel_id), np.int8)


def sim(g, fuel, wind, from_deg, hours=8, rh=20.0, spot=0.0, backend="cpu", seed=1):
    t = tables(rh, hours=hours)
    n = g["nx"] // 2
    h = LEAD + hours + 2
    fn = kernel_cpu.simulate if backend == "cpu" else kernel_gpu.simulate
    arr, ros, fli, sp = fn(g, fuel, t, np.full(h, wind), np.full(h, from_deg), [(n, n, 0.0)], hours,
                           spot_rate=spot, hour0=LEAD, seed=seed)
    return arr, n


def test_wetter_fuel_is_slower_and_extinguishes():
    r0 = [tables(rh)["r0"][0, fuels.row_of(102)] for rh in (15, 70, 98)]
    assert r0[0] > r0[1] > r0[2]
    assert r0[2] < 1e-3


def test_rain_wets_dead_fuel():
    n = 24
    dry = moisture.dead_moisture(np.full(n, 35.0), np.full(n, 15.0), np.zeros(n))
    rain = np.zeros(n); rain[10:14] = 3.0
    wet = moisture.dead_moisture(np.full(n, 35.0), np.full(n, 15.0), rain)
    assert wet[0, 13] > dry[0, 13] + 0.015          # 4 h of 3 mm/h rain raises 1 h moisture (WRF-SFIRE wetting is slow: about +2 points here)
    assert wet[0, 23] < wet[0, 13]                  # and it dries again afterwards


def test_wind_pushes_fire_downwind():
    g, f = grid()
    arr, n = sim(g, f, wind=6.0, from_deg=270)      # from the west: fire runs east
    east, north, west = arr[n, n + 15], arr[n - 15, n], arr[n, n - 15]
    assert east < north < west


def test_still_air_is_symmetric():
    g, f = grid()
    arr, n = sim(g, f, wind=0.0, from_deg=0)
    d = [arr[n, n + 6], arr[n, n - 6], arr[n + 6, n], arr[n - 6, n]]      # GR2 spreads 0.5 m/min: 6 cells = 180 m is reached in 8 h
    assert np.isfinite(d).all() and max(d) / min(d) < 1.03
    diag = arr[n + 4, n + 4] * 6 / (4 * 2 ** 0.5)      # time per unit distance must match the axis directions
    assert abs(diag - d[0]) / d[0] < 0.06


def test_uphill_faster_than_downhill():
    g, f = grid(slope=0.4, up_az=0.0)               # upslope is north
    arr, n = sim(g, f, wind=0.0, from_deg=0)
    assert arr[n - 15, n] * 3 < arr[n + 15, n]


def test_non_burnable_barrier_blocks_fire():
    g, f = grid()
    n = g["nx"] // 2
    f[:, n + 20:n + 23] = 0
    arr, _ = sim(g, f, wind=6.0, from_deg=270)
    assert np.isinf(arr[n, n + 60])
    assert np.isfinite(arr[n, n - 5])


def test_spotting_can_cross_a_firebreak():
    g, f = grid(fuel_id=188)
    n = g["nx"] // 2
    f[:, n + 12:n + 14] = 0
    arr, _ = sim(g, f, wind=20.0, from_deg=270, hours=6, rh=5.0, spot=0.5)   # TL8 reaches ~1200 kW/m here (spotting needs > 800)
    assert np.isfinite(arr[n, n + 40])
    arr0, _ = sim(g, f, wind=20.0, from_deg=270, hours=6, rh=5.0, spot=0.0)
    assert np.isinf(arr0[n, n + 40])                   # without embers the break holds


def test_albini_distance_matches_reference_and_behaves():
    """The kernels' Albini (1983) distance equals the FIRELAB reference implementation (pkfire/vendor/spotting.py) and behaves physically."""
    from pkfire.vendor import spotting as ref
    for (A, B) in ((69.6, -0.818), (129.0, -1.238), (83.9, -1.04)):
        for I in (150, 400, 1000, 3000, 8000):
            for u in (2, 5, 10, 18):
                assert abs(kernel_cpu.max_spot_m(float(I), float(u), A, B) - float(ref.max_spot_distance_m(I, u, (A, B)))) < 0.5
    d = [kernel_cpu.max_spot_m(float(I), 10.0, 69.6, -0.818) for I in (200, 1000, 4000)]
    assert d[0] <= d[1] <= d[2] and d[2] > d[0]
    w = [kernel_cpu.max_spot_m(2000.0, float(u), 69.6, -0.818) for u in (3, 6, 12)]
    assert w[0] < w[1] < w[2]
    assert kernel_cpu.max_spot_m(2000.0, 10.0, 0.0, 0.0) == 0.0          # no coefficient: grass and crop fires do not spot


def test_grass_and_crop_fuels_never_spot():
    g, f = grid(fuel_id=102)
    n = g["nx"] // 2
    f[:, n + 12:n + 14] = 0
    arr, _ = sim(g, f, wind=25.0, from_deg=270, hours=6, rh=4.0, spot=0.9)
    assert np.isinf(arr[n, n + 40])


def test_ignition_probability_is_physical():
    p = moisture.ignition_probability(np.array([30.0, 30.0, 30.0]), np.array([0.03, 0.08, 0.20]))
    assert p[0] > p[1] > p[2] >= 0 and p[0] <= 1.0


def test_length_to_breadth_uses_metres_per_second():
    """Moderate wind must give a moderate ellipse: head/back ratio near Anderson's value, not ~2x more elongated."""
    g, f = grid(n=401)
    arr, n = sim(g, f, wind=5.0, from_deg=270, hours=12)
    east, west = arr[n, n + 20] - 0.0, arr[n, n - 6]
    assert np.isfinite(east) and np.isfinite(west)
    ros_head = 20 * 30.0 / east
    ros_back = 6 * 30.0 / west
    assert 1.5 < ros_head / ros_back < 25.0         # with the mph bug this ratio exceeded 40


def test_crosswalk_marks_desert_and_water_non_burnable():
    from pkfire import terrain as t
    lc = np.array([[1, 2, 5, 7, 8, 9, 11, 11]], np.uint8)
    dem = np.array([[100, 1500, 200, 200, 200, 3000, 200, 200]], np.float32)
    ndvi = np.array([[0.3, 0.4, 0.3, 0.3, 0.1, 0.3, 0.05, 0.3]], np.float32)
    row, cont, _ = t.fuel_from_lulc(lc, dem, ndvi)
    assert list(row[0] == 0) == [True, False, False, True, True, True, True, False]
    assert row[0, 1] == fuels.row_of(fuels.F_CONIFER_LITTER)


@pytest.mark.skipif(not kernel_gpu.available(), reason="no CUDA device")
def test_gpu_equals_cpu():
    n = 301
    rng = np.random.default_rng(0)
    g = dict(ny=n, nx=n, dx=30.0, slope=np.clip(rng.normal(0.15, 0.1, (n, n)), 0, 1).astype(np.float32),
             upaz=np.full((n, n), 0.7, np.float32), cont=np.ones((n, n), np.float32))
    f = np.full((n, n), fuels.row_of(122), np.int8)
    f[:, 200:204] = 0
    f[80:100, 80:240] = fuels.row_of(188)
    a1, _ = sim(g, f, 7.0, 250.0, backend="cpu")
    a2, _ = sim(g, f, 7.0, 250.0, backend="gpu")
    m1, m2 = np.isfinite(a1) & (a1 <= 480), np.isfinite(a2) & (a2 <= 480)
    assert (m1 ^ m2).sum() == 0
    assert np.abs(a1[m1 & m2] - a2[m1 & m2]).max() < 0.01


def _run_tl8(wind, rh, crown=True, hours=4):
    g, f = grid(fuel_id=188)
    n = g["nx"] // 2
    t = tables(rh, hours=hours)
    if not crown:
        t = dict(t)
        t["crown_cbd"] = np.zeros_like(t["crown_cbd"])
    h = LEAD + hours + 2
    arr, ros, fli, sp = kernel_cpu.simulate(g, f, t, np.full(h, wind), np.full(h, 270.0), [(n, n, 0.0)], hours, spot_rate=0.0, hour0=LEAD)
    return arr, ros, fli, n


def fuels_i0():
    return float(fuels.build_tables(moisture.dead_moisture(np.full(3, 30.0), np.full(3, 20.0), np.zeros(3)), 0.5, 0.9)["crown_i0"][fuels.row_of(188)])


def test_crown_fire_needs_intensity_and_raises_rate():
    ros_w, fli_w = _run_tl8(3.0, 25.0, True)[1:3]
    ros_w0, fli_w0 = _run_tl8(3.0, 25.0, False)[1:3]
    assert np.allclose(ros_w, ros_w0) and np.allclose(fli_w, fli_w0)             # a weak fire stays on the surface: nothing changes
    ros_s, fli_s = _run_tl8(20.0, 5.0, True)[1:3]
    ros_s0, fli_s0 = _run_tl8(20.0, 5.0, False)[1:3]
    assert fli_s0.max() > fuels_i0()                                             # the surface fire is intense enough to initiate
    assert ros_s.max() > ros_s0.max() * 1.2 and fli_s.max() > fli_s0.max() * 1.5   # active crown fire spreads faster and burns hotter


def test_van_wagner_threshold_value():
    assert abs(fuels_i0() - (0.010 * fuels.CROWN[188][0] * (460 + 25.9 * 100)) ** 1.5) < 1.0


def test_suppression_and_road_factors_reduce_spread():
    from pkfire import adjust
    n = 60
    sc = dict(ny=n, nx=n, dx=50.0, fuel=np.ones((n, n), np.int8), cont=np.ones((n, n), np.float32), lc=np.full((n, n), 11, np.uint8), dem=np.zeros((n, n), np.float32))
    sc["roadf"] = np.ones((n, n), np.float32); sc["roadf"][:, 30] = 0.1; sc["roads_ok"] = True
    w = dict(wind=np.ones(10), wdir=np.zeros(10), temp=np.ones(10), rh=np.ones(10), precip=np.zeros(10), source="x")
    a = adjust.apply(sc, w, dict(road=1.0, suppress=0.6), 4, 0)
    assert a["cont"][10, 30] < 0.1 and a["cont"][10, 2] > a["cont"][10, 29]       # the road cell is slow; far from the road suppression fades
    assert a["cont"][10, 2] > 0.9 * 1.0 * (1 - 0.6 * np.exp(-28 * 50 / 2000.0)) - 1e-3


def test_assimilated_detections_start_later_and_seed_more_cells():
    g, f = grid()
    n = g["nx"] // 2
    t = tables(20.0, hours=6)
    h = LEAD + 8
    one, *_ = kernel_cpu.simulate(g, f, t, np.full(h, 6.0), np.full(h, 270.0), [(n, n, 0.0)], 6, spot_rate=0.0, hour0=LEAD)
    two, *_ = kernel_cpu.simulate(g, f, t, np.full(h, 6.0), np.full(h, 270.0), [(n, n, 0.0), (n, n + 25, 120.0)], 6, spot_rate=0.0, hour0=LEAD)
    assert np.isfinite(two[n, n + 25]) and two[n, n + 25] <= 120.0 + 1e-3          # a detection seeded at 120 min is burning by then
    assert np.isfinite(two).sum() >= np.isfinite(one).sum()


# ---------------------------------------------------------------- mass-conserving terrain wind (pkfire/windfield.py)
def _hill(n=120, dx=240.0, H=400.0, sig=2500.0):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    c = n / 2
    return (H * np.exp(-((x - c) ** 2 + (y - c) ** 2) * dx * dx / (2 * sig ** 2))).astype(np.float32), dx, int(c)


def test_windfield_flat_terrain_changes_nothing():
    from pkfire import windfield as wf
    r, a, info = wf.solve(np.zeros((60, 60), np.float32), 240.0, 225.0, np.full((60, 60), 0.1, np.float32), backend="cpu")
    assert np.allclose(r, 1.0, atol=1e-3) and np.abs(a).max() < 0.1


def test_windfield_is_mass_conserving_and_matches_theory_on_a_hill():
    from pkfire import windfield as wf
    dem, dx, c = _hill()
    r, a, info = wf.solve(dem, dx, 270.0, np.full(dem.shape, 0.1, np.float32), backend="cpu")
    assert info["max_div_after"] < 1e-3 * info["max_div_before"]                 # divergence removed by 3 orders of magnitude
    jh = 1 + 1.6 * 400.0 / (1.18 * 2500.0)                                       # Jackson & Hunt style crest speed-up (1.22)
    assert abs(r[c, c] - jh) < 0.08, r[c, c]
    assert r[c + 10, c] > 1.05 and r[c + 10, c] < r[c, c]                        # the flanks speed up, but less than the crest
    assert r[5, 5] < 1.02 and r[5, c] < 1.02                                     # far from the hill the wind is unchanged
    assert np.abs(a).max() > 2.0                                                 # the flow is deflected around the hill


def test_windfield_mirror_symmetry_and_gpu_equals_cpu():
    from pkfire import windfield as wf
    dem, dx, c = _hill(n=100)
    z0 = np.full(dem.shape, 0.1, np.float32)
    west, rw, _ = wf.solve(dem, dx, 270.0, z0, backend="cpu")[0], None, None
    east = wf.solve(dem, dx, 90.0, z0, backend="cpu")[0]
    d = np.abs(west - east[:, ::-1])
    assert d.mean() < 0.01 and d[5:-5, 5:-5].max() < 0.05                        # mirrored winds give mirrored fields (staircase noise at the edges only)
    if kernel_gpu.available():
        g = wf.solve(dem, dx, 270.0, z0, backend="gpu")[0]
        assert np.abs(g - west).max() < 0.01


def test_wind_rotation_and_ratio_reach_the_kernel():
    g, f = grid()
    n = g["nx"] // 2
    t = tables(20.0, hours=8)
    h = LEAD + 10
    base, *_ = kernel_cpu.simulate(g, f, t, np.full(h, 6.0), np.full(h, 270.0), [(n, n, 0.0)], 8, spot_rate=0.0, hour0=LEAD)
    g2 = dict(g, wrot=np.full(g["slope"].shape, np.radians(90.0), np.float32))   # wind turned clockwise by 90 degrees: from the north
    turned, *_ = kernel_cpu.simulate(g2, f, t, np.full(h, 6.0), np.full(h, 270.0), [(n, n, 0.0)], 8, spot_rate=0.0, hour0=LEAD)
    assert base[n, n + 12] < base[n - 12, n] and turned[n + 12, n] < turned[n, n + 12]   # east head becomes a south head
    g3 = dict(g, wmod=np.full(g["slope"].shape, 1.5, np.float32))
    fast, *_ = kernel_cpu.simulate(g3, f, t, np.full(h, 6.0), np.full(h, 270.0), [(n, n, 0.0)], 8, spot_rate=0.0, hour0=LEAD)
    assert fast[n, n + 12] < base[n, n + 12]


@pytest.mark.skipif(not kernel_gpu.available(), reason="no CUDA device")
def test_batched_ensemble_equals_separate_runs():
    n = 161
    rng = np.random.default_rng(3)
    g = dict(ny=n, nx=n, dx=30.0, slope=np.clip(rng.normal(0.15, 0.1, (n, n)), 0, 1).astype(np.float32), upaz=np.full((n, n), 0.7, np.float32), cont=np.ones((n, n), np.float32))
    f = np.full((n, n), fuels.row_of(122), np.int8)
    f[:, 110:113] = 0
    f[60:80, 40:120] = fuels.row_of(188)
    hours, h = 6, LEAD + 8
    members = []
    for k, (ws, wd, rh) in enumerate(((7.0, 250.0, 20.0), (4.0, 200.0, 30.0), (10.0, 300.0, 15.0))):
        members.append((tables(rh, hours=hours), np.full(h, ws), np.full(h, wd)))
    ign = [(80, 80, 0.0)]
    single = [kernel_cpu.simulate(g, f, t, w, d, ign, hours, spot_rate=0.0, hour0=LEAD)[0] for t, w, d in members]
    batch = kernel_gpu.simulate_batch(g, f, [m[0] for m in members], [m[1] for m in members], [m[2] for m in members], ign, hours, spot_rate=0.0, hour0=LEAD)
    assert batch.shape == (3, n, n)
    for k in range(3):
        a, b = np.isfinite(single[k]) & (single[k] <= hours * 60), np.isfinite(batch[k]) & (batch[k] <= hours * 60)
        assert (a ^ b).sum() == 0
        assert np.abs(single[k][a] - batch[k][a]).max() < 0.01
    assert len({int((np.isfinite(b) & (b <= hours * 60)).sum()) for b in batch}) == 3        # the members really differ


def _plane_scene(facing_south, deg=20.0, n=60, dx=50.0):
    from rasterio.transform import from_origin
    rows = np.arange(n)[:, None] * np.ones((1, n))
    h = (n - rows) * dx * math.tan(math.radians(deg)) if facing_south else rows * dx * math.tan(math.radians(deg))      # high ground north (south-facing) or south
    return dict(ny=n, nx=n, dx=dx, dem=(2000 + h).astype(np.float32), slope=np.full((n, n), math.tan(math.radians(deg)), np.float32),
                upaz=np.full((n, n), 0.0 if facing_south else math.pi, np.float32), lc=np.full((n, n), 11, np.uint8), ndvi=np.full((n, n), 0.2, np.float32),
                fuel=np.ones((n, n), np.int8), crs="EPSG:32643", transform=from_origin(520000, 3760000, dx, -dx))


def test_sun_position_summer_solstice_noon():
    import datetime as dt
    from pkfire import microclimate as mc
    el, az = mc.sun_position(34.0, 73.0, dt.datetime(2026, 6, 21, 7, 8))
    assert 77 < math.degrees(el) < 81 and abs(math.degrees(az) - 180) < 25
    el, az = mc.sun_position(34.0, 73.0, dt.datetime(2026, 6, 21, 20, 0))
    assert math.degrees(el) < 0


def test_sun_facing_slope_is_drier_than_shaded_slope():
    import datetime as dt
    from pkfire import microclimate as mc
    n = 100
    w = dict(temp=np.full(n, 28.0), rh=np.full(n, 35.0), wind=np.full(n, 3.0), wdir=np.zeros(n), precip=np.zeros(n))
    start = dt.datetime(2026, 4, 10, 4, 0)
    ds, _ = mc.moisture_delta(_plane_scene(True), w, start, 12, 0)
    dn, _ = mc.moisture_delta(_plane_scene(False), w, start, 12, 0)
    assert ds.mean() < -0.0005 and dn.mean() > ds.mean() + 0.004                       # south slope drier, north slope moister than flat open ground


def test_canopy_shade_and_elevation_moisture():
    import datetime as dt
    from pkfire import microclimate as mc
    n = 100
    w = dict(temp=np.full(n, 28.0), rh=np.full(n, 35.0), wind=np.full(n, 3.0), wdir=np.zeros(n), precip=np.zeros(n))
    sc = _plane_scene(True, deg=0.001)
    sc["lc"][:, :30] = 2                                                             # trees on the left
    d, _ = mc.moisture_delta(sc, w, dt.datetime(2026, 4, 10, 4, 0), 12, 0)
    assert d[:, :30].mean() > d[:, 40:].mean() + 0.005                               # forest floor stays moister than open ground
    hi = _plane_scene(True, deg=0.001); hi["dem"] = hi["dem"] + 800                   # same place 800 m higher than the weather point
    hi["dem"][30, 30] = 2000.0
    sc2 = _plane_scene(True, deg=0.001)
    sc2["dem"][:, :] = 2000.0; sc2["dem"][:, 50:] = 2800.0                           # right half 800 m above the centre-row reference
    d2, _ = mc.moisture_delta(sc2, w, dt.datetime(2026, 4, 10, 4, 0), 12, 0)
    assert d2[:, 55:].mean() > d2[:, 5:20].mean()                                    # cooler higher ground is moister


def test_rate_multiplier_follows_moisture_and_is_unity_when_off():
    import datetime as dt
    from pkfire import microclimate as mc
    n = 100
    w = dict(temp=np.full(n, 28.0), rh=np.full(n, 35.0), wind=np.full(n, 3.0), wdir=np.zeros(n), precip=np.zeros(n))
    sc = _plane_scene(True)
    sc["fuel"][:] = fuels.row_of(2) if hasattr(fuels, "row_of") else 1
    m = mc.rate_multiplier(sc, w, dt.datetime(2026, 4, 10, 4, 0), 12, 0, 1.0)
    assert m is not None and m.mean() > 1.0                                          # a drier-than-reference slope spreads faster
    assert mc.rate_multiplier(sc, w, dt.datetime(2026, 4, 10, 4, 0), 12, 0, 0.0) is None


def test_stability_lowers_alpha_and_changes_flow():
    from pkfire import adjust, windfield
    import datetime as dt
    n = 80
    sc = _plane_scene(True)
    w = dict(temp=np.full(120, 20.0), rh=np.full(120, 50.0), wind=np.full(120, 2.0), wdir=np.zeros(120), precip=np.zeros(120))
    a_night = adjust.chan_alpha(sc, w, dt.datetime(2026, 4, 10, 17, 0), 8, 0, 1.0)     # evening and night, light wind
    assert adjust.chan_alpha(sc, w, dt.datetime(2026, 4, 10, 17, 0), 8, 0, 0.0) == adjust.NEUTRAL_ALPHA
    assert 0.1 < a_night < adjust.NEUTRAL_ALPHA
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    dem = (300 * np.exp(-(((x - 40) ** 2 + (y - 40) ** 2) / (2 * 10.0 ** 2)))).astype(np.float32)          # a hill
    z0 = np.full(dem.shape, 0.1, np.float32)
    r1, a1, _ = windfield.solve(dem, 60.0, 270.0, z0, alpha=adjust.NEUTRAL_ALPHA, backend="numpy")
    r2, a2, _ = windfield.solve(dem, 60.0, 270.0, z0, alpha=a_night, backend="numpy")
    assert np.abs(a2).max() > np.abs(a1).max() * 1.1                                   # stable air turns the flow more around the hill


def test_ensemble_widths_scale_spread():
    from pkfire import run as runmod
    w = dict(temp=np.full(100, 25.0), rh=np.full(100, 30.0), wind=np.full(100, 5.0), wdir=np.full(100, 90.0), precip=np.zeros(100))
    _, ws0, wd0 = runmod.ensemble_inputs(w, 0.2, 1.0, 12, 5, dict(wind=0.0, wdir=0.0, moist=0.0, ndvi=0.0))
    _, ws1, wd1 = runmod.ensemble_inputs(w, 0.2, 1.0, 12, 5, None)
    assert np.allclose(np.array(ws0), 5.0) and np.allclose(np.array(wd0), 90.0)
    assert np.std([x[0] for x in ws1]) > 0.3 and np.std([x[0] for x in wd1]) > 3.0


def test_pbs_district_lookup():
    from pkfire import exposure
    r = exposure.pbs_district(73.2, 34.4)                                              # Mansehra area
    assert r is None or (r["pop_2023"] > 1e5 and 0 <= r["kacha_pct"] <= 100)


def test_canopy_crown_values_and_per_cell_kernel_input():
    row = fuels.row_of(188)
    fr = np.array([[row, row, row, 1]], np.int8)
    mean = np.array([[10.0, 2.0, 12.0, 10.0]], np.float32)
    p90 = np.array([[20.0, 3.0, 12.0, 20.0]], np.float32)
    i0, cbd = fuels.canopy_crown(fr, mean, p90)
    assert abs(i0[0, 0] - (0.010 * 0.35 * 20.0 * (460 + 25.9 * 100)) ** 1.5) < 1.0       # CBH 7 m
    assert abs(cbd[0, 0] - 0.15 * 0.5) < 1e-6 and abs(cbd[0, 2] - 0.15) < 1e-6          # bulk density follows the filled share of the cell
    assert i0[0, 1] == 0 and cbd[0, 1] == 0 and cbd[0, 3] == 0                           # below 4 m, and non-pine fuel: no crown fire
    g, f = grid(fuel_id=188)
    t = tables(5.0, hours=4)
    h = LEAD + 6
    ci0, ccbd = fuels.crown_cells(g, t, f)
    assert np.allclose(ccbd, t["crown_cbd"][f.ravel()])                                  # without measured canopy the per-fuel defaults apply
    g2 = dict(g, crown_ci0=np.zeros(g["ny"] * g["nx"], np.float32), crown_cbd=np.zeros(g["ny"] * g["nx"], np.float32))
    n = g["nx"] // 2
    a = kernel_cpu.simulate(g, f, t, np.full(h, 20.0), np.full(h, 270.0), [(n, n, 0.0)], 4, spot_rate=0.0, hour0=LEAD)
    b = kernel_cpu.simulate(g2, f, t, np.full(h, 20.0), np.full(h, 270.0), [(n, n, 0.0)], 4, spot_rate=0.0, hour0=LEAD)
    assert a[2].max() > b[2].max() * 1.5                                                 # canopy-free cells carry no crown fire
