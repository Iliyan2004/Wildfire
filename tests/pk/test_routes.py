"""Route tests with Flask's test client against a synthetic run (no simulation is run)."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RID = "00000000-000000-pytest"


@pytest.fixture(scope="module")
def client():
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_synth_run.py"), RID], check=True, capture_output=True)
    from pkfire import app as appmod
    appmod.app.config["TESTING"] = True
    yield appmod.app.test_client()
    import shutil
    shutil.rmtree(os.path.join(ROOT, "data", "runs", RID), ignore_errors=True)


def test_pages(client):
    land = client.get("/").data
    assert b"PAKFIRE" in land and b'href="/app"' in land and b'href="/instant"' in land and b"/static/landing/scene.js" in land   # landing page: scene module and both buttons
    assert client.get("/static/landing/scene.js").status_code == 200 and client.get("/static/vendor/three/three.module.min.js").status_code == 200
    assert client.get("/api/pakistan.geojson").get_json()["features"][0]["properties"]["kind"] == "mask"
    assert client.get("/manual").status_code in (200, 404)                                     # manual route
    inst = client.get("/instant").data
    assert b"PAKFIRE Instant" in inst and b"__MAPBOX_TOKEN__" not in inst
    app = client.get("/app").data
    assert b"welcomeOverlay" in app and b'id="ptabs"' in app and b"__MAPBOX_TOKEN__" not in app                       # the application, with the Mapbox token inserted
    assert client.get("/static/agn-logo.png").status_code == 200
    assert client.get(f"/runs/{RID}/mapbox3d").status_code == 200
    assert client.get("/runs/not-a-run/mapbox3d").status_code == 404
    assert client.get("/runs/..%2f..%2fetc/summary.json").status_code == 404


def test_run_files(client):
    s = client.get(f"/runs/{RID}/summary.json").get_json()
    assert s["summary"]["burned_ha"] > 0 and "arrival" in s["layers"]
    for layer in ("arrival", "intensity", "prob"):
        r = client.get(f"/runs/{RID}/overlay/{layer}.png")
        assert r.status_code == 200 and r.data[:4] == b"\x89PNG" and json.loads(r.headers["X-Coordinates"])
    assert client.get(f"/runs/{RID}/overlay/nope.png").status_code == 404
    assert len(client.get(f"/api/runs/{RID}/arrival.bin").data) == 400 * 400 * 2


def test_reports_and_compare(client):
    assert client.get(f"/runs/{RID}/report.pdf").data[:5] == b"%PDF-"
    assert client.get(f"/batch-report.pdf?ids={RID}").data[:5] == b"%PDF-"
    c = client.post("/compare-runs", json=dict(a=RID, b=RID)).get_json()
    assert c["only_a_ha"] == 0 and c["only_b_ha"] == 0 and c["both_ha"] == c["area_a_ha"]


def test_validate_observed(client):
    obs = dict(type="FeatureCollection", features=[dict(type="Feature", properties={}, geometry=dict(type="Polygon", coordinates=[[[73.1, 33.7], [73.25, 33.7], [73.25, 33.82], [73.1, 33.82], [73.1, 33.7]]]))])
    import io
    r = client.post(f"/runs/{RID}/validate-observed", data=dict(file=(io.BytesIO(json.dumps(obs).encode()), "obs.geojson")), content_type="multipart/form-data").get_json()
    assert 0 < r["IoU"] < 1 and 0 < r["Dice"] < 1
    miss = dict(obs); miss["features"][0]["geometry"]["coordinates"] = [[[60, 24], [60.1, 24], [60.1, 24.1], [60, 24.1], [60, 24]]]
    r = client.post(f"/runs/{RID}/validate-observed", data=dict(file=(io.BytesIO(json.dumps(miss).encode()), "obs.geojson")), content_type="multipart/form-data")
    assert r.status_code == 400


def test_exposure(client):
    e = client.get(f"/runs/{RID}/exposure.json").get_json()
    assert e["population"]["available"] and e["population"]["in_burn"] > 0
    assert e["facilities"]["hospitals"]["total_in_window"] >= e["facilities"]["hospitals"]["burned"]
    a = [i["arrival_min"] for i in e["facilities"]["hospitals"]["items"] if i["status"] == "burned"]
    assert a == sorted(a)


def test_evacuation(client):
    e = client.get(f"/runs/{RID}/evacuation.json").get_json()
    if "error" in e:                                                    # road database not reachable from this machine
        assert "road" in e["error"]
        return
    assert e["graph"]["vertices"] > 100
    for o in e["origins"]:
        assert o["status"] in ("escapes", "cut off", "escapes only if leaving earlier", "no road within 2 km")
        if o["status"] == "escapes":
            assert o["travel_min"] > 0 and o["latest_departure_min"] >= o["depart_min"] - 8


def test_area_and_monitor(client):
    sq = dict(type="Polygon", coordinates=[[[73.0, 33.6], [73.2, 33.6], [73.2, 33.8], [73.0, 33.8], [73.0, 33.6]]])
    a = client.post("/area/polygon", json=dict(geometry=sq)).get_json()
    assert a["district"]["name"] == "Islamabad" and a["sim_possible"]
    assert client.post("/area/polygon", json=dict(geometry=dict(type="Polygon", coordinates=[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]))).status_code == 400
    w = client.post("/area/weather-series", json=dict(lat=33.7, lon=73.1, hours=3, wind=dict(speed_ms=8, from_deg=250, temp_c=36, rh=20))).get_json()
    assert len(w["wind"]) == 4 and w["source"] == "manual"
    assert client.post("/run", json=dict(params={})).status_code == 400            # no ignition
    n0 = len(client.get("/monitor/list").get_json())
    client.post("/monitor/watch", json=dict(centre=[73.1, 33.7], radius_km=10, every_h=6, name="pytest"))
    lst = client.get("/monitor/list").get_json()
    assert len(lst) == n0 + 1
    client.post("/monitor/delete", json=dict(id=lst[-1]["id"]))
    assert len(client.get("/monitor/list").get_json()) == n0


def test_mitigation_needs_interventions(client):
    r = client.post("/area/mitigation", json=dict(ignition=[73.1, 33.7], params={}, interventions={}))
    assert r.status_code == 400


def test_danger_class():
    from pkfire.app import danger_class
    assert danger_class(dict(dead1h_min_72h_pct=3.5, wind_peak_72h=11, rain_prior_48h_mm=0))["cls"] == "extreme"
    assert danger_class(dict(dead1h_min_72h_pct=14, wind_peak_72h=2, rain_prior_48h_mm=12))["cls"] == "low"


def test_data_status(client):
    d = client.get("/data-status").get_json()
    assert d["missing_required"] == [] and d["ndvi_composites"] > 500


def test_perimeters_and_delete(client):
    g = client.get(f"/runs/{RID}/perimeters.geojson?step_min=120").get_json()
    ts = [f["properties"]["t_min"] for f in g["features"]]
    ar = [f["properties"]["area_ha"] for f in g["features"]]
    assert len(ts) >= 3 and ts == sorted(ts) and ar == sorted(ar)
    assert all(f["geometry"]["type"] in ("Polygon", "MultiPolygon") for f in g["features"])
    from subprocess import run
    import sys as _s
    run([_s.executable, os.path.join(ROOT, "tools", "make_synth_run.py"), "00000000-000000-todelete"], check=True, capture_output=True)
    assert client.delete("/runs/00000000-000000-todelete").status_code == 200
    assert client.get("/runs/00000000-000000-todelete/summary.json").status_code == 404
    assert client.delete("/runs/..%2f..%2fdata").status_code in (404, 405)
