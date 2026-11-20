"""Headless-browser smoke test of the PAKFIRE page: loads it, places an ignition, opens a run's 3D scene, scrubs, screenshots,
and reports console errors. Needs a Playwright browser build matching the installed package (or CHROME=/path/to/chrome).
Usage: python tools/smoke.py <run_id> [out.png]   (use a synthetic run from tools/make_synth_run.py for a quick check)"""
import json, os, sys
from playwright.sync_api import sync_playwright
BASE = os.environ.get("BASE", "http://localhost:8020")
RID = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "smoke.png"
args = ["--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"]
with sync_playwright() as p:
    b = p.chromium.launch(args=args, executable_path=os.environ.get("CHROME"))
    pg = b.new_page(viewport={"width": 1500, "height": 900}); msgs = []
    pg.on("console", lambda m: m.type == "error" and msgs.append(m.text[:240])); pg.on("pageerror", lambda e: msgs.append("PAGEERR " + str(e)[:300]))
    pg.goto(BASE + "/app"); pg.wait_for_function("typeof map !== 'undefined' && map.isStyleLoaded()", timeout=240000)
    pg.evaluate("setIgnition(73.06, 33.745)"); pg.wait_for_timeout(1500)
    print("ignition set:", pg.evaluate("ign"))
    pg.evaluate(f"openRun('{RID}')"); pg.wait_for_timeout(8000)
    fr = [f for f in pg.frames if "mapbox3d" in f.url][0]
    fr.wait_for_function("typeof S !== 'undefined' && S.R", timeout=120000)
    fr.evaluate("pause(); setFollow(true); seek(S.meta.t_end_min*0.6)"); pg.wait_for_timeout(15000)
    print(json.dumps(fr.evaluate("({T:S.T, nb:S.R.nb, clock:document.getElementById('timeSub').textContent, mode:S.mode})")))
    pg.screenshot(path=OUT)
    for m in dict.fromkeys(msgs): print("ERR", m)
    b.close()
