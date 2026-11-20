"""Screenshots for the PAKFIRE user manual, with numbered callouts on the controls the text describes.
   CHROME=/path/to/chrome python tools/manual/capture.py <run_id>      (the app must be running on BASE, default http://localhost:8020)
Writes docs/manual/shots/<name>.png. Each shot lists CSS selectors; their boxes get an orange outline and a number."""
import os, sys
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

BASE = os.environ.get("BASE", "http://localhost:8020")
RID = sys.argv[1]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "manual", "shots")
os.makedirs(OUT, exist_ok=True)
W, H = 1500, 900
ARGS = ["--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"]
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except Exception:
    FONT = ImageFont.load_default()


def boxes(frame, sels):
    out = []
    for s in sels:
        b = frame.evaluate("""(s) => { const e = document.querySelector(s); if (!e) return null; const r = e.getBoundingClientRect();
                              return r.width > 0 ? [r.left, r.top, r.width, r.height] : null; }""", s)
        out.append(b)
    return out


def snap(pg, name, sels=(), frame=None, offset=(0, 0)):
    path = os.path.join(OUT, name + ".png")
    pg.screenshot(path=path, timeout=180000)
    bx = boxes(frame or pg, sels)
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    for k, b in enumerate(bx, 1):
        if not b:
            print("  missing", sels[k - 1]); continue
        x, y, w, h = b[0] + offset[0], b[1] + offset[1], b[2], b[3]
        d.rounded_rectangle([x - 3, y - 3, x + w + 3, y + h + 3], radius=6, outline=(232, 89, 12), width=3)
        cx, cy = min(max(x + w - 2, 14), W - 14), max(y - 4, 14)
        d.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], fill=(232, 89, 12), outline=(255, 255, 255), width=2)
        t = str(k); tw = d.textlength(t, font=FONT); d.text((cx - tw / 2, cy - 9), t, fill="white", font=FONT)
    im.save(path, optimize=True)
    print("shot", name, flush=True)


with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS, executable_path=os.environ.get("CHROME"))
    if not os.environ.get("SKIP_APP"):
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto(BASE + "/app", wait_until="domcontentloaded", timeout=180000); pg.wait_for_function("typeof map !== 'undefined' && map.isStyleLoaded()", timeout=240000); pg.wait_for_timeout(4000)
        snap(pg, "welcome", ["#welcomeOverlay .action", "#welcomeOverlay .wr .secondary:nth-of-type(1)", "#welcomeOverlay .wr .secondary:nth-of-type(2)"])
        pg.evaluate("closeWelcome()"); pg.wait_for_timeout(800)
        snap(pg, "screen", ["#geocoderSlot", ".layer-switch", ".hdr-nav", "#queueChip", "#themeToggle", ".mapboxgl-ctrl-top-left", ".maptools", "#ptabs", "#viewShowBtn"])
        pg.evaluate("setIgnition(73.06, 33.745); map.jumpTo({center:[73.06,33.745], zoom:10.5})"); pg.wait_for_timeout(9000)
        snap(pg, "fire_tab", ["#bIgn", "#bDraw", "#areaInfo", "#ctxBox"])
        pg.evaluate("document.querySelector('#rightCol').scrollTop = 560"); pg.wait_for_timeout(800)
        snap(pg, "fire_weather", ["#wSrc", "#startInput", "#hoursInput", "#wLoadBtn"])
        pg.evaluate("document.querySelector('#rightCol').scrollTop = 1180"); pg.wait_for_timeout(800)
        snap(pg, "fire_solver", ["#winSel", "#calSel", "#spotChk", "#asmChk", "#runBtn"])
        pg.evaluate(f"openRun('{RID}')"); pg.wait_for_timeout(12000)
        fr = [f for f in pg.frames if "mapbox3d" in f.url][0]
        fr.wait_for_function("typeof S !== 'undefined' && S.R", timeout=180000)
        fr.evaluate("pause(); seek(S.meta.t_end_min*0.55)"); pg.wait_for_timeout(5000)
        pg.evaluate("document.querySelector('#rightCol').scrollTop = 0"); pg.wait_for_timeout(500)
        snap(pg, "result", ["#runOut .stat-row", "#viewExpandBtn", "#viewHideBtn", "#v3dFull", "#legend2d"])
        pg.evaluate("toggleViewMain()"); pg.wait_for_timeout(6000)
        off = pg.evaluate("(()=>{const r=document.getElementById('view3dFrame').getBoundingClientRect();return [r.left,r.top]})()")
        snap(pg, "view3d", [".tiles", "#playPauseBtn", "#timeSlider", "#speedSelect", "#chartBtn", "#followBtn", "#exportToggle"], frame=fr, offset=off)
        pg.evaluate("toggleViewMain()"); pg.wait_for_timeout(1500)
        pg.evaluate("showPane('runs')"); pg.wait_for_timeout(1200)
        snap(pg, "runs_tab", ["#layerPick", "#repLink", "#tlLink", "#cmpA", "#hist"])
        pg.evaluate("showPane('impact'); impactLoad('expo')"); pg.wait_for_timeout(12000)
        snap(pg, "impact_tab", ["#impExpoBtn", "#impExpoBox", "#impLead", "#impEvacBtn"])
        pg.evaluate("showPane('mitigate')"); pg.wait_for_timeout(1000)
        snap(pg, "mitigate_tab", ["#bBreak", "#sugBrk", "#bTreat", "#trMode", "#mitBtn"])
        pg.evaluate("showPane('discover')"); pg.wait_for_timeout(1000)
        snap(pg, "discover_tab", ["#dMetric", "#distBox", "#hMetric", "#wRad", "#wBtn"])
        pg.evaluate("showPane('fire'); toggleTheme()"); pg.wait_for_timeout(2000)
        snap(pg, "light", ["#themeToggle"])
        pg.evaluate("toggleTheme()")
        pg.close()
    # Instant
    pg = b.new_page(viewport={"width": W, "height": H})
    pg.goto(BASE + "/instant", wait_until="domcontentloaded", timeout=180000); pg.wait_for_function("typeof map !== 'undefined' && map.isStyleLoaded()", timeout=240000); pg.wait_for_timeout(3000)
    snap(pg, "instant_welcome", ["#wSearch", "#wChips", "#wGo"])
    pg.close(); pg = b.new_page(viewport={"width": W, "height": H})
    pg.goto(BASE + "/instant#ign=73.06,33.745&h=6", wait_until="domcontentloaded", timeout=180000); pg.wait_for_function("document.querySelectorAll('#stats .stat').length>2", timeout=240000); pg.wait_for_timeout(4000)
    snap(pg, "instant_result", ["#barSearch", "#pChips", "#ws", "#wd", "#rh", "#hrs", "#stats", "#legend", "#fullLink2"])
    pg.close()
    # start page
    pg = b.new_page(viewport={"width": W, "height": H})
    pg.goto(BASE + "/?q=low&scale=0.6&notune&frames=4&rise=0", wait_until="domcontentloaded", timeout=180000); pg.wait_for_function("window.__stillDone === true", timeout=400000)
    snap(pg, "landing", [])
    b.close()
