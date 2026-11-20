"""Interface review of the PAKFIRE page in a headless browser (desktop and phone width).
   CHROME=/path/to/chrome python tools/ui_review.py <run_id> <out_dir>
Checks: console and page errors, failed requests, horizontal overflow of the page and of panels, controls without an accessible name, targets smaller than
24 px, text smaller than 11 px, every tier and detail panel opening without error, the exposure and evacuation panels filling in, and the 3D scene at
close zoom (true-scale flames). The welcome card, light theme and closed drawer are also captured. Screenshots are written next to the JSON report."""
import json, os, sys
from playwright.sync_api import sync_playwright

BASE = os.environ.get("BASE", "http://localhost:8020")
RID, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
ARGS = ["--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"]

AUDIT = """() => {
  const out = {overflowX: document.documentElement.scrollWidth > innerWidth + 1, wide: [], noname: [], small: [], tiny: []};
  const vis = (e) => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e); return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  for (const e of document.querySelectorAll('body *')) {
    if (!vis(e)) continue;
    const r = e.getBoundingClientRect();
    if (r.right > innerWidth + 2 && !e.closest('.mapboxgl-map') && !e.closest('iframe')) out.wide.push((e.id || e.className || e.tagName).toString().slice(0, 40));
  }
  for (const e of document.querySelectorAll('button, input, select, textarea, a[href]')) {
    if (!vis(e)) continue;
    const name = (e.getAttribute('aria-label') || e.title || e.innerText || e.value || (e.labels && e.labels[0] && e.labels[0].innerText) || e.placeholder || '').trim();
    if (!name) out.noname.push((e.id || e.className || e.tagName).toString().slice(0, 40));
    const r = e.getBoundingClientRect();
    if ((r.width < 24 || r.height < 24) && e.type !== 'range' && e.type !== 'checkbox' && e.type !== 'radio') out.small.push(((e.id || e.innerText || e.tagName) + ' ' + Math.round(r.width) + 'x' + Math.round(r.height)).slice(0, 50));
  }
  for (const e of document.querySelectorAll('body *')) {
    if (!vis(e) || !e.childNodes.length) continue;
    const t = [...e.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim().length > 2);
    if (t && parseFloat(getComputedStyle(e).fontSize) < 11) out.tiny.push(((e.id || e.className || e.tagName) + ' ' + getComputedStyle(e).fontSize).toString().slice(0, 50));
  }
  for (const k of Object.keys(out)) if (Array.isArray(out[k])) out[k] = [...new Set(out[k])].slice(0, 25);
  return out;
}"""


def review(p, name, vp):
    b = p.chromium.launch(args=ARGS, executable_path=os.environ.get("CHROME"))
    pg = b.new_page(viewport=vp)
    errs, failed = [], []
    pg.on("console", lambda m: m.type == "error" and errs.append(m.text[:200]))
    pg.on("pageerror", lambda e: errs.append("PAGEERR " + str(e)[:240]))
    pg.on("requestfailed", lambda r: failed.append(r.url[:120]))
    pg.on("response", lambda r: r.status >= 400 and failed.append(f"{r.status} {r.url[:110]}"))
    pg.goto(BASE + "/app")
    pg.wait_for_function("typeof map !== 'undefined' && map.isStyleLoaded()", timeout=240000)
    rep = dict(viewport=vp)
    pg.screenshot(path=f"{OUT}/{name}_welcome.png")
    pg.evaluate("closeWelcome()")
    pg.wait_for_timeout(600)
    rep["initial"] = pg.evaluate(AUDIT)
    pg.screenshot(path=f"{OUT}/{name}_initial.png")
    for mode in ("satellite", "streets", "dark"):
        pg.evaluate(f"setMapLayer('{mode}')")
    pg.evaluate("toggleTheme()")
    pg.wait_for_timeout(600)
    rep["light"] = pg.evaluate(AUDIT)
    pg.screenshot(path=f"{OUT}/{name}_light.png")
    pg.evaluate("toggleTheme()")
    pg.evaluate("toggleDrawer()")
    pg.wait_for_timeout(700)
    pg.screenshot(path=f"{OUT}/{name}_drawer_closed.png")
    pg.evaluate("toggleDrawer()")
    pg.wait_for_timeout(500)
    pg.evaluate("setIgnition(73.06, 33.745)")
    pg.wait_for_timeout(1500)
    tiers = pg.evaluate("[...document.querySelectorAll('[data-tier],.tier,button.tab')].map(e=>e.id||e.innerText).slice(0,12)")
    rep["tier_buttons"] = tiers
    pg.evaluate(f"openRun('{RID}')")
    pg.wait_for_timeout(9000)
    pg.evaluate("showPane('impact'); impactLoad('expo'); impactLoad('evac')")
    pg.wait_for_timeout(15000)
    rep["exposure_text"] = pg.evaluate("$('impExpoBox').innerText")
    rep["evac_text"] = pg.evaluate("$('impEvacBox').innerText")
    pg.screenshot(path=f"{OUT}/{name}_pane_impact.png")
    for pane in ("fire", "runs", "mitigate", "discover"):
        pg.evaluate(f"showPane('{pane}')"); pg.wait_for_timeout(700)
        rep[f"pane_{pane}"] = pg.evaluate(AUDIT)
        pg.screenshot(path=f"{OUT}/{name}_pane_{pane}.png")
    pg.evaluate("showPane('runs')")
    rep["result"] = pg.evaluate(AUDIT)
    pg.screenshot(path=f"{OUT}/{name}_result.png")
    if name == "desktop":
        fr = [f for f in pg.frames if "mapbox3d" in f.url][0]
        fr.wait_for_function("typeof S !== 'undefined' && S.R", timeout=120000)
        fr.evaluate("pause(); setFollow(false); seek(S.meta.t_end_min*0.5)")
        pg.wait_for_timeout(8000)
        head = fr.evaluate("(()=>{const R=S.R,m=S.meta;const k=R.ord[Math.min(R.nb-1, Math.max(0, (R.ord.length*0.5)|0))];return [m.bounds[0]+((k%R.w)+0.5)*(m.bounds[2]-m.bounds[0])/R.w, m.bounds[3]-(((k/R.w)|0)+0.5)*(m.bounds[3]-m.bounds[1])/R.h]})()")
        fr.evaluate(f"map.jumpTo({{center:[{head[0]},{head[1]}],zoom:16.8,pitch:55}})")
        pg.wait_for_timeout(12000)
        pg.screenshot(path=f"{OUT}/desktop_flames_true_scale.png")
        fr.evaluate("S.fh = 8")
        pg.wait_for_timeout(5000)
        pg.screenshot(path=f"{OUT}/desktop_flames_x8.png")
    rep["console_errors"] = list(dict.fromkeys(errs))
    rep["failed_requests"] = list(dict.fromkeys(failed))[:15]
    b.close()
    return rep


with sync_playwright() as p:
    out = {}
    for name, vp in [x for x in (("desktop", {"width": 1500, "height": 900}), ("phone", {"width": 390, "height": 800})) if x[0] in os.environ.get("ONLY", "desktop,phone")]:
        out[name] = review(p, name, vp)
        json.dump(out, open(f"{OUT}/ui_review.json", "w"), indent=1)
        print("done", name, flush=True)
