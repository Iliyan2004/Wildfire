"""Screenshots of the landing page at fixed moments (software GL).
   CHROME=/path/to/chrome python tools/landing_shots.py <out_dir> [width height scale quality]
Uses the page's debug parameters: t (seconds of idle fire to pre-run), rise (0..1, the click animation, pre-run to that point), flash, scale, q (low|mid|high)."""
import os, sys
from playwright.sync_api import sync_playwright

BASE = os.environ.get("BASE", "http://localhost:8020")
OUT = sys.argv[1]
W, H = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (1500, 900)
SCALE = sys.argv[4] if len(sys.argv) > 4 else "0.6"
QUAL = sys.argv[5] if len(sys.argv) > 5 else "mid"
STATES = [("calm", dict(rise=0)), ("rise20", dict(rise=0.2)), ("rise40", dict(rise=0.4)), ("rise60", dict(rise=0.6)),
          ("rise80", dict(rise=0.8)), ("rise95", dict(rise=0.95, flash=0.3))]
only = os.environ.get("ONLY")
os.makedirs(OUT, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(args=["--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"], executable_path=os.environ["CHROME"])
    for name, q in STATES:
        if only and name not in only.split(","):
            continue
        pg = b.new_page(viewport={"width": W, "height": H})
        logs = []
        pg.on("console", lambda m: logs.append(m.text[:300]))
        pg.on("pageerror", lambda e: logs.append("PAGEERR " + str(e)[:300]))
        qs = "&".join(f"{k}={v}" for k, v in q.items()) + f"&scale={SCALE}&q={QUAL}&notune&frames=4"
        pg.goto(f"{BASE}/?{qs}")
        pg.wait_for_function("window.__stillDone === true", timeout=400000)
        pg.screenshot(path=f"{OUT}/{name}.png", timeout=120000)
        print(name, "ok", [l for l in logs if "arn" in l or "rror" in l][:4], flush=True)
        pg.close()
    b.close()
