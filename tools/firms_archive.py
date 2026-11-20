"""Downloads NASA FIRMS VIIRS S-NPP active-fire detections for Pakistan, 5 days per request (the API maximum), resumable.
  python tools/firms_archive.py 2024-01-01 2026-10-20
Standard-processing (SP) archive for older dates, near-real-time (NRT) for the last ~3 months. One CSV per chunk in
data/external/firms/. The key is read from .secrets/firms_map_key and never printed."""
import datetime as dt, os, sys, time
import requests

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = open(os.path.join(HERE, ".secrets", "firms_map_key")).read().strip()
OUT = os.path.join(HERE, "data", "external", "firms")
AREA = "60.8,23.5,77.9,37.2"
start, end = (dt.date.fromisoformat(a) for a in sys.argv[1:3])
nrt_from = dt.date.today() - dt.timedelta(days=85)
d = start
while d <= end:
    n = min(5, (end - d).days + 1)
    path = os.path.join(OUT, f"viirs_snpp_{d.isoformat()}_{n}d.csv")
    if not os.path.exists(path):
        src = "VIIRS_SNPP_NRT" if d >= nrt_from else "VIIRS_SNPP_SP"
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{src}/{AREA}/{n}/{d.isoformat()}"
        for attempt in range(6):
            try:
                r = requests.get(url, timeout=90)
                if r.status_code == 200 and r.text.startswith("latitude"):
                    open(path + ".part", "w").write(r.text); os.replace(path + ".part", path)
                    print(d, src, len(r.text.splitlines()) - 1, "rows", flush=True); break
                print(d, "HTTP", r.status_code, r.text[:80].replace("\n", " "), flush=True)
            except Exception as e:
                print(d, "error", type(e).__name__, flush=True)
            time.sleep(10 * (attempt + 1))
        time.sleep(1.5)
    d += dt.timedelta(days=n)
print("finished")
