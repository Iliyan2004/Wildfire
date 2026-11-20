"""Downloads NASA FIRMS yearly country archives for Pakistan (no key needed), resumable.   python tools/firms_yearly.py [sensor ...]
Source: https://firms.modaps.eosdis.nasa.gov/data/country/<sensor>/<year>/<sensor>_<year>_Pakistan.csv  (VIIRS S-NPP from 2012, NOAA-20 from 2018)."""
import os, sys, time
import requests
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "external", "firms_yearly")
FIRST = {"viirs-snpp": 2012, "viirs-jpss1": 2018}
for sensor in (sys.argv[1:] or ["viirs-snpp"]):
    for y in range(FIRST[sensor], 2026):
        p = os.path.join(OUT, f"{sensor}_{y}_Pakistan.csv")
        if os.path.exists(p):
            continue
        url = f"https://firms.modaps.eosdis.nasa.gov/data/country/{sensor}/{y}/{sensor}_{y}_Pakistan.csv"
        for a in range(5):
            try:
                r = requests.get(url, timeout=300)
                if r.status_code == 404:
                    print(sensor, y, "not published", flush=True); break
                r.raise_for_status()
                if not r.text.startswith("latitude"):
                    raise ValueError(r.text[:80])
                open(p + ".part", "w").write(r.text); os.replace(p + ".part", p)
                print(sensor, y, len(r.text.splitlines()) - 1, "rows", flush=True); break
            except Exception as e:
                print(sensor, y, "retry", type(e).__name__, str(e)[:60], flush=True); time.sleep(10 * (a + 1))
print("finished")
