"""Downloads the Meta/WRI high-resolution canopy height tiles (s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm) that cover Pakistan.
   python tools/download_meta_chm.py [workers]   Resumable: finished tiles are skipped. Tile list is the quadkey index tiles.geojson."""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import shape, box

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "external", "meta_chm")
BASE = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/"
PK = box(60.8, 23.6, 77.9, 37.1)


def tiles():
    idx = os.path.join(OUT, "tiles.geojson")
    if not os.path.exists(idx):
        urllib.request.urlretrieve(BASE + "tiles.geojson", idx)
    return sorted(f["properties"]["tile"] for f in json.load(open(idx))["features"] if shape(f["geometry"]).intersects(PK))


def get(t):
    dst = os.path.join(OUT, t + ".tif")
    if os.path.exists(dst):
        return t, "have"
    for k in range(4):
        try:
            urllib.request.urlretrieve(BASE + f"chm/{t}.tif", dst + ".part")
            os.replace(dst + ".part", dst)
            return t, "ok"
        except Exception as e:
            err = str(e)[:80]
    return t, "FAIL " + err


if __name__ == "__main__":
    ts = tiles()
    print(len(ts), "tiles", flush=True)
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        for i, (t, s) in enumerate(ex.map(get, ts)):
            print(i + 1, t, s, flush=True)
