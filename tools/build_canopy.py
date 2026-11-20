"""Reduces the Meta/WRI 1 m canopy height tiles to a national 30 m raster the scene builder can window-read.
   python tools/build_canopy.py [workers]      input data/external/meta_chm/*.tif (EPSG:3857, uint8 m, 1.19 m)
   output data/national/canopy_height_pakistan.tif   EPSG:4326, 1/3600 deg (~30 m), uint8, tiled 512, deflate, overviews
     band 1 = mean canopy height (m) of the 30 m cell, 0 where no canopy; band 2 = 90th percentile height of the 1 m pixels (m)
   Stage 1: every tile is block-reduced 32 x 32 (mean and 90th percentile) into data/cache/canopy_tiles/<tile>.npz (resumable).
   Stage 2: the reduced tiles are warped (average) into the mosaic."""
import glob, os, sys
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "external", "meta_chm")
CACHE = os.path.join(ROOT, "data", "cache", "canopy_tiles")
OUT = os.path.join(ROOT, "data", "national", "canopy_height_pakistan.tif")
B = 32
BOUNDS = (60.8, 23.6, 77.9, 37.1)
RES = 1.0 / 3600


def reduce_tile(path):
    t = os.path.splitext(os.path.basename(path))[0]
    dst = os.path.join(CACHE, t + ".npz")
    if os.path.exists(dst):
        return t
    with rasterio.open(path) as r:
        h, w = r.shape
        mean = np.zeros((h // B, w // B), np.uint8); p90 = np.zeros_like(mean)
        for i in range(0, h, B * 32):
            a = r.read(1, window=((i, i + B * 32), (0, w)))
            a = np.where(a > 100, 0, a)                      # 255 or other fill values are not heights (trees above 100 m do not exist in Pakistan)
            blk = a.reshape(32, B, w // B, B).transpose(0, 2, 1, 3).reshape(32, w // B, B * B)
            mean[i // B: i // B + 32] = np.round(blk.mean(-1)).astype(np.uint8)
            p90[i // B: i // B + 32] = np.partition(blk, int(0.9 * B * B), axis=-1)[..., int(0.9 * B * B)]
        tr = r.transform * r.transform.scale(B, B)
        np.savez(dst + ".tmp.npz", mean=mean, p90=p90, tr=np.array(tr)[:6], crs=str(r.crs))
        os.replace(dst + ".tmp.npz", dst)
    return t


def mosaic():
    w = int(round((BOUNDS[2] - BOUNDS[0]) / RES)); h = int(round((BOUNDS[3] - BOUNDS[1]) / RES))
    tr = from_origin(BOUNDS[0], BOUNDS[3], RES, RES)
    prof = dict(driver="GTiff", dtype="uint8", count=2, width=w, height=h, crs="EPSG:4326", transform=tr, tiled=True, blockxsize=512, blockysize=512,
                compress="deflate", predictor=2, nodata=None, BIGTIFF="YES")
    files = sorted(glob.glob(os.path.join(CACHE, "*.npz")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rasterio.open(OUT + ".part", "w", **prof).close()      # create empty, then reopen read-write so tiles can be max-merged
    with rasterio.open(OUT + ".part", "r+") as o:
        for k, f in enumerate(files):
            z = np.load(f)
            src_tr = rasterio.Affine(*z["tr"]); crs = str(z["crs"])
            from rasterio.warp import transform_bounds
            hh, ww = z["mean"].shape
            l, b, r_, t = rasterio.transform.array_bounds(hh, ww, src_tr)
            bb = transform_bounds(crs, "EPSG:4326", l, b, r_, t)
            c0 = max(int((bb[0] - BOUNDS[0]) / RES) - 1, 0); c1 = min(int((bb[2] - BOUNDS[0]) / RES) + 2, w)
            r0 = max(int((BOUNDS[3] - bb[3]) / RES) - 1, 0); r1 = min(int((BOUNDS[3] - bb[1]) / RES) + 2, h)
            if c1 <= c0 or r1 <= r0:
                continue
            win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            wtr = rasterio.windows.transform(win, tr)
            for band, key in ((1, "mean"), (2, "p90")):
                cur = o.read(band, window=win)
                new = np.zeros(cur.shape, np.float32)
                reproject(z[key].astype(np.float32), new, src_transform=src_tr, src_crs=crs, dst_transform=wtr, dst_crs="EPSG:4326", resampling=Resampling.average)
                o.write(np.maximum(cur, np.round(new).astype(np.uint8)), band, window=win)
            print("mosaic", k + 1, len(files), flush=True)
    os.replace(OUT + ".part", OUT)
    with rasterio.open(OUT, "r+") as o:
        o.build_overviews([2, 4, 8, 16], Resampling.average)
        o.update_tags(ns="rio_overview", resampling="average")
        o.update_tags(SOURCE="Meta/WRI high resolution canopy height (s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float), 32x32 block reduction", BAND1="mean height m", BAND2="p90 height m")


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.tif")))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with ProcessPoolExecutor(n) as ex:
        for i, t in enumerate(ex.map(reduce_tile, files)):
            print("reduced", i + 1, len(files), t, flush=True)
    mosaic()
