"""MODIS MCD64A1 burned area (500 m, burn day) as an independent truth for burned extent and timing.

The monthly tiles (data/national/modis_burned/<tile>/<YYYY-MM>.tif, burn day of year, 0 unburned, -1 unmapped, -2 water) are converted to days
since 2024-01-01 and mosaicked once into one EPSG:4326 raster (0.0045 deg, about 500 m). Pixel dates come from the day of year; the year comes
from the file label, except that a late-year day of year in a January or February file belongs to the previous year (the monthly files are
labelled by the search window, so a 'May' file can hold early-June burn dates)."""
from __future__ import annotations

import datetime as dt
import glob
import os

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILES = os.path.join(HERE, "data", "national", "modis_burned")
MOSAIC = os.path.join(HERE, "data", "validation", "mcd_burn_days_2000.tif")
W, S, E_, N, RES = 60.8, 23.5, 77.9, 37.2, 0.0045
EPOCH = dt.date(2000, 1, 1)


def build(force=False):
    if os.path.exists(MOSAIC) and not force:
        return MOSAIC
    os.makedirs(os.path.dirname(MOSAIC), exist_ok=True)
    cols, rows = int(round((E_ - W) / RES)), int(round((N - S) / RES))
    tr = from_origin(W, N, RES, RES)
    out = np.zeros((rows, cols), np.int16)
    base = np.array([(dt.date(yy, 1, 1) - EPOCH).days for yy in range(2010, 2028)], np.int16)
    for f in sorted(glob.glob(os.path.join(TILES, "*", "*.tif"))):
        y, m = (int(x) for x in os.path.basename(f)[:-4].split("-"))
        with rasterio.open(f) as d:
            doy = d.read(1)
            days = np.zeros(doy.shape, np.int16)
            pos = doy > 0
            year = np.where((doy > 300) & (m <= 2), y - 1, y)
            days[pos] = (base[(year - 2010)] + doy)[pos]
            tmp = np.zeros((rows, cols), np.int16)
            reproject(days, tmp, src_transform=d.transform, src_crs=d.crs, dst_transform=tr, dst_crs="EPSG:4326", resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        out = np.where(tmp > 0, tmp, out)
    with rasterio.open(MOSAIC, "w", driver="GTiff", height=rows, width=cols, count=1, dtype="int16", crs="EPSG:4326", transform=tr, nodata=0, compress="deflate") as dst:
        dst.write(out, 1)
    return MOSAIC


def burned_days_on(scene):
    """Burn day (days since 2000-01-01, 0 = none) resampled onto a scene grid (nearest)."""
    build()
    dst = np.zeros((scene["ny"], scene["nx"]), np.int16)
    with rasterio.open(MOSAIC) as d:
        reproject(rasterio.band(d, 1), dst, dst_transform=scene["transform"], dst_crs=scene["crs"], src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
    return dst
