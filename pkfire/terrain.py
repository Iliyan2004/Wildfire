"""
Builds the simulation grid around an ignition from the national datasets on the
shared drive: SRTM terrain (29 m), ESRI 10 m land cover (2025) and MODIS 1 km NDVI.

Whole-of-Pakistan coverage works by cutting a local UTM window around each
ignition (default +-50 km); nothing is pre-clipped, so any point in the country
can be ignited.
"""
from __future__ import annotations

import datetime as dt
import glob
import math
import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy import ndimage

from . import fuels as fm

SHARE = os.environ.get("FIRE_SHARE", "/mnt/external-ssd/DGX SHARED")
DEM_TIF = os.environ.get("FIRE_DEM", f"{SHARE}/3. Raster Data/DEM_SRTM/pakDEM/Pak_srtm.tif")
LULC_TIF = os.environ.get("FIRE_LULC", f"{SHARE}/2. ESRI_LandCover Data/lc2025/lc2025Pakistan.tif")
CGLS_TIF = os.environ.get("FIRE_CGLS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "national", "cgls_lc100_pakistan.tif"))
NDVI_DIR = os.environ.get("FIRE_NDVI", f"{SHARE}/3. Raster Data/MODIS_NDVI_1km")

# ESRI 10-class LULC -> role
LC_WATER, LC_TREES, LC_FLOODED, LC_CROPS, LC_BUILT, LC_BARE, LC_SNOW, LC_CLOUD, LC_RANGE = 1, 2, 4, 5, 7, 8, 9, 10, 11

FOREST_ELEV_M = 1200.0       # trees above this are conifer/understory timber (FM10), below litter (FM9)


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_grid(lat: float, lon: float, half_km: float, cell_m: float):
    """Local UTM grid centred on (lat, lon). Returns (crs, transform, ny, nx)."""
    crs = CRS.from_epsg(utm_epsg(lon, lat))
    from pyproj import Transformer
    x0, y0 = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    n = int(round(2 * half_km * 1000 / cell_m))
    tr = from_origin(x0 - n * cell_m / 2, y0 + n * cell_m / 2, cell_m, cell_m)
    return crs, tr, n, n


def _read_to_grid(path, crs, tr, ny, nx, resampling, dtype, nodata, band=1):
    """Window-read a national raster (decimated to ~grid resolution) and warp it to the grid."""
    with rasterio.open(path) as src:
        l, b, r, t = rasterio.transform.array_bounds(ny, nx, tr)
        sl, sb, sr, st = transform_bounds(crs, src.crs, l, b, r, t, densify_pts=21)
        pad_x, pad_y = (sr - sl) * 0.02, (st - sb) * 0.02
        win = from_bounds(sl - pad_x, sb - pad_y, sr + pad_x, st + pad_y, src.transform)
        win = win.round_offsets().round_lengths()
        # pick a read size no finer than ~1x the target cell so huge windows stay cheap
        src_cell = abs(src.transform.a) if not src.crs.is_projected else abs(src.transform.a)
        tgt_in_src = (sr - sl) / nx
        step = max(tgt_in_src / src_cell, 1.0) if src_cell else 1.0
        out_w = max(int(win.width / step), 2)
        out_h = max(int(win.height / step), 2)
        data = src.read(band, window=win, out_shape=(out_h, out_w), boundless=True,
                        fill_value=src.nodata if src.nodata is not None else 0,
                        resampling=Resampling.nearest if resampling == Resampling.nearest else Resampling.average)
        s_tr = rasterio.windows.transform(win, src.transform) * Affine.scale(win.width / out_w, win.height / out_h)
        dst = np.full((ny, nx), nodata, dtype)
        reproject(data, dst, src_transform=s_tr, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=tr, dst_crs=crs, dst_nodata=nodata, resampling=resampling)
    return dst


def load_dem(crs, tr, ny, nx):
    dem = _read_to_grid(DEM_TIF, crs, tr, ny, nx, Resampling.bilinear, np.float32, np.nan)
    bad = ~np.isfinite(dem) | (dem < -500)
    if bad.all():
        raise ValueError("no terrain data at this location (outside the SRTM extent)")
    if bad.any():
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        dem = dem[tuple(idx)]
    return dem


OVERSAMPLE = 3
LC_CLASSES = (1, 2, 4, 5, 7, 8, 9, 11)


def lulc_path(year):
    """ESRI 10 m land cover of the given year (2017-2025 exist on the server; other years use the nearest)."""
    y = min(max(int(year), 2017), 2025)
    p = os.path.join(SHARE, "2. ESRI_LandCover Data", f"lc{y}", f"lc{y}Pakistan.tif")
    return p if os.path.exists(p) else LULC_TIF


def load_lulc(crs, tr, ny, nx, path=None):
    """Land cover on the grid as the majority class of each cell plus the share of burnable land in it.

    The 10 m classes are read on a grid three times finer than the cell and reduced by counting, so fields, roads, bare patches
    and buildings inside a 50-100 m cell are represented instead of the single pixel nearest sampling would pick.
    Returns (majority class uint8, burnable fraction float32)."""
    k = OVERSAMPLE
    tr3 = tr * Affine.scale(1.0 / k)
    lc3 = _read_to_grid(path or LULC_TIF, crs, tr3, ny * k, nx * k, Resampling.nearest, np.uint8, 0)
    frac = np.empty((len(LC_CLASSES), ny, nx), np.float32)
    for n, c in enumerate(LC_CLASSES):
        frac[n] = (lc3 == c).reshape(ny, k, nx, k).sum((1, 3)).astype(np.float32)
    valid = np.maximum(frac.sum(0), 1.0)
    mode = np.where(frac.sum(0) > 0, np.array(LC_CLASSES, np.uint8)[frac.argmax(0)], 0).astype(np.uint8)
    fb = (frac[1] + frac[2] + frac[3] + frac[7] + 0.15 * frac[5]) / valid        # trees, flooded vegetation, crops, rangeland, a little for bare ground
    return mode, fb.astype(np.float32)


def load_foresttype(crs, tr, ny, nx):
    """Copernicus CGLS-LC100 (2019, 100 m) forest type: 1 needleleaf, 2 broadleaf, 0 unknown, mixed or not forest."""
    if not os.path.exists(CGLS_TIF):
        return np.zeros((ny, nx), np.uint8)
    c = _read_to_grid(CGLS_TIF, crs, tr, ny, nx, Resampling.nearest, np.uint8, 255)
    out = np.zeros((ny, nx), np.uint8)
    out[np.isin(c, (111, 121))] = 1
    out[np.isin(c, (112, 114, 122, 124))] = 2
    return out


CANOPY_TIF = os.environ.get("FIRE_CANOPY", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "national", "canopy_height_pakistan.tif"))


def load_canopy(crs, tr, ny, nx):
    """Meta/WRI canopy height reduced to 30 m (tools/build_canopy.py): (mean height, 90th-percentile height) in metres, zeros when the file is absent."""
    if not os.path.exists(CANOPY_TIF):
        return np.zeros((ny, nx), np.float32), np.zeros((ny, nx), np.float32)
    return tuple(_read_to_grid(CANOPY_TIF, crs, tr, ny, nx, Resampling.average, np.float32, 0, band=b) for b in (1, 2))


def canopy_for_scene(scene):
    """Adds canopy_mean / canopy_p90 to a scene on first use (scenes cached before the canopy map existed do not carry them)."""
    if "canopy_p90" not in scene:
        scene["canopy_mean"], scene["canopy_p90"] = load_canopy(scene["crs"], scene["transform"], scene["ny"], scene["nx"])
    return scene["canopy_mean"], scene["canopy_p90"]


def pick_ndvi_file(date: dt.date) -> str:
    """The MODIS composite nearest in time to ``date`` when the date lies inside the record (up to 16 days away); for later dates (a live run)
    the newest composite within 16 days of the day of year."""
    files = sorted(glob.glob(os.path.join(NDVI_DIR, "pakistan_ndvi_1km_*.tif")))
    if not files:
        return ""
    dates = [dt.date.fromisoformat(os.path.basename(f)[-14:-4]) for f in files]
    if date <= max(dates):
        k = min(range(len(files)), key=lambda i: abs((dates[i] - date).days))
        return files[k]
    best, best_key = files[-1], (True, 0, 1e9)
    for f, d in zip(files, dates):
        diff = abs((d.timetuple().tm_yday - date.timetuple().tm_yday + 183) % 366 - 183)
        key = (diff > 16, -d.year if diff <= 16 else 0, diff)
        if key < best_key:
            best, best_key = f, key
    return best


def load_ndvi(crs, tr, ny, nx, date: dt.date):
    f = pick_ndvi_file(date)
    if not f:
        return np.full((ny, nx), 0.25, np.float32), ""
    nd = _read_to_grid(f, crs, tr, ny, nx, Resampling.bilinear, np.float32, np.nan)
    nd = np.where(np.isfinite(nd), nd, np.nan)
    if np.isnan(nd).all():
        return np.full((ny, nx), 0.25, np.float32), os.path.basename(f)
    nd = np.where(np.isnan(nd), np.nanmedian(nd), nd)
    return nd.astype(np.float32), os.path.basename(f)


def terrain(dem, cell_m):
    """Slope (tan) and upslope azimuth (radians clockwise from north)."""
    dzdy_south, dzdx = np.gradient(dem, cell_m)       # rows grow southward
    dzdn = -dzdy_south
    tan = np.hypot(dzdx, dzdn).astype(np.float32)
    up = np.arctan2(dzdx, dzdn).astype(np.float32)
    return tan, up


def fuel_from_lulc(lc, dem, ndvi, ftype=None, fb=None):
    """Fuel-model table row per cell + a fuel-continuity multiplier (sparse fuels spread slowly).

    Crosswalk (ESRI 10-class land cover -> Scott & Burgan / Anderson), see technical reference ch. 6:
      needleleaf forest (CGLS) TL8 long-needle litter; broadleaf forest TU1; forest of unknown type: TL8 above 1000 m, TU1 below
      rangeland        GR2 / GS2 / FM5 by NDVI (0.10-0.20 / 0.20-0.40 / >0.40), continuity from NDVI
      crops            GR2 stubble, continuity 0.7        flooded vegetation  FM3 reeds
      bare ground      GR2 only where NDVI >= 0.20         built, water, snow, ice: do not burn
    """
    lc = lc.copy()
    invalid = (lc == 0) | (lc == LC_CLOUD)
    if invalid.any() and not invalid.all():
        idx = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
        lc = lc[tuple(idx)]
    row = np.zeros(lc.shape, np.int8)
    cont = np.ones(lc.shape, np.float32)
    trees = lc == LC_TREES
    ft = np.zeros(lc.shape, np.uint8) if ftype is None else ftype
    row[trees & ((ft == 1) | ((ft == 0) & (dem >= FOREST_ELEV_M)))] = fm.row_of(fm.F_CONIFER_LITTER)
    row[trees & ((ft == 2) | ((ft == 0) & (dem < FOREST_ELEV_M)))] = fm.row_of(fm.F_TIMBER_UNDERSTORY)
    row[lc == LC_FLOODED] = fm.row_of(fm.F_REEDS)
    row[lc == LC_CROPS] = fm.row_of(fm.F_STUBBLE)
    cont[lc == LC_CROPS] = 0.7
    rng = lc == LC_RANGE
    row[rng & (ndvi > 0.40)] = fm.row_of(fm.F_SHRUB)
    row[rng & (ndvi > 0.20) & (ndvi <= 0.40)] = fm.row_of(fm.F_GRASSSHRUB)
    row[rng & (ndvi >= 0.10) & (ndvi <= 0.20)] = fm.row_of(fm.F_GRASS)
    cont[rng] = np.clip((ndvi[rng] - 0.05) / 0.25, 0.1, 1.0)
    bare = (lc == LC_BARE) & (ndvi >= 0.20)
    row[bare] = fm.row_of(fm.F_GRASS)
    cont[bare] = 0.15
    if fb is not None:                                   # fragmentation inside the cell (fields, roads, bare patches)
        cont *= np.clip(fb, 0.1, 1.0)
    return row, cont, lc


def build_scene(lat, lon, half_km, cell_m, date: dt.date):
    crs, tr, ny, nx = make_grid(lat, lon, half_km, cell_m)
    dem = load_dem(crs, tr, ny, nx)
    lc, fb = load_lulc(crs, tr, ny, nx, lulc_path(date.year))
    ftype = load_foresttype(crs, tr, ny, nx)
    ndvi, ndvi_file = load_ndvi(crs, tr, ny, nx, date)
    tan, up = terrain(dem, cell_m)
    row, cont, lc = fuel_from_lulc(lc, dem, ndvi, ftype, fb)
    return dict(ftype=ftype, fb=fb, crs=crs, transform=tr, ny=ny, nx=nx, dx=cell_m, dem=dem, lc=lc, ndvi=ndvi,
                ndvi_file=ndvi_file, slope=tan, upaz=up, fuel=row, cont=cont)
