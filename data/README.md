# data/

The repository tracks only small curated files here. Everything else is large (about 13 GB) and is rebuilt or downloaded:

| Folder | In git | How to get it |
|---|---|---|
| `external/pbs_districts.json`, `fire_stations.geojson`, `informal/SOURCES.txt` | yes | built by `tools/build_pbs.py`; stations from OSM; informal settlements: see SOURCES.txt (Zenodo 18606153) |
| `external/firms*`, `external/informal/*.gpkg` | no | `tools/firms_archive.py`, `tools/firms_yearly.py`; informal settlements from the Zenodo record |
| `national/canopy_height_pakistan.tif` (580 MB) | no | `tools/download_meta_chm.py` then `tools/build_canopy.py` (71 GB of tiles; the raster is optional, the `canopy` option is off) |
| `national/cgls_lc100_pakistan.tif`, `national/modis_burned/` | no | Copernicus CGLS-LC100 (2019) forest type; MODIS MCD64A1 tiles (pystac-client) |
| `validation/*.json`, `validation/earlier_527_event_set/` | yes | scored results, calibrations and event lists |
| `validation/cache/`, `validation/wind/`, `validation/*.tif` | no | `python tools/run_validation.py prepare`, `tools/wind_precompute.py` (hours) |
| `runs/`, `raw/weather/`, `cache/windfield/`, `monitor.json` | no | written by the application as it runs |

The model also reads the Pakistan SRTM DEM, the ESRI 10 m land cover, MODIS NDVI, district boundaries and WorldPop from the shared drive; the paths and override variables are in `docs/PAKFIRE_ARCH_Document.pdf` (Chapter 5).
