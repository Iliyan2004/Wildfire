# PAKFIRE

Wildfire spread simulation for the whole of Pakistan: place an ignition anywhere, run a physical spread model (Rothermel/Albini surface fire, Finney minimum travel time, spotting, crown fire, terrain wind) on the GPU (CPU fallback), and see the result as a map, an animated 3D scene, an exposure and evacuation analysis, and a PDF report. A research prototype, calibrated and tested against satellite-observed fires; see `docs/PAKFIRE_Technical_Reference.pdf` for what it can and cannot do.

## Run

```
~/flood/venv/bin/python -m pkfire.app              # http://<host>:8020/   (FIRE_PORT changes the port)
ssh -L 8020:localhost:8020 aliyan@<host>           # to open it from another machine at http://localhost:8020/
~/flood/venv/bin/python -m pytest tests/pk -q      # 41 tests, about a minute
node tools/forestfire_test.mjs                     # fire model of the start page
```

Pages: `/` start page (a 3D forest where a fire spreads when you press a button), `/instant` PAKFIRE Instant (fire footprint in about a second), `/app` the full model (four tiers: instant footprint, spread run, ensemble, mitigation; discovery, monitoring, exposure, evacuation, reports).

## Layout

| Path | Content |
|---|---|
| `pkfire/` | the engine and the Flask application; `vendor/` holds Rothermel and Albini spotting; `calibration.json` is the shipped calibration |
| `web/pk/` | `landing.html`, `instant.html`, `app.html`; `landing/` (3D scene and its fire model); `viewer/` (animated 3D fire scene); `vendor/three` |
| `tools/` | data downloads, validation, calibration, interface review, start-page screenshots and test |
| `tests/pk/` | pytest suite (physics, GPU = CPU, routes) |
| `docs/` | Technical Reference and Architecture Document (`.pdf` and `.html`), `figures/`, `background/` |
| `data/` | `external/` (FIRMS, fire stations, census, informal settlements), `national/` (canopy, forest type, burned area), `runs/` (saved runs), `validation/` (events, results, caches), `raw/weather/`, `cache/windfield/` |
| `logs/`, `.secrets/` | runtime logs; `mapbox_token` and `firms_map_key` (never committed) |

## Documents

- `docs/PAKFIRE_Technical_Reference.pdf`: physics, data, features, validation record, bugs found, limitations.
- `docs/PAKFIRE_ARCH_Document.pdf`: every file, input and output path, the process of one run, the API, libraries and versions.
- `LICENSES.md`: data and software credits.

## Notes

- The evacuation panel needs the SimCity PostGIS road database (port 5455); without it the panel says the road database is unavailable.
- Waiting on data from the owner: an official fire-station list (`FIRE_STATIONS`), surveyed fire perimeters, local wind observations.
- The earlier Margalla platform (FIRELAB) that PAKFIRE grew from is no longer in this folder: it was moved on 7 November 2026 to `../Fire Simulation Model - FIRELAB legacy (archived 2026-11-07)`. Delete that folder when it is no longer wanted.
