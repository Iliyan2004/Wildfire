# Accuracy improvements implemented 16 October 2026

Restart `python scripts/server.py`, then reload the live page to use these changes.

- Ignitions on non-burnable cells may move at most 90 m to a burnable cell centre.
  Empty fuel grids, nonfinite coordinates, and distant ignitions are rejected.
  Run JSON records displacement for every ignition; the page displays it.
  The limit is an operational policy, not a measured ignition uncertainty.
- Dead moisture retains Simard EMC and 1/10/100 h dry relaxation, adding the
  WRF-SFIRE rain-wetting term. Rain approaches saturation 2.5 kg/kg at a rate
  `(1 - exp(-(rain_mm_h - 0.05)/8))/14` per hour above 0.05 mm/h.
  Reference: https://github.com/wrf-model/WRF/blob/master/phys/module_fr_fire_phys.F
  (`advance_moisture`, rain branch). This hybrid does not implement WRF-SFIRE's
  full drying/wetting hysteresis, and coefficients are not locally calibrated.
  Hourly precipitation belongs to the preceding hour, is held constant within
  that interval, and is used during conditioning and the run. Missing rain
  values do not wet fuels; they should not be interpreted as observed zero rain.
- Run results expose input quality, missing processes, observation source,
  moisture method, and ignition displacement. Build readiness is not accuracy.
- Area-matched scores are explicitly shape diagnostics using observed area;
  suppression may affect shape. Run-end overlap is not a verified timed score.
  Automatic scar comparisons record the pre/post satellite observation window.
- `validate_all.py --years 2019-2023` now saves `validation_2019-2023.*` and
  a metadata JSON with skipped events and assumptions. The 2024-2026 split
  has separate filenames. Events rejected by the snap limit are recorded.

Verification: 54 tests pass, including rain response, drying, precipitation
timing, antecedent rain, bounded/empty-grid ignition behavior, and API runs
that save input-quality metadata or reject an invalid ignition. Python and
page JavaScript syntax checks pass. Browser interaction was not exercised.

Saved historical validation results predate these changes. No new real-fire
validation has been run and improved predictive accuracy is not established.
Before releasing an accuracy claim, rebuild development evaluations, review
ignition and observation uncertainty, calibrate using development data only,
then evaluate the held-back years once decisions are frozen.

Still requires data or separate implementation: local fuel-load and moisture
measurements; verified ignition/containment times and time-stamped perimeters;
uncertainty ensembles; crown initiation/spread with CBH/CBD/foliar moisture;
suppression and long-range spotting from torching trees.
