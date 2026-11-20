#!/bin/bash
# Long-history validation: events 2012-2026 (development before 2024, test from 2024), prepare, terrain wind, baseline, calibrate, score.
cd "$(dirname "$0")/.."
export VAL_SET=v2 PY=~/flood/venv/bin/python
$PY -c "from pkfire import validate as v; v.build_events(full=True, force=True)" > logs/v2_events.log 2>&1
FORCE=1 $PY tools/run_validation.py prepare 4 > logs/v2_prepare.log 2>&1
$PY tools/wind_precompute.py > logs/v2_wind.log 2>&1
cp data/validation/calib_none.json data/validation/calib_v2base.json 2>/dev/null; echo '{}' > data/validation/calib_v2base.json
$PY tools/run_validation.py score v2base > logs/v2_base.log 2>&1
BASE=v2base CAL_MAX=450 SWEEPS=3 $PY tools/calibrate.py v2cal 8 > logs/v2_calibrate.log 2>&1
$PY tools/run_validation.py score v2cal > logs/v2_score.log 2>&1
echo exit >> logs/v2_score.log
