#!/usr/bin/env bash
# Vision-Exp checks: score the check that images reach the model; if they do, run the dry run and the
# arm C pilot, and score them. Requires the judge on :8901.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

while pgrep -f "generate.py --run phase0_vision" >/dev/null; do sleep 30; done
$PY scripts/judge.py --run phase0_vision
mkdir -p runs/phase0
$PY scripts/analyze.py --runs phase0_vision --vision | tee runs/phase0/check2_vision.json
if ! grep -q '"pass": true' runs/phase0/check2_vision.json; then
  echo "CHECK 2 FAILED: images do not clearly help; vision tower may not be loading. Stopping."
  exit 2
fi

$PY scripts/generate.py --run phase0_dryrun_vision --set dry_run --arms C,D --seeds 0
$PY scripts/generate.py --run pilot --set flip_pilot --arms C
$PY scripts/judge.py --run phase0_dryrun_vision
$PY scripts/judge.py --run pilot
echo VISION_PHASE0_DONE
