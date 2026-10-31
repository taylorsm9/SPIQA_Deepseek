#!/usr/bin/env bash
# Vision-Exp block. After MinerU's standard-tier run into data/mineru/zips: freeze -> vision-arm inputs -> judge up -> score dry run + pilot ->
# gate on dry-run truncation/parse rates -> scored run of C -> score.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY scripts/freeze_mineru.py
$PY scripts/build_inputs.py --arms C --set questions

mkdir -p logs
(nohup scripts/serve_judge.sh > logs/judge_server.log 2>&1 &)
until curl -sf localhost:8901/v1/models >/dev/null; do sleep 15; done

while pgrep -f "[g]enerate.py --run (phase0_dryrun_vision|pilot)" >/dev/null; do sleep 30; done
$PY scripts/judge.py --run phase0_dryrun_vision
$PY scripts/judge.py --run pilot

# Gate: stop if the dry run truncates or fails to parse more than 10% of answers
$PY - <<'EOF'
import json, sys
g = [json.loads(l) for l in open("runs/phase0_dryrun_vision/generations.jsonl")]
bad = sum(1 for r in g if "error" in r or r.get("truncated") or not r.get("parsed"))
print(f"dry run: {bad}/{len(g)} errored, truncated or unparsed")
sys.exit(3 if bad > 0.10 * len(g) else 0)
EOF

$PY scripts/generate.py --run stage1 --set questions --arms C --model vision
$PY scripts/judge.py --run stage1
echo VISION_BLOCK_DONE_PAUSED
