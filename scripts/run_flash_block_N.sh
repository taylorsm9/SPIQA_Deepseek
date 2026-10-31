#!/usr/bin/env bash
# Arm N scored run, after the A/O block finishes generating. Same settings as every arm; truncated
# answers score 0. Separate run dir so two writers never share a generations file and A/O latency is unaffected.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while pgrep -f "[g]enerate.py --run stage1_flash --set questions --arms A,O" >/dev/null; do sleep 60; done
$PY scripts/generate.py --run stage1_flash_n --set arm_n --arms N --model flash
$PY scripts/generate.py --run stage1_flash_n --set arm_n --arms N --model flash
$PY scripts/count_answer_tokens.py --run stage1_flash_n --model flash
while pgrep -f "[j]udge.py --run stage1_flash$" >/dev/null; do sleep 60; done
$PY scripts/judge.py --run stage1_flash_n
$PY - <<'PY'
import json
last = {}
for l in open("runs/stage1_flash_n/generations.jsonl"):
    r = json.loads(l)
    if "error" not in r or r["id"] not in last:
        last[r["id"]] = r
rs = list(last.values()); n = len(rs)
err = sum("error" in r for r in rs); tr = sum(bool(r.get("truncated")) for r in rs)
un = sum("error" not in r and not r.get("parsed") for r in rs)
print(f"stage1_flash_n N: n={n} errors={err} truncated={tr} unparsed={un}",
      "OVER_2PCT" if max(err, tr, un) / n > 0.02 else "ok")
PY
echo N_BLOCK_DONE
