#!/usr/bin/env bash
# Arm G (V4-Flash 0731 reading DeepSeek-OCR-2's own conversion of each PDF). Runs the OCR-2 pipeline on the
# 4090 (judge stopped), builds inputs, restarts the judge, then dry run -> gate -> scored run (967) -> gate ->
# judge. Same settings as every arm. Writes only data/ocr2_pipeline, data/inputs/G, runs/dryrun_g, runs/stage1_g.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
source .env   # DEEPSEEK_BASE_URL: the OpenAI-compatible endpoint serving the DeepSeek models

if pgrep -f "vllm serve models/Qwen3.8-27B" >/dev/null; then
  echo "stopping judge: $(pgrep -af 'vllm serve models/Qwen3.8-27B' | cut -c1-120)"
  pkill -f "vllm serve models/Qwen3.8-27B"
fi
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 10; done

envs/mineru/bin/python scripts/ocr2_pipeline.py
# Stop if OCR-2 lost many pages or many figure readings ran into the limit
$PY - <<'EOF'
import json, sys
s = json.load(open("data/ocr2_pipeline/stats.json"))
figs = [json.loads(l) for l in open("data/ocr2_pipeline/figures.jsonl")]
cut = sum(f["finish"] != "stop" for f in figs)
print(f"OCR-2 pipeline: {s}; figure readings cut off: {cut}/{len(figs)}")
sys.exit(5 if s["papers"] != 300 or s["pages_dropped"] > 0.1 * s["pages"] or cut > 0.05 * len(figs) else 0)
EOF

$PY scripts/build_inputs.py --arms G --set questions
$PY scripts/build_inputs.py --arms G --set dry_run

mkdir -p logs
(nohup scripts/serve_judge.sh > logs/judge_server.log 2>&1 &)
until curl -sf localhost:8901/v1/models >/dev/null; do sleep 15; done
curl -sf "$DEEPSEEK_BASE_URL/models" | grep -q deepseek-v4-flash-0731

check() {
$PY - "$1" "$2" <<'EOF'
import json, sys
run, max_rate = sys.argv[1], float(sys.argv[2])
last = {}
for l in open(f"runs/{run}/generations.jsonl"):
    r = json.loads(l)
    if "error" not in r or r["id"] not in last:
        last[r["id"]] = r
rs = list(last.values()); n = len(rs)
err = sum("error" in r for r in rs); tr = sum(bool(r.get("truncated")) for r in rs)
un = sum("error" not in r and not r.get("parsed") for r in rs)
bad = max(err, tr, un) / n > max_rate
print(f"{run} G: n={n} errors={err} truncated={tr} unparsed={un} {'FAIL' if bad else 'ok'}")
sys.exit(3 if bad else 0)
EOF
}

$PY scripts/generate.py --run dryrun_g --set dry_run --arms G --model flash
check dryrun_g 0
$PY scripts/generate.py --run stage1_g --set questions --arms G --model flash
$PY scripts/generate.py --run stage1_g --set questions --arms G --model flash
$PY scripts/count_answer_tokens.py --run stage1_g --model flash
check stage1_g 0.02
$PY scripts/judge.py --run dryrun_g
$PY scripts/judge.py --run stage1_g
echo ARM_G_DONE
