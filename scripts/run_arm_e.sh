#!/usr/bin/env bash
# Arm E (V4-Flash 0731, A's text + MinerU image analysis of each figure). After MinerU's advanced-tier run into
# data/mineru_adv/zips: freezes it, builds inputs, starts the judge, then dry run -> gate -> scored run (967) -> gate -> judge.
# Same settings as every arm; writes only data/mineru_adv, data/inputs/E, runs/dryrun_e and runs/stage1_e.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
source .env   # DEEPSEEK_BASE_URL: the OpenAI-compatible endpoint serving the DeepSeek models

$PY scripts/freeze_mineru.py --out mineru_adv_out | tail -3
$PY scripts/build_inputs.py --arms E --set questions
$PY scripts/build_inputs.py --arms E --set dry_run
$PY - <<'EOF'
import glob, json
rows = [json.load(open(f)) for f in glob.glob("data/inputs/E/*.json")]
print(f"E inputs: {len(rows)}; figures with a description: {sum(r.get('figures_described', 0) for r in rows)} "
      f"(summed over questions)")
EOF
# With descriptions blanked, every E input must equal A's frozen input: E differs from A only by the descriptions
$PY - <<'EOF'
import json, sys
sys.path.insert(0, ".")
import intake.prompts as P
from intake.spiqa import load_all, load_config
cfg = load_config(); papers = load_all(cfg)
qmap = {q.qid: (q, p) for p in papers.values() for q in p.questions}
P.attach_descriptions = lambda blocks, adv, label_of: 0
bad = 0
for qid in json.load(open("data/samples/stage1.json"))["questions"]:
    q, p = qmap[qid]
    e = P.build("E", q, p, cfg)["messages"]
    a = json.load(open(f"data/inputs/A/{qid.replace('/', '__')}.json"))["messages"]
    bad += e != a
print(f"E-minus-descriptions vs A: {bad} mismatches")
sys.exit(4 if bad else 0)
EOF

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
print(f"{run} E: n={n} errors={err} truncated={tr} unparsed={un} {'FAIL' if bad else 'ok'}")
sys.exit(3 if bad else 0)
EOF
}

$PY scripts/generate.py --run dryrun_e --set dry_run --arms E --model flash
check dryrun_e 0
$PY scripts/generate.py --run stage1_e --set questions --arms E --model flash
$PY scripts/generate.py --run stage1_e --set questions --arms E --model flash
$PY scripts/count_answer_tokens.py --run stage1_e --model flash
check stage1_e 0.02
$PY scripts/judge.py --run dryrun_e
$PY scripts/judge.py --run stage1_e
echo ARM_E_DONE
