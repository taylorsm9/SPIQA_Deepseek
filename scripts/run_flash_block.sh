#!/usr/bin/env bash
# V4-Flash 0731 block: dry run of A, O, N -> gate -> scored run of A (967),
# O (967), N (150) interleaved -> answer-token counts -> gate -> judge. Run with 0731 serving on :8888 and
# the judge on :8901. Writes only runs/dryrun_flash and runs/stage1_flash.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
source .env   # DEEPSEEK_BASE_URL: the OpenAI-compatible endpoint serving the DeepSeek models

# gate RUN MAX_RATE: per arm, fail if errors, truncations or unparsed answers exceed MAX_RATE
gate() {
$PY - "$1" "$2" <<'EOF'
import json, sys
from collections import defaultdict
run, max_rate = sys.argv[1], float(sys.argv[2])
last = {}
for l in open(f"runs/{run}/generations.jsonl"):
    r = json.loads(l)
    if "error" not in r or r["id"] not in last:
        last[r["id"]] = r
arms = defaultdict(list)
for r in last.values():
    arms[r["arm"]].append(r)
fail = False
for arm, rs in sorted(arms.items()):
    n = len(rs)
    err = sum("error" in r for r in rs)
    trunc = sum(bool(r.get("truncated")) for r in rs)
    unp = sum("error" not in r and not r.get("parsed") for r in rs)
    bad = max(err, trunc, unp) / n > max_rate
    fail |= bad
    print(f"{run} {arm}: n={n} errors={err} truncated={trunc} unparsed={unp} {'FAIL' if bad else 'ok'}")
sys.exit(3 if fail else 0)
EOF
}

curl -sf "$DEEPSEEK_BASE_URL/models" | grep -q deepseek-v4-flash-0731
curl -sf localhost:8901/v1/models >/dev/null

$PY scripts/generate.py --run dryrun_flash --set dry_run --arms A,O,N --model flash
gate dryrun_flash 0
$PY scripts/count_answer_tokens.py --run dryrun_flash --model flash
$PY scripts/judge.py --run dryrun_flash

$PY scripts/generate.py --run stage1_flash --set questions --arms A,O,N --arm-set N=arm_n --model flash
# one resume pass picks up any request that failed all three attempts
$PY scripts/generate.py --run stage1_flash --set questions --arms A,O,N --arm-set N=arm_n --model flash
$PY scripts/count_answer_tokens.py --run stage1_flash --model flash
gate stage1_flash 0.02
$PY scripts/judge.py --run stage1_flash
echo FLASH_BLOCK_DONE
