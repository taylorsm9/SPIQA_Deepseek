#!/usr/bin/env bash
# Qwen3.8-27B W4A16 judge on the 4090. MinerU must be stopped first (they share the card in sequence).
set -euo pipefail
cd "$(dirname "$0")/.."
exec envs/judge/bin/vllm serve models/Qwen3.8-27B-W4A16-AutoRound \
  --served-model-name judge \
  --port 8901 \
  --max-model-len 4096 \
  --max-logprobs 20 \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 16 \
  --seed 0
