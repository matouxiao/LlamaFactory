#!/usr/bin/env bash
# 一键：用 checkpoint-6128 对 data/demo.jsonl 批量生成，无需任何手动输入。
# 入口虽为 llamafactory-cli train，但 do_train=false，只跑 predict。
#
#   cd /mnt/workspace/xjz/LlamaFactory
#   bash examples/inference/run_demo_jsonl_predict.sh
#
# 输出：models/qwen2_5_omni-7b/lora/predict_demo/generated_predictions.jsonl

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
exec llamafactory-cli train examples/inference/qwen2_5_omni_lora_predict_demo.yaml
