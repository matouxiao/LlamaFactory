#!/usr/bin/env bash
# Qwen2.5-Omni 特征可视化（无需训练 / 无需重新 SFT）
#
# 用法:
#   cd /mnt/workspace/xjz/LlamaFactory
#   bash examples/analysis/run_visualize_omni_features.sh
#
# 可选：对比已有 LoRA（audio_tower 通常不变，主要看 text 与配对指标）
#   ADAPTER_PATH=/mnt/workspace/xjz/models/qwen2_5_omni-7b/lora/grpo_SFTge85_3070/checkpoint-300 \
#     bash examples/analysis/run_visualize_omni_features.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/xjz/models/qwen2_5_omni}"
JSONL="${JSONL:-/mnt/workspace/xjz/LlamaFactory/data/asr_ge85_noconflict_plus_hallucfix_10k_shuffled.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/workspace/xjz/models/qwen2_5_omni-7b/analysis/feature_viz_ge85}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXTRA=()
if [[ -n "${ADAPTER_PATH:-}" ]]; then
  EXTRA+=(--adapter_path "$ADAPTER_PATH")
fi

export CUDA_VISIBLE_DEVICES
python examples/analysis/visualize_omni_features.py \
  --model_path "$MODEL_PATH" \
  --jsonl "$JSONL" \
  --num_samples "$NUM_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --max_audio_seconds 120 \
  --reduce umap \
  "${EXTRA[@]}"

echo "See PNG + cosine_stats.json under: $OUTPUT_DIR"
