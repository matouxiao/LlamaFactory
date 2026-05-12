#!/usr/bin/env bash
# 批量评测多个 LoRA checkpoint。
# 默认从 qwen2_5_omni_lora_predict_demo.yaml 里的 adapter_name_or_path 推导 CKPT_ROOT，
# 扫描其中所有 checkpoint-* 子目录，并为每个 checkpoint 单独生成输出目录。
# 若某个 checkpoint 缺少 LoRA 必要文件（如 adapter_config.json），会自动跳过，不中断整批任务。
#
# 用法：
#   cd /mnt/workspace/xjz/LlamaFactory
#   bash examples/inference/run_demo_jsonl_predict_multi_ckpt.sh
#
# 可选环境变量：
#   CUDA_VISIBLE_DEVICES=0
#   CKPT_ROOT=/mnt/workspace/xjz/models/qwen2_5_omni-7b/lora/sft_dora_no429_noconflict2
#   OUTPUT_ROOT=/mnt/workspace/xjz/models/qwen2_5_omni-7b/lora/predict_batch
#   CHECKPOINTS=checkpoint-447,checkpoint-894,checkpoint-1341

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

BASE_YAML="${ROOT}/examples/inference/qwen2_5_omni_lora_predict_demo.yaml"

readarray -t yaml_defaults < <(python3 - <<'PY' "$BASE_YAML"
import sys

yaml_path = sys.argv[1]
adapter = ""
output_dir = ""
with open(yaml_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("adapter_name_or_path:"):
            adapter = line.split(":", 1)[1].strip()
        elif line.startswith("output_dir:"):
            output_dir = line.split(":", 1)[1].strip()

if not adapter:
    raise SystemExit("adapter_name_or_path not found in base yaml")
if not output_dir:
    raise SystemExit("output_dir not found in base yaml")

print(adapter)
print(output_dir)
PY
)

BASE_ADAPTER_PATH="${yaml_defaults[0]}"
BASE_OUTPUT_DIR="${yaml_defaults[1]}"
CKPT_ROOT="${CKPT_ROOT:-$(dirname "$BASE_ADAPTER_PATH")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "$BASE_OUTPUT_DIR")/predict_batch}"
CHECKPOINTS="${CHECKPOINTS:-}"

mkdir -p "$OUTPUT_ROOT"

declare -a ckpt_paths=()
declare -a success_paths=()
declare -a skipped_paths=()
declare -a failed_paths=()

if [[ -n "$CHECKPOINTS" ]]; then
  IFS=',' read -r -a ckpt_names <<< "$CHECKPOINTS"
  for ckpt_name in "${ckpt_names[@]}"; do
    ckpt_name="${ckpt_name// /}"
    [[ -z "$ckpt_name" ]] && continue
    ckpt_paths+=("${CKPT_ROOT}/${ckpt_name}")
  done
else
  while IFS= read -r ckpt_path; do
    ckpt_paths+=("$ckpt_path")
  done < <(python3 - <<'PY' "$CKPT_ROOT"
import os
import re
import sys

root = sys.argv[1]
if not os.path.isdir(root):
    raise SystemExit(f"CKPT_ROOT not found: {root}")

items = []
for name in os.listdir(root):
    path = os.path.join(root, name)
    if os.path.isdir(path) and re.fullmatch(r"checkpoint-\d+", name):
        step = int(name.split("-")[-1])
        items.append((step, path))

for _, path in sorted(items):
    print(path)
PY
)
fi

if [[ "${#ckpt_paths[@]}" -eq 0 ]]; then
  echo "No checkpoints found under: $CKPT_ROOT" >&2
  exit 1
fi

for ckpt_path in "${ckpt_paths[@]}"; do
  ckpt_name="$(basename "$ckpt_path")"
  output_dir="${OUTPUT_ROOT}/${ckpt_name}"
  temp_yaml="$(mktemp "/tmp/${ckpt_name}.XXXXXX.yaml")"

  if [[ ! -d "$ckpt_path" ]]; then
    echo "Skip missing checkpoint dir: $ckpt_path" >&2
    skipped_paths+=("$ckpt_path")
    rm -f "$temp_yaml"
    continue
  fi

  if [[ ! -f "$ckpt_path/adapter_config.json" ]]; then
    echo "Skip invalid checkpoint (missing adapter_config.json): $ckpt_path" >&2
    skipped_paths+=("$ckpt_path")
    rm -f "$temp_yaml"
    continue
  fi

  echo "=============================="
  echo "Running predict for: $ckpt_path"
  echo "Output dir: $output_dir"

  if ! python3 - <<'PY' "$BASE_YAML" "$temp_yaml" "$ckpt_path" "$output_dir"
import sys

base_yaml, temp_yaml, ckpt_path, output_dir = sys.argv[1:5]
with open(base_yaml, "r", encoding="utf-8") as f:
    text = f.read()

updated = []
for line in text.splitlines():
    if line.startswith("adapter_name_or_path:"):
        updated.append(f"adapter_name_or_path: {ckpt_path}")
    elif line.startswith("output_dir:"):
        updated.append(f"output_dir: {output_dir}")
    else:
        updated.append(line)

with open(temp_yaml, "w", encoding="utf-8") as f:
    f.write("\n".join(updated) + "\n")
PY
  then
    echo "Skip checkpoint due to yaml generation failure: $ckpt_path" >&2
    failed_paths+=("$ckpt_path")
    rm -f "$temp_yaml"
    continue
  fi

  if llamafactory-cli train "$temp_yaml"; then
    success_paths+=("$ckpt_path")
  else
    echo "Predict failed, continue to next checkpoint: $ckpt_path" >&2
    failed_paths+=("$ckpt_path")
  fi

  rm -f "$temp_yaml"
done

echo "=============================="
echo "Batch predict finished."
echo "Total checkpoints: ${#ckpt_paths[@]}"
echo "Succeeded: ${#success_paths[@]}"
echo "Skipped: ${#skipped_paths[@]}"
echo "Failed: ${#failed_paths[@]}"

if [[ "${#skipped_paths[@]}" -gt 0 ]]; then
  echo "Skipped checkpoints:"
  printf '  %s\n' "${skipped_paths[@]}"
fi

if [[ "${#failed_paths[@]}" -gt 0 ]]; then
  echo "Failed checkpoints:"
  printf '  %s\n' "${failed_paths[@]}"
fi
