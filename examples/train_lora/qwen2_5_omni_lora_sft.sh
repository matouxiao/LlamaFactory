#!/usr/bin/env bash
# Qwen2.5-Omni-7B LoRA SFT（correct_model）
# 在 LlamaFactory 仓库根目录执行：bash examples/train_lora/qwen2_5_omni_lora_sft.sh
#
# GPU：默认仅使用物理 GPU 4–7（共 4 卡）。映射后进程内为 cuda:0..3，torchrun 进程数=4。
# 覆盖示例：CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/train_lora/qwen2_5_omni_lora_sft.sh
# 也可显式指定进程数（一般不必）：NPROC_PER_NODE=4 ...

set -euo pipefail

cd "$(dirname "$0")/../.."

# 未设置时默认用物理 4–7；若已在环境中 export 过则尊重现有值
export CUDA_VISIBLE_DEVICES=4,5,6,7
# 避免 tokenizer 再开多进程，与 DDP/CUDA 叠加时易卡住
export TOKENIZERS_PARALLELISM=false

CONFIG_FILE="${CONFIG_FILE:-examples/train_lora/qwen2_5_omni_lora_sft.yaml}"

# 若需 ZeRO-3：CONFIG_FILE=examples/train_lora/qwen2_5_omni_lora_sft_ds3.yaml bash ...

llamafactory-cli train "$CONFIG_FILE"
