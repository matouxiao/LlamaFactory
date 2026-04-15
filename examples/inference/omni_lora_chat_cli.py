#!/usr/bin/env python3
"""终端多轮对话：为 Qwen2.5-Omni 传入 audios（官方 llamafactory-cli chat 不支持音频）。

若已有 data/demo.jsonl 想「自动批量」测模型，不要用本脚本，请用：
  bash examples/inference/run_demo_jsonl_predict.sh
（见 qwen2_5_omni_lora_predict_demo.yaml）

用法（在 LlamaFactory 仓库根目录）：
  CUDA_VISIBLE_DEVICES=0 python examples/inference/omni_lora_chat_cli.py examples/inference/qwen2_5_omni_lora_chat.yaml

首轮输入音频，二选一格式的**一行**：
  - 绝对路径或相对 media_dir 的音频文件，如 audio/train/.../x.opus
  - 或 JSON 切片（与训练数据一致），例如：
    {"path":"audio/train/youtube/B00000/Y0000000009_-0p8pYdlfjY.opus","start_time":46.39,"end_time":150.17}

随后输入用户文本（须含与训练一致的 <audio> 与任务说明；可与 demo.jsonl 的 prompt+query 拼接成一条 user 内容）。

命令：exit 退出，clear 清空历史。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 保证从仓库根目录运行时能 import llamafactory
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

from llamafactory.chat.chat_model import ChatModel  # noqa: E402
from llamafactory.extras.misc import torch_gc  # noqa: E402


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_audio_line(line: str, media_dir: str | None) -> str | dict:
    line = line.strip()
    if line.startswith("{"):
        d = json.loads(line)
        if not isinstance(d, dict):
            raise ValueError("JSON 音频须为对象，含 path / start_time / end_time 等")
        p = d.get("path") or d.get("audio")
        if isinstance(p, str) and media_dir and not os.path.isabs(p):
            d = dict(d)
            d["path"] = os.path.normpath(os.path.join(media_dir, p.lstrip("/")))
        return d
    if os.path.isabs(line) or not media_dir:
        return line
    return os.path.normpath(os.path.join(media_dir, line.lstrip("/")))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    cfg_path = Path(sys.argv[1]).resolve()
    if not cfg_path.is_file():
        print(f"找不到配置: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = _load_yaml(cfg_path)
    media_dir = cfg.get("media_dir")

    # 与 llamafactory-cli chat config.yaml 一致：用 dict 初始化
    chat_model = ChatModel(cfg)

    messages: list[dict[str, str]] = []
    print(
        "Omni 终端对话（带音频）。\n"
        "第一轮：粘贴**一行**音频（文件路径或 JSON 切片）。\n"
        "第二轮起：直接输入 user 文本（可不含新音频则沿用上一轮音频逻辑需自行 clear 后重贴）。\n"
        "简化：每轮都先贴音频行，再贴文本行（两次 input）。\n"
        "命令 exit / clear。\n",
        flush=True,
    )

    last_audio: str | dict | None = None

    while True:
        try:
            a_line = input("\n[音频一行；回车沿用上一轮] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if a_line.lower() == "exit":
            break
        if a_line.lower() == "clear":
            messages = []
            last_audio = None
            torch_gc()
            print("历史已清空。", flush=True)
            continue

        if a_line:
            last_audio = _resolve_audio_line(a_line, media_dir)

        try:
            text = input("[用户文本] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text.lower() == "exit":
            break
        if text.lower() == "clear":
            messages = []
            last_audio = None
            torch_gc()
            print("历史已清空。", flush=True)
            continue

        if not text:
            print("跳过：文本为空。", flush=True)
            continue

        audios = None
        if last_audio is not None:
            audios = [last_audio]
        else:
            print(
                "警告：尚未提供音频，模型仅收文本，与纠错训练条件不一致。",
                file=sys.stderr,
                flush=True,
            )

        messages.append({"role": "user", "content": text})
        print("\nAssistant: ", end="", flush=True)
        response = ""
        for new_text in chat_model.stream_chat(messages, audios=audios):
            print(new_text, end="", flush=True)
            response += new_text
        print(flush=True)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
