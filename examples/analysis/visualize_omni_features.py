#!/usr/bin/env python3
# Copyright 2025. Feature visualization for Qwen2.5-Omni (Thinker) before/without full SFT.
#
# Extracts:
#   - audio: thinker.get_audio_features() -> mean-pooled vectors (audio_tower path)
#   - text:  thinker.model.embed_tokens(prompt+query) -> mean-pooled vectors
#
# Does NOT require training. Optional --adapter_path loads LoRA for bookkeeping only;
# audio_tower stays frozen under default LoRA, so audio vectors usually match base.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 python examples/analysis/visualize_omni_features.py \
#     --jsonl /mnt/workspace/xjz/LlamaFactory/data/asr_ge85_noconflict_plus_hallucfix_10k_shuffled.jsonl \
#     --num_samples 300 \
#     --output_dir /mnt/workspace/xjz/models/qwen2_5_omni-7b/analysis/feature_viz_ge85

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from transformers import AutoConfig, AutoProcessor

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None  # type: ignore

try:
    from transformers import AutoModelForTextToWaveform
except ImportError:
    AutoModelForTextToWaveform = None  # type: ignore

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


THINKING_EMPTY_RE = re.compile(r"<thinking>\s*</thinking>", re.IGNORECASE)


def _force_submodule_attn_impl(module: torch.nn.Module, impl: str) -> None:
    """Keep audio_tower on FA2 when LM uses sdpa (see trl grpo_qwen2_5_omni_audio.py)."""
    cfg = getattr(module, "config", None)
    if cfg is not None and hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = impl
    for sub in module.modules():
        sub_cfg = getattr(sub, "config", None)
        if sub_cfg is not None and hasattr(sub_cfg, "_attn_implementation"):
            sub_cfg._attn_implementation = impl


def load_thinker(model_path: str, dtype: torch.dtype, attn_implementation: str | None) -> torch.nn.Module:
    trust_remote_code = True
    model_kwargs: dict = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    full_model = None
    errors: list[str] = []

    if AutoModelForMultimodalLM is not None:
        try:
            if type(config) in AutoModelForMultimodalLM._model_mapping.keys():
                full_model = AutoModelForMultimodalLM.from_pretrained(model_path, **model_kwargs)
        except Exception as exc:
            errors.append(f"AutoModelForMultimodalLM: {exc}")

    if full_model is None and AutoModelForTextToWaveform is not None:
        try:
            if type(config) in AutoModelForTextToWaveform._model_mapping.keys():
                full_model = AutoModelForTextToWaveform.from_pretrained(model_path, **model_kwargs)
        except Exception as exc:
            errors.append(f"AutoModelForTextToWaveform: {exc}")

    if full_model is None:
        raise RuntimeError("Failed to load Qwen2.5-Omni model:\n" + "\n".join(errors))

    if getattr(full_model.config, "model_type", None) == "qwen2_5_omni" and hasattr(full_model, "thinker"):
        thinker = full_model.thinker
    else:
        thinker = full_model

    if attn_implementation and attn_implementation != "flash_attention_2":
        for name in ("audio_tower", "visual"):
            sub = getattr(thinker, name, None)
            if sub is not None:
                _force_submodule_attn_impl(sub, "flash_attention_2")

    thinker.eval()
    return thinker


def load_samples(jsonl: str, num_samples: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if num_samples < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, num_samples)
    return rows


def label_need_fix(response: str) -> str:
    return "no_fix" if THINKING_EMPTY_RE.search(response or "") else "need_fix"


def count_thinking_items(response: str) -> int:
    m = re.search(r"<thinking>(.*?)</thinking>", response or "", re.DOTALL | re.IGNORECASE)
    if not m or not m.group(1).strip():
        return 0
    return len(re.findall(r"\d+\.\s*(replace|delete)", m.group(1), re.IGNORECASE))


@torch.inference_mode()
def extract_audio_vector(
    thinker: torch.nn.Module,
    processor,
    audio_path: str,
    max_audio_seconds: float | None,
    sampling_rate: int,
) -> np.ndarray:
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)

    duration = None
    if max_audio_seconds is not None and max_audio_seconds > 0:
        duration = max_audio_seconds

    waveform, _ = librosa.load(audio_path, sr=sampling_rate, duration=duration)
    feature_extractor = processor.feature_extractor
    fe_out = feature_extractor(
        [waveform],
        sampling_rate=sampling_rate,
        return_attention_mask=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_features = fe_out["input_features"].to(thinker.device, dtype=thinker.dtype)
    feature_attention_mask = fe_out["attention_mask"].to(thinker.device)
    feature_attention_mask = feature_attention_mask.bool()

    if not hasattr(thinker, "get_audio_features"):
        raise AttributeError("thinker has no get_audio_features(); upgrade transformers.")

    audio_hidden = thinker.get_audio_features(
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
    )
    vec = audio_hidden.float().mean(dim=0).cpu().numpy()
    return vec


@torch.inference_mode()
def extract_text_vector(
    thinker: torch.nn.Module,
    processor,
    text: str,
    max_text_tokens: int,
) -> np.ndarray:
    tokenizer = processor.tokenizer
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_text_tokens,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(thinker.device)
    attention_mask = enc["attention_mask"].to(thinker.device).bool()

    embed = thinker.get_input_embeddings()
    hidden = embed(input_ids).float()
    mask = attention_mask.unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.squeeze(0).cpu().numpy()


def paired_cosine_stats(audio_mat: np.ndarray, text_mat: np.ndarray) -> dict[str, float]:
    n = audio_mat.shape[0]
    a = torch.from_numpy(audio_mat).float()
    t = torch.from_numpy(text_mat).float()
    a = F.normalize(a, dim=-1)
    t = F.normalize(t, dim=-1)
    paired = (a * t).sum(dim=-1)
    perm = torch.randperm(n)
    mismatched = (a * t[perm]).sum(dim=-1)
    return {
        "paired_mean": float(paired.mean()),
        "paired_std": float(paired.std()),
        "mismatch_mean": float(mismatched.mean()),
        "mismatch_std": float(mismatched.std()),
        "margin_mean": float((paired - mismatched).mean()),
    }


def reduce_2d(matrix: np.ndarray, method: str, seed: int) -> np.ndarray:
    n = matrix.shape[0]
    if n < 3:
        return np.zeros((n, 2), dtype=np.float32)

    if method == "umap":
        if not HAS_UMAP:
            print("[warn] umap-learn not installed, falling back to t-SNE.", file=sys.stderr)
            method = "tsne"
        else:
            reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=min(15, n - 1))
            return reducer.fit_transform(matrix)

    perplexity = min(30, max(5, n // 4))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca", learning_rate="auto")
    return tsne.fit_transform(matrix)


def plot_scatter(
    coords: np.ndarray,
    labels: list[str],
    kinds: list[str],
    title: str,
    out_path: Path,
) -> None:
    uniq_labels = sorted(set(labels))
    uniq_kinds = sorted(set(kinds))
    label_colors = {lab: plt.cm.tab10(i % 10) for i, lab in enumerate(uniq_labels)}
    kind_markers = {k: m for k, m in zip(uniq_kinds, ["o", "s", "^", "D", "v", "P", "*"])}

    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(len(labels)):
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            c=[label_colors[labels[i]]],
            marker=kind_markers.get(kinds[i], "o"),
            s=36,
            alpha=0.75,
            edgecolors="none",
        )

    for lab in uniq_labels:
        ax.scatter([], [], c=[label_colors[lab]], label=lab, s=40)
    for k in uniq_kinds:
        ax.scatter([], [], c="gray", marker=kind_markers.get(k, "o"), label=f"kind={k}", s=40)

    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni feature visualization (audio_tower + text embed).")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/workspace/xjz/models/qwen2_5_omni",
        help="Base Omni model directory.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional LoRA adapter dir (usually does not change audio_tower outputs).",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="/mnt/workspace/xjz/LlamaFactory/data/asr_ge85_noconflict_plus_hallucfix_10k_shuffled.jsonl",
    )
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/workspace/xjz/models/qwen2_5_omni-7b/analysis/feature_viz",
    )
    parser.add_argument("--attn_implementation", type=str, default="sdpa", help="LM attention; audio_tower forced FA2.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=120.0,
        help="Truncate each wav for speed (None = full length).",
    )
    parser.add_argument("--max_text_tokens", type=int, default=512)
    parser.add_argument("--reduce", type=str, default="umap", choices=["umap", "tsne"])
    parser.add_argument("--skip_audio", action="store_true", help="Only text embeddings.")
    parser.add_argument("--skip_text", action="store_true", help="Only audio embeddings.")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processor from {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    sampling_rate = int(getattr(processor, "audio_sampling_rate", 16000))

    print(f"Loading thinker from {args.model_path} (attn={args.attn_implementation}) ...")
    thinker = load_thinker(args.model_path, dtype=dtype, attn_implementation=args.attn_implementation)

    if args.adapter_path:
        if PeftModel is None:
            raise ImportError("peft is required for --adapter_path")
        print(f"Loading adapter from {args.adapter_path} ...")
        thinker = PeftModel.from_pretrained(thinker, args.adapter_path)
        thinker.eval()

    rows = load_samples(args.jsonl, args.num_samples, args.seed)
    print(f"Processing {len(rows)} samples from {args.jsonl} ...")

    audio_vecs: list[np.ndarray] = []
    text_vecs: list[np.ndarray] = []
    meta_rows: list[dict] = []
    failures = 0

    for idx, row in enumerate(rows):
        audio_path = row.get("audio") or (row.get("audios") or [None])[0]
        prompt = row.get("prompt", "")
        query = row.get("query", "")
        response = row.get("response", "")
        text = f"{prompt}\n{query}".strip()
        fix_label = label_need_fix(response)
        n_fix = count_thinking_items(response)

        record = {
            "index": idx,
            "audio_path": audio_path,
            "label": fix_label,
            "n_fix_items": n_fix,
        }

        try:
            if not args.skip_audio and audio_path:
                record["audio_seconds"] = float(
                    librosa.get_duration(path=audio_path)
                    if args.max_audio_seconds is None
                    else min(librosa.get_duration(path=audio_path), args.max_audio_seconds)
                )
                audio_vecs.append(
                    extract_audio_vector(
                        thinker,
                        processor,
                        audio_path,
                        max_audio_seconds=args.max_audio_seconds,
                        sampling_rate=sampling_rate,
                    )
                )
            if not args.skip_text and text:
                text_vecs.append(
                    extract_text_vector(thinker, processor, text, max_text_tokens=args.max_text_tokens)
                )
            meta_rows.append(record)
            if (idx + 1) % 20 == 0:
                print(f"  [{idx + 1}/{len(rows)}] ok (failures={failures})")
        except Exception as exc:
            failures += 1
            print(f"  [skip] index={idx} audio={audio_path!r}: {exc}", file=sys.stderr)

    if failures:
        print(f"Finished with {failures} skipped samples.", file=sys.stderr)

    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in meta_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = [r["label"] for r in meta_rows]

    if audio_vecs and text_vecs and len(audio_vecs) == len(text_vecs):
        audio_mat = np.stack(audio_vecs, axis=0)
        text_mat = np.stack(text_vecs, axis=0)
        np.save(out_dir / "audio_embeddings.npy", audio_mat)
        np.save(out_dir / "text_embeddings.npy", text_mat)

        stats = paired_cosine_stats(audio_mat, text_mat)
        stats_path = out_dir / "cosine_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print("\n=== Audio–Text cosine (normalized) ===")
        print(f"  paired (same sample):  mean={stats['paired_mean']:.4f}  std={stats['paired_std']:.4f}")
        print(f"  mismatch (shuffled):   mean={stats['mismatch_mean']:.4f}  std={stats['mismatch_std']:.4f}")
        print(f"  margin (paired-mismatch): {stats['margin_mean']:.4f}")
        print(f"  -> margin > 0 suggests same-sample audio/text are closer than random pairs.")
        print(f"  Saved: {stats_path}")

        combined = np.concatenate([audio_mat, text_mat], axis=0)
        kinds = ["audio"] * len(audio_mat) + ["text"] * len(text_mat)
        comb_labels = labels + labels
        coords = reduce_2d(combined, args.reduce, args.seed)
        plot_scatter(
            coords,
            comb_labels,
            kinds,
            f"Qwen2.5-Omni features ({args.reduce}) — audio vs text",
            out_dir / f"scatter_combined_{args.reduce}.png",
        )

    if audio_vecs:
        audio_mat = np.stack(audio_vecs, axis=0)
        np.save(out_dir / "audio_embeddings.npy", audio_mat)
        coords_a = reduce_2d(audio_mat, args.reduce, args.seed)
        plot_scatter(
            coords_a,
            labels[: len(audio_vecs)],
            ["audio"] * len(audio_vecs),
            f"Audio tower features ({args.reduce})",
            out_dir / f"scatter_audio_{args.reduce}.png",
        )

    if text_vecs:
        text_mat = np.stack(text_vecs, axis=0)
        np.save(out_dir / "text_embeddings.npy", text_mat)
        coords_t = reduce_2d(text_mat, args.reduce, args.seed)
        plot_scatter(
            coords_t,
            labels[: len(text_vecs)],
            ["text"] * len(text_vecs),
            f"Text embed features ({args.reduce})",
            out_dir / f"scatter_text_{args.reduce}.png",
        )

    print(f"\nDone. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
