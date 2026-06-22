#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Aggregate predict_batch/checkpoint-*/predict_results.json into all_results_evaluated.json.
# Fills missing CER fields from sibling generated_predictions.jsonl when needed.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


def _load_row(pr_path: Path) -> dict | None:
    sys.path.insert(0, str(_repo_src()))
    from llamafactory.train.sft.metric import compute_correction_metrics_from_jsonl  # noqa: PLC0415

    with pr_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ckpt = pr_path.parent.name
    if not ckpt.startswith("checkpoint-"):
        return None
    step = int(ckpt.split("-")[-1])
    n_samples = int(data.get("predict_answer_no_punct_samples") or 0)

    cer_keys = (
        "predict_answer_cer",
        "predict_before_correction_answer_cer",
        "predict_improvement_answer_cer",
    )
    if not all(k in data for k in cer_keys):
        jl = pr_path.parent / "generated_predictions.jsonl"
        if jl.is_file():
            extra = compute_correction_metrics_from_jsonl(str(jl))
            for k in cer_keys:
                if k in extra:
                    data[k] = extra[k]

    def g(key: str) -> float | None:
        v = data.get(key)
        return float(v) if v is not None else None

    ak = [
        "predict_answer_no_punct_bleu-4",
        "predict_answer_no_punct_rouge-1",
        "predict_answer_no_punct_rouge-2",
        "predict_answer_no_punct_rouge-l",
    ]
    av = [g(k) for k in ak]
    answer_no_punct_mean = sum(av) / len(av) if all(x is not None for x in av) else None

    ik = [
        "predict_improvement_no_punct_bleu-4",
        "predict_improvement_no_punct_rouge-1",
        "predict_improvement_no_punct_rouge-2",
        "predict_improvement_no_punct_rouge-l",
    ]
    iv = [g(k) for k in ik]
    improvement_no_punct_mean = sum(iv) / len(iv) if all(x is not None for x in iv) else None

    dk = ["predict_bleu-4", "predict_rouge-1", "predict_rouge-2", "predict_rouge-l"]
    dv = [g(k) for k in dk]
    default_metric_mean = sum(dv) / len(dv) if all(x is not None for x in dv) else None

    row: dict = {
        "checkpoint": ckpt,
        "step": step,
        "path": str(pr_path),
        "n_samples": n_samples,
        "predict_answer_no_punct_bleu-4": g("predict_answer_no_punct_bleu-4"),
        "predict_answer_no_punct_rouge-1": g("predict_answer_no_punct_rouge-1"),
        "predict_answer_no_punct_rouge-2": g("predict_answer_no_punct_rouge-2"),
        "predict_answer_no_punct_rouge-l": g("predict_answer_no_punct_rouge-l"),
        "predict_improvement_no_punct_bleu-4": g("predict_improvement_no_punct_bleu-4"),
        "predict_improvement_no_punct_rouge-1": g("predict_improvement_no_punct_rouge-1"),
        "predict_improvement_no_punct_rouge-2": g("predict_improvement_no_punct_rouge-2"),
        "predict_improvement_no_punct_rouge-l": g("predict_improvement_no_punct_rouge-l"),
        "predict_bleu-4": g("predict_bleu-4"),
        "predict_rouge-1": g("predict_rouge-1"),
        "predict_rouge-2": g("predict_rouge-2"),
        "predict_rouge-l": g("predict_rouge-l"),
        "predict_before_correction_answer_cer": g("predict_before_correction_answer_cer"),
        "predict_answer_cer": g("predict_answer_cer"),
        "predict_improvement_answer_cer": g("predict_improvement_answer_cer"),
        "answer_no_punct_mean": answer_no_punct_mean,
        "improvement_no_punct_mean": improvement_no_punct_mean,
        "default_metric_mean": default_metric_mean,
    }
    return row


def _sort(rows: list[dict], key: str, *, reverse: bool) -> list[dict]:
    valid = [r for r in rows if r.get(key) is not None]
    return sorted(valid, key=lambda x: x[key], reverse=reverse)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate predict_batch checkpoint metrics.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/workspace/xjz/models/qwen2_5_omni-7b/lora/predict_batch"),
        help="Directory containing checkpoint-*/predict_results.json",
    )
    args = parser.parse_args()
    root: Path = args.root

    paths = sorted(root.glob("checkpoint-*/predict_results.json"), key=lambda p: int(p.parent.name.split("-")[-1]))
    rows: list[dict] = []
    for p in paths:
        row = _load_row(p)
        if row is not None:
            rows.append(row)

    by_n: dict[int, list[dict]] = {}
    for r in rows:
        by_n.setdefault(int(r["n_samples"]), []).append(r)

    def rankings_for(subset: list[dict]) -> dict[str, list[dict]]:
        return {
            "by_answer_mean": _sort(subset, "answer_no_punct_mean", reverse=True),
            "by_default_mean": _sort(subset, "default_metric_mean", reverse=True),
            "by_improvement_mean": _sort(subset, "improvement_no_punct_mean", reverse=True),
            "by_answer_cer": _sort(subset, "predict_answer_cer", reverse=False),
            "by_improvement_cer": _sort(subset, "predict_improvement_answer_cer", reverse=True),
        }

    out: dict = {
        "all_checkpoints": len(rows),
        "by_sample_count": {str(k): len(v) for k, v in sorted(by_n.items())},
        "cer_note": (
            "CER: character-level Levenshtein / len(gold <answer>), after strip_punctuation on ASR / pred / gold answer. "
            "Lower predict_answer_cer is better. predict_improvement_answer_cer = before_cer - after_cer (higher is better)."
        ),
        "rankings_all_by_answer_mean": _sort(rows, "answer_no_punct_mean", reverse=True),
        "rankings_all_by_default_mean": _sort(rows, "default_metric_mean", reverse=True),
        "rankings_all_by_improvement_mean": _sort(rows, "improvement_no_punct_mean", reverse=True),
        "rankings_all_by_answer_cer": _sort(rows, "predict_answer_cer", reverse=False),
        "rankings_all_by_improvement_cer": _sort(rows, "predict_improvement_answer_cer", reverse=True),
    }

    for n, subset in sorted(by_n.items()):
        rk = rankings_for(subset)
        out[f"rankings_n{n}_by_answer_mean"] = rk["by_answer_mean"]
        out[f"rankings_n{n}_by_default_mean"] = rk["by_default_mean"]
        out[f"rankings_n{n}_by_improvement_mean"] = rk["by_improvement_mean"]
        out[f"rankings_n{n}_by_answer_cer"] = rk["by_answer_cer"]
        out[f"rankings_n{n}_by_improvement_cer"] = rk["by_improvement_cer"]

    out_path = root / "all_results_evaluated.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
