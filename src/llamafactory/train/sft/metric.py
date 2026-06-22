# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# 解析 user 块时兼容两种结束符：Qwen 系为 redacted_im_end，其他 chat 模板常为 im_end。
USER_PATTERN = re.compile(
    r"<\|im_start\|>user\n(.*?)<\|(?:im_end|redacted_im_end)\|>\n<\|im_start\|>assistant\n",
    re.DOTALL,
)


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()


def _compute_text_similarity(pred: str, label: str) -> dict[str, float]:
    hypothesis = list(jieba.cut(pred))
    reference = list(jieba.cut(label))

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    metric_result = {}
    for k, v in result.items():
        metric_result[k] = round(v["f"] * 100, 4)

    bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
    metric_result["bleu-4"] = round(bleu_score * 100, 4)
    return metric_result


def _extract_answer(text: str) -> str:
    matched = ANSWER_PATTERN.search(text)
    return matched.group(1).strip() if matched is not None else text.strip()


def _extract_raw_query(prompt: str) -> str:
    r"""从 prompt 的 user 段取出「纠错前」ASR 文本。

    约定：首行可为音频占位；下一行为任务说明（可含字面量 ``<answer>`` 等字样）；其后为待纠错转写直至 user 段结束。
    """
    matched = USER_PATTERN.search(prompt)
    user_block = matched.group(1).strip() if matched is not None else prompt.strip()
    lines = [line for line in user_block.splitlines() if line.strip()]

    if lines and (lines[0].startswith("<|audio_bos|>") or lines[0].startswith("<audio>")):
        lines = lines[1:]

    if lines and ("<thinking>" in lines[0] or "<answer>" in lines[0] or "纠错" in lines[0]):
        lines = lines[1:]

    return "\n".join(lines).strip()


def _strip_punctuation(text: str) -> str:
    return "".join(char for char in text if not unicodedata.category(char).startswith(("P", "S")))


def _levenshtein_distance(a: str, b: str) -> int:
    r"""Classic Levenshtein distance on Unicode codepoints (character-level for Chinese)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def _cer(reference: str, hypothesis: str) -> float:
    r"""Character Error Rate: edit distance / len(reference). Lower is better.

    If reference is empty, returns 0.0 when hypothesis is also empty, else 1.0.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein_distance(reference, hypothesis) / len(reference)


def _average_metric_dict(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}

    return {key: float(np.mean([metric[key] for metric in metric_dicts])) for key in metric_dicts[0].keys()}


def compute_correction_metrics_from_jsonl(prediction_file: str) -> dict[str, float]:
    r"""Compute answer-only no-punctuation metrics, CER on answer, and raw-ASR baseline."""
    if not os.path.isfile(prediction_file):
        return {}

    before_metrics = []
    after_metrics = []
    cer_before_list: list[float] = []
    cer_after_list: list[float] = []
    sample_count = 0

    with open(prediction_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            before = _strip_punctuation(_extract_raw_query(sample["prompt"]))
            after = _strip_punctuation(_extract_answer(sample["predict"]))
            label = _strip_punctuation(_extract_answer(sample["label"]))

            before_metrics.append(_compute_text_similarity(before, label))
            after_metrics.append(_compute_text_similarity(after, label))
            cer_before_list.append(_cer(label, before))
            cer_after_list.append(_cer(label, after))
            sample_count += 1

    before_result = _average_metric_dict(before_metrics)
    after_result = _average_metric_dict(after_metrics)
    metric_names = tuple(before_result.keys())

    result = {"predict_answer_no_punct_samples": float(sample_count)}
    for metric_name in metric_names:
        result[f"predict_before_correction_no_punct_{metric_name}"] = before_result[metric_name]
        result[f"predict_answer_no_punct_{metric_name}"] = after_result[metric_name]
        result[f"predict_improvement_no_punct_{metric_name}"] = after_result[metric_name] - before_result[metric_name]

    if sample_count:
        mean_before_cer = float(np.mean(cer_before_list))
        mean_after_cer = float(np.mean(cer_after_list))
        result["predict_before_correction_answer_cer"] = round(mean_before_cer, 6)
        result["predict_answer_cer"] = round(mean_after_cer, 6)
        result["predict_improvement_answer_cer"] = round(mean_before_cer - mean_after_cer, 6)

    return result
