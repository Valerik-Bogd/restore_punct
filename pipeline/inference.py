"""
Demo helpers for raw strings.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer

from .env import MAX_LEN, MODEL_DIR
from .labels import id2label
from .models import BERT_CRF, build_model

if TYPE_CHECKING:
    from .config import RunConfig


_WORD_RE = re.compile(r"[\w]+(?:-[\w]+)*")


def load_for_inference(cfg: "RunConfig"):
    run_dir = os.path.join(MODEL_DIR, cfg.name)
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"no saved model at {run_dir}")
    tokenizer = AutoTokenizer.from_pretrained(run_dir)
    model = build_model(cfg.architecture, load_from=run_dir)
    model.eval()
    return model, tokenizer


def restore_punctuation(model, tokenizer, text: str, max_len: int = MAX_LEN) -> str:
    """
    Feed a raw unpunctuated string and stitch punctuation back
    """
    device = next(model.parameters()).device
    words = [m.group() for m in _WORD_RE.finditer(text)]
    if not words:
        return ""

    inputs = tokenizer(
        words,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        is_split_into_words=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(model, "crf"):
            emissions = outputs[1] if isinstance(outputs, tuple) else outputs
            mask = inputs["attention_mask"].byte()
            predictions = model.crf.decode(emissions, mask=mask)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if isinstance(logits, tuple):
                logits = logits[1] if len(logits) > 1 else logits[0]
            predictions = torch.argmax(logits, dim=2)[0].cpu().tolist()

    word_ids = inputs.word_ids()
    restored = ""
    previous_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        original_word = words[word_idx]
        pred_id = predictions[i] if i < len(predictions) else 0
        punct = id2label.get(int(pred_id), "O")
        restored += original_word + (" " if punct == "O" else punct + " ")
        previous_word_idx = word_idx

    return re.sub(r"\s+", " ", restored).strip()


__all__ = ["load_for_inference", "restore_punctuation"]
