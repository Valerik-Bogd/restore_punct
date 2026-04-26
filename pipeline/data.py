"""Dataset loading, label alignment, and train-set mixing.

Hides the ``datasets.load_dataset + concatenate + tokenize`` dance so the
training notebook never has to touch it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import DataCollatorForTokenClassification

from .env import DATA_DIR, MAX_LEN

if TYPE_CHECKING:
    from .config import RunConfig


def align_labels_with_tokens(examples, tokenizer, max_len: int = MAX_LEN):
    """Subword-aware label alignment.

    First subword of a word keeps the word's label; all later subwords and
    special tokens get -100 (ignored by the loss).
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_len,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def _load_json_split(path: str, split: str = "train") -> Dataset:
    return load_dataset("json", data_files={split: path})[split]


def _concat_files(paths: list[str], split: str = "train") -> Dataset | None:
    if not paths:
        return None
    parts = [_load_json_split(p, split=split) for p in paths]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _load_replay(replay_files: list[dict]) -> Dataset | None:
    """Each entry: {"path": str, "n": int, "seed": int (optional, default 42)}.

    Loads, shuffles with the entry's seed, takes the first ``n`` rows, then
    concatenates all replay parts into one dataset.
    """
    if not replay_files:
        return None

    parts: list[Dataset] = []
    for entry in replay_files:
        path = entry["path"]
        n = int(entry["n"])
        seed = int(entry.get("seed", 42))
        ds = _load_json_split(path, split="train").shuffle(seed=seed)
        n = min(n, len(ds))
        parts.append(ds.select(range(n)))
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def build_training_dataset(cfg: "RunConfig", tokenizer) -> dict[str, Dataset]:
    """Builds tokenized ``{"train": ..., "validation": ...}`` per the config.

    Training set = concat(train_files) + replay_pool, shuffled with cfg.seed.
    Validation set = concat(val_files), no shuffle.
    """
    train_base = _concat_files(cfg.train_files, split="train")
    if train_base is None:
        raise ValueError(f"cfg.train_files is empty for run '{cfg.name}'")

    replay_pool = _load_replay(cfg.replay_files)
    if replay_pool is not None:
        train_ds = concatenate_datasets([train_base, replay_pool])
    else:
        train_ds = train_base
    train_ds = train_ds.shuffle(seed=cfg.seed)

    val_ds = _concat_files(cfg.val_files, split="validation")
    if val_ds is None:
        raise ValueError(f"cfg.val_files is empty for run '{cfg.name}'")

    def _tok(batch):
        return align_labels_with_tokens(batch, tokenizer)

    tokenized_train = train_ds.map(
        _tok,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_val = val_ds.map(
        _tok,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    return {"train": tokenized_train, "validation": tokenized_val}


DEFAULT_BENCHMARKS: dict[str, str] = {
    "General_Test": f"{DATA_DIR}/bench_test_all.json",
    "GERA_Test": f"{DATA_DIR}/bench_test_gera.json",
    "LORuGEC_Test": f"{DATA_DIR}/bench_test_lorugec.json",
}


def build_benchmark_datasets(cfg: "RunConfig", tokenizer) -> dict[str, Dataset]:
    """Tokenize every benchmark. cfg.benchmarks overrides defaults if provided."""
    bench_paths = cfg.benchmarks or DEFAULT_BENCHMARKS

    def _tok(batch):
        return align_labels_with_tokens(batch, tokenizer)

    out: dict[str, Dataset] = {}
    for name, path in bench_paths.items():
        raw = _load_json_split(path, split="test")
        out[name] = raw.map(_tok, batched=True, remove_columns=raw.column_names)
    return out


# --- Advanced utilities: CRF transition prior & O-masking collator ---------


def compute_empirical_transitions(
    train_files: list[str],
    num_labels: int,
    smoothing: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count label-pair frequencies in raw training files and return log-priors.

    Reads each file's ``ner_tags`` lists (no tokenizer needed), counts

    - transitions: every consecutive ``(prev, next)`` pair,
    - starts:      the first label of each sequence,
    - ends:        the last label of each sequence,

    applies add-``smoothing`` Laplace smoothing so zero counts don't become
    ``-inf`` after taking the log, row-normalizes transitions to a valid
    conditional distribution, then returns

    - ``log_transitions`` : shape ``[K, K]``,  ``log P(next | prev)``
    - ``log_start``       : shape ``[K]``,     ``log P(first label)``
    - ``log_end``          : shape ``[K]``,     ``log P(last label)``

    Suitable for copying straight into torchcrf's ``transitions`` /
    ``start_transitions`` / ``end_transitions`` buffers.
    """
    trans = torch.full((num_labels, num_labels), smoothing, dtype=torch.float64)
    start = torch.full((num_labels,), smoothing, dtype=torch.float64)
    end = torch.full((num_labels,), smoothing, dtype=torch.float64)

    for path in train_files:
        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        for ex in examples:
            tags = ex["ner_tags"]
            if not tags:
                continue
            start[tags[0]] += 1
            end[tags[-1]] += 1
            for a, b in zip(tags[:-1], tags[1:]):
                trans[a, b] += 1

    trans_probs = trans / trans.sum(dim=1, keepdim=True)
    start_probs = start / start.sum()
    end_probs = end / end.sum()

    return (
        torch.log(trans_probs).float(),
        torch.log(start_probs).float(),
        torch.log(end_probs).float(),
    )


class OMaskingCollator:
    """Wrapper around :class:`DataCollatorForTokenClassification` that
    randomly drops a fraction ``p`` of O-label tokens from the loss by
    relabeling them to ``-100`` per-batch.

    Runs after the base collator has stacked + padded labels, so padding
    positions (already ``-100``) are untouched. Re-randomized every batch,
    so across an epoch every O token has a chance to contribute.
    """

    def __init__(self, base_collator, o_label_id: int, p: float):
        if not 0.0 < p < 1.0:
            raise ValueError(f"o_mask_prob must be in (0, 1), got {p}")
        self.base_collator = base_collator
        self.o_label_id = int(o_label_id)
        self.p = float(p)

    def __call__(self, features):
        batch = self.base_collator(features)
        labels = batch["labels"]
        o_mask = labels == self.o_label_id
        drop = torch.rand_like(labels, dtype=torch.float) < self.p
        batch["labels"] = torch.where(o_mask & drop, torch.full_like(labels, -100), labels)
        return batch

    # HF Trainer sometimes inspects collator attributes; delegate gracefully
    def __getattr__(self, name):
        return getattr(self.base_collator, name)


def make_data_collator(tokenizer, cfg: "RunConfig"):
    """Return a collator honoring ``cfg.o_mask_prob``. For CRF architectures we
    intentionally skip O-masking — the CRF forward maps ``-100`` back to 0
    ("O") via ``safe_labels``, so masking would be a silent no-op.
    """
    base = DataCollatorForTokenClassification(tokenizer)
    if cfg.o_mask_prob <= 0.0 or cfg.architecture == "bert+crf":
        return base
    from .labels import label2id
    return OMaskingCollator(base, o_label_id=label2id["O"], p=cfg.o_mask_prob)


__all__ = [
    "align_labels_with_tokens",
    "build_training_dataset",
    "build_benchmark_datasets",
    "DEFAULT_BENCHMARKS",
    "compute_empirical_transitions",
    "OMaskingCollator",
    "make_data_collator",
]
