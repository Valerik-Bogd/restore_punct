"""Trainer factory + top-level ``train_run`` entry point.

``train_run(cfg)`` performs: build model -> build datasets -> build trainer ->
``trainer.train()`` -> persist artifacts. The returned model object is handy
for a quick inference demo inside the template notebook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments

from custom_trainer import PUNCT_WEIGHTS, CRFTrainer, FocalLossTrainer

from .data import (
    build_training_dataset,
    compute_empirical_transitions,
    make_data_collator,
)
from .env import BATCH_SIZE, GRAD_ACCUM, MODEL_DIR
from .labels import LABELS, NUM_LABELS, label2id
from .models import BERT_CRF, BERT_LSTM, build_model, get_tokenizer, save_model_artifacts

if TYPE_CHECKING:
    from .config import RunConfig


def _compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        LABELS[pred_id]
        for pred_seq, lab_seq in zip(predictions, labels)
        for pred_id, lab_id in zip(pred_seq, lab_seq)
        if lab_id != -100
    ]
    true_labels = [
        LABELS[lab_id]
        for pred_seq, lab_seq in zip(predictions, labels)
        for pred_id, lab_id in zip(pred_seq, lab_seq)
        if lab_id != -100
    ]

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(true_labels, true_predictions)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def _focal_alpha_tensor() -> torch.Tensor:
    """Build per-class weights and normalize them to mean 1.

    - Strict coverage: every PUNCT_WEIGHTS key must map to a real label, and
      every label must have an explicit entry. Silent drift (old bug) is out.
    - Mean normalization: per-token loss magnitude stays comparable to plain
      CE. Previously max alpha=15 was used multiplicatively on top of .mean()
      reduction, which inflated the effective gradient ~5-10x and caused the
      focal baselines to over-predict rare classes and crash O-class F1.

      After normalization sum(alpha)=NUM_LABELS and mean(alpha)=1, so relative
      class emphasis is preserved but the overall loss scale matches CE.
    """
    missing_in_weights = set(label2id) - set(PUNCT_WEIGHTS)
    extra_in_weights = set(PUNCT_WEIGHTS) - set(label2id)
    if missing_in_weights:
        raise KeyError(f"PUNCT_WEIGHTS is missing entries for labels: {sorted(missing_in_weights)}")
    if extra_in_weights:
        raise KeyError(f"PUNCT_WEIGHTS has keys not present in label_map: {sorted(extra_in_weights)}")

    alpha = torch.ones(NUM_LABELS)
    for label_name, weight in PUNCT_WEIGHTS.items():
        alpha[label2id[label_name]] = float(weight)

    alpha = alpha * (NUM_LABELS / alpha.sum())
    return alpha


def build_trainer(cfg: "RunConfig", model, train_ds, val_ds, tokenizer):
    loss = cfg.resolved_loss()
    data_collator = make_data_collator(tokenizer, cfg)

    args = TrainingArguments(
        output_dir=f"{MODEL_DIR}/{cfg.name}_hf_tmp",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=cfg.num_epochs,
        weight_decay=0.01,
        logging_steps=50,
        gradient_accumulation_steps=GRAD_ACCUM,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=cfg.seed,
        remove_unused_columns=False,
        warmup_ratio=cfg.warmup_ratio,
    )

    if loss == "crf":
        return CRFTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
        )

    if loss == "ce" or cfg.architecture == BERT_LSTM:
        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
        )

    if loss == "focal":
        return FocalLossTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
            alpha_tensor=_focal_alpha_tensor(),
        )

    raise ValueError(f"unknown loss '{loss}' (expected 'ce' | 'focal' | 'crf')")


def train_run(cfg: "RunConfig"):
    """Train a model per cfg and save artifacts to ``models/<cfg.name>/``."""
    tokenizer = get_tokenizer()

    print(f"[{cfg.name}] architecture={cfg.architecture} loss={cfg.resolved_loss()} "
          f"epochs={cfg.num_epochs} lr={cfg.learning_rate}")
    if cfg.init_from:
        print(f"[{cfg.name}] warm-starting from {cfg.init_from}")
    if cfg.o_mask_prob > 0.0:
        effective = 0.0 if cfg.architecture == BERT_CRF else cfg.o_mask_prob
        print(f"[{cfg.name}] O-masking p={cfg.o_mask_prob} (effective={effective})")

    # CRF-only extras: class-weighted auxiliary loss + empirical transition init.
    crf_alpha = None
    crf_aux_mode = "none"
    if cfg.architecture == BERT_CRF and cfg.crf_aux_loss != "none":
        crf_aux_mode = cfg.crf_aux_loss
        crf_alpha = _focal_alpha_tensor()
        print(f"[{cfg.name}] CRF aux loss={crf_aux_mode} weight={cfg.crf_aux_weight} gamma={cfg.crf_aux_gamma}")

    model = build_model(
        cfg.architecture,
        load_from=cfg.init_from,
        crf_aux_mode=crf_aux_mode,
        crf_aux_weight=cfg.crf_aux_weight,
        crf_aux_gamma=cfg.crf_aux_gamma,
        crf_alpha=crf_alpha,
    )

    if cfg.architecture == BERT_CRF and cfg.crf_init_transitions:
        print(f"[{cfg.name}] computing empirical CRF transitions from {cfg.train_files}...")
        log_trans, log_start, log_end = compute_empirical_transitions(cfg.train_files, NUM_LABELS)
        model.init_transitions_from_prior(log_trans, log_start, log_end)
        print(f"[{cfg.name}] CRF transitions warm-started from data")

    ds = build_training_dataset(cfg, tokenizer)
    trainer = build_trainer(cfg, model, ds["train"], ds["validation"], tokenizer)

    trainer.train()
    save_model_artifacts(model, tokenizer, cfg.name, cfg.architecture)

    del trainer
    torch.cuda.empty_cache()
    return model


__all__ = ["train_run", "build_trainer"]
