"""Benchmark evaluation + two-store result persistence.

- ``evaluate_run(cfg)``     -> {test_name: classification_report_dict}
- ``save_run_results(...)`` -> updates results/models_db.json, writes
  results/<name>.json and results/<name>.xlsx
- ``evaluate_and_save(cfg)`` -> convenience = the two combined
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import asdict
from typing import TYPE_CHECKING

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorForTokenClassification

from .data import build_benchmark_datasets
from .env import BATCH_SIZE, MODEL_DIR, MODELS_DB_PATH, RESULTS_DIR
from .labels import id2label
from .models import BERT_CRF, build_model, get_tokenizer

if TYPE_CHECKING:
    from .config import RunConfig


def _evaluate_model_on_data(model, dataset, data_collator, use_crf: bool):
    """Inference loop that handles plain / focal / CRF / LSTM output shapes."""
    model.eval()
    device = next(model.parameters()).device

    # drop anything the collator can't tensorize (stray text columns)
    valid_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    drop_cols = [c for c in dataset.column_names if c not in valid_cols]
    ds = dataset.remove_columns(drop_cols)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=data_collator)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            if use_crf:
                emissions = model(**batch)
                if isinstance(emissions, tuple):
                    emissions = emissions[1] if len(emissions) > 1 else emissions[0]
                mask = batch["attention_mask"].byte()
                seqs = model.crf.decode(emissions, mask=mask)
                for i, pred_seq in enumerate(seqs):
                    for j, p in enumerate(pred_seq):
                        lbl = labels[i, j].item()
                        if lbl != -100:
                            all_preds.append(p)
                            all_labels.append(lbl)
            else:
                outputs = model(**batch)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, (tuple, list)):
                    logits = outputs[1] if len(outputs) > 1 else outputs[0]
                else:
                    logits = outputs
                preds = torch.argmax(logits, dim=2)
                for i in range(labels.size(0)):
                    for j in range(labels.size(1)):
                        lbl = labels[i, j].item()
                        if lbl != -100:
                            all_preds.append(preds[i, j].item())
                            all_labels.append(lbl)

    true_tags = [id2label[l] for l in all_labels]
    pred_tags = [id2label[p] for p in all_preds]
    return classification_report(true_tags, pred_tags, output_dict=True, zero_division=0)


def evaluate_run(cfg: "RunConfig") -> dict:
    """Load ``models/<cfg.name>/`` and score it against every benchmark."""
    run_dir = os.path.join(MODEL_DIR, cfg.name)
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"no saved model at {run_dir} for run '{cfg.name}'")

    tokenizer = get_tokenizer()
    model = build_model(cfg.architecture, load_from=run_dir)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    bench = build_benchmark_datasets(cfg, tokenizer)

    use_crf = cfg.architecture == BERT_CRF
    reports: dict[str, dict] = {}
    for name, ds in bench.items():
        print(f"[{cfg.name}] evaluating on {name} (n={len(ds)})")
        reports[name] = _evaluate_model_on_data(model, ds, data_collator, use_crf=use_crf)

    del model
    torch.cuda.empty_cache()
    return reports


def _write_run_excel(run_name: str, reports: dict, out_path: str) -> None:
    """Per-run .xlsx: Summary sheet (one row) + Per-Class Details sheet.

    Same multi-index shape the legacy notebook produced, generalized to N tests.
    """
    test_names = list(reports.keys())

    # Summary shows BOTH macro and weighted so O-class domination can't hide
    # real differences between methods. Macro F1 treats all 28 classes
    # equally; weighted F1 is ~80%-dominated by the "O" class.
    sum_row = {"Model": run_name}
    labels: set[str] = set()
    for test_name in test_names:
        rep = reports[test_name]
        wa = rep.get("weighted avg", {})
        ma = rep.get("macro avg", {})
        sum_row[(test_name, "Macro_F1")] = round(ma.get("f1-score", 0.0) * 100, 2)
        sum_row[(test_name, "Macro_P")] = round(ma.get("precision", 0.0) * 100, 2)
        sum_row[(test_name, "Macro_R")] = round(ma.get("recall", 0.0) * 100, 2)
        sum_row[(test_name, "Weighted_F1")] = round(wa.get("f1-score", 0.0) * 100, 2)
        sum_row[(test_name, "Weighted_P")] = round(wa.get("precision", 0.0) * 100, 2)
        sum_row[(test_name, "Weighted_R")] = round(wa.get("recall", 0.0) * 100, 2)
        sum_row[(test_name, "Accuracy")] = round(rep.get("accuracy", 0.0) * 100, 2)
        labels.update(
            k for k in rep.keys()
            if k not in ("accuracy", "macro avg", "weighted avg")
        )

    detail_rows = []
    for label in sorted(labels):
        row = {"Model": run_name, "Punctuation": label}
        for test_name in test_names:
            rep = reports[test_name]
            if label in rep:
                m = rep[label]
                row[(test_name, "F1")] = round(m["f1-score"] * 100, 2)
                row[(test_name, "Precision")] = round(m["precision"] * 100, 2)
                row[(test_name, "Recall")] = round(m["recall"] * 100, 2)
                row[(test_name, "Support")] = m["support"]
        detail_rows.append(row)

    df_sum = pd.DataFrame([sum_row]).set_index("Model")
    df_sum.columns = pd.MultiIndex.from_tuples(df_sum.columns)

    df_det = pd.DataFrame(detail_rows)
    if not df_det.empty:
        df_det = df_det.set_index(["Model", "Punctuation"])
        df_det.columns = pd.MultiIndex.from_tuples(df_det.columns)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_sum.to_excel(writer, sheet_name="Summary")
        if not df_det.empty:
            df_det.to_excel(writer, sheet_name="Per-Class Details")


def save_run_results(cfg: "RunConfig", reports: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) update the central BERT-side DB
    if os.path.exists(MODELS_DB_PATH):
        with open(MODELS_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
    else:
        db = {}

    entry = {
        "timestamp": str(datetime.datetime.now()),
        "config": asdict(cfg),
        **reports,
    }
    db[cfg.name] = entry

    with open(MODELS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    print(f"Updated {MODELS_DB_PATH} (entry: {cfg.name})")

    # 2) per-run JSON snapshot
    per_run_json = os.path.join(RESULTS_DIR, f"{cfg.name}.json")
    with open(per_run_json, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    print(f"Wrote {per_run_json}")

    # 3) per-run Excel (Summary + Per-Class Details)
    per_run_xlsx = os.path.join(RESULTS_DIR, f"{cfg.name}.xlsx")
    _write_run_excel(cfg.name, reports, per_run_xlsx)
    print(f"Wrote {per_run_xlsx}")


def evaluate_and_save(cfg: "RunConfig") -> dict:
    reports = evaluate_run(cfg)
    save_run_results(cfg, reports)
    return reports


__all__ = ["evaluate_run", "save_run_results", "evaluate_and_save"]
