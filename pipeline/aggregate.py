"""Merge ``models_db.json`` + ``yandex_db.json`` into ``master_summary.xlsx``.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from .env import MASTER_XLSX_PATH, MODELS_DB_PATH, YANDEX_DB_PATH

_RESERVED_TOP = {"timestamp", "config"}
_RESERVED_CLASS = {"accuracy", "macro avg", "weighted avg"}


def _load_db(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_tests(entry: dict):
    for k, v in entry.items():
        if k in _RESERVED_TOP:
            continue
        if isinstance(v, dict) and ("weighted avg" in v or any(isinstance(vv, dict) for vv in v.values())):
            yield k, v


def _summary_row(source: str, run_name: str, entry: dict, all_tests: list[str]) -> dict:
    row = {"Source": source, "Run": run_name}
    for test_name in all_tests:
        rep = entry.get(test_name)
        if not isinstance(rep, dict):
            continue
        wa = rep.get("weighted avg", {})
        ma = rep.get("macro avg", {})
        if not wa and not ma:
            continue
        row[(test_name, "Macro_F1")] = round(ma.get("f1-score", 0.0) * 100, 2)
        row[(test_name, "Macro_P")] = round(ma.get("precision", 0.0) * 100, 2)
        row[(test_name, "Macro_R")] = round(ma.get("recall", 0.0) * 100, 2)
        row[(test_name, "Weighted_F1")] = round(wa.get("f1-score", 0.0) * 100, 2)
        row[(test_name, "Weighted_P")] = round(wa.get("precision", 0.0) * 100, 2)
        row[(test_name, "Weighted_R")] = round(wa.get("recall", 0.0) * 100, 2)
        row[(test_name, "Accuracy")] = round(rep.get("accuracy", 0.0) * 100, 2)
    return row


def _detail_rows(source: str, run_name: str, entry: dict, all_tests: list[str]) -> list[dict]:
    labels: set[str] = set()
    for test_name in all_tests:
        rep = entry.get(test_name)
        if isinstance(rep, dict):
            labels.update(k for k in rep.keys() if k not in _RESERVED_CLASS)

    rows = []
    for label in sorted(labels):
        row = {"Source": source, "Run": run_name, "Punctuation": label}
        for test_name in all_tests:
            rep = entry.get(test_name)
            if not isinstance(rep, dict):
                continue
            m = rep.get(label)
            if not isinstance(m, dict):
                continue
            row[(test_name, "F1")] = round(m.get("f1-score", 0.0) * 100, 2)
            row[(test_name, "Precision")] = round(m.get("precision", 0.0) * 100, 2)
            row[(test_name, "Recall")] = round(m.get("recall", 0.0) * 100, 2)
            row[(test_name, "Support")] = m.get("support", 0)
        rows.append(row)
    return rows


def rebuild_master_excel(out_path: str = MASTER_XLSX_PATH) -> str:
    models_db = _load_db(MODELS_DB_PATH)
    yandex_db = _load_db(YANDEX_DB_PATH)

    if not models_db and not yandex_db:
        print("Nothing to aggregate yet: both JSON stores are empty.")
        return out_path

    preferred = ["General_Test", "GERA_Test", "LORuGEC_Test"]
    seen_tests: list[str] = []
    def _collect(db):
        for entry in db.values():
            for tname, _ in _iter_tests(entry):
                if tname not in seen_tests:
                    seen_tests.append(tname)
    _collect(models_db)
    _collect(yandex_db)
    all_tests = [t for t in preferred if t in seen_tests] + [t for t in seen_tests if t not in preferred]

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    for source, db in (("bert", models_db), ("yandex", yandex_db)):
        for run_name, entry in db.items():
            summary_rows.append(_summary_row(source, run_name, entry, all_tests))
            detail_rows.extend(_detail_rows(source, run_name, entry, all_tests))

    df_sum = pd.DataFrame(summary_rows)
    if not df_sum.empty:
        df_sum = df_sum.set_index(["Source", "Run"])
        df_sum.columns = pd.MultiIndex.from_tuples(df_sum.columns)

    df_det = pd.DataFrame(detail_rows)
    if not df_det.empty:
        df_det = df_det.set_index(["Source", "Run", "Punctuation"])
        df_det.columns = pd.MultiIndex.from_tuples(df_det.columns)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if not df_sum.empty:
            df_sum.to_excel(writer, sheet_name="Summary")
        if not df_det.empty:
            df_det.to_excel(writer, sheet_name="Per-Class Details")

    print(f"Rebuilt master table -> {out_path}")
    print(f"  BERT runs   : {len(models_db)}")
    print(f"  Yandex runs : {len(yandex_db)}")
    return out_path


__all__ = ["rebuild_master_excel"]
