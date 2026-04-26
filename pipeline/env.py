"""Environment detection + shared paths / batch sizes.

Colab vs. local is auto-detected. All downstream modules import from here
instead of hard-coding paths or batch sizes.
"""

from __future__ import annotations

import os
import sys

try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except ImportError:
    IS_COLAB = False


MODEL_NAME = "DeepPavlov/rubert-base-cased-sentence"
MAX_LEN = 512

if IS_COLAB:
    ROOT_DIR = "/content/drive/MyDrive/omg_diploma_2025/restore_punct"
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    GRAD_ACCUM = 2
else:
    # When this file is imported from notebooks/ we still want ROOT_DIR to
    # point to the project root (one level up from pipeline/).
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # RTX 5060 (8 GB VRAM)
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    GRAD_ACCUM = 4

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

for _d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


MODELS_DB_PATH = os.path.join(RESULTS_DIR, "models_db.json")
YANDEX_DB_PATH = os.path.join(RESULTS_DIR, "yandex_db.json")
MASTER_XLSX_PATH = os.path.join(RESULTS_DIR, "master_summary.xlsx")


__all__ = [
    "IS_COLAB",
    "ROOT_DIR",
    "DATA_DIR",
    "MODEL_DIR",
    "RESULTS_DIR",
    "MODEL_NAME",
    "MAX_LEN",
    "BATCH_SIZE",
    "NUM_WORKERS",
    "GRAD_ACCUM",
    "MODELS_DB_PATH",
    "YANDEX_DB_PATH",
    "MASTER_XLSX_PATH",
]
