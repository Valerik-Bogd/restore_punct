"""Label map loader. Reads ``data/label_map.json`` once and exposes the usual
helpers (``id2label``, ``label2id``, ``LABELS``, ``NUM_LABELS``).
"""

from __future__ import annotations

import json
import os

from .env import DATA_DIR

_LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.json")

with open(_LABEL_MAP_PATH, "r", encoding="utf-8") as _f:
    _raw = json.load(_f)

id2label: dict[int, str] = {int(k): v for k, v in _raw.items()}
label2id: dict[str, int] = {v: k for k, v in id2label.items()}
LABELS: list[str] = list(id2label.values())
NUM_LABELS: int = len(LABELS)

__all__ = ["id2label", "label2id", "LABELS", "NUM_LABELS"]
