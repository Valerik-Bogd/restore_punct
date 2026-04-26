"""Research pipeline for Russian punctuation restoration.

Every experiment is expressed as a single `RunConfig` and executed by
`train_run` + `evaluate_and_save`. All artifacts land under `models/<name>/`
and `results/<name>.{json,xlsx}`, and the two JSON stores
(`results/models_db.json`, `results/yandex_db.json`) are merged into
`results/master_summary.xlsx` by `rebuild_master_excel`.
"""

from .config import RunConfig

__all__ = ["RunConfig"]
