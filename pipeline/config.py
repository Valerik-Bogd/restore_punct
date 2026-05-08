"""`RunConfig` dataclass: the single source of truth for any training run.

Every experiment notebook is `RunConfig(...)` + three boilerplate calls.
Persisted verbatim into `results/models_db.json` under the run's key, so
any past run is fully reproducible from its saved config alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunConfig:
    # unique folder name; drives models/<name>/ and results/<name>.{json,xlsx}
    name: str

    # "bert" | "bert+crf" | "bert+lstm"
    architecture: str = "bert"

    # "ce" | "focal" | "crf"   (crf is forced whenever architecture == "bert+crf")
    loss: str = "ce"

    # absolute or ROOT_DIR-relative paths; concatenated then shuffled
    train_files: list[str] = field(default_factory=list)
    val_files: list[str] = field(default_factory=list)

    # replay pool: each entry = {"path": "...", "n": 15000, "seed": 42}
    # Sampled, concatenated after train_files, everything re-shuffled with cfg.seed.
    replay_files: list[dict] = field(default_factory=list)

    # path to a previous models/<name>/ dir to warm-start from (fine-tuning)
    init_from: Optional[str] = None

    num_epochs: int = 3
    learning_rate: float = 2e-5

    # override the default 3-benchmark eval set if needed
    # {"test_name": "path/to/file.json"}; None => use defaults from data.py
    benchmarks: Optional[dict] = None

    seed: int = 42

    # --- Advanced knobs (all default-off; compose freely) -------------------
    # Warm-start the CRF transition matrix from empirical (prev -> next) label
    # pair frequencies in the training data, with add-one smoothing. Fixes
    # CRF's random-transition cold start. Only applies to architecture="bert+crf".
    crf_init_transitions: bool = False

    # Auxiliary per-token loss on CRF emissions to inject class-imbalance signal
    # (CRF log-likelihood is sequence-level and inherently ignores PUNCT_WEIGHTS).
    # "none" | "ce_weighted" | "focal". Uses the same PUNCT_WEIGHTS alpha as
    # FocalLossTrainer. Only applies to architecture="bert+crf".
    crf_aux_loss: str = "none"
    crf_aux_weight: float = 0.2   # lambda on the auxiliary term
    crf_aux_gamma: float = 2.0    # focal gamma when crf_aux_loss="focal"

    # At every training step, randomly relabel O tokens to -100 with this
    # probability, rebalancing the 81/19 O/punctuation ratio. Evaluation is
    # unaffected. No-op for "bert+crf" (CRF's safe_labels hack re-maps -100
    # back to O); intended for CE/focal paths.
    o_mask_prob: float = 0.0

    # LR schedule: fraction of total steps used for linear warmup.
    # 0.0 = no warmup (backward-compatible default).
    warmup_ratio: float = 0.0

    # free-form metadata (e.g. {"stage": "1-baseline-loss-sweep"})
    tags: dict = field(default_factory=dict)

    def resolved_loss(self) -> str:
        """Force 'crf' loss for CRF architecture so users can't mis-configure."""
        if self.architecture == "bert+crf":
            return "crf"
        return self.loss


__all__ = ["RunConfig"]
