"""
`RunConfig` dataclass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunConfig:
    name: str

    # "bert" | "bert+crf" | "bert+lstm"
    architecture: str = "bert"

    # "ce" | "focal" | "crf"   (crf is forced when architecture == "bert+crf")
    loss: str = "ce"

    train_files: list[str] = field(default_factory=list)
    val_files: list[str] = field(default_factory=list)

    # replay pool
    replay_files: list[dict] = field(default_factory=list)

    # path to a previous models/<name>/ dir to warm-start
    init_from: Optional[str] = None

    num_epochs: int = 3
    learning_rate: float = 2e-5

    benchmarks: Optional[dict] = None

    seed: int = 42

    # Warm-start the CRF transition matrix from empirical label
    crf_init_transitions: bool = False

    # Aux per-token loss on CRF emissions to add class-imbalance signal
    # "none" | "ce_weighted" | "focal"
    crf_aux_loss: str = "none"
    crf_aux_weight: float = 0.2   # lambda on the auxiliary term
    crf_aux_gamma: float = 2.0    # focal gamma when crf_aux_loss="focal"

    # At every training step, randomly relabel O tokens to -100 with this prob
    o_mask_prob: float = 0.0

    # LR schedule
    warmup_ratio: float = 0.0

    # metadata
    tags: dict = field(default_factory=dict)

    def resolved_loss(self) -> str:
        if self.architecture == "bert+crf":
            return "crf"
        return self.loss


__all__ = ["RunConfig"]
