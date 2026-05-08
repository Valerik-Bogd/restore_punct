"""
- "bert"      -> HuggingFace ``AutoModelForTokenClassification`` (save_pretrained / from_pretrained)
- "bert+crf"  -> custom head, state_dict serialized as ``pytorch_model.bin``
- "bert+lstm" -> custom head, state_dict serialized as ``pytorch_model.bin``
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer

from .env import MODEL_DIR, MODEL_NAME
from .labels import NUM_LABELS, id2label, label2id


def _load_with_local_fallback(loader_fn, *args, **kwargs):
    """
    try normally (online + cache)
    or retry once with ``local_files_only=True``
    or raise error
    """
    try:
        return loader_fn(*args, **kwargs)
    except Exception as original:
        try:
            out = loader_fn(*args, **kwargs, local_files_only=True)
            ident = args[0] if args else kwargs.get("pretrained_model_name_or_path", "?")
            print(f"[pipeline] Hub unreachable -> loaded '{ident}' from HF cache")
            return out
        except Exception:
            raise original

try:
    from torchcrf import CRF
except ImportError:
    CRF = None


BERT = "bert"
BERT_CRF = "bert+crf"
BERT_LSTM = "bert+lstm"
_VALID_ARCH = (BERT, BERT_CRF, BERT_LSTM)

CUSTOM_WEIGHTS_FILE = "pytorch_model.bin"


_VALID_AUX_MODES = ("none", "ce_weighted", "focal")


class BertCRFForTokenClassification(nn.Module):
    """
    BERT + linear head + linear-chain CRF, 
    optionally + weighted-CE or focal aux loss on the emissions
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        aux_mode: str = "none",
        aux_weight: float = 0.2,
        aux_gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
    ):
        super().__init__()
        if CRF is None:
            raise ImportError("torchcrf is required for 'bert+crf' (`pip install pytorch-crf`).")
        if aux_mode not in _VALID_AUX_MODES:
            raise ValueError(f"aux_mode must be one of {_VALID_AUX_MODES}, got {aux_mode!r}")

        self.num_labels = num_labels
        self.bert = _load_with_local_fallback(AutoModel.from_pretrained, model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        self.aux_mode = aux_mode
        self.aux_weight = float(aux_weight)
        self.aux_gamma = float(aux_gamma)

        if alpha is None:
            alpha = torch.ones(num_labels)
        self.register_buffer("_alpha", alpha.float())

    def init_transitions_from_prior(
        self,
        log_transitions: torch.Tensor,
        log_start: torch.Tensor | None = None,
        log_end: torch.Tensor | None = None,
    ) -> None:
        """Overwrite torchcrf's random transition init with empirical."""
        with torch.no_grad():
            self.crf.transitions.data.copy_(log_transitions.to(self.crf.transitions.device))
            if log_start is not None:
                self.crf.start_transitions.data.copy_(log_start.to(self.crf.start_transitions.device))
            if log_end is not None:
                self.crf.end_transitions.data.copy_(log_end.to(self.crf.end_transitions.device))

    def _aux_loss(self, emissions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat_emi = emissions.reshape(-1, self.num_labels)
        flat_lab = labels.reshape(-1)
        valid = flat_lab != -100
        if not valid.any():
            return emissions.sum() * 0.0

        v_emi = flat_emi[valid]
        v_lab = flat_lab[valid]
        alpha = self._alpha.to(v_emi.device)

        if self.aux_mode == "ce_weighted":
            return F.cross_entropy(v_emi, v_lab, weight=alpha)

        # focal
        ce = F.cross_entropy(v_emi, v_lab, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.aux_gamma) * ce
        return (focal * alpha[v_lab]).mean()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            mask = attention_mask.type(torch.uint8)
            safe_labels = torch.where(labels == -100, torch.zeros_like(labels), labels)
            crf_nll = -self.crf(emissions, safe_labels, mask=mask, reduction="mean")

            if self.aux_mode == "none":
                return (crf_nll, emissions)

            aux = self._aux_loss(emissions, labels)
            return (crf_nll + self.aux_weight * aux, emissions)
        return emissions


class BertLSTMForTokenClassification(nn.Module):
    def __init__(self, model_name: str, num_labels: int, lstm_hidden: int = 256):
        super().__init__()
        self.num_labels = num_labels
        self.bert = _load_with_local_fallback(AutoModel.from_pretrained, model_name)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        logits = self.classifier(lstm_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return (loss, logits)
        return logits


def build_model(
    architecture: str,
    load_from: str | None = None,
    crf_aux_mode: str = "none",
    crf_aux_weight: float = 0.2,
    crf_aux_gamma: float = 2.0,
    crf_alpha: torch.Tensor | None = None,
) -> nn.Module:
    """
    - ``"bert"``       -> ``AutoModelForTokenClassification`` (``save_pretrained`` / ``from_pretrained``)
    - ``"bert+crf"``   -> custom module, state_dict in ``pytorch_model.bin``.
    - ``"bert+lstm"``  -> custom module, state_dict in ``pytorch_model.bin``.
    """
    if architecture not in _VALID_ARCH:
        raise ValueError(f"unknown architecture '{architecture}', expected one of {_VALID_ARCH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if architecture == BERT:
        src = load_from or MODEL_NAME
        model = _load_with_local_fallback(
            AutoModelForTokenClassification.from_pretrained,
            src,
            num_labels=NUM_LABELS,
            id2label=id2label,
            label2id=label2id,
        )
        return model.to(device)

    if architecture == BERT_CRF:
        model = BertCRFForTokenClassification(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            aux_mode=crf_aux_mode,
            aux_weight=crf_aux_weight,
            aux_gamma=crf_aux_gamma,
            alpha=crf_alpha,
        )
    else:
        model = BertLSTMForTokenClassification(MODEL_NAME, num_labels=NUM_LABELS)

    if load_from is not None:
        weights_path = os.path.join(load_from, CUSTOM_WEIGHTS_FILE)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        # Older checkpoints don't carry the `_alpha` buffer; tolerate it.
        model.load_state_dict(state, strict=False)

    return model.to(device)


def save_model_artifacts(model: nn.Module, tokenizer, run_name: str, architecture: str) -> str:
    """
    model + tokenizer into ``models/<run_name>/``.
    """
    save_dir = os.path.join(MODEL_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)

    if architecture == BERT:
        model.save_pretrained(save_dir)
    elif architecture in (BERT_CRF, BERT_LSTM):
        torch.save(model.state_dict(), os.path.join(save_dir, CUSTOM_WEIGHTS_FILE))
    else:
        raise ValueError(f"unknown architecture '{architecture}'")

    tokenizer.save_pretrained(save_dir)
    print(f"Model saved -> {save_dir}")
    return save_dir


def get_tokenizer():
    return _load_with_local_fallback(AutoTokenizer.from_pretrained, MODEL_NAME)


__all__ = [
    "BERT",
    "BERT_CRF",
    "BERT_LSTM",
    "BertCRFForTokenClassification",
    "BertLSTMForTokenClassification",
    "build_model",
    "save_model_artifacts",
    "get_tokenizer",
]
