import torch
import torch.nn as nn
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Per-class weights for CrossEntropy/Focal/CRF-aux losses.
#
# Every key is a label string that appears VERBATIM in data/label_map.json;
# pipeline/training.py::_focal_alpha_tensor asserts full coverage at build time
# so silent misses (old bug: ' "', '" ', '...' were never applied; labels
# '?"', '!"', '" -' got no weight at all) are no longer possible.
#
# Tier rationale (empirical frequency on train_all.json):
#   O  81.0%   -> 0.2   (massively down-weighted; still seen billions of times)
#   ,  7.9%,  .  5.3%   -> 2.0  (frequent; mild boost)
#   "  2.5%             -> 8.0
#   compounds 0.1-0.5%  -> 10-12
#   rare singletons / compounds <0.1% -> 15
#
# Absolute magnitudes are irrelevant because _focal_alpha_tensor normalizes
# the returned tensor to mean 1.0 -- only the RATIOS between classes matter.
PUNCT_WEIGHTS = {
    # majority class
    "O":      0.2,
    # frequent singletons
    ",":      2.0,
    ".":      2.0,
    '"':      8.0,
    # moderate singletons
    "-":      5.0,
    ":":     10.0,
    ";":     10.0,
    # common compounds (0.1-0.5% of tokens)
    '".':    10.0,
    '. "':   10.0,
    '",':    10.0,
    '", -':  10.0,
    '. -':   12.0,
    ', -':   12.0,
    # rare singletons
    "!":     15.0,
    "?":     15.0,
    # rare compounds
    '" -':   12.0,
    '? -':   15.0,
    ', "':   12.0,
    ': "':   12.0,
    '- "':   12.0,
    '?"':    15.0,
    '!"':    15.0,
    '""':    15.0,
    # very rare compounds (<0.01% of tokens)
    '"?':    15.0,
    '"!':    15.0,
    '?" -':  15.0,
    '!" -':  15.0,
    '! -':   15.0,
}


class WeightedTrainer(Trainer):
    def __init__(self, class_weights_tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # weights on CPU initially
        # device dynamically during forward pass
        self.class_weights = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # weights to device
        weights = self.class_weights.to(model.device)
        
        # Loss
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        
        # logits: (Batch * Seq_Len, Num_Labels)
        # labels: (Batch * Seq_Len)
        num_labels = self.model.config.num_labels
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


# class FocalLossTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)

#         if hasattr(outputs, "logits"):
#             logits = outputs.logits
#         elif isinstance(outputs, (tuple, list)):
#             # Common pattern: (loss, logits) or (logits,)
#             logits = outputs[-1]
#         else:
#             logits = outputs

#         num_labels = getattr(model, "num_labels", None)
#         if num_labels is None:
#             num_labels = logits.shape[-1]

#         logits = logits.view(-1, num_labels)
#         labels = labels.view(-1)

#         ce_loss = torch.nn.functional.cross_entropy(
#             logits, labels, reduction="none", ignore_index=-100
#         )
#         pt = torch.exp(-ce_loss)
#         focal_loss = (1 - pt) ** 2.0 * ce_loss  # standard Gamma=2.0
        
#         focal_loss = focal_loss.mean()
        
#         return (focal_loss, outputs) if return_outputs else focal_loss
    
import torch
from transformers import Trainer

class FocalLossTrainer(Trainer):
    def __init__(self, *args, alpha_tensor=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_tensor = alpha_tensor
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[1]
        else:
            logits = outputs

        num_labels = logits.shape[-1]
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        valid_mask = labels_flat != -100
        valid_logits = logits_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]

        if valid_labels.numel() == 0:
            loss = valid_logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        ce_loss = torch.nn.functional.cross_entropy(
            valid_logits, valid_labels, reduction="none"
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha_tensor is not None:
            alpha_weights = self.alpha_tensor.to(valid_labels.device)
            focal_loss = focal_loss * alpha_weights[valid_labels]

        loss = focal_loss.mean()
        
        return (loss, outputs) if return_outputs else loss


class CRFTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs[0]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs[0] if "labels" in inputs else None
            emissions = outputs[1] if "labels" in inputs else outputs
            # attention mask for decoding
            mask = inputs["attention_mask"].byte()
            # Viterbi
            preds = model.crf.decode(emissions, mask=mask)
            max_len = emissions.shape[1]
            padded_preds = [p + [0] * (max_len - len(p)) for p in preds]
            preds_tensor = torch.tensor(padded_preds, device=emissions.device)
        if prediction_loss_only:
            return (loss, None, None)
        # convert 2D Viterbi preds into 3D one-hot vectors
        one_hot_preds = torch.nn.functional.one_hot(preds_tensor, num_classes=model.num_labels).float()
        return (loss, one_hot_preds, inputs.get("labels"))
