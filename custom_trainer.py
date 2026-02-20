import torch
import torch.nn as nn
from transformers import Trainer

PUNCT_WEIGHTS = {
    "O": 0.2,
    ".": 2.0,
    ",": 2.0,
    "!": 15.0,
    "?": 15.0,
    ":": 10.0,
    ";": 10.0,
    "-": 5.0,

    '"': 8.0,
    ' "': 8.0,
    '" ': 8.0,
    ', "': 10.0,
    ': "': 10.0,
    '. "': 10.0,
    '",': 10.0,
    '".': 10.0,
    '"?': 15.0,
    '"!': 15.0,
    '...': 15.0,

    '- "': 10.0,
    '", -': 12.0,
    '!" -': 15.0,
    '?" -': 15.0,
    '. -': 12.0,
    '""': 15.0,

    "! -": 15.0,
    "? -": 15.0,
    ", -": 12.0,
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
