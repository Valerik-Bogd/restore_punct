"""Smoke test for the new advanced knobs in pipeline/.

Exercises, without running a full training:
  1. compute_empirical_transitions on train_all.json (shape + finiteness)
  2. BertCRFForTokenClassification forward pass with aux_mode in {none, ce_weighted, focal}
  3. init_transitions_from_prior actually writes into crf.transitions
  4. OMaskingCollator drops O tokens at ~target rate
  5. make_data_collator routes correctly (focal->OMasking, bert+crf->plain)
  6. build_model accepts + round-trips CRF aux state via save/load

Run with:  anaconda3/envs/neural/bin/python scripts/_smoke_new_paths.py
"""

import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import DataCollatorForTokenClassification

from pipeline.config import RunConfig
from pipeline.data import (
    compute_empirical_transitions,
    make_data_collator,
    OMaskingCollator,
)
from pipeline.labels import NUM_LABELS, label2id
from pipeline.models import (
    BertCRFForTokenClassification,
    build_model,
    get_tokenizer,
    save_model_artifacts,
)


def check(cond, msg):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {msg}")
    if not cond:
        raise SystemExit(1)


def test_empirical_transitions():
    print("\n--- 1. compute_empirical_transitions ---")
    log_trans, log_start, log_end = compute_empirical_transitions(
        [f"{ROOT}/data/train_all.json"], NUM_LABELS
    )
    check(log_trans.shape == (NUM_LABELS, NUM_LABELS), f"log_trans shape {log_trans.shape}")
    check(log_start.shape == (NUM_LABELS,), f"log_start shape {log_start.shape}")
    check(log_end.shape == (NUM_LABELS,), f"log_end shape {log_end.shape}")
    check(torch.isfinite(log_trans).all().item(), "log_trans all finite")
    check(torch.isfinite(log_start).all().item(), "log_start all finite")
    check(torch.isfinite(log_end).all().item(), "log_end all finite")
    # rows should sum to 1 in prob space -> logsumexp == 0 (within FP tol)
    row_lse = torch.logsumexp(log_trans, dim=1)
    check(torch.allclose(row_lse, torch.zeros_like(row_lse), atol=1e-5),
          f"log_trans rows normalize (max dev {row_lse.abs().max().item():.2e})")
    print(f"     O->O log-prob = {log_trans[label2id['O'], label2id['O']].item():.3f}")
    return log_trans, log_start, log_end


def test_crf_aux_forward():
    print("\n--- 2. BertCRFForTokenClassification aux forward/backward ---")
    torch.manual_seed(0)
    B, T = 2, 16
    input_ids = torch.randint(0, 1000, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    labels = torch.randint(0, NUM_LABELS, (B, T))
    labels[0, 0] = -100  # ignore in aux

    for mode in ("none", "ce_weighted", "focal"):
        alpha = torch.rand(NUM_LABELS) + 0.5
        m = BertCRFForTokenClassification(
            "DeepPavlov/rubert-base-cased-sentence",
            num_labels=NUM_LABELS,
            aux_mode=mode,
            aux_weight=0.3,
            aux_gamma=2.0,
            alpha=alpha,
        )
        loss, emissions = m(input_ids, attention_mask=attention_mask, labels=labels)
        check(torch.is_tensor(loss) and loss.dim() == 0, f"aux_mode={mode} loss is scalar")
        check(torch.isfinite(loss).item(), f"aux_mode={mode} loss finite ({loss.item():.3f})")
        loss.backward()  # must not raise
        check(True, f"aux_mode={mode} backward OK")


def test_transition_init(log_trans, log_start, log_end):
    print("\n--- 3. init_transitions_from_prior ---")
    m = BertCRFForTokenClassification(
        "DeepPavlov/rubert-base-cased-sentence", num_labels=NUM_LABELS,
    )
    before = m.crf.transitions.data.clone()
    m.init_transitions_from_prior(log_trans, log_start, log_end)
    after = m.crf.transitions.data
    check(not torch.allclose(before, after), "CRF transitions actually changed")
    check(torch.allclose(after, log_trans), "CRF transitions == log_trans prior")


def test_omasking():
    print("\n--- 4. OMaskingCollator ---")
    tok = get_tokenizer()
    base = DataCollatorForTokenClassification(tok)
    o_id = label2id["O"]
    p = 0.5
    collator = OMaskingCollator(base, o_label_id=o_id, p=p)

    features = [
        {"input_ids": [101, 200, 300, 400, 500, 102],
         "attention_mask": [1, 1, 1, 1, 1, 1],
         "labels":         [-100, o_id, o_id, o_id, o_id, -100]}
        for _ in range(200)
    ]
    torch.manual_seed(1)
    batch = collator(features)
    labels = batch["labels"]
    dropped = (labels == -100).sum().item()
    o_plus_pad = sum(1 for f in features for x in f["labels"] if x == -100 or x == o_id)
    pad = sum(1 for f in features for x in f["labels"] if x == -100)
    drop_rate = (dropped - pad) / (o_plus_pad - pad)
    check(abs(drop_rate - p) < 0.05, f"O drop rate ~ p (got {drop_rate:.3f}, p={p})")


def test_routing():
    print("\n--- 5. make_data_collator routing ---")
    tok = get_tokenizer()
    cfg_ce   = RunConfig(name="t", architecture="bert",     loss="ce",    o_mask_prob=0.3)
    cfg_foc  = RunConfig(name="t", architecture="bert",     loss="focal", o_mask_prob=0.3)
    cfg_off  = RunConfig(name="t", architecture="bert",     loss="focal", o_mask_prob=0.0)
    cfg_crf  = RunConfig(name="t", architecture="bert+crf", loss="crf",   o_mask_prob=0.3)

    check(isinstance(make_data_collator(tok, cfg_ce),  OMaskingCollator), "ce+o_mask -> OMasking")
    check(isinstance(make_data_collator(tok, cfg_foc), OMaskingCollator), "focal+o_mask -> OMasking")
    check(not isinstance(make_data_collator(tok, cfg_off), OMaskingCollator), "focal no o_mask -> plain")
    check(not isinstance(make_data_collator(tok, cfg_crf), OMaskingCollator), "bert+crf -> plain (no-op)")


def test_build_and_save():
    print("\n--- 6. build_model + save/load round-trip (CRF + aux) ---")
    m = build_model(
        "bert+crf",
        crf_aux_mode="focal",
        crf_aux_weight=0.25,
        crf_aux_gamma=2.0,
        crf_alpha=torch.arange(NUM_LABELS).float() + 1.0,
    )
    check(m.aux_mode == "focal", f"aux_mode set ({m.aux_mode})")
    check(abs(m.aux_weight - 0.25) < 1e-9, f"aux_weight set ({m.aux_weight})")
    check(m._alpha[0].item() == 1.0, "_alpha set")

    tok = get_tokenizer()
    from pipeline.env import MODEL_DIR as _MD
    run_name = "smoke_crf_tmp"
    save_dir = os.path.join(_MD, run_name)
    try:
        save_model_artifacts(m, tok, run_name, "bert+crf")
        check(os.path.isfile(f"{save_dir}/pytorch_model.bin"), "saved pytorch_model.bin")
        # reload, passing fresh alpha; saved buffer should override it
        m2 = build_model(
            "bert+crf",
            load_from=save_dir,
            crf_aux_mode="focal",
            crf_aux_weight=0.25,
            crf_aux_gamma=2.0,
            crf_alpha=torch.zeros(NUM_LABELS),  # different; should be overwritten
        )
        check(torch.allclose(m2._alpha.cpu(), m._alpha.cpu()),
              "reloaded _alpha matches original")
    finally:
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)


if __name__ == "__main__":
    log_trans, log_start, log_end = test_empirical_transitions()
    test_crf_aux_forward()
    test_transition_init(log_trans, log_start, log_end)
    test_omasking()
    test_routing()
    test_build_and_save()
    print("\nAll smoke tests passed.")
