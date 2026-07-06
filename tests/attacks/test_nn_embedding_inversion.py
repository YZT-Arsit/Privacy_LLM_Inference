"""NN embedding inversion + EDNN differential."""

from __future__ import annotations

import torch

from pllo.attacks.nn_embedding_inversion import run_nn_embedding_inversion
from pllo.attacks.representations import AttackInputs, toy_representations


def test_exact_embedding_top1():
    inp = toy_representations(vocab_size=128, hidden_size=32, num_samples=40, seed=0, obfuscation="none")
    r = run_nn_embedding_inversion(inp, target_method="plaintext_gpu", use_protected=False)
    assert r.status == "measured" and r.metrics["token_recovery_top1"] > 0.99


def test_orthogonal_rotation_drops_recovery():
    inp = toy_representations(vocab_size=128, hidden_size=32, num_samples=40, seed=1, obfuscation="orthogonal")
    r = run_nn_embedding_inversion(inp, target_method="obfuscatune_qwen_orthogonal", use_protected=True)
    assert r.metrics["token_recovery_top1"] < 0.2


def test_closed_model_blocked():
    inp = AttackInputs(token_ids=torch.arange(4))
    r = run_nn_embedding_inversion(inp, target_method="ours_amulet_style",
                                   threat_model="closed_model_no_weight_access")
    assert r.status == "blocked" and r.validate() == []


def test_ednn_differential_runs():
    inp = toy_representations(vocab_size=64, hidden_size=16, num_samples=20, seed=2, obfuscation="permutation")
    r = run_nn_embedding_inversion(inp, target_method="stip_qwen", metric="differential", use_protected=True)
    assert r.status == "measured"
    assert "EDNN" in r.attack_name or "differential" in r.notes
