"""ArrowMatch, permutation/multiset leakage, frequency attacks."""

from __future__ import annotations

import torch

from pllo.attacks.arrowmatch_attack import arrowmatch_single, run_arrowmatch_attack
from pllo.attacks.frequency_distribution_attack import run_frequency_distribution_attack
from pllo.attacks.permutation_multiset_attack import structural_leakage_probe
from pllo.attacks.representations import AttackInputs, toy_representations


def test_arrowmatch_recovers_permutation():
    # public weights, obfuscated = columns permuted + positively scaled
    g = torch.Generator().manual_seed(0)
    W = torch.randn(24, 24, generator=g)
    perm = torch.randperm(24, generator=g)
    scales = torch.rand(24, generator=g) + 0.5
    obf = (W * scales.unsqueeze(0))[:, perm]
    sigma, _ = arrowmatch_single(obf, W, bijection=True)
    # obf col i = W col perm[i] scaled -> sigma[i] should equal perm[i]
    acc = (sigma == perm).float().mean().item()
    assert acc > 0.99


def test_arrowmatch_blocked_without_weights():
    inp = toy_representations(seed=0, obfuscation="orthogonal")  # no model_weights
    r = run_arrowmatch_attack(inp, target_method="obfuscatune_qwen_orthogonal")
    assert r.status == "blocked" and r.threat_model == "weight_leakage_worst_case"


def test_arrowmatch_worst_case_flags_weights():
    inp = toy_representations(seed=0, obfuscation="permutation")  # sets model_weights
    r = run_arrowmatch_attack(inp, target_method="stip_qwen")
    assert r.status == "measured"
    assert r.attacker_knowledge["has_model_weights"] is True
    assert r.metrics["attack_success_rate"] > 0.9 and r.validate() == []


def test_multiset_leakage_high_for_permutation():
    inp = toy_representations(vocab_size=64, hidden_size=16, num_samples=30, seed=0, obfuscation="permutation")
    r = structural_leakage_probe(inp, target_method="stip_qwen")
    assert r.metrics["multiset_leakage_score"] > 0.9      # permutation preserves multiset


def test_multiset_leakage_low_for_random_matrix():
    inp = toy_representations(vocab_size=64, hidden_size=16, num_samples=30, seed=0, obfuscation="random")
    r = structural_leakage_probe(inp, target_method="obfuscatune_qwen_random")
    assert r.metrics["multiset_leakage_score"] < 0.9      # matrix mixing breaks the multiset


def test_frequency_norm_preserved_under_orthogonal():
    inp = toy_representations(vocab_size=64, hidden_size=16, num_samples=30, seed=0, obfuscation="orthogonal")
    r = run_frequency_distribution_attack(inp, target_method="obfuscatune_qwen_orthogonal")
    assert r.metrics["multiset_leakage_score"] > 0.9      # orthogonal preserves per-token norm
