"""KPA: linear recovery with enough pairs; fresh pad fails; permutation recovery."""

from __future__ import annotations

from pllo.attacks.kpa_known_plaintext import kpa_linear_single, run_kpa_known_plaintext
from pllo.attacks.representations import AttackInputs, toy_representations


def test_linear_recovered_with_enough_pairs():
    inp = toy_representations(vocab_size=256, hidden_size=32, num_samples=200, seed=0, obfuscation="orthogonal")
    X, Xstar = inp.known_plaintext_pairs
    under = kpa_linear_single(X, Xstar, num_pairs=8, seed=0)["deobfuscation_relative_error"]
    over = kpa_linear_single(X, Xstar, num_pairs=64, seed=0)["deobfuscation_relative_error"]
    assert over < 1e-4 and under > over


def test_run_kpa_success_and_threat_model():
    inp = toy_representations(vocab_size=256, hidden_size=32, num_samples=200, seed=0, obfuscation="orthogonal")
    r = run_kpa_known_plaintext(inp, target_method="obfuscatune_qwen_orthogonal", mode="linear")
    assert r.threat_model == "known_plaintext"
    assert r.attacker_knowledge["has_known_plaintext_pairs"] is True
    assert r.metrics["attack_success_rate"] > 0.99 and r.validate() == []


def test_fresh_padding_defeats_kpa():
    inp = toy_representations(vocab_size=256, hidden_size=32, num_samples=200, seed=1, obfuscation="fresh_pad")
    r = run_kpa_known_plaintext(inp, target_method="ours_amulet_style", mode="linear")
    assert r.metrics["attack_success_rate"] < 0.5 and "INTENDED PROTECTION" in r.notes


def test_permutation_kpa_recovers():
    inp = toy_representations(vocab_size=256, hidden_size=32, num_samples=100, seed=0, obfuscation="permutation")
    r = run_kpa_known_plaintext(inp, target_method="stip_qwen", mode="permutation")
    assert r.metrics["permutation_recovery_accuracy"] > 0.99


def test_blocked_without_pairs():
    r = run_kpa_known_plaintext(AttackInputs(), target_method="toy")
    assert r.status == "blocked"
