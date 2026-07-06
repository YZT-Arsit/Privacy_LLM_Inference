"""Real-Qwen 4-method representation builder + Gram recovery attack.

Uses a tiny *random* Qwen2 (no download) so the same code path that runs on
Qwen-7B is exercised in CI. Asserts the qualitative security story each defense
must produce on real hidden states / real weights.
"""

from __future__ import annotations

import torch

import pllo.attacks as A
from pllo.attacks.permutation_multiset_attack import structural_leakage_probe
from pllo.attacks.real_qwen_representations import build_real_representations
from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config


def _reps():
    model = load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=0, dtype=torch.float32)
    ids = torch.randint(0, model.config.vocab_size, (1, 24),
                        generator=torch.Generator().manual_seed(0))
    return build_real_representations(model, ids, seed=0, include_fresh_pad=True)


def test_builder_produces_all_methods():
    reps = _reps()
    for m in ("plaintext_gpu", "stip_qwen", "obfuscatune_qwen_orthogonal",
              "ours_amulet_style", "ours_amulet_style_fresh_pad"):
        assert m in reps
        assert reps[m].secret_metadata_present_but_not_revealed is True


def test_plaintext_is_the_upper_bound():
    reps = _reps()
    r = A.run_nn_embedding_inversion(reps["plaintext_gpu"], target_method="plaintext_gpu",
                                     use_protected=True)
    assert r.metrics["token_recovery_top1"] == 1.0     # no protection -> full recovery
    # every masked method hides the tokens from a direct NN
    for m in ("stip_qwen", "obfuscatune_qwen_orthogonal", "ours_amulet_style"):
        rm = A.run_nn_embedding_inversion(reps[m], target_method=m, use_protected=True)
        assert rm.metrics["token_recovery_top1"] == 0.0


def test_multiset_leaks_for_permutation_not_orthogonal():
    reps = _reps()
    stip = structural_leakage_probe(reps["stip_qwen"], target_method="stip_qwen")
    obf = structural_leakage_probe(reps["obfuscatune_qwen_orthogonal"],
                                   target_method="obfuscatune_qwen_orthogonal")
    assert stip.metrics["multiset_leakage_score"] == 1.0        # permutation preserves multiset
    assert obf.metrics["multiset_leakage_score"] < 0.5          # orthogonal mixes coordinates


def test_per_token_norm_leaks_for_all_orthogonal_family():
    reps = _reps()
    for m in ("stip_qwen", "obfuscatune_qwen_orthogonal", "ours_amulet_style",
              "ours_amulet_style_fresh_pad"):
        f = A.run_frequency_distribution_attack(reps[m], target_method=m)
        assert f.metrics["frequency_rank_correlation"] > 0.99   # norm is preserved -> leaks


def test_gram_breaks_permutation_and_signed_perm_not_orthogonal():
    reps = _reps()
    stip = A.run_gram_weight_recovery(reps["stip_qwen"], target_method="stip_qwen")
    obf = A.run_gram_weight_recovery(reps["obfuscatune_qwen_orthogonal"],
                                     target_method="obfuscatune_qwen_orthogonal")
    ours = A.run_gram_weight_recovery(reps["ours_amulet_style"], target_method="ours_amulet_style")
    assert stip.metrics["attack_success_rate"] == 1.0          # permutation recovered
    assert ours.metrics["attack_success_rate"] == 1.0          # signed-perm recovered (sign-invariant)
    assert obf.metrics["attack_success_rate"] == 0.0           # orthogonal mix is NOT a Gram perm


def test_arrowmatch_underperforms_gram_on_signed_perm():
    reps = _reps()
    arrow = A.run_arrowmatch_attack(reps["ours_amulet_style"], target_method="ours_amulet_style")
    gram = A.run_gram_weight_recovery(reps["ours_amulet_style"], target_method="ours_amulet_style")
    # cosine matching is defeated by per-column sign flips; Gram is not
    assert arrow.metrics["permutation_recovery_accuracy"] < 1.0
    assert gram.metrics["attack_success_rate"] == 1.0


def test_fresh_pad_defeats_global_known_plaintext():
    reps = _reps()
    ours = A.run_kpa_known_plaintext(reps["ours_amulet_style"], target_method="ours_amulet_style",
                                     mode="linear")
    fresh = A.run_kpa_known_plaintext(reps["ours_amulet_style_fresh_pad"],
                                      target_method="ours_amulet_style_fresh_pad", mode="linear")
    assert ours.status == "measured"          # a single global mask -> KPA applies
    assert fresh.status == "blocked"          # fresh per-token mask -> no global pairs
