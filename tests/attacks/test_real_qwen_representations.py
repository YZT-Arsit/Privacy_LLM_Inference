"""Real-Qwen multi-method representation builder + Gram recovery + norm audit.

Uses a tiny *random* Qwen2 (no download) so the same code path that runs on
Qwen-7B is exercised in CI. Asserts the qualitative security story each defense
must produce on real hidden states / real weights.
"""

from __future__ import annotations

import torch

import pllo.attacks as A
from pllo.attacks.norm_leakage_audit import audit_norm_leakage
from pllo.attacks.permutation_multiset_attack import structural_leakage_probe
from pllo.attacks.real_qwen_representations import BUILDABLE_METHODS, build_real_representations
from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config

OBF = "obfuscatune_qwen_orthogonal"
SIGNED = "ours_amulet_style_signed_perm"
FRESH = "ours_amulet_style_fresh_pad"
NONISO = "ours_non_isometric_variant"


def _reps():
    model = load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=0, dtype=torch.float32)
    ids = torch.randint(0, model.config.vocab_size, (1, 24),
                        generator=torch.Generator().manual_seed(0))
    return build_real_representations(model, ids, seed=0)


def test_builder_produces_all_buildable_methods():
    reps = _reps()
    for m in BUILDABLE_METHODS:
        assert m in reps
        assert reps[m].secret_metadata_present_but_not_revealed is True
        assert reps[m].known_plaintext_pairs is not None      # KPA runs on every method


def test_plaintext_is_the_upper_bound():
    reps = _reps()
    r = A.run_nn_embedding_inversion(reps["plaintext_gpu"], target_method="plaintext_gpu",
                                     use_protected=True)
    assert r.metrics["token_recovery_top1"] == 1.0
    for m in ("stip_qwen", OBF, SIGNED, FRESH, NONISO):
        rm = A.run_nn_embedding_inversion(reps[m], target_method=m, use_protected=True)
        assert rm.metrics["token_recovery_top1"] == 0.0


def test_gram_breaks_static_permutation_and_signed_perm_not_orthogonal():
    reps = _reps()
    stip = A.run_gram_weight_recovery(reps["stip_qwen"], target_method="stip_qwen")
    obf = A.run_gram_weight_recovery(reps[OBF], target_method=OBF)
    ours = A.run_gram_weight_recovery(reps[SIGNED], target_method=SIGNED)
    assert stip.metrics["attack_success_rate"] == 1.0
    assert ours.metrics["attack_success_rate"] == 1.0
    assert obf.metrics["attack_success_rate"] == 0.0


def test_norm_audit_isometry_leaks_nonisometry_breaks():
    reps = _reps()
    rep = audit_norm_leakage(reps)["per_method"]
    # every isometry preserves norm exactly -> leaks (corr ~ 1)
    for m in (OBF, "stip_qwen", SIGNED, FRESH):
        assert rep[m]["norm_preserved_exactly_max_rel_err"] < 1e-4
        assert rep[m]["measured_norm_leakage_pearson"] > 0.99
        assert rep[m]["breaks_norm_leakage"] is False
    # the non-isometric design candidate does NOT preserve norm
    assert rep[NONISO]["norm_preserved_exactly_max_rel_err"] > 1e-2
    assert rep[NONISO]["is_norm_preserving_theoretically"] is False


def test_fresh_and_noniso_are_marked_fresh():
    reps = _reps()
    assert reps[FRESH].defense_metadata["whether_fresh_per_sample"] is True
    assert reps[NONISO].defense_metadata["whether_fresh_per_sample"] is True
    assert reps[SIGNED].defense_metadata["whether_fresh_per_sample"] is False
    assert reps[OBF].defense_metadata["whether_fresh_per_sample"] is False


def test_split_downstream_inverts_plaintext_not_masked():
    from pllo.attacks.qwen_split_downstream import build_split_attack_inputs
    model = load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=0, dtype=torch.float32)
    ids = torch.randint(0, model.config.vocab_size, (1, 8),
                        generator=torch.Generator().manual_seed(0))
    plain, _ = build_split_attack_inputs(model, ids, k=1, method="plaintext_gpu", seed=0)
    obf, _ = build_split_attack_inputs(model, ids, k=1, method=OBF, seed=0)
    # BRE forward: matches the smashed state; plaintext is invertible, masked is not.
    rp = A.run_bre_bisr_attack(plain, target_method="plaintext_gpu", num_steps=60)
    ro = A.run_bre_bisr_attack(obf, target_method=OBF, num_steps=60)
    assert rp.status == "measured" and ro.status == "measured"   # measured, NOT blocked
    assert rp.metrics["token_recovery_top1"] > ro.metrics["token_recovery_top1"]
