"""Tests for the CryptoGen baseline (arXiv:2602.08798).

Verify the plaintext-faithful reimplementation of CryptoGen's core algorithmic
contributions: (a) the diagonal CT x PT matvec equals the plaintext product,
(b) the ARCC folding-sum reduction is O(log d) and correct, (c) ARCC attention
scores equal q K^T, (d) slot-aware KV-cache concatenation keeps the ciphertext
count ~constant and reconstructs exactly, and (e) the cost model reproduces the
O(L^2)->O(L) attention scaling and the CT x PT m-independence of the paper.
"""
from __future__ import annotations

import math

import torch

from pllo.baselines.cryptogen import CryptoGen, CryptoGenConfig


def _cg(hidden=64, head_dim=64, n=8192):
    return CryptoGen(CryptoGenConfig(hidden=hidden, head_dim=head_dim,
                                     poly_modulus_degree=n, dtype="float64"))


def test_diagonal_matvec_matches_plaintext():
    cg = _cg()
    g = torch.Generator().manual_seed(0)
    A = torch.randn(64, 64, generator=g, dtype=torch.float64)
    x = torch.randn(64, generator=g, dtype=torch.float64)
    out = cg.diagonal_matvec(A, x)
    assert out["max_abs_err"] < 1e-9
    assert out["he_multiplications"] == 64
    assert out["he_rotations"] == 63


def test_fold_sum_is_log_depth_and_correct():
    cg = _cg()
    d = 64
    ct = torch.zeros(cg.config.poly_modulus_degree, dtype=torch.float64)
    ct[:d] = torch.arange(1, d + 1, dtype=torch.float64)
    summed, rots = cg.fold_sum(ct, d)
    assert abs(summed[0].item() - d * (d + 1) / 2) < 1e-9
    assert rots == int(math.log2(d))          # 6 rotations for d=64


def test_arcc_scores_match_plaintext():
    cg = _cg()
    g = torch.Generator().manual_seed(1)
    q = torch.randn(64, generator=g, dtype=torch.float64)
    K = torch.randn(10, 64, generator=g, dtype=torch.float64)   # 10 cached keys
    out = cg.arcc_scores(q, K)
    assert out["max_abs_err"] < 1e-9
    assert out["reduction_complexity"] == "O(log d)"
    assert out["rotations_per_key"] == 6                        # log2(64)


def test_kv_concat_constant_ciphertexts_and_exact():
    cg = _cg(head_dim=64, n=8192)
    g = torch.Generator().manual_seed(2)
    tokens = torch.randn(300, 64, generator=g, dtype=torch.float64)
    out = cg.kv_concat(tokens)
    # B = 8192//64 = 128 tokens/ct -> ceil(300/128) = 3 ciphertexts vs 300 naive
    assert out["tokens_per_ciphertext"] == 128
    assert out["cache_ciphertexts"] == 3
    assert out["naive_ciphertexts"] == 300
    assert out["reconstruct_max_abs_err"] < 1e-12


def test_cost_model_scaling():
    cg = _cg(hidden=768, head_dim=64, n=8192)
    c = cg.cost_model(m=128, k=8)
    # BOLT attention O(k^2), CryptoGen O(k)
    assert c["attention_ctxct"]["bolt"] == 64
    assert c["attention_ctxct"]["cryptogen"] == 8
    # CT x PT per-step: CryptoGen independent of m, BOLT proportional to m
    assert c["ctxpt_gen_mult_asymptotic"]["bolt"] > c["ctxpt_gen_mult_asymptotic"]["cryptogen"]
    assert c["ctxpt_gen_mult_asymptotic"]["cryptogen_independent_of_m"] is True
    # reduction rotations O(log d1)
    assert "O(log d1)" in c["reduction_rotations_order"]["cryptogen"]
    # concrete Table I: CryptoGen per-token Gen cheaper than BOLT
    assert c["paper_table_i_cited"]["mult_gen_per_token"]["cryptogen"] == 64
    assert c["paper_table_i_cited"]["mult_gen_per_token"]["bolt"] == 768


def test_structural_audit_contrast():
    cg = _cg()
    a = cg.structural_audit()
    assert a["hardware_trust_anchor"].startswith("none")
    assert a["interactive"] is True
    assert a["scales_to_7b"] is False
    assert a["contrast_ours_A_rightmul"]["scales_to_7b"] is True
    assert a["contrast_ours_A_rightmul"]["interactive"] is False


def test_self_declaration():
    d = CryptoGen().declare.to_dict()
    assert d["name"] == "cryptogen"
    assert d["exact_primitive_implemented"] is True
    assert d["full_system_reproduced"] is False
    assert d["requires_crypto_library"] is True
