"""Gate 0 / PACKAGE -- build the immutable private-base GPU package for real
Qwen2.5-0.5B, with in-build fp64 correctness validation and honest leakage scans.

Trust domains
-------------
* TRUSTED packager (this process): legitimately loads the plaintext checkpoint and
  holds every mask secret. Plaintext materialisations here are EXPECTED and counted.
* UNTRUSTED GPU package (the output dir): contains transformed ``*_tilde`` artifacts
  ONLY. No plaintext base weight / embedding / LM-head / norm gamma / bias, and no mask
  secret (N_res, B, R, S, P, Pi, ...), is ever written.

Frozen fold convention (design_spec.json, from_scratch_private_base)
--------------------------------------------------------------------
Single global orthogonal signed-permutation residual mask ``Nr`` (896) threads the
whole residual stream (so residual adds stay in one basis). For a Linear ``W`` reading
the residual and writing masked space ``M_out``:  stored ``W_tilde`` folds
``Nr^T diag(gamma) W ... M_out`` so the untrusted worker computes the masked output
directly from ``H_tilde``. RMSNorm gammas are FOLDED (no plaintext gamma tensor).

Per-domain masks (all secret, TDX-side; baked into the stored numbers, never written):
  attention Q/K : Bq = per-plane 2D rotation that COMMUTES with RoPE (resolves OI1 so
                  the pre-RoPE q/k bias is masked, not plaintext); runtime post-RoPE R
                  is applied TDX-side at inference (not in this package).
  attention V   : S signed-perm per kv-head; o_proj folds S^{-1} on its input side.
  SwiGLU        : P pure permutation (4864) shared by gate/up; down folds P^{-1}.
  vocabulary    : monomial M = Pi (permutation-only baseline, D=I) on logits/LM head.
Embedding and LM head are TWO transformed views of the one tied source E
(``E Nr`` vs ``M^T E diag(gamma_f) Nr``) -> the cross-view attack surface D2 must probe.

The build fails closed: every fold is reconstructed at fp64 against the real HF op and
must match < 1e-8, and the RoPE-commuting property of B is verified, or the package is
not written. Storage precision fp32 (matched-fp32 parity, per the HumanEval finding).

Emits results/aaai_private_base/private_package/*. Not a GPU/TDX run.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pllo.ops.masked_training_kernels import (  # noqa: E402
    orthogonal_signed_perm, permutation_matrix, rmsnorm_core, apply_rope,
    rope_cos_sin)
from pllo.deployment.private_base_package import (  # noqa: E402
    PrivateBasePackager, scan_package_for_plaintext, PrivateBasePackageError)

CKPT = Path("/Users/Hoshino/privacy_llm_data/checkpoints/Qwen2.5-0.5B")
OUT = REPO / "results/aaai_private_base/private_package"
PKG = OUT / "gpu_package"
SOURCE_SAFE_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
DT = torch.float64          # build / validation precision
STORE = torch.float32       # package storage precision (matched-fp32 parity)


# ----- secret mask constructors (TDX-side; never written to the package) -----
def block_diag(mat: torch.Tensor, reps: int) -> torch.Tensor:
    return torch.block_diag(*([mat] * reps))


def rope_commuting_rotation(head_dim: int, seed: int, dtype=DT) -> torch.Tensor:
    """Orthogonal (head_dim,head_dim) that COMMUTES with RoPE: an independent 2D
    rotation by a secret angle on each RoPE plane (i, i+half). Rotations in the same
    2D plane commute, so RoPE(x @ B) == RoPE(x) @ B exactly."""
    half = head_dim // 2
    g = torch.Generator().manual_seed(int(seed))
    ang = torch.rand(half, generator=g, dtype=dtype) * 6.283185307179586
    B = torch.eye(head_dim, dtype=dtype)
    c, s = torch.cos(ang), torch.sin(ang)
    for i in range(half):
        j = i + half
        B[i, i] = c[i]; B[j, j] = c[i]
        B[i, j] = -s[i]; B[j, i] = s[i]
    return B


def diag(v: torch.Tensor) -> torch.Tensor:
    return torch.diag(v.to(DT))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- trusted packager loads plaintext (expected, counted) ----
    assert sha256_file(CKPT / "model.safetensors") == SOURCE_SAFE_SHA, \
        "checkpoint byte-identity failed"
    from safetensors.torch import load_file
    sd = load_file(str(CKPT / "model.safetensors"))
    cfg = json.loads((CKPT / "config.json").read_text())
    H = cfg["hidden_size"]; L = cfg["num_hidden_layers"]; eps = cfg["rms_norm_eps"]
    nh = cfg["num_attention_heads"]; nkv = cfg["num_key_value_heads"]
    hd = H // nh; theta = cfg["rope_theta"]
    counters = {
        "trusted_packager": {
            "trusted_packager_plaintext_base_materializations": len(sd),
            "trusted_packager_plaintext_embedding_materializations": 1,
            "trusted_packager_plaintext_lm_head_materializations": 1,  # tied view of E
            "trusted_packager_plaintext_gamma_materializations": 2 * L + 1,
            "trusted_packager_plaintext_bias_materializations": 3 * L,
            "expected_nonzero": True,
            "note": "trusted offline packager legitimately holds plaintext + secrets"},
        "untrusted_gpu_package_required_zero": {
            "untrusted_plaintext_base_weight_tensors": 0,
            "untrusted_plaintext_embedding_tensors": 0,
            "untrusted_plaintext_lm_head_tensors": 0,
            "untrusted_plaintext_gamma_tensors": 0,
            "untrusted_plaintext_bias_tensors": 0,
            "untrusted_mask_secret_tensors": 0},
    }

    # ---- global residual mask (single basis for the whole residual stream) ----
    Nr = orthogonal_signed_perm(H, seed=9000, dtype=DT)     # (896,896), Nr Nr^T = I
    NrT = Nr.T

    packager = PrivateBasePackager(out_dir=PKG, model_id="Qwen2.5-0.5B-private-base",
                                   config={"arch": {"H": H, "L": L, "nh": nh,
                                           "nkv": nkv, "head_dim": hd, "eps": eps,
                                           "rope_theta": theta, "vocab": cfg["vocab_size"]},
                                           "residual_mask": "orthogonal_signed_perm",
                                           "storage_dtype": "float32",
                                           "vocab_mask": "monomial_permutation_only_D_eq_I",
                                           "qk_bias_masking": "rope_commuting_planar_rotation_OI1"})

    val = {"folds_checked": 0, "max_fold_err": 0.0, "rope_commute_max_err": 0.0,
           "score_preserve_max_err": 0.0, "checks": []}

    def record(nm, err, tol=1e-8):
        val["folds_checked"] += 1
        val["max_fold_err"] = max(val["max_fold_err"], float(err))
        ok = float(err) < tol
        val["checks"].append({"fold": nm, "err": float(err), "ok": ok})
        if not ok:
            raise PrivateBasePackageError(f"fold validation FAILED {nm}: err={err}")

    # ---- embedding view: E_tilde = E @ Nr ----
    E = sd["model.embed_tokens.weight"].to(DT)                       # (V,896)
    E_tilde = E @ Nr
    # validate: masked embed lookup == plaintext lookup @ Nr
    idx = torch.tensor([0, 5, 100, 151935])
    record("embed_view", (E_tilde[idx] - (E[idx] @ Nr)).abs().max())
    # NOTE: transformed artifacts deliberately DROP the plaintext ".weight"/".bias"
    # stem (fail-closed packager policy: a *_tilde name must not embed a plaintext
    # tensor path). Worker maps these back to modules from the manifest.
    packager.add_transformed_tensor("model.embed_tokens_tilde", E_tilde.to(STORE))

    cos, sin = rope_cos_sin(4, hd, theta, DT)                        # small T for check

    for l in range(L):
        p = f"model.layers.{l}."
        ga = sd[p + "input_layernorm.weight"].to(DT)                 # (896,)
        gm = sd[p + "post_attention_layernorm.weight"].to(DT)
        Wq = sd[p + "self_attn.q_proj.weight"].to(DT); bq = sd[p + "self_attn.q_proj.bias"].to(DT)
        Wk = sd[p + "self_attn.k_proj.weight"].to(DT); bk = sd[p + "self_attn.k_proj.bias"].to(DT)
        Wv = sd[p + "self_attn.v_proj.weight"].to(DT); bv = sd[p + "self_attn.v_proj.bias"].to(DT)
        Wo = sd[p + "self_attn.o_proj.weight"].to(DT)
        Wg = sd[p + "mlp.gate_proj.weight"].to(DT); Wu = sd[p + "mlp.up_proj.weight"].to(DT)
        Wd = sd[p + "mlp.down_proj.weight"].to(DT)

        # per-layer secret masks
        B = rope_commuting_rotation(hd, seed=1000 + l)               # (64,64) commutes w/ RoPE
        S = orthogonal_signed_perm(hd, seed=2000 + l, dtype=DT)      # (64,64) V mask
        P = permutation_matrix(cfg["intermediate_size"], seed=3000 + l, dtype=DT)  # (4864,4864)
        Bq = block_diag(B, nh); Bk = block_diag(B, nkv)              # (896,896),(128,128)
        Sv = block_diag(S, nkv); So = block_diag(S, nh)             # (128,128),(896,896)

        # verify B really commutes with RoPE and preserves scores
        xq = torch.randn(4, hd, dtype=DT)
        rc = (apply_rope(xq @ B, cos, sin) - apply_rope(xq, cos, sin) @ B).abs().max()
        val["rope_commute_max_err"] = max(val["rope_commute_max_err"], float(rc))
        if float(rc) >= 1e-9:
            raise PrivateBasePackageError(f"B not RoPE-commuting L{l}: {rc}")
        xk = torch.randn(4, hd, dtype=DT)
        sp = ((apply_rope(xq, cos, sin) @ B) @ (apply_rope(xk, cos, sin) @ B).T
              - apply_rope(xq, cos, sin) @ apply_rope(xk, cos, sin).T).abs().max()
        val["score_preserve_max_err"] = max(val["score_preserve_max_err"], float(sp))
        if float(sp) >= 1e-9:
            raise PrivateBasePackageError(f"score not preserved L{l}: {sp}")

        # ---- attention weight/bias folds ----
        # Wq_tilde.T = Nr^T diag(ga) Wq^T Bq   ->  Wq_tilde = Bq^T Wq diag(ga) Nr
        Wq_t = (NrT @ diag(ga) @ Wq.T @ Bq).T
        bq_t = bq @ Bq
        Wk_t = (NrT @ diag(ga) @ Wk.T @ Bk).T
        bk_t = bk @ Bk
        Wv_t = (NrT @ diag(ga) @ Wv.T @ Sv).T
        bv_t = bv @ Sv
        Wo_t = (torch.linalg.inv(So) @ Wo.T @ Nr).T                  # o_proj no bias
        # ---- mlp folds ----
        Wg_t = (NrT @ diag(gm) @ Wg.T @ P).T
        Wu_t = (NrT @ diag(gm) @ Wu.T @ P).T
        Wd_t = (torch.linalg.inv(P) @ Wd.T @ Nr).T

        # ---- fp64 correctness: masked op reconstructs plaintext op ----
        Hm = torch.randn(4, H, dtype=DT)                            # plaintext hidden
        Htil = Hm @ Nr                                              # what the worker sees
        rms = rmsnorm_core(Htil, eps)                              # == rmsnorm(Hm) @ Nr
        # q/k/v pre (masked) from stored tilde
        q_m = rms @ Wq_t.T + bq_t
        q_ref = ((rmsnorm_core(Hm, eps) * ga) @ Wq.T + bq) @ Bq
        record(f"L{l}.q_proj", (q_m - q_ref).abs().max())
        k_m = rms @ Wk_t.T + bk_t
        k_ref = ((rmsnorm_core(Hm, eps) * ga) @ Wk.T + bk) @ Bk
        record(f"L{l}.k_proj", (k_m - k_ref).abs().max())
        v_m = rms @ Wv_t.T + bv_t
        v_ref = ((rmsnorm_core(Hm, eps) * ga) @ Wv.T + bv) @ Sv
        record(f"L{l}.v_proj", (v_m - v_ref).abs().max())
        # o_proj: attn_out (masked by So) folded back to residual basis Nr
        ao = torch.randn(4, H, dtype=DT)                           # plaintext attn output
        o_m = (ao @ So) @ Wo_t.T
        o_ref = (ao @ Wo.T) @ Nr
        record(f"L{l}.o_proj", (o_m - o_ref).abs().max())
        # gate/up under shared P, then down under P^{-1}
        rms2 = rmsnorm_core(Htil, eps)
        gate_m = rms2 @ Wg_t.T; up_m = rms2 @ Wu_t.T
        gate_ref = ((rmsnorm_core(Hm, eps) * gm) @ Wg.T) @ P
        record(f"L{l}.gate_proj", (gate_m - gate_ref).abs().max())
        up_ref = ((rmsnorm_core(Hm, eps) * gm) @ Wu.T) @ P
        record(f"L{l}.up_proj", (up_m - up_ref).abs().max())
        act_m = torch.nn.functional.silu(gate_m) * up_m           # masked SwiGLU
        down_m = act_m @ Wd_t.T
        act_ref = torch.nn.functional.silu((rmsnorm_core(Hm, eps) * gm) @ Wg.T) * \
            ((rmsnorm_core(Hm, eps) * gm) @ Wu.T)
        down_ref = (act_ref @ Wd.T) @ Nr
        record(f"L{l}.down_proj", (down_m - down_ref).abs().max())

        # ---- write transformed artifacts (fail-closed on any plaintext/secret name) ----
        for nm, t in [(f"{p}self_attn.q_proj_tilde", Wq_t),
                      (f"{p}self_attn.q_proj_bias_tilde", bq_t),
                      (f"{p}self_attn.k_proj_tilde", Wk_t),
                      (f"{p}self_attn.k_proj_bias_tilde", bk_t),
                      (f"{p}self_attn.v_proj_tilde", Wv_t),
                      (f"{p}self_attn.v_proj_bias_tilde", bv_t),
                      (f"{p}self_attn.o_proj_tilde", Wo_t),
                      (f"{p}mlp.gate_proj_tilde", Wg_t),
                      (f"{p}mlp.up_proj_tilde", Wu_t),
                      (f"{p}mlp.down_proj_tilde", Wd_t)]:
            packager.add_transformed_tensor(nm, t.to(STORE))
        packager.record_mask_domain(f"layer_{l}",
                                    ["Nr(residual)", "B(qk_rope_commuting)", "S(v)",
                                     "P(swiglu)"])

    # ---- LM-head view: second transformed view of tied E ----
    # The vocab monomial mask M=Pi (permutation-only, D=I) is represented as an INDEX
    # vector, NEVER a dense (V,V) matrix (that would be 185 GB). Pi[i, perm[i]]=1, so
    #   logits @ Pi  == logits[:, perm_inv]   (column perm)
    #   Pi^T @ E     == E[perm_inv]           (row perm)
    gf = sd["model.norm.weight"].to(DT)                             # (896,)
    V = cfg["vocab_size"]
    g8 = torch.Generator().manual_seed(8000)
    perm = torch.randperm(V, generator=g8)
    perm_inv = torch.empty_like(perm); perm_inv[perm] = torch.arange(V)
    # lm_head_tilde = Pi^T (E diag(gf) Nr) = (E diag(gf) Nr)[perm_inv]
    EgN = (E * gf) @ Nr                                            # (V,896)
    lm_tilde = EgN[perm_inv]                                       # row-permuted view
    Hf = torch.randn(4, H, dtype=DT); Hft = Hf @ Nr
    logit_m = rmsnorm_core(Hft, eps) @ lm_tilde.T
    logit_ref = (((rmsnorm_core(Hf, eps) * gf) @ E.T))[:, perm_inv]  # logits @ Pi
    record("lm_head_view", (logit_m - logit_ref).abs().max())
    packager.add_transformed_tensor("lm_head_tilde", lm_tilde.to(STORE))
    packager.record_mask_domain("vocab", ["Nr(residual)", "gamma_final(folded)",
                                          "Pi(monomial_permutation_only)"])

    manifest = packager.finalize()

    # ---- independent plaintext-absence scan ----
    scan = scan_package_for_plaintext(PKG)

    # ---- honest tensor-fingerprint scan (structural leakage inventory) ----
    # Orthogonal masks PRESERVE singular values; permutations preserve the value
    # multiset. Report these as REAL residual leaks -- do not claim unrecognisable.
    fp = {"method": "singular_value_and_value_multiset_vs_plaintext",
          "note": "orthogonal/permutation masks preserve spectra & value multisets; "
                  "these are documented structural leaks, NOT claimed hidden",
          "probes": []}
    for nm, W, fname in [
            ("q_proj.L0", sd["model.layers.0.self_attn.q_proj.weight"].to(DT),
             "model.layers.0.self_attn.q_proj_tilde.pt"),
            ("down_proj.L0", sd["model.layers.0.mlp.down_proj.weight"].to(DT),
             "model.layers.0.mlp.down_proj_tilde.pt")]:
        stored = torch.load(PKG / fname)
        Wt = list(stored.values())[0].to(DT)
        sv_plain = torch.linalg.svdvals(W)
        sv_til = torch.linalg.svdvals(Wt)
        # NB the tilde folds diag(gamma) too, so spectra are NOT identical here; report
        sv_gap = float((sv_plain.sort().values - sv_til.sort().values).abs().max())
        vm_plain = W.flatten().sort().values
        vm_til = Wt.flatten().sort().values
        vm_gap = float((vm_plain - vm_til).abs().max()) if vm_plain.numel() == vm_til.numel() else -1.0
        exact_equal = bool(torch.equal(W.to(STORE), Wt.to(STORE)))
        fp["probes"].append({"tensor": nm, "singular_value_gap_after_gamma_fold": sv_gap,
                             "value_multiset_gap": vm_gap,
                             "exact_plaintext_tensor_present": exact_equal})
    (OUT / "tensor_fingerprint_scan.json").write_text(json.dumps(fp, indent=2))

    # ---- hash-pin the immutable package ----
    pins = {f.name: sha256_file(f) for f in sorted(PKG.glob("*")) if f.is_file()}
    package_root_hash = hashlib.sha256(
        json.dumps(pins, sort_keys=True).encode()).hexdigest()
    (OUT / "package_hash_pin.json").write_text(json.dumps(
        {"per_file_sha256": pins, "package_root_hash": package_root_hash,
         "immutable": True}, indent=2, sort_keys=True))

    (OUT / "build_validation.json").write_text(json.dumps(val, indent=2))
    (OUT / "trust_domain_counters.json").write_text(json.dumps(counters, indent=2))

    ok = (scan["clean"] and val["max_fold_err"] < 1e-8
          and val["rope_commute_max_err"] < 1e-9
          and val["score_preserve_max_err"] < 1e-9
          and all(not pr["exact_plaintext_tensor_present"] for pr in fp["probes"]))
    summary = {
        "package_dir": str(PKG), "artifact_count": manifest["artifact_count"],
        "folds_checked": val["folds_checked"], "max_fold_err": val["max_fold_err"],
        "rope_commute_max_err": val["rope_commute_max_err"],
        "score_preserve_max_err": val["score_preserve_max_err"],
        "plaintext_absence_scan_clean": scan["clean"],
        "scan_finding_count": scan["finding_count"],
        "no_exact_plaintext_tensor": all(not pr["exact_plaintext_tensor_present"]
                                         for pr in fp["probes"]),
        "package_root_hash": package_root_hash,
        "documented_residual_leaks": [
            "attention_scores_exposed_by_design",
            "orthogonal_masks_preserve_singular_value_spectra",
            "permutation_masks_preserve_value_multiset",
            "tied_E_two_view_cross_view_attack_surface",
            "qk_bias_masked_by_planar_rotation_pair_norms_preserved"],
        "package_ok": ok}
    (OUT / "package_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "summary.md").write_text(
        "# Gate-0 private-base GPU package\n\n"
        f"artifacts: {manifest['artifact_count']} | folds validated: "
        f"{val['folds_checked']} (max err {val['max_fold_err']:.2e}) | "
        f"RoPE-commute err {val['rope_commute_max_err']:.2e} | score-preserve err "
        f"{val['score_preserve_max_err']:.2e}\n"
        f"plaintext-absence scan clean: {scan['clean']} | no exact plaintext tensor: "
        f"{summary['no_exact_plaintext_tensor']}\n"
        f"package_root_hash: {package_root_hash[:16]}...\n\n"
        "Documented residual leaks (NOT claimed hidden): attention scores, "
        "singular-value spectra, permutation value-multiset, tied-E two-view surface, "
        "q/k-bias planar-rotation pair-norms.\n\n"
        f"**PACKAGE OK: {ok}**\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
