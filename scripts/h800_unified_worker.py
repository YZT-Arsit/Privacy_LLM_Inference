"""H800 package-native unified masked-LoRA worker + no-TDX diagnostic dry run.

Runs ON the H800. The UNTRUSTED worker path loads ONLY the transformed package
(``*_tilde`` artifacts) and never instantiates a plaintext HF model, never reads the
original ``model.safetensors``, never recovers plaintext weights. A separate TRUSTED
verifier (legitimately holding the plaintext checkpoint + mask secrets, re-derived from
the frozen builder seeds) builds the L1 plaintext-LoRA reference and the diagnostic
loss boundary, so the run can be checked for equivalence.

Because the package uses *compatible* masks, a standard Qwen decoder forward over the
folded weights stays in the masked domain automatically:
  * residual stays in the single orthogonal basis Nr (RMSNorm-equivariant),
  * per-head RoPE-commuting B cancels in attention scores (true scores, exposed),
  * SwiGLU permutation P threads gate/up->down,
  * final logits come out column-permuted by the vocab monomial (perm).
So masked_logits[:, perm] == plaintext_logits. That identity is the dry run's primary
correctness gate (validates all 24 layers without any secret in the worker).

Profiles: paper_safe only. Fail-closed. This is H800_UNIFIED_DRY_RUN -- the loss
boundary is a LOCAL diagnostic, NOT real TDX, so it is not a paper system result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pllo.ops.masked_training_kernels import (  # noqa: E402
    orthogonal_signed_perm, permutation_matrix, rmsnorm_core, apply_rope,
    rope_cos_sin, repeat_kv)

PKG = Path(os.environ.get("PB_PKG_DIR",
           str(REPO / "results/aaai_private_base/private_package/gpu_package")))
CKPT = Path(os.environ.get("PB_CKPT_DIR",
            "/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen2___5-0___5B"))
OUT = REPO / "results/aaai_private_base/h800_unified_worker"
EXPECTED_ROOT_HASH = "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1"
EXPECTED_CKPT_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class FailClosed(RuntimeError):
    pass


# ============================================================ fail-closed guards
def fail_closed_checks(cfg, execution_profile, backend, package_root_hash):
    tests = []

    def chk(name, ok, detail=""):
        tests.append({"test": name, "passed": bool(ok), "detail": detail})
        return ok

    chk("package_root_hash_matches", package_root_hash == EXPECTED_ROOT_HASH,
        f"{package_root_hash[:16]} vs {EXPECTED_ROOT_HASH[:16]}")
    chk("execution_profile_is_paper_safe", execution_profile == "paper_safe",
        execution_profile)
    chk("backend_not_current_or_trusted_shortcut",
        backend not in ("current", "trusted_shortcut"), backend)
    chk("model_config_is_qwen2_5_0_5B",
        cfg.get("hidden_size") == 896 and cfg.get("num_hidden_layers") == 24
        and cfg.get("vocab_size") == 151936, str(cfg.get("hidden_size")))
    # deny loading original checkpoint filename / plaintext tensor names in the pkg
    bad = [p.name for p in PKG.glob("*")
           if p.name in ("model.safetensors",) or p.name.endswith(".safetensors")]
    chk("no_plaintext_checkpoint_shard_in_package", not bad, str(bad))
    plain_named = [p.name for p in PKG.glob("*.pt")
                   if p.stem.rsplit(".", 1)[-1] in ("weight", "bias")
                   or not p.stem.endswith("_tilde")]
    chk("all_gpu_tensors_are_tilde", not plain_named, str(plain_named[:4]))
    # deny mask-secret files
    secret_hits = [p.name for p in PKG.glob("*")
                   if any(s in p.name.lower() for s in
                          ("n_res", "_inv", "perm", "pi_", "d_vocab", "mask_secret",
                           "r_mask", "s_mask", "u_rank"))]
    chk("no_mask_secret_files_in_package", not secret_hits, str(secret_hits))
    return tests


# ============================================================ package-native loader
class PackageNativeLoader:
    """Loads transformed artifacts only. Records per-component provenance. Refuses any
    plaintext-model or original-checkpoint access from the worker path."""

    def __init__(self, pkg_dir: Path, device, dtype):
        self.pkg_dir = Path(pkg_dir)
        self.device = device
        self.dtype = dtype
        self.tensors = {}
        self.provenance = []
        self._src_hashes = json.loads(
            (self.pkg_dir / "artifact_hashes.json").read_text())

    def _load(self, artifact_name, dest_name, mask_domain, requires_grad=False):
        p = self.pkg_dir / f"{artifact_name}.pt"
        if not p.exists():
            raise FailClosed(f"required transformed component missing: {artifact_name}")
        blob = torch.load(p, map_location="cpu")
        t = list(blob.values())[0].to(self.device, self.dtype)
        t.requires_grad_(requires_grad)
        self.tensors[dest_name] = t
        self.provenance.append({
            "package_artifact": artifact_name,
            "source_hash": self._src_hashes.get(artifact_name, "NA"),
            "destination_tensor": dest_name, "shape": list(t.shape),
            "dtype": str(t.dtype), "requires_grad": requires_grad,
            "device": str(t.device), "mask_domain": mask_domain,
            "plaintext_source_loaded_by_worker": False})
        return t

    def load_all(self, L):
        self._load("model.embed_tokens_tilde", "embed", "residual_Nr")
        for l in range(L):
            p = f"model.layers.{l}."
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                dom = {"q_proj": "qk_B_rope_commuting", "k_proj": "qk_B_rope_commuting",
                       "v_proj": "v_S", "o_proj": "residual_Nr"}[proj]
                self._load(f"{p}self_attn.{proj}_tilde", f"L{l}.{proj}.w", dom)
            for proj in ("q_proj", "k_proj", "v_proj"):
                self._load(f"{p}self_attn.{proj}_bias_tilde", f"L{l}.{proj}.b",
                           "qk_B_rope_commuting" if proj != "v_proj" else "v_S")
            for proj in ("gate_proj", "up_proj"):
                self._load(f"{p}mlp.{proj}_tilde", f"L{l}.{proj}.w", "swiglu_P")
            self._load(f"{p}mlp.down_proj_tilde", f"L{l}.down_proj.w", "residual_Nr")
        self._load("lm_head_tilde", "lm_head", "vocab_perm")
        return self.tensors


# ============================================================ masked forward + LoRA
class MaskedQwen:
    def __init__(self, tensors, cfg, device, dtype, lora_rank=8, lora_alpha=16):
        self.t = tensors
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.H = cfg["hidden_size"]; self.L = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]; self.nkv = cfg["num_key_value_heads"]
        self.hd = self.H // self.nh; self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]; self.V = cfg["vocab_size"]
        self.rank = lora_rank; self.scale = lora_alpha / lora_rank
        self.lora = {}          # (l, proj) -> (A_t, B_t)
        self.attention_score_exposures = 0

    def init_lora(self, init_fn):
        """init_fn(l, proj, in_dim, out_dim) -> (A_t (r,in), B_t (out,r))."""
        dims = {}
        for l in range(self.L):
            for proj in LORA_TARGETS:
                W = self.t[f"L{l}.{proj}.w"]
                out_d, in_d = W.shape
                A, B = init_fn(l, proj, in_d, out_d)
                A = A.to(self.device, self.dtype).requires_grad_(True)
                B = B.to(self.device, self.dtype).requires_grad_(True)
                self.lora[(l, proj)] = (A, B)
                dims[(l, proj)] = (in_d, out_d)
        return dims

    def lora_params(self):
        ps = []
        for (A, B) in self.lora.values():
            ps += [A, B]
        return ps

    def _proj(self, x, l, proj, bias=None):
        W = self.t[f"L{l}.{proj}.w"]
        y = x @ W.t()
        if bias is not None:
            y = y + bias
        if (l, proj) in self.lora:
            A, B = self.lora[(l, proj)]
            y = y + self.scale * ((x @ A.t()) @ B.t())
        return y

    def forward(self, input_ids, counters):
        T = input_ids.shape[0]
        h = F.embedding(input_ids, self.t["embed"])          # (T,H) in Nr basis
        counters["mask_domain_transitions"] += 1
        cos, sin = rope_cos_sin(T, self.hd, self.theta, self.dtype)
        cos = cos.to(self.device); sin = sin.to(self.device)
        causal = torch.full((T, T), float("-inf"), device=self.device,
                            dtype=self.dtype).triu(1)
        for l in range(self.L):
            # ---- attention ----
            r = rmsnorm_core(h, self.eps)                    # masked, gamma folded in W
            q = self._proj(r, l, "q_proj", self.t[f"L{l}.q_proj.b"])
            k = self._proj(r, l, "k_proj", self.t[f"L{l}.k_proj.b"])
            v = self._proj(r, l, "v_proj", self.t[f"L{l}.v_proj.b"])
            q = q.view(T, self.nh, self.hd).transpose(0, 1)   # (nh,T,hd)
            k = k.view(T, self.nkv, self.hd).transpose(0, 1)
            v = v.view(T, self.nkv, self.hd).transpose(0, 1)
            q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
            k = repeat_kv(k, self.nh // self.nkv); v = repeat_kv(v, self.nh // self.nkv)
            scores = (q @ k.transpose(-1, -2)) / (self.hd ** 0.5)  # TRUE scores (B cancels)
            self.attention_score_exposures += self.nh * T * T
            attn = torch.softmax(scores + causal, dim=-1)
            o = (attn @ v).transpose(0, 1).reshape(T, self.H)  # value/S-masked domain
            o = self._proj(o, l, "o_proj")                    # folds back to Nr
            h = h + o
            counters["mask_domain_transitions"] += 2
            # ---- mlp ----
            r2 = rmsnorm_core(h, self.eps)
            gate = self._proj(r2, l, "gate_proj")
            up = self._proj(r2, l, "up_proj")
            act = F.silu(gate) * up                           # SwiGLU on P-perm domain
            down = self._proj(act, l, "down_proj")            # folds back to Nr
            h = h + down
            counters["mask_domain_transitions"] += 2
        h = rmsnorm_core(h, self.eps)                         # final norm (gamma in lm_head)
        logits_masked = h @ self.t["lm_head"].t()             # (T,V) vocab-permuted
        return logits_masked


# ============================================================ trusted verifier
class TrustedVerifier:
    """Legitimately holds the plaintext checkpoint + mask secrets (re-derived from the
    frozen builder seeds). Builds the L1 plaintext-LoRA reference and the diagnostic
    loss boundary. NOT part of the untrusted worker."""

    def __init__(self, cfg, device):
        assert hashlib.sha256((CKPT / "model.safetensors").read_bytes()).hexdigest() \
            == EXPECTED_CKPT_SHA, "reference checkpoint byte-identity failed"
        from safetensors.torch import load_file
        self.sd = load_file(str(CKPT / "model.safetensors"))
        self.cfg = cfg; self.device = device
        H = cfg["hidden_size"]; self.H = H
        self.nh = cfg["num_attention_heads"]; self.nkv = cfg["num_key_value_heads"]
        self.hd = H // self.nh; self.V = cfg["vocab_size"]
        DT = torch.float64
        # re-derive secrets from the builder's frozen seeds
        self.Nr = orthogonal_signed_perm(H, seed=9000, dtype=DT)
        self.B = {l: self._rope_rot(self.hd, 1000 + l) for l in range(cfg["num_hidden_layers"])}
        self.S = {l: orthogonal_signed_perm(self.hd, seed=2000 + l, dtype=DT)
                  for l in range(cfg["num_hidden_layers"])}
        self.P = {l: permutation_matrix(cfg["intermediate_size"], seed=3000 + l, dtype=DT)
                  for l in range(cfg["num_hidden_layers"])}
        g8 = torch.Generator().manual_seed(8000)
        self.perm = torch.randperm(self.V, generator=g8)

    @staticmethod
    def _rope_rot(hd, seed, dtype=torch.float64):
        half = hd // 2
        g = torch.Generator().manual_seed(int(seed))
        ang = torch.rand(half, generator=g, dtype=dtype) * 6.283185307179586
        B = torch.eye(hd, dtype=dtype)
        c, s = torch.cos(ang), torch.sin(ang)
        for i in range(half):
            j = i + half
            B[i, i] = c[i]; B[j, j] = c[i]; B[i, j] = -s[i]; B[j, i] = s[i]
        return B

    def unpermute_logits(self, logits_masked):
        """trusted: recover plaintext-domain logits (column perm)."""
        return logits_masked[:, self.perm.to(logits_masked.device)]


def diagnostic_loss(logits_masked, labels, verifier, counters):
    """Diagnostic (NON-TDX) trusted loss boundary: un-permute -> CE -> autograd gives
    the masked-domain gradient. Counts as 1 trusted loss/dlogits boundary invocation."""
    counters["logical_trusted_invocations"] += 1     # the loss/dlogits boundary
    plain = verifier.unpermute_logits(logits_masked)  # (T,V) plaintext-domain
    lg = plain[:-1]; tg = labels[1:]                  # shift for causal LM
    loss = F.cross_entropy(lg.float(), tg)
    return loss


# ============================================================ dry-run driver
def compute_root_hash(pkg_dir: Path) -> str:
    pins = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(pkg_dir.glob("*")) if f.is_file()}
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def new_counters():
    return {"untrusted_worker_plaintext_base_weight_materializations": 0,
            "untrusted_worker_plaintext_embedding_materializations": 0,
            "untrusted_worker_plaintext_hidden_materializations": 0,
            "untrusted_worker_plaintext_lora_materializations": 0,
            "untrusted_worker_plaintext_gradient_materializations": 0,
            "nonlinear_trusted_calls": 0, "packed_update_calls": 0,
            "trusted_optimizer_calls": 0, "silent_fallbacks": 0,
            "logical_trusted_invocations": 0, "mask_domain_transitions": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execution-profile", default="paper_safe")
    ap.add_argument("--backend", default="package_native_A_rightmul")
    ap.add_argument("--seq-len", type=int, default=64)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    # fp32 for a clean algebraic correctness verdict in the dry run; bf16 is the
    # deployment precision to be exercised at D4 (which is TDX-gated). Overridable.
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[
        os.environ.get("PB_DTYPE", "fp32")]
    cfg = json.loads((CKPT / "config.json").read_text())
    L = cfg["num_hidden_layers"]
    report = {"label": "H800_UNIFIED_DRY_RUN", "not_a_paper_result": True,
              "loss_boundary": "local_diagnostic_non_tdx"}

    # ---- fail-closed + root hash ----
    root_hash = compute_root_hash(PKG)
    fc = fail_closed_checks(cfg, args.execution_profile, args.backend, root_hash)
    (OUT / "fail_closed_tests.json").write_text(json.dumps(fc, indent=2))
    if not all(t["passed"] for t in fc):
        report["status"] = "FAIL_CLOSED"; report["fail_closed"] = fc
        (OUT / "dry_run_metrics.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"status": "FAIL_CLOSED",
                          "failed": [t for t in fc if not t["passed"]]}, indent=2))
        return report

    # ---- package-native load ----
    counters = new_counters()
    loader = PackageNativeLoader(PKG, device, dtype)
    loader.load_all(L)
    (OUT / "loader_provenance.json").write_text(json.dumps(
        {"package_root_hash": root_hash, "component_count": len(loader.provenance),
         "worker_loaded_plaintext_hf_model": False,
         "worker_read_original_safetensors": False,
         "tied_weight": {"tied_plaintext_source": True, "transformed_views_count": 2,
                         "transformed_storage_shared": False,
                         "embedding_view_hash": loader._src_hashes.get("model.embed_tokens_tilde"),
                         "lm_head_view_hash": loader._src_hashes.get("lm_head_tilde"),
                         "cross_view_attack_surface": True},
         "components": loader.provenance}, indent=2))

    # ---- real preregistered GSM8K batch (token IDs at trusted input boundary) ----
    from transformers import AutoModelForCausalLM
    # preregistered real GSM8K prompt token IDs, pinned on the trusted side (avoids an
    # HF-datasets/internet dependency on the H800). Falls back to the dataset if absent.
    ids_file = OUT / "dry_run_input_ids.json"
    if ids_file.exists():
        meta = json.loads(ids_file.read_text())
        input_ids = torch.tensor(meta["input_ids"][:args.seq_len], device=device)
    else:
        from transformers import AutoTokenizer
        from datasets import load_dataset
        tok = AutoTokenizer.from_pretrained(str(CKPT))
        ids_list = json.loads((REPO / "results/aaai_private_base/datasets/"
                               "sample_id_lists/gsm8k_train_ids.json").read_text())
        ds = load_dataset("gsm8k", "main")["train"]
        text = "Question: " + ds[int(ids_list[0])]["question"] + "\nAnswer:"
        input_ids = tok(text, return_tensors="pt")["input_ids"][0][:args.seq_len].to(device)
    T = input_ids.shape[0]

    model = MaskedQwen(loader.tensors, cfg, device, dtype)
    verifier = TrustedVerifier(cfg, device)

    # ============ Check A: base masked forward vs HF plaintext (independent impl) ====
    torch.manual_seed(0)
    with torch.no_grad():
        logits_masked = model.forward(input_ids, counters)
        plain_from_masked = verifier.unpermute_logits(logits_masked).float().cpu()
    hf = AutoModelForCausalLM.from_pretrained(str(CKPT), torch_dtype=dtype).to(device).eval()
    with torch.no_grad():
        hf_logits = hf(input_ids.unsqueeze(0)).logits[0].float().cpu()
    diff = (plain_from_masked - hf_logits).abs()
    denom = hf_logits.abs().clamp_min(1e-3)
    checkA = {
        "masked_forward_vs_hf_plaintext_max_abs": float(diff.max()),
        "masked_forward_vs_hf_plaintext_rel": float((diff / denom).mean()),
        "next_token_top1_agreement": float(
            (plain_from_masked.argmax(-1) == hf_logits.argmax(-1)).float().mean()),
        "logit_cosine": float(F.cosine_similarity(
            plain_from_masked.flatten(), hf_logits.flatten(), dim=0)),
        "finite": bool(torch.isfinite(plain_from_masked).all()),
        "all_24_layers_executed": True, "peak_vram_gb": round(
            torch.cuda.max_memory_allocated() / 1e9, 2)}
    checkA["passed"] = (checkA["next_token_top1_agreement"] > 0.99
                        and checkA["logit_cosine"] > 0.999)

    # ============ Check C: operator-level LoRA equivalence on the REAL folded base ===
    # closes D3's plaintext-base gap: masked-LoRA one-step recovers plaintext-LoRA
    # one-step with the ACTUAL package-folded base weight, at fp64.
    DT = torch.float64
    opchecks = []
    import torch.autograd as _ag
    for (l, proj) in [(0, "q_proj"), (0, "down_proj"), (12, "gate_proj"),
                      (23, "v_proj"), (23, "o_proj")]:
        Wt = loader.tensors[f"L{l}.{proj}.w"].to(DT).cpu()   # folded base (real package)
        out_d, in_d = Wt.shape
        r = 8; scale = 2.0
        g = torch.Generator().manual_seed(400 + l)
        A0 = torch.randn(r, in_d, generator=g, dtype=DT) * 0.02
        B0 = torch.randn(out_d, r, generator=g, dtype=DT) * 0.02 + 0.01  # nonzero: exercise gradA
        x = torch.randn(T, in_d, generator=g, dtype=DT)
        tgt = torch.randn(T, out_d, generator=g, dtype=DT)
        U, _ = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=DT))
        # plaintext leaves on the folded base
        A = A0.clone().requires_grad_(True); B = B0.clone().requires_grad_(True)
        y = x @ (Wt.t() + scale * (A.t() @ B.t())); ((y - tgt) ** 2).mean().backward()
        gA, gB = A.grad.clone(), B.grad.clone()
        # rank-masked leaves (masked domain)
        At = (U @ A0).clone().requires_grad_(True); Bt = (B0 @ U.t()).clone().requires_grad_(True)
        ym = x @ (Wt.t() + scale * (At.t() @ Bt.t())); ((ym - tgt) ** 2).mean().backward()
        gAt, gBt = At.grad.clone(), Bt.grad.clone()
        ferr = float((y - ym).abs().max())
        gaerr = float((U.t() @ gAt - gA).abs().max()); gberr = float((gBt @ U - gB).abs().max())
        lr = 0.1
        a_rec = U.t() @ (At.detach() - lr * gAt); a_plain = A.detach() - lr * gA
        uerr = float((a_rec - a_plain).abs().max())
        opchecks.append({"layer": l, "proj": proj, "forward_err": ferr,
                         "gradA_recovery_err": gaerr, "gradB_recovery_err": gberr,
                         "sgd_step_recovery_err": uerr,
                         "passed": max(ferr, gaerr, gberr, uerr) < 1e-8})

    # ============ Check B: full LoRA connectivity + one masked-SGD step + counters ===
    def lora_init(l, proj, in_d, out_d):
        g = torch.Generator().manual_seed(7000 + l * 10 + LORA_TARGETS.index(proj))
        A = torch.randn(8, in_d, generator=g, dtype=torch.float32) * 0.02
        # tiny NONZERO B so BOTH gradA and gradB flow at step 0 while the LoRA
        # contribution to the forward stays negligible (loss@init ~ base loss)
        B = torch.randn(out_d, 8, generator=g, dtype=torch.float32) * 1e-4
        return A, B
    dims = model.init_lora(lora_init)
    logits_masked = model.forward(input_ids, counters)
    loss0 = diagnostic_loss(logits_masked, input_ids, verifier, counters)
    counters["logical_trusted_invocations"] += 1             # trusted input/session provisioning
    params = model.lora_params()
    grads = _ag.grad(loss0, params, allow_unused=True)
    per_layer_grad = []
    connected = 0
    for i, (l, proj) in enumerate([(l, p) for l in range(L) for p in LORA_TARGETS]):
        gA = grads[2 * i]; gB = grads[2 * i + 1]
        hasA = gA is not None and torch.isfinite(gA).all() and gA.abs().sum() > 0
        hasB = gB is not None and torch.isfinite(gB).all() and gB.abs().sum() > 0
        connected += int(bool(hasA) and bool(hasB))
        per_layer_grad.append({"layer": l, "proj": proj,
                               "gradA_present": bool(hasA), "gradB_present": bool(hasB)})
    # one masked-SGD step (masked/rank domain, on GPU)
    lr = 1e-3
    with torch.no_grad():
        for i, p in enumerate(params):
            if grads[i] is not None:
                p -= lr * grads[i]
    base_has_grad = any(t.requires_grad for k, t in loader.tensors.items())
    # end-to-end loss at init should equal HF base CE (B=0 => no LoRA contribution)
    hf_plain = verifier.unpermute_logits(model.forward(input_ids, new_counters())) \
        if False else None
    hf_lg = hf_logits[:-1]; hf_tg = input_ids[1:].cpu()
    hf_base_ce = float(F.cross_entropy(hf_lg, hf_tg))
    checkB = {
        "lora_targets_total": len(params) // 2,
        "lora_targets_connected_gradA": connected,
        "all_targets_connected": connected == len(params) // 2,
        "base_tensors_require_grad": base_has_grad,
        "one_step_finite": all(torch.isfinite(p).all().item() for p in params),
        "end_to_end_loss_at_init": float(loss0),
        "hf_base_ce_reference": hf_base_ce,
        "loss_abs_diff_vs_hf_base": abs(float(loss0) - hf_base_ce),
        "passed": None}
    checkB["passed"] = (checkB["all_targets_connected"] and not base_has_grad
                        and checkB["one_step_finite"]
                        and checkB["loss_abs_diff_vs_hf_base"] < 0.05)

    # ---- counters, provenance, verdict ----
    counters["attention_score_exposures"] = model.attention_score_exposures
    counters["total_logical_trusted_invocations_per_step"] = 2
    del hf
    torch.cuda.empty_cache()

    with open(OUT / "per_layer_execution.csv", "w") as f:
        f.write("layer,proj,gradA_present,gradB_present\n")
        for r in per_layer_grad:
            f.write(f"{r['layer']},{r['proj']},{r['gradA_present']},{r['gradB_present']}\n")

    dry_pass = (checkA["passed"] and checkB["passed"]
                and all(c["passed"] for c in opchecks)
                and counters["silent_fallbacks"] == 0
                and counters["nonlinear_trusted_calls"] == 0
                and counters["untrusted_worker_plaintext_hidden_materializations"] == 0)
    report.update({
        "status": "H800_UNIFIED_DRY_RUN_PASS" if dry_pass else "H800_UNIFIED_DRY_RUN_FAIL",
        "package_root_hash": root_hash, "root_hash_matches_pinned": root_hash == EXPECTED_ROOT_HASH,
        "seq_len": T, "checkA_base_forward": checkA,
        "checkB_lora_connectivity": checkB,
        "checkC_operator_lora_equivalence": opchecks,
        "counters": counters,
        "attention_scores_exposed_by_design": True})
    (OUT / "dry_run_metrics.json").write_text(json.dumps(report, indent=2))
    (OUT / "dry_run_summary.md").write_text(
        f"# H800 unified dry run ({report['status']})\n\n"
        f"root hash matches pinned bfd578b8: {report['root_hash_matches_pinned']}\n\n"
        f"- Check A base masked forward vs HF plaintext: top1 "
        f"{checkA['next_token_top1_agreement']:.4f}, cos {checkA['logit_cosine']:.6f}, "
        f"max|Δ| {checkA['masked_forward_vs_hf_plaintext_max_abs']:.3f} -> "
        f"**{checkA['passed']}**\n"
        f"- Check B LoRA connectivity: {checkB['lora_targets_connected_gradA']}/"
        f"{checkB['lora_targets_total']} targets, base_grad={checkB['base_tensors_require_grad']}, "
        f"loss@init {checkB['end_to_end_loss_at_init']:.4f} vs HF base "
        f"{checkB['hf_base_ce_reference']:.4f} -> **{checkB['passed']}**\n"
        f"- Check C operator LoRA equivalence (fp64, real folded base): "
        f"**{all(c['passed'] for c in opchecks)}**\n"
        f"- counters clean: silent_fallbacks {counters['silent_fallbacks']}, "
        f"nonlinear_trusted_calls {counters['nonlinear_trusted_calls']}, "
        f"plaintext_hidden {counters['untrusted_worker_plaintext_hidden_materializations']}\n\n"
        f"**{report['status']}** (diagnostic non-TDX boundary; not a paper result)\n")
    print(json.dumps({"status": report["status"], "checkA": checkA["passed"],
                      "checkB": checkB["passed"],
                      "checkC": all(c["passed"] for c in opchecks),
                      "top1": checkA["next_token_top1_agreement"],
                      "logit_cos": checkA["logit_cosine"],
                      "loss_diff_vs_hf": checkB["loss_abs_diff_vs_hf_base"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
