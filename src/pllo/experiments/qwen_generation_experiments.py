"""E1/E2 no-LoRA Qwen2.5-7B masked-generation experiments (paper-grade).

Reuses the validated Stage 8.4 masked execution
(:mod:`pllo.hf_wrappers.qwen_memory_optimized`) to produce the main no-LoRA
results under the fixed paper configuration (bs=1, seq_len=128, max_new_tokens=64,
28/28 layers, bf16 on H800). Three decoding modes:

* **greedy** -- deterministic; exact token-level plain-vs-masked + HF comparison.
* **teacher-forced** -- fixed-prefix per-step correctness (the main long-horizon
  metric for bf16): top-1 agreement hf/plain/masked, top-k overlap, logits errors.
* **sampling** -- temperature/top_p with a fixed seed; realistic service stress;
  no exact-match requirement (completion + top-k overlap + logits errors + latency).

Nothing of ours runs in a TEE (``tee_used_on_gpu=False``); the masked decoder /
attention / MLP / KV cache / LM head run on the untrusted GPU. The byte/boundary
accounting follows the TEE<->GPU protocol model (masked tensors cross; plaintext
hidden / input ids / recovered logits / sampled tokens stay trusted).

The real Qwen2.5-7B path is gated on CUDA + a local checkpoint; a tiny-random
Qwen2 ``--dry-run`` validates the runner end-to-end on CPU (never reported as a
paper result).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from pllo.hf_wrappers.qwen_memory_optimized import (
    MemoryOptimizedConfig,
    masked_prefill_full_logits,
    run_memory_optimized_masked,
)

__all__ = [
    "protocol_accounting",
    "topk_overlap",
    "teacher_forced_block",
    "run_e1_nolora",
    "run_e2_token_scaling",
]

_DT_BYTES = {"float16": 2, "bf16": 2, "bfloat16": 2, "fp16": 2,
             "float32": 4, "fp32": 4}


def _dtype_bytes(name: str) -> int:
    return _DT_BYTES.get((name or "").lower(), 4)


def protocol_accounting(seq_len: int, hidden: int, vocab: int,
                        max_new_tokens: int, dtype: str) -> dict[str, int]:
    """TEE<->GPU byte/boundary accounting for one masked generation.

    GPU sees masked hidden in + masked logits out per call; the boundary keeps the
    recovered logits + sampled token + does embedding masking. One prefill + (N-1)
    decode calls; each decode step also does a trusted embed/recover/sample."""
    b = _dtype_bytes(dtype)
    decode_steps = max(0, max_new_tokens - 1)
    gpu_bytes = (seq_len * hidden + vocab) * b              # prefill in+out
    gpu_bytes += decode_steps * (hidden + vocab) * b       # per-step in+out
    # trusted: embedding masking (prompt + each new token) + recover + sample
    trusted_bytes = (seq_len * hidden + decode_steps * hidden) * b
    trusted_bytes += max_new_tokens * (vocab * b + 8)
    boundary_calls = {
        "embed_and_mask": 1 + decode_steps,
        "recover_logits": max_new_tokens,
        "sample": max_new_tokens,
    }
    return {"gpu_bytes": int(gpu_bytes), "trusted_bytes": int(trusted_bytes),
            "boundary_calls": boundary_calls}


def topk_overlap(a_logits: torch.Tensor, b_logits: torch.Tensor,
                 k: int = 5) -> float:
    """Mean per-row top-k index overlap (|A∩B|/k) between two logit tensors."""
    k = min(k, a_logits.shape[-1])
    ta = a_logits.topk(k, dim=-1).indices
    tb = b_logits.topk(k, dim=-1).indices
    rows = ta.reshape(-1, k)
    rb = tb.reshape(-1, k)
    ov = []
    for i in range(rows.shape[0]):
        ov.append(len(set(rows[i].tolist()) & set(rb[i].tolist())) / k)
    return float(sum(ov) / len(ov)) if ov else 0.0


def _err_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    diff = a - b
    denom = float(torch.linalg.norm(b)) or 1.0
    return {"max_abs_error": float(diff.abs().max()),
            "mean_abs_error": float(diff.abs().mean()),
            "relative_l2_error": float(torch.linalg.norm(diff) / denom)}


def teacher_forced_block(model: Any, mc: Any, prompt_ids: torch.Tensor,
                         ref_ids: torch.Tensor, cfg: MemoryOptimizedConfig,
                         topk: int = 5) -> dict[str, Any]:
    """Per-step top-1 agreement + top-k overlap + logits errors under a fixed
    prefix (HF reference tokens). Avoids free-running cascade."""
    device = torch.device(cfg.device)
    teacher = torch.cat([prompt_ids, ref_ids], dim=1).to(device)
    L = int(prompt_ids.shape[1])
    N = int(ref_ids.shape[1])
    tf_cfg = replace(cfg, seq_len=int(teacher.shape[1]), max_new_tokens=1)
    plain_logits, recovered = masked_prefill_full_logits(model, mc, teacher, tf_cfg)
    with torch.no_grad():
        hf_logits = model(input_ids=teacher).logits.to(plain_logits.dtype)

    hp = hm = pm = 0
    em_max = em_mean = em_l2 = 0.0
    ov = 0.0
    hf_rows = []
    mk_rows = []
    for t in range(N):
        pos = L - 1 + t
        hf_t = hf_logits[:, pos, :]
        pl_t = plain_logits[:, pos, :]
        mk_t = recovered[:, pos, :]
        hp += int((hf_t.argmax(-1) == pl_t.argmax(-1)).all())
        hm += int((hf_t.argmax(-1) == mk_t.argmax(-1)).all())
        pm += int((pl_t.argmax(-1) == mk_t.argmax(-1)).all())
        ov += topk_overlap(hf_t, mk_t, topk)
        hf_rows.append(hf_t)
        mk_rows.append(mk_t)
    if N:
        es = _err_stats(torch.cat(mk_rows), torch.cat(hf_rows))
        em_max, em_mean, em_l2 = (es["max_abs_error"], es["mean_abs_error"],
                                  es["relative_l2_error"])
    inv = 1.0 / N if N else 0.0
    return {
        "teacher_forced_steps_evaluated": N,
        "teacher_forced_top1_match_rate_hf_plain": hp * inv,
        "teacher_forced_top1_match_rate_hf_masked": hm * inv,
        "teacher_forced_top1_match_rate_plain_masked": pm * inv,
        "logits_max_abs_error": em_max,
        "logits_mean_abs_error": em_mean,
        "logits_relative_l2_error": em_l2,
        "topk_overlap": ov * inv,
        "topk": topk,
    }


def _hf_generate(model, input_ids, max_new_tokens, *, do_sample, temperature=0.7,
                 top_p=0.9, seed=0):
    import torch as _t
    kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=1,
              pad_token_id=getattr(model.config, "eos_token_id", None))
    if do_sample:
        _t.manual_seed(seed)
        kw.update(temperature=temperature, top_p=top_p)
    with _t.no_grad():
        out = model.generate(input_ids=input_ids, **kw)
    return out[:, input_ids.shape[1]:]                      # only new tokens


def run_e1_nolora(model: Any, mc: Any, input_ids: torch.Tensor,
                  cfg: MemoryOptimizedConfig, *, modes=("greedy", "teacher_forced",
                  "sampling"), topk: int = 5, temperature: float = 0.7,
                  top_p: float = 0.9, sample_seed: int = 0) -> dict[str, Any]:
    """Run the E1 no-LoRA masked-generation experiment for the given input."""
    import time
    device = torch.device(cfg.device)
    input_ids = input_ids.to(device)
    H = int(mc.hidden_size)
    V = int(mc.vocab_size)
    T = int(input_ids.shape[1])
    N = int(cfg.max_new_tokens)
    acct = protocol_accounting(T, H, V, N, cfg.dtype)

    report: dict[str, Any] = {
        "stage": "E1_nolora_qwen_generation",
        "model_type": str(getattr(mc, "model_type", "qwen2")),
        "batch_size": int(input_ids.shape[0]), "seq_len": T,
        "max_new_tokens": N, "num_layers": cfg.num_layers, "dtype": cfg.dtype,
        "tee_used_on_gpu": False,
        "gpu_visible_plaintext_fields": [], "leaked_secret_fields": [],
        "trusted_bytes": acct["trusted_bytes"], "gpu_bytes": acct["gpu_bytes"],
        "boundary_calls": acct["boundary_calls"],
        "peak_gpu_memory_mb": None,
        "modes": {},
    }

    # --- greedy (deterministic exact comparison) ---------------------------
    if "greedy" in modes:
        t0 = time.perf_counter()
        gcfg = replace(cfg, seq_len=T, max_new_tokens=N)
        masked_rep = run_memory_optimized_masked(model, mc, input_ids, gcfg)
        latency = time.perf_counter() - t0
        hf_new = _hf_generate(model, input_ids, N, do_sample=False)
        masked_tokens = masked_rep.get("generated_masked_tokens")
        seq_exact = None
        if masked_tokens is not None:
            mt = torch.as_tensor(masked_tokens).reshape(1, -1)[:, :N]
            seq_exact = bool(torch.equal(mt.to(hf_new.device),
                                         hf_new[:, :mt.shape[1]]))
        peak = (masked_rep.get("peak_memory") or {})
        report["peak_gpu_memory_mb"] = peak.get("max_allocated_mb") \
            if isinstance(peak, dict) else None
        report["modes"]["greedy"] = {
            "plain_vs_masked_token_match_rate":
                masked_rep.get("greedy_token_match", masked_rep.get(
                    "top1_match_rate")),
            "top1_match_rate": masked_rep.get("top1_match_rate"),
            "logits_max_abs_error": masked_rep.get("max_abs_error"),
            "sequence_exact_match_hf_masked": seq_exact,
            "latency_s": latency,
            "executed_layers": masked_rep.get("executed_layers",
                                              masked_rep.get("requested_layers")),
        }

    # --- teacher-forced (main long-horizon correctness) -------------------
    if "teacher_forced" in modes:
        t0 = time.perf_counter()
        ref = _hf_generate(model, input_ids, N, do_sample=False)
        tf = teacher_forced_block(model, mc, input_ids, ref, cfg, topk=topk)
        tf["latency_s"] = time.perf_counter() - t0
        report["modes"]["teacher_forced"] = tf

    # --- sampling (realistic service stress) ------------------------------
    if "sampling" in modes:
        t0 = time.perf_counter()
        sref = _hf_generate(model, input_ids, N, do_sample=True,
                            temperature=temperature, top_p=top_p, seed=sample_seed)
        stf = teacher_forced_block(model, mc, input_ids, sref, cfg, topk=topk)
        report["modes"]["sampling"] = {
            "generation_completion_tokens": int(sref.shape[1]),
            "temperature": temperature, "top_p": top_p, "seed": sample_seed,
            "topk_overlap": stf["topk_overlap"],
            "logits_max_abs_error": stf["logits_max_abs_error"],
            "logits_relative_l2_error": stf["logits_relative_l2_error"],
            "teacher_forced_top1_match_rate_hf_masked":
                stf["teacher_forced_top1_match_rate_hf_masked"],
            "latency_s": time.perf_counter() - t0,
        }
    return report


def run_e2_token_scaling(model: Any, mc: Any, input_ids: torch.Tensor,
                         cfg: MemoryOptimizedConfig,
                         token_grid=(1, 8, 16, 32, 64), *,
                         modes=("greedy", "teacher_forced"), topk: int = 5
                         ) -> dict[str, Any]:
    """Fixed model/prompt/seq_len; sweep ``max_new_tokens`` over ``token_grid``."""
    rows = []
    for nt in token_grid:
        ecfg = replace(cfg, max_new_tokens=int(nt))
        r = run_e1_nolora(model, mc, input_ids, ecfg, modes=modes, topk=topk)
        flat = {
            "max_new_tokens": int(nt), "seq_len": r["seq_len"],
            "dtype": r["dtype"], "num_layers": r["num_layers"],
            "tee_used_on_gpu": False,
            "trusted_bytes": r["trusted_bytes"], "gpu_bytes": r["gpu_bytes"],
            "boundary_calls": r["boundary_calls"],
            "peak_gpu_memory_mb": r["peak_gpu_memory_mb"],
        }
        if "greedy" in r["modes"]:
            g = r["modes"]["greedy"]
            flat["greedy_plain_vs_masked_token_match_rate"] = \
                g["plain_vs_masked_token_match_rate"]
            flat["greedy_latency_s"] = g["latency_s"]
        if "teacher_forced" in r["modes"]:
            tf = r["modes"]["teacher_forced"]
            flat["tf_top1_hf_masked"] = \
                tf["teacher_forced_top1_match_rate_hf_masked"]
            flat["tf_top1_plain_masked"] = \
                tf["teacher_forced_top1_match_rate_plain_masked"]
            flat["tf_logits_max_abs_error"] = tf["logits_max_abs_error"]
            flat["tf_topk_overlap"] = tf["topk_overlap"]
        rows.append(flat)
    return {"stage": "E2_token_scaling", "token_grid": list(token_grid),
            "tee_used_on_gpu": False, "rows": rows}
