#!/usr/bin/env python
"""CONJFORMER (arXiv:2606.16461) real-model reproduction on Qwen2.5-7B-Instruct.

Faithful client/server split (paper Fig. 1 + §2.2):
  - CLIENT keeps embed_tokens + lm_head + the secret basis {U, O_b, R_b, P_b}.
  - SERVER holds the *conjugated* decoder body and runs the WHOLE forward pass in
    the rotated basis (X_0 U^T -> X_B U^T), never seeing an unrotated hidden.
The obfuscation is exactly equivariant (verified numerically here), so a
CONJFORMER completion is token-identical to the retrofit (scalar-RMSNorm) model.
The utility question is therefore: what does the required scalar-RMSNorm
architecture change cost, with NO fine-tune (the client cannot re-run Qwen's
instruct recipe)? We measure that against the unmodified plaintext model.

Modes (default all):
  verify    numerically confirm server-on-rotated == retrofit-on-plain (real 7B)
  generate  greedy-decode N prompts on {plaintext, conjformer} -> quality + samples
  latency   per-token latency + throughput of the real obfuscated pipeline vs plaintext
  finetune  retrofit fine-tuning throughput (tok/s) -> wall-time to recover (paper D.3)

All server-side compute is a normal Qwen2 forward (same FLOPs as plaintext) plus
a client-side d x d rotate/de-rotate per step -> no extra TEE round-trip (contrast
our folded_remote, which pays a boundary round-trip + fp32 logits wire per token).
Secret matrices are never serialised.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import copy

from pllo.baselines.conjformer import (
    conjugate_qwen2,
    retrofit_scalar_rmsnorm,
    rotate_in,
    rotate_out,
    sample_secrets,
    structural_audit,
    verify_equivariance,
)

DEFAULT_PROMPTS = [
    "Write a 3-sentence summary of why orthogonal matrices preserve vector norms. "
    "Do not use the word 'rotation'.",
    "List exactly 5 fruits, each on its own line, numbered 1 to 5.",
    "Explain what a trusted execution environment is in under 60 words.",
    "Write a Python function is_prime(n) that returns True iff n is prime. "
    "Respond with only the code.",
    "Give three bullet points about differential privacy. Each bullet must start with the word 'It'.",
]


def load_prompts(path: str | None, field: str, n: int) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS[:n]
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj[field])
            if len(prompts) >= n:
                break
    return prompts


@torch.no_grad()
def manual_greedy(embed, lm_head, body, secrets, input_ids, max_new_tokens,
                  eos_ids, device):
    """One faithful greedy decode. secrets=None => plaintext path (no rotation);
    secrets set => CONJFORMER (rotate embeddings in, run server body in rotated
    basis with KV cache, de-rotate the returned hidden, apply client lm_head).

    Returns (generated_ids[list], n_prompt_tokens, per_step_seconds[list])."""
    from transformers import DynamicCache

    cache = DynamicCache()
    cur = input_ids.to(device)
    n_prompt = cur.shape[1]
    past = 0
    out_ids: list[int] = []
    step_s: list[float] = []
    for _ in range(max_new_tokens):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        emb = embed(cur)                                  # client embedding
        if secrets is not None:
            emb = rotate_in(emb, secrets)                 # client -> server basis
        L = cur.shape[1]
        cache_pos = torch.arange(past, past + L, device=device)
        pos_ids = cache_pos.unsqueeze(0)
        attn = torch.ones(1, past + L, device=device, dtype=torch.long)
        res = body(inputs_embeds=emb, past_key_values=cache, use_cache=True,
                   cache_position=cache_pos, position_ids=pos_ids,
                   attention_mask=attn)
        cache = res.past_key_values
        h = res.last_hidden_state[:, -1:, :]              # rotated normed hidden
        if secrets is not None:
            h = rotate_out(h, secrets)                    # server -> client basis
        logits = lm_head(h)                               # client lm_head
        nxt = int(logits[0, -1].argmax().item())
        if device == "cuda":
            torch.cuda.synchronize()
        step_s.append(time.perf_counter() - t0)
        out_ids.append(nxt)
        past += L
        cur = torch.tensor([[nxt]], device=device)
        if eos_ids and nxt in eos_ids:
            break
    return out_ids, n_prompt, step_s


def degeneration_flags(ids: list[int]) -> dict:
    """Cheap degeneration signal: longest run of an identical token + repeat ratio."""
    if not ids:
        return {"len": 0, "max_run": 0, "repeat_ratio": 0.0}
    max_run = run = 1
    for i in range(1, len(ids)):
        run = run + 1 if ids[i] == ids[i - 1] else 1
        max_run = max(max_run, run)
    uniq = len(set(ids))
    return {"len": len(ids), "max_run": max_run,
            "repeat_ratio": round(1.0 - uniq / len(ids), 3)}


def build(model_path, dtype, device, seed, rope_compatible):
    """Memory-lean build: hold at most 2 full 7B copies (plain + server).

    server is made by deep-copying plain ONCE, then retrofitting scalar-RMSNorm
    and conjugating IN PLACE -> peak = plain + server = 2 copies (fp32 7B ~56GB).
    The scalar-RMSNorm reference model (`retro`) needed by verify/finetune is
    built transiently inside those blocks and freed, so only they briefly peak
    at 3 copies. plain's embed_tokens / lm_head double as the CLIENT modules
    (retrofit never touches embeddings)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    td = {"float32": torch.float32, "bfloat16": torch.bfloat16,
          "float16": torch.float16}[dtype]
    print(f"[build] loading {model_path} ({dtype}) ...", flush=True)
    plain = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=td).to(device).eval()
    secrets = sample_secrets(plain.config, seed=seed, rope_compatible=rope_compatible)
    print("[build] deepcopy -> retrofit scalar-RMSNorm -> conjugate (server, in place) ...", flush=True)
    server = copy.deepcopy(plain)
    n_norm = retrofit_scalar_rmsnorm(server)
    conjugate_qwen2(server, secrets)
    server = server.to(device).eval()
    audit = structural_audit()
    audit["norm_layers_retrofitted"] = n_norm
    return tok, plain, server, secrets, audit


def make_retro(plain, device):
    """Transient scalar-RMSNorm (unconjugated) reference; caller frees it."""
    retro = copy.deepcopy(plain)
    retrofit_scalar_rmsnorm(retro)
    return retro.to(device).eval()


def chat_ids(tok, prompt, device):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt")["input_ids"].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="all",
                    choices=["all", "verify", "generate", "latency", "finetune"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"],
                    help="bfloat16 default fits 3x7B on 80GB; CONJFORMER's orthogonal "
                         "conjugation is numerically benign so bf16 is faithful (verify "
                         "confirms). fp32 'all' needs ~84GB -> run modes separately.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rope-compatible", action="store_true",
                    help="use full (non-RoPE-safe) O_b; breaks logit invariance (ablation)")
    ap.add_argument("--prompts-jsonl", default=None)
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--n-prompts", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--ft-tokens-per-step", type=int, default=4096)
    ap.add_argument("--ft-steps", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-out", default=None)
    a = ap.parse_args()
    device = a.device
    rope_compatible = not a.no_rope_compatible

    tok, plain, server, secrets, audit = build(
        a.model, a.dtype, device, a.seed, rope_compatible)
    # move the secret basis to the device ONCE (else rotate would copy a 100MB
    # fp64 d x d matrix host->device every decode step -> fake latency overhead)
    secrets.U = secrets.U.to(device)
    vocab_size = plain.config.vocab_size
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))
    report = {"model": a.model, "dtype": a.dtype, "seed": a.seed,
              "rope_compatible_O": rope_compatible, "audit": audit}

    client_embed = plain.get_input_embeddings()      # embeddings unchanged by retrofit
    client_head = plain.get_output_embeddings()

    # ------------------------------------------------------------------ verify
    if a.mode in ("all", "verify"):
        ids = chat_ids(tok, DEFAULT_PROMPTS[0], device)[:, :24]
        retro = make_retro(plain, device)
        rep = verify_equivariance(retro, server, secrets, ids)
        del retro
        if device == "cuda":
            torch.cuda.empty_cache()
        rep["note"] = ("server(rotated) == retrofit(plain) => the obfuscation itself "
                       "adds no utility cost; any gap vs plaintext is the scalar-RMSNorm change")
        report["verify"] = rep
        print(f"[verify] logit_top1_agreement={rep['logit_top1_agreement']:.4f} "
              f"hidden_rel_err={rep['hidden_rel_err']:.2e} equivariant={rep['equivariant']}",
              flush=True)

    # ---------------------------------------------------------------- generate
    if a.mode in ("all", "generate"):
        prompts = load_prompts(a.prompts_jsonl, a.prompt_field, a.n_prompts)
        samples, gstats = [], {"plaintext": [], "conjformer": []}
        for pi, prompt in enumerate(prompts):
            ids = chat_ids(tok, prompt, device)
            p_ids, _, _ = manual_greedy(client_embed, client_head, plain.model, None,
                                        ids, a.max_new_tokens, eos_ids, device)
            c_ids, _, _ = manual_greedy(client_embed, client_head, server.model, secrets,
                                        ids, a.max_new_tokens, eos_ids, device)
            p_txt = tok.decode(p_ids, skip_special_tokens=True)
            c_txt = tok.decode(c_ids, skip_special_tokens=True)
            samples.append({"prompt": prompt, "plaintext": p_txt, "conjformer": c_txt,
                            "plaintext_deg": degeneration_flags(p_ids),
                            "conjformer_deg": degeneration_flags(c_ids)})
            gstats["plaintext"].append(degeneration_flags(p_ids))
            gstats["conjformer"].append(degeneration_flags(c_ids))
            print(f"[generate] {pi+1}/{len(prompts)} plain_len={len(p_ids)} "
                  f"conj_len={len(c_ids)} conj_maxrun={degeneration_flags(c_ids)['max_run']}",
                  flush=True)
        report["generate"] = {
            "n_prompts": len(prompts), "max_new_tokens": a.max_new_tokens,
            "plaintext_mean_maxrun": round(sum(d["max_run"] for d in gstats["plaintext"]) / len(prompts), 2),
            "conjformer_mean_maxrun": round(sum(d["max_run"] for d in gstats["conjformer"]) / len(prompts), 2),
            "plaintext_degenerate": sum(1 for d in gstats["plaintext"] if d["max_run"] >= 20),
            "conjformer_degenerate": sum(1 for d in gstats["conjformer"] if d["max_run"] >= 20),
            "note": "conjformer == retrofit(scalar-RMSNorm, NO fine-tune); "
                    "difference vs plaintext is the architecture-change cost",
        }
        sp = a.samples_out or (str(Path(a.out).with_suffix("")) + "_samples.json")
        with open(sp, "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"[generate] wrote samples -> {sp}", flush=True)

    # ----------------------------------------------------------------- latency
    if a.mode in ("all", "latency"):
        ids = chat_ids(tok, DEFAULT_PROMPTS[2], device)
        # warmup
        manual_greedy(client_embed, client_head, plain.model, None, ids, 8, None, device)
        manual_greedy(client_embed, client_head, server.model, secrets, ids, 8, None, device)
        _, np_p, p_steps = manual_greedy(client_embed, client_head, plain.model, None,
                                         ids, a.max_new_tokens, None, device)
        _, np_c, c_steps = manual_greedy(client_embed, client_head, server.model, secrets,
                                         ids, a.max_new_tokens, None, device)
        # step 0 is prefill; decode steps are 1..; report decode-token latency
        def summ(steps):
            dec = steps[1:] if len(steps) > 1 else steps
            ms = sorted(x * 1000 for x in dec)
            return {"prefill_ms": round(steps[0] * 1000, 2),
                    "decode_ms_median": round(ms[len(ms) // 2], 3),
                    "decode_ms_mean": round(sum(ms) / len(ms), 3),
                    "decode_tok_per_s": round(1000.0 / (sum(ms) / len(ms)), 2),
                    "n_decode_steps": len(dec)}
        report["latency"] = {"prompt_tokens": np_p, "plaintext": summ(p_steps),
                             "conjformer": summ(c_steps),
                             "conjformer_overhead_x": round(
                                 (sum(c_steps[1:]) / max(len(c_steps) - 1, 1)) /
                                 (sum(p_steps[1:]) / max(len(p_steps) - 1, 1)), 3),
                             "note": "conjformer = server full-forward (same FLOPs) + client d x d "
                                     "rotate/de-rotate; NO TEE round-trip. Contrast ours ~9x/token."}
        print(f"[latency] plaintext {report['latency']['plaintext']['decode_tok_per_s']} tok/s | "
              f"conjformer {report['latency']['conjformer']['decode_tok_per_s']} tok/s | "
              f"overhead {report['latency']['conjformer_overhead_x']}x", flush=True)

    # ---------------------------------------------------------------- finetune
    if a.mode in ("all", "finetune"):
        # retrofit fine-tuning throughput: the scalar-RMSNorm model must be
        # fine-tuned to recover quality (paper D.3 ~2000 retrofit iters). We
        # measure the COMPUTE-bound throughput (fwd+bwd) of a real train step on
        # the 7B and extrapolate wall-time. Free the inference copies first;
        # full-param AdamW states (56GB fp32) don't fit alongside them, and the
        # paper itself fine-tunes on 4xA100 -- so we time fwd+bwd+SGD (no extra
        # optimizer state) with gradient checkpointing, the compute lower bound.
        retro = make_retro(plain, device)
        try:
            del server
        except NameError:
            pass
        del plain
        if device == "cuda":
            torch.cuda.empty_cache()
        retro.train()
        if hasattr(retro, "gradient_checkpointing_enable"):
            retro.gradient_checkpointing_enable()
            retro.config.use_cache = False
        opt = torch.optim.SGD(retro.parameters(), lr=1e-4)
        seq = min(a.ft_tokens_per_step, 2048)
        bsz = max(1, a.ft_tokens_per_step // seq)
        g = torch.Generator().manual_seed(0)
        vocab = vocab_size
        ft_ms = []
        for step in range(a.ft_steps):
            ids = torch.randint(0, vocab, (bsz, seq), generator=g).to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = retro(input_ids=ids, labels=ids)
            out.loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if step >= 2:  # drop warmup
                ft_ms.append(dt)
            print(f"[finetune] step {step} loss={out.loss.item():.3f} {dt*1000:.0f}ms", flush=True)
        del retro
        if device == "cuda":
            torch.cuda.empty_cache()
        mean_s = sum(ft_ms) / max(len(ft_ms), 1)
        tok_per_step = bsz * seq
        report["finetune"] = {
            "tokens_per_step": tok_per_step, "step_s_mean": round(mean_s, 4),
            "train_tok_per_s": round(tok_per_step / mean_s, 1),
            "optimizer": "SGD + gradient_checkpointing (compute-bound fwd+bwd; "
                         "full-param AdamW needs 56GB states / multi-GPU, paper uses 4xA100)",
            "est_retrofit_2000iter_min": round(2000 * mean_s / 60.0, 1),
            "est_retrofit_2000iter_min_note": "paper D.3 retrofit ~2000 iters @ 8192 tok/iter; "
                    "single H800 lower bound at this per-step compute",
            "note": "CONJFORMER needs this retrofit fine-tune (plus, for an instruct "
                    "model, an instruct-preserving fine-tune it never demonstrates) to "
                    "recover quality; ours needs ZERO fine-tuning (serves the unmodified "
                    "instruct model). Wall-time here scales with tokens_per_step.",
        }
        print(f"[finetune] {report['finetune']['train_tok_per_s']} tok/s | "
              f"~{report['finetune']['est_retrofit_2000iter_min']} min for 2000 retrofit iters",
              flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
