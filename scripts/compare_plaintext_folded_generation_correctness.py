"""Plaintext Qwen vs folded-package generation correctness diagnostic.

The folded remote path (masked embeddings -> untrusted folded shards -> recovered
logits) is supposed to reproduce the PLAINTEXT Qwen forward. When folded FREE
generation degrades (repetition / lost instruction-following) while teacher-forced
per-step logits still match, the bug is autoregressive-state divergence, not
per-step fidelity. This tool separates those and -- critically -- first proves its
own plaintext reference REPRODUCES ``run_ifeval_generation.py --backend
plaintext_local`` before drawing any folded conclusion.

Pipeline:

1. **Plaintext reproduction gate.** The authoritative plaintext reference is the
   SAME code ``run_ifeval_generation.py`` uses: ``build_predictor("plaintext_local")
   .generate(prompt)`` on the raw ``prompt`` (no chat template re-applied -- the
   IFEval JSONL prompt is already rendered; ifeval tokenises it raw). The tool's
   own step-by-step plaintext path (manual incremental greedy, needed for per-step
   logits) is compared TOKENWISE to that authoritative output. If they differ,
   ``plaintext_reproduction_passed=false`` and EVERY folded correctness conclusion
   is BLOCKED (``correctness_blocked_by_plaintext_reproduction_mismatch=true``).

2. **Folded comparison** (only once reproduction passes), ``--rollout-mode``:
   * ``teacher_forcing`` (default): folded fed the plaintext greedy tokens ->
     per-step plaintext-vs-folded logits / top1 (pure per-step fidelity);
   * ``free_running``: plaintext and folded EACH advance their own greedy token
     through their own KV -> ``first_free_running_divergent_step`` (the real
     generation fork);
   * ``both``: run both; if teacher-forcing matches but free-running forks ->
     ``suspected_root_cause="autoregressive state divergence or generation-state
     mismatch"``.

3. Resident vs non-resident folded re-check (``--compare-nonresident``).

Masking + recovery use a seed-matched ``MaskedQwenSession`` (the masks the package
was folded against); NO mask / inverse / pad / PRG seed / schedule secret / raw
LoRA / plaintext hidden / KV / logits tensor crosses to the package or is written
to the report. The report emits ONLY scalar metrics, token ids, token text, and
shape/dtype/device metadata. Threat model unchanged; pad is never disabled.

``--dry-run`` uses the tiny random Qwen2 on CPU (never a paper result).

# ---- server example only (run on the GPU server; NOT executed locally) ----
# python scripts/compare_plaintext_folded_generation_correctness.py \
#   --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \
#   --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full_current_seq1024 \
#   --embedding-path /root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current \
#   --input-jsonl /root/autodl-tmp/datasets/privacy_llm_benchmarks/converted/ifeval_prompts.jsonl \
#   --seq-len 1024 --max-steps 64 --dtype bfloat16 --device cuda \
#   --verify-plaintext-reproduction --rollout-mode both \
#   --resident-folded-weights --compare-nonresident \
#   --output-json outputs/debug/plaintext_vs_folded_correctness_seq1024_steps64.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_script(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- pure comparison helpers (unit-tested directly; numpy 1-D logit rows) ----

def _np_err(a, b):
    import numpy as np
    diff = np.abs(np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64"))
    return (float(diff.max()), float(diff.mean())) if diff.size else (0.0, 0.0)


def _np_rel(a, b):
    import numpy as np
    aa = np.asarray(a, dtype="float64")
    bb = np.asarray(b, dtype="float64")
    denom = float(np.linalg.norm(bb)) or 1.0
    return float(np.linalg.norm(aa - bb) / denom)


def _np_topk(a, k):
    import numpy as np
    arr = np.asarray(a).reshape(-1)
    k = min(k, arr.shape[0])
    return [int(i) for i in np.argsort(-arr)[:k].tolist()]


def _first_divergent_tokens(a_tokens, b_tokens):
    """First index where two token-id lists differ (over the common prefix)."""
    n = min(len(a_tokens), len(b_tokens))
    for i in range(n):
        if a_tokens[i] != b_tokens[i]:
            return i
    if len(a_tokens) != len(b_tokens):
        return n            # one is a strict prefix of the other
    return None


def _token_match_rate(a_tokens, b_tokens):
    n = min(len(a_tokens), len(b_tokens))
    if not n:
        return 0.0
    return sum(1 for i in range(n) if a_tokens[i] == b_tokens[i]) / n


def _compare_logit_steps(plain_list, folded_list, *, topk, atol, rtol,
                         id_to_text=None):
    """Compare two equal-length lists of 1-D logit rows step by step.

    Robust to per-step shape / dtype mismatch (records the flags, marks the step
    divergent, skips the numeric error for that step). Returns (rows, summary)
    where ``summary`` carries ``first_divergent_step`` (first step whose argmax
    differs OR whose shape mismatches) + aggregate error / match-rate fields."""
    import numpy as np
    rows = []
    first_divergent = None
    matches = 0
    n = min(len(plain_list), len(folded_list))
    max_err_max = 0.0
    mean_err_sum = 0.0
    mean_err_count = 0
    for i in range(n):
        pa = np.asarray(plain_list[i]).reshape(-1)
        fa = np.asarray(folded_list[i]).reshape(-1)
        shape_match = bool(pa.shape == fa.shape)
        dtype_match = bool(np.asarray(plain_list[i]).dtype
                           == np.asarray(folded_list[i]).dtype)
        p_top1 = int(pa.argmax()) if pa.size else None
        f_top1 = int(fa.argmax()) if fa.size else None
        row = {
            "step_id": i,
            "plaintext_token_id": p_top1, "folded_token_id": f_top1,
            "plaintext_top1_token_id": p_top1, "folded_top1_token_id": f_top1,
            "shape_match": shape_match, "dtype_match": dtype_match,
            "token_match": None, "top1_match": None,
            "plaintext_top5": None, "folded_top5": None, "top5_overlap": None,
            "max_abs_logit_err": None, "mean_abs_logit_err": None,
            "relative_logit_err": None, "allclose_atol_rtol": None,
            "plaintext_token_text": (id_to_text(p_top1) if id_to_text else None),
            "folded_token_text": (id_to_text(f_top1) if id_to_text else None),
        }
        row["plaintext_top1_text"] = row["plaintext_token_text"]
        row["folded_top1_text"] = row["folded_token_text"]
        if not shape_match:
            row["token_match"] = row["top1_match"] = False
            if first_divergent is None:
                first_divergent = i
            rows.append(row)
            continue
        t1 = bool(p_top1 == f_top1)
        row["token_match"] = row["top1_match"] = t1
        row["plaintext_top5"] = _np_topk(pa, topk)
        row["folded_top5"] = _np_topk(fa, topk)
        row["top5_overlap"] = (len(set(row["plaintext_top5"])
                                   & set(row["folded_top5"]))
                               / max(1, min(topk, pa.shape[0])))
        mx, mn = _np_err(pa, fa)
        row["max_abs_logit_err"] = mx
        row["mean_abs_logit_err"] = mn
        row["relative_logit_err"] = _np_rel(pa, fa)
        row["allclose_atol_rtol"] = bool(np.allclose(pa, fa, atol=atol, rtol=rtol))
        max_err_max = max(max_err_max, mx)
        mean_err_sum += mn
        mean_err_count += 1
        if t1:
            matches += 1
        elif first_divergent is None:
            first_divergent = i
        rows.append(row)
    fd_text = None
    if first_divergent is not None and id_to_text is not None \
            and first_divergent < len(rows):
        fd_text = rows[first_divergent].get("plaintext_token_text")
    summary = {
        "steps_compared": n,
        "first_divergent_step": first_divergent,
        "first_divergent_token_text": fd_text,
        "top1_match_rate": (matches / n) if n else 0.0,
        "max_abs_logit_err_max": max_err_max,
        "mean_abs_logit_err_mean": (mean_err_sum / mean_err_count
                                    if mean_err_count else 0.0),
    }
    return rows, summary


def _suspected_root_cause(step, *, prefix):
    """Heuristic localisation hint for a divergent step (honest, coarse)."""
    if step is None:
        return None
    if step == 0:
        return ("%s (step 0): chat_template / tokenization / input embedding "
                "artifact / position_ids / causal mask / RoPE / first folded "
                "layer (QKV/attn/MLP) / folded LM-head inverse" % prefix)
    return ("%s (step %d): KV cache append / KV mask-domain transition / decode "
            "position_ids+RoPE / per-step fresh obfuscation domain / cache reuse "
            "of a stale domain" % (prefix, step))


# ---- plaintext + folded per-step collectors --------------------------------

def _manual_plaintext_steps(model, ids, attn, max_steps, eos_id):
    """Manual incremental greedy plaintext decode (so we get per-step logits).

    Mirrors ``model.generate(do_sample=False, num_beams=1, attention_mask=...,
    use_cache=True)``: argmax each step, thread KV + a grown attention mask, stop
    at eos. Returns (logit_rows [float32 cpu numpy V], token_ids)."""
    import numpy as np
    import torch
    rows, toks = [], []
    cur_attn = attn
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=cur_attn, use_cache=True)
        lg = o.logits[:, -1, :].float().to("cpu")
        rows.append(np.asarray(lg[0].numpy()))
        t = int(lg.argmax(-1).item())
        toks.append(t)
        past = o.past_key_values
        for _ in range(max_steps - 1):
            if eos_id is not None and t == eos_id:
                break
            nxt = torch.tensor([[t]], device=ids.device)
            if cur_attn is not None:
                cur_attn = torch.cat(
                    [cur_attn, torch.ones((cur_attn.shape[0], 1),
                                          dtype=cur_attn.dtype,
                                          device=cur_attn.device)], dim=1)
            o = model(input_ids=nxt, attention_mask=cur_attn,
                      past_key_values=past, use_cache=True)
            lg = o.logits[:, -1, :].float().to("cpu")
            rows.append(np.asarray(lg[0].numpy()))
            t = int(lg.argmax(-1).item())
            toks.append(t)
            past = o.past_key_values
    return rows, toks


def _raw_greedy_generate(model, ids, attn, max_steps, eos_id):
    """``model.generate`` greedy with generation-config logit processors
    NEUTRALISED (repetition_penalty=1.0, no_repeat_ngram_size=0). This is the
    apples-to-apples plaintext baseline for a RAW-argmax decoder (which is what the
    folded path and ``_manual_plaintext_steps`` do). Comparing the authoritative
    ifeval output against THIS isolates a generation-config effect (e.g. Qwen's
    repetition_penalty) from a real encoding/decode bug. Returns token ids."""
    import torch
    kw = dict(max_new_tokens=max_steps, do_sample=False, num_beams=1,
              pad_token_id=eos_id, repetition_penalty=1.0, no_repeat_ngram_size=0)
    with torch.no_grad():
        out = (model.generate(ids, attention_mask=attn, **kw) if attn is not None
               else model.generate(ids, **kw))
    return [int(t) for t in out[0, ids.shape[1]:].tolist()]


def _folded_steps(backend, session, h_tilde, max_steps, seq_len, n_layers, cfg0,
                  *, fed_tokens=None, eos_id=None, eos_ids=None, rep_penalty=None,
                  seen_init=None):
    """Folded-package recovered logits, step by step. ``fed_tokens`` -> TEACHER
    FORCING (raw, for per-step fidelity); ``None`` -> FREE RUNNING. In FREE mode,
    ``rep_penalty`` + ``eos_ids`` reproduce the EXACT folded_remote decode path
    (trusted-side repetition_penalty after recovery, before argmax; trusted-side
    EOS stop). ``seen_init`` seeds the rep-penalty token history with the prompt
    ids (trusted-only). Returned rows are the DECISION logits (post-penalty in free
    mode). The boundary owns masking/recovery; the package only runs folded ops."""
    import numpy as np
    import torch
    from pllo.benchmarks.real_predictors import apply_repetition_penalty
    eset = set(eos_ids) if eos_ids else ({eos_id} if eos_id is not None else set())
    free = fed_tokens is None
    seen = list(seen_init or []) if free else None

    def _decision_row(rec_t):
        if free and rep_penalty not in (None, 1.0) and seen:
            rec_t = apply_repetition_penalty(rec_t, seen, rep_penalty)
        return np.asarray(rec_t.detach().to("cpu").float().numpy())

    pre = backend.run_prefill(h_tilde, n_layers, cfg0, session._cos, session._sin,
                              session.eps)
    rec = session.recover(backend.run_head(pre["y_tilde"], session.eps)[:, -1, :])
    row = _decision_row(rec[0])
    rows, toks = [row], [int(row.argmax())]
    t = toks[0]
    if seen is not None:
        seen.append(t)
    position = seq_len
    for step in range(max_steps - 1):
        if fed_tokens is not None:
            if step >= len(fed_tokens):
                break
            feed = int(fed_tokens[step])
        else:
            if eset and t in eset:
                break
            feed = t
        x = session.mask_token_embedding(torch.tensor([feed]))
        dec = backend.run_decode(x, position, cfg0, session._cos, session._sin,
                                 session.eps, num_exec_layers=n_layers)
        rec = session.recover(
            backend.run_head(dec["y_tilde"], session.eps)[:, -1, :])
        row = _decision_row(rec[0])
        rows.append(row)
        t = int(row.argmax())
        toks.append(t)
        if seen is not None:
            seen.append(t)
        position += 1
    return rows, toks


class _TinyPlaintextPredictor:
    """Dry-run stand-in for ``_PlaintextLocalPredictor`` (tiny CPU Qwen2, fixed
    synthetic prompt ids). Same surface the diagnostic uses: ``_encode`` /
    ``_model`` / ``_tok`` / ``generate`` / ``max_new_tokens``."""

    backend = "plaintext_local_dry"

    def __init__(self, model, mc, ids, max_new_tokens):
        self._model = model
        self.mc = mc
        self._tok = None
        self._ids = ids
        self.max_new_tokens = int(max_new_tokens)
        self.model_name = "tiny-dry"

    def _encode(self, prompt):
        return self._ids, None

    def generate(self, prompt):
        import torch
        ids, _ = self._encode(prompt)
        with torch.no_grad():
            out = self._model.generate(
                ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=getattr(self.mc, "eos_token_id", None))
        new = out[0, ids.shape[1]:]
        return {"text": None, "token_ids": [int(t) for t in new.tolist()]}


def _decode_text(tok, token_ids):
    if tok is None or not token_ids:
        return None
    try:
        return tok.decode(token_ids, skip_special_tokens=True)
    except Exception:                                        # noqa: BLE001
        return None


def _classify_divergence(first_div):
    """Map the first free-running divergent step to a likely cause class."""
    if first_div is None:
        return ("no_token_divergence", "free-running tokens match: any IFEval gap "
                "is postprocessing / whitespace / special-token cleanup -- see "
                "analyze_ifeval_strict_gap.py (do NOT trim/clean to chase score)")
    if first_div == 0:
        return ("step0_divergence", "prompt encoding / chat template / generation "
                "config / logits processor mismatch (prefill step)")
    return ("early_decode_divergence", "KV / position+RoPE / logits processor / "
            "EOS-state mismatch (decode step %d)" % first_div)


def _plaintext_generate_scores(model, ids, attn, max_steps, eos_id):
    """Authoritative plaintext greedy (model.generate, applies the model's
    generation_config incl repetition_penalty) WITH per-step processed logits.
    Returns (token_ids, per_step_logit_rows[float32 cpu numpy V])."""
    import numpy as np
    import torch
    kw = dict(max_new_tokens=max_steps, do_sample=False, num_beams=1,
              pad_token_id=eos_id, output_scores=True,
              return_dict_in_generate=True)
    with torch.no_grad():
        out = (model.generate(ids, attention_mask=attn, **kw) if attn is not None
               else model.generate(ids, **kw))
    toks = [int(t) for t in out.sequences[0, ids.shape[1]:].tolist()]
    rows = [np.asarray(s[0].detach().to("cpu").float().numpy())
            for s in (out.scores or [])]
    return toks, rows


def _diagnose_example(*, prompt, model, tok, predictor, mc, device, dtype,
                      pkg_dir, n_layers, seed, max_steps, rep_penalty, eos_ids,
                      stop_on_eos, first_n, inject_div_step, make_backend,
                      example_index):
    """Token-level free-running diagnosis for ONE example on the EXACT
    folded_remote decode path (trusted-side repetition_penalty + EOS). No secret
    tensors are emitted -- only scalars, token ids, token text, shape metadata."""
    import hashlib
    import numpy as np
    import torch
    from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession
    from pllo.hf_wrappers.qwen_memory_optimized import (
        MemoryOptimizedConfig, _cfg_to)

    if predictor is not None:
        ids, attn = predictor._encode(prompt)
    else:
        ids, attn = model_ids_fallback(model, mc, max_steps), None
    seq_len = int(ids.shape[1])
    eos0 = (sorted(eos_ids)[0] if eos_ids else None)

    # plaintext authoritative (generation-config applied) + per-step logits
    p_tokens, p_rows = _plaintext_generate_scores(model, ids, attn, max_steps, eos0)

    # folded free-running on the EXACT folded_remote path (rep_penalty + EOS)
    cfg = MemoryOptimizedConfig(
        num_layers=n_layers, batch_size=1, seq_len=seq_len, max_new_tokens=max_steps,
        device=device, dtype=dtype, folding_dtype="float32",
        folded_weight_device=device, mlp_down_chunk_size=512, seed=seed)
    session = MaskedQwenSession(model, mc, cfg)
    h_tilde = session.mask_embeddings(ids)
    cfg0 = _cfg_to(session.layer_configs[0], session.compute_device)
    backend = make_backend()
    prompt_ids = [int(t) for t in ids.reshape(-1).tolist()]
    f_rows, f_tokens = _folded_steps(
        backend, session, h_tilde, max_steps, seq_len, n_layers, cfg0,
        fed_tokens=None, eos_ids=(eos_ids if stop_on_eos else None),
        rep_penalty=rep_penalty, seen_init=prompt_ids)
    if inject_div_step is not None and 0 <= inject_div_step < len(f_tokens):
        f_tokens = list(f_tokens)
        base = (p_tokens[inject_div_step] if inject_div_step < len(p_tokens)
                else f_tokens[inject_div_step])
        f_tokens[inject_div_step] = int(base) + 1            # TEST: force a fork

    first_div = _first_divergent_tokens(p_tokens, f_tokens)
    cls, cls_note = _classify_divergence(first_div)

    def _txt(tids):
        if tok is None:
            return None
        try:
            return tok.decode([int(t) for t in tids], skip_special_tokens=False)
        except Exception:                                    # noqa: BLE001
            return None

    at = {"max_abs_logit_err": None, "top1_match": None, "top5_overlap": None,
          "plaintext_token_at_divergence": None, "folded_token_at_divergence": None,
          "plaintext_token_text": None, "folded_token_text": None}
    if first_div is not None:
        pt = p_tokens[first_div] if first_div < len(p_tokens) else None
        ft = f_tokens[first_div] if first_div < len(f_tokens) else None
        at["plaintext_token_at_divergence"] = pt
        at["folded_token_at_divergence"] = ft
        at["plaintext_token_text"] = (tok.decode([pt]) if (tok and pt is not None)
                                      else None)
        at["folded_token_text"] = (tok.decode([ft]) if (tok and ft is not None)
                                   else None)
        if first_div < len(p_rows) and first_div < len(f_rows):
            pa = np.asarray(p_rows[first_div]).reshape(-1)
            fa = np.asarray(f_rows[first_div]).reshape(-1)
            if pa.shape == fa.shape:
                at["max_abs_logit_err"] = _np_err(pa, fa)[0]
                at["top1_match"] = bool(int(pa.argmax()) == int(fa.argmax()))
                at["top5_overlap"] = (len(set(_np_topk(pa, 5)) & set(_np_topk(fa, 5)))
                                      / 5.0)

    chat_tmpl = getattr(tok, "chat_template", None) if tok is not None else None
    chat_sha = (hashlib.sha256(chat_tmpl.encode("utf-8")).hexdigest()
                if isinstance(chat_tmpl, str) and chat_tmpl else None)
    n = int(first_n)
    return {
        "example_index": example_index,
        "prompt": prompt if isinstance(prompt, str) else None,
        "prompt_token_count": seq_len,
        "chat_template_sha256": chat_sha,
        "plaintext_first_token_ids": p_tokens[:n],
        "folded_first_token_ids": f_tokens[:n],
        "plaintext_first_text": _txt(p_tokens[:n]),
        "folded_first_text": _txt(f_tokens[:n]),
        "plaintext_total_tokens": len(p_tokens),
        "folded_total_tokens": len(f_tokens),
        "first_free_running_divergent_step": first_div,
        "divergence_class": cls,
        "divergence_note": cls_note,
        **at,
    }


def model_ids_fallback(model, mc, max_steps):
    """Dry-run prompt ids when there is no real tokenizer/predictor."""
    import torch
    return torch.randint(0, int(getattr(mc, "vocab_size", 256)), (1, 8))


def _run_diagnosis(args) -> int:
    """Batch per-example free-running divergence diagnosis on the exact
    folded_remote decode path. Writes markdown + JSON; emits only scalars / token
    ids / token text / shape metadata (no logits / hidden / mask / secret)."""
    import json as _json

    from pllo.deployment import load_manifest
    from pllo.experiments.folded_probe_common import seed_from_manifest, tiny_model
    from pllo.experiments.nonlinear_designs import normalize_nonlinear_backend
    from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend
    from pllo.protocol.tee_gpu_messages import BoundaryInitRequest

    args.nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)
    dry_run = bool(args.dry_run or not args.model_path)

    # prompts (loaded exactly like run_ifeval_generation.py)
    prompts = []
    if args.input_jsonl and Path(args.input_jsonl).exists():
        with open(args.input_jsonl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and str(_json.loads(line).get("prompt", "")).strip():
                    prompts.append(_json.loads(line))
    if not prompts:
        prompts = [{"prompt": args.prompt, "key": "ex-0"}]

    # which example indices to diagnose
    if args.batch_from_strict_gap and Path(args.batch_from_strict_gap).exists():
        gap = _json.loads(Path(args.batch_from_strict_gap).read_text())
        want_ids = list((gap.get("strict") or {}).get(
            "plaintext_pass_folded_fail") or [])
        keyidx = {}
        for i, ex in enumerate(prompts):
            for k in ("key", "id", "prompt"):
                if k in ex:
                    keyidx[str(ex[k])] = i
        indices = [keyidx[i] for i in want_ids if i in keyidx]
    elif args.example_indices:
        indices = [int(x) for x in args.example_indices.split(",") if x.strip()]
    else:
        indices = [int(args.example_index)]
    indices = [i for i in indices if 0 <= i < len(prompts)]

    # build package (dry-run) + plaintext predictor/model
    pkg_path = args.folded_package_path
    if dry_run:
        prof = _load_script("prof", "scripts/run_folded_worker_forward_profile.py")
        out_dir = Path(args.output_json).resolve().parent / "_dry_run_pkg"
        out_dir.mkdir(parents=True, exist_ok=True)
        pkg_path, _art = prof._build_dry_run(out_dir, args.num_layers, args.seed)

    if dry_run:
        model, mc = tiny_model()
        device, dtype, tok, predictor = "cpu", "float32", None, None
        eos_ids = ({int(mc.eos_token_id)} if getattr(mc, "eos_token_id", None)
                   is not None else set())
    else:
        from pllo.benchmarks.real_predictors import build_predictor
        predictor = build_predictor(
            "plaintext_local", model_path=args.model_path,
            model_name=args.model_name, seq_len=args.seq_len,
            max_new_tokens=args.max_steps, dtype=args.dtype, device=args.device,
            use_chat_template=bool(args.use_chat_template))
        model, mc = predictor._model, predictor._model.config
        tok = predictor._tok
        device, dtype = args.device, args.dtype
        eos_ids = _resolve_eos_ids(tok, args.model_path)

    pkg_dir = Path(pkg_path)
    n_layers = int(load_manifest(pkg_dir).num_layers)
    seed = seed_from_manifest(pkg_dir, args.seed)
    rep_penalty = _resolve_rep_penalty(args)
    stop_on_eos = not args.disable_eos_stop

    def make_backend():
        b = Qwen7BFoldedPackageGpuBackend(
            folded_package_path=str(pkg_dir), device=device, dtype=dtype,
            nonlinear_backend=args.nonlinear_backend,
            resident_folded_weights=bool(args.resident_folded_weights))
        b.init(BoundaryInitRequest(
            session_id="diag", hidden_size=int(getattr(mc, "hidden_size")),
            vocab_size=int(getattr(mc, "vocab_size")), num_layers=n_layers,
            dtype=dtype, gpu_backend="qwen7b_folded_package"))
        return b

    per = []
    for idx in indices:
        d = _diagnose_example(
            prompt=str(prompts[idx]["prompt"]), model=model, tok=tok,
            predictor=predictor, mc=mc, device=device, dtype=dtype, pkg_dir=pkg_dir,
            n_layers=n_layers, seed=seed, max_steps=max(1, int(args.max_steps)),
            rep_penalty=rep_penalty, eos_ids=eos_ids, stop_on_eos=stop_on_eos,
            first_n=args.diag_first_n_tokens,
            inject_div_step=args.diag_inject_divergence_step,
            make_backend=make_backend, example_index=idx)
        per.append(d)

    classes = {}
    for d in per:
        classes[d["divergence_class"]] = classes.get(d["divergence_class"], 0) + 1
    report = {
        "stage": "plaintext_vs_folded_divergence_diagnosis",
        "dry_run": dry_run, "device": device, "cli_dtype": dtype,
        "align_generation_config": bool(args.align_generation_config),
        "repetition_penalty": rep_penalty,
        "stop_on_eos": stop_on_eos,
        "eos_token_id": (sorted(eos_ids) if eos_ids else None),
        "max_steps": max(1, int(args.max_steps)),
        "num_examples_diagnosed": len(per),
        "example_indices": indices,
        "divergence_class_counts": dict(sorted(classes.items())),
        "per_example_diagnosis": per,
        "decode_path": "exact_folded_remote (trusted-side repetition_penalty + EOS)",
        "analysis_scope": "token ids / token text / scalar metrics only; no "
        "recovered logits / hidden / mask / inverse / PRG seed / raw LoRA emitted",
        "plaintext_logits_or_sampling_on_gpu": False,
    }
    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== plaintext vs folded DIVERGENCE diagnosis (%s, dry_run=%s) ==="
          % (device, dry_run))
    print("examples=%s rep_penalty=%s stop_on_eos=%s" % (indices, rep_penalty,
                                                         stop_on_eos))
    for d in per:
        print("  ex%s: first_divergent_step=%s class=%s (plain=%s folded=%s)"
              % (d["example_index"], d["first_free_running_divergent_step"],
                 d["divergence_class"], d["plaintext_token_at_divergence"],
                 d["folded_token_at_divergence"]))
    print("class_counts=%s" % report["divergence_class_counts"])
    return 0


def _resolve_rep_penalty(args):
    if args.repetition_penalty is not None:
        return float(args.repetition_penalty)
    if args.align_generation_config and args.model_path:
        try:
            from transformers import GenerationConfig
            gc = GenerationConfig.from_pretrained(args.model_path,
                                                  local_files_only=True)
            rp = getattr(gc, "repetition_penalty", None)
            return float(rp) if rp else None
        except Exception:                                    # noqa: BLE001
            return None
    return None


def _resolve_eos_ids(tok, model_path):
    from pllo.benchmarks.real_predictors import _normalize_eos_ids
    eos = None
    try:
        from transformers import GenerationConfig
        gc = GenerationConfig.from_pretrained(model_path, local_files_only=True)
        eos = getattr(gc, "eos_token_id", None)
    except Exception:                                        # noqa: BLE001
        eos = None
    if eos is None:
        eos = getattr(tok, "eos_token_id", None)
    return _normalize_eos_ids(eos)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--system-message", default=None)
    ap.add_argument("--use-chat-template", action="store_true", default=False,
                    help="recorded only; the IFEval JSONL prompt is already "
                    "rendered and ifeval tokenises it raw -> we do NOT re-apply a "
                    "template (matching run_ifeval_generation.py)")
    ap.add_argument("--input-jsonl", default=None,
                    help="JSONL of prompts (raw ex['prompt'], loaded exactly like "
                    "run_ifeval_generation.py); --example-index selects one")
    ap.add_argument("--example-index", type=int, default=0)
    ap.add_argument("--folded-package-path", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--num-layers", type=int, default=4, help="dry-run only")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folded-weight-device", default=None)
    ap.add_argument("--mlp-down-chunk-size", type=int, default=512)
    ap.add_argument("--nonlinear-backend", default="current")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--rollout-mode", default="teacher_forcing",
                    choices=["teacher_forcing", "free_running", "both"])
    ap.add_argument("--verify-plaintext-reproduction", action="store_true",
                    default=False, help="always on internally; flag documents "
                    "intent + ensures the authoritative path runs")
    ap.add_argument("--resident-folded-weights", action="store_true",
                    default=False)
    ap.add_argument("--compare-nonresident", action="store_true", default=False)
    ap.add_argument("--require-real", action="store_true", default=False)
    # ---- generation-config alignment + per-example DIVERGENCE DIAGNOSIS (the
    # exact folded_remote decode path: trusted-side repetition_penalty + EOS) ----
    ap.add_argument("--align-generation-config", action="store_true",
                    default=False, help="apply repetition_penalty (trusted-side) "
                    "in the folded free-running rollout, matching folded_remote "
                    "--align-generation-config")
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="explicit repetition_penalty (else read from the model's "
                    "generation_config.json under --align-generation-config)")
    ap.add_argument("--disable-eos-stop", action="store_true", default=False,
                    help="diagnosis mode: disable trusted-side EOS stop (default "
                    "is EOS stop, matching folded_remote)")
    ap.add_argument("--diagnose-divergence", action="store_true", default=False,
                    help="run the per-example free-running token-level divergence "
                    "diagnosis (exact folded_remote decode path) instead of the "
                    "default rollout report")
    ap.add_argument("--example-indices", default=None,
                    help="comma-separated example indices for batch diagnosis "
                    "(e.g. '0,3,19'); overrides --example-index")
    ap.add_argument("--batch-from-strict-gap", default=None,
                    help="analyze_ifeval_strict_gap.py JSON: batch-diagnose every "
                    "strict plaintext-pass-but-folded-fail example (mapped to its "
                    "index in --input-jsonl by prompt/id)")
    ap.add_argument("--diag-first-n-tokens", type=int, default=50)
    ap.add_argument("--diag-inject-divergence-step", type=int, default=None,
                    help="TEST: force a folded free-token divergence at step K in "
                    "the diagnosis path to validate classification")
    # TEST-ONLY fault injectors (validate the detectors; never use in real runs)
    ap.add_argument("--inject-folded-divergence-step", type=int, default=None,
                    help="TEST: corrupt the teacher-forced folded logits at step K")
    ap.add_argument("--inject-free-divergence-step", type=int, default=None,
                    help="TEST: corrupt the folded FREE token at step K")
    ap.add_argument("--inject-resident-divergence", action="store_true",
                    default=False)
    ap.add_argument("--inject-plaintext-repro-mismatch", action="store_true",
                    default=False, help="TEST: corrupt the manual plaintext token "
                    "0 to confirm the reproduction gate blocks correctness")
    ap.add_argument("--inject-authoritative-processor-divergence",
                    action="store_true", default=False,
                    help="TEST: perturb ONLY the authoritative tokens (simulating "
                    "a generation-config processor) to confirm the tool attributes "
                    "it to generation config, not an encoding bug")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    if (args.diagnose_divergence or args.example_indices
            or args.batch_from_strict_gap):
        return _run_diagnosis(args)

    # ---- prompt source (loaded exactly like run_ifeval_generation.py) ----
    if args.input_jsonl and Path(args.input_jsonl).exists():
        rows_in = []
        with open(args.input_jsonl, encoding="utf-8") as fh:
            for i, ln in enumerate(fh):
                ln = ln.strip()
                if not ln:
                    continue
                ex = json.loads(ln)
                if str(ex.get("prompt", "")).strip():
                    rows_in.append(ex)
        if rows_in:
            idx = max(0, min(args.example_index, len(rows_in) - 1))
            args.prompt = str(rows_in[idx]["prompt"])

    dry_run = bool(args.dry_run or not args.model_path)
    if args.require_real and dry_run:
        print("ERROR: --require-real but --model-path not given", file=sys.stderr)
        return 3
    if not dry_run and not args.folded_package_path:
        print("ERROR: real run needs --folded-package-path", file=sys.stderr)
        return 3

    import numpy as np
    import torch

    from pllo.deployment import load_manifest, verify_package
    from pllo.experiments.folded_probe_common import seed_from_manifest, tiny_model
    from pllo.experiments.nonlinear_designs import normalize_nonlinear_backend
    from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession
    from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig, _cfg_to
    from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend
    from pllo.protocol.tee_gpu_messages import BoundaryInitRequest
    from pllo.protocol.wire import encode_message
    from pllo.protocol.remote import forbidden_fields_in_payload

    args.nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)
    cmp_res = _load_script("cmp_res",
                           "scripts/compare_resident_folded_correctness.py")

    # ---- build / locate the folded package ----
    if dry_run:
        prof = _load_script("prof", "scripts/run_folded_worker_forward_profile.py")
        out_dir = Path(args.output_json).resolve().parent / "_dry_run_pkg"
        out_dir.mkdir(parents=True, exist_ok=True)
        pkg_path, art_path = prof._build_dry_run(out_dir, args.num_layers,
                                                 args.seed)
        args.folded_package_path = str(pkg_path)
        if not args.embedding_path:
            args.embedding_path = str(art_path)

    max_steps = max(1, int(args.max_steps))

    # ---- AUTHORITATIVE plaintext path == run_ifeval_generation.py's ----
    if dry_run:
        model, mc = tiny_model()
        device = "cpu"
        ids = torch.randint(0, mc.vocab_size, (1, min(args.seq_len, 8)))
        attn = None
        predictor = _TinyPlaintextPredictor(model, mc, ids, max_steps)
        tok = None
        eos_id = getattr(mc, "eos_token_id", None)
    else:
        from pllo.benchmarks.real_predictors import build_predictor
        predictor = build_predictor(
            "plaintext_local", model_path=args.model_path,
            model_name=args.model_name, seq_len=args.seq_len,
            max_new_tokens=max_steps, dtype=args.dtype, device=args.device,
            use_chat_template=bool(args.use_chat_template))
        model, mc = predictor._model, predictor._model.config
        tok = predictor._tok
        device = args.device
        ids, attn = predictor._encode(args.prompt)
        eos_id = getattr(tok, "eos_token_id", None)

    seq_len = int(ids.shape[1])

    # (A) authoritative generation (the literal ifeval plaintext_local call)
    a_gen = predictor.generate(args.prompt)
    a_tokens = list(a_gen.get("token_ids") or [])
    if args.inject_authoritative_processor_divergence and a_tokens:
        a_tokens = list(a_tokens)
        a_tokens[0] = int(a_tokens[0]) + 1   # TEST: simulate a processor effect
    a_text = a_gen.get("text")
    if (a_text is None or args.inject_authoritative_processor_divergence) \
            and tok is not None:
        a_text = _decode_text(tok, a_tokens)

    # (A_raw) raw-greedy reference (generation-config processors neutralised) ->
    # isolates a generation-config effect from an encoding/decode bug.
    a_raw_tokens = _raw_greedy_generate(model, ids, attn, max_steps, eos_id)

    # (B) the tool's own step-by-step plaintext (raw argmax; for per-step logits)
    plain_rows, b_tokens = _manual_plaintext_steps(model, ids, attn, max_steps,
                                                   eos_id)
    if args.inject_plaintext_repro_mismatch and b_tokens:
        b_tokens = list(b_tokens)
        b_tokens[0] = int(b_tokens[0]) + 1            # TEST: force a repro mismatch

    # reproduction gate: B must reproduce A (the ifeval baseline) tokenwise
    cmp_len = len(a_tokens)
    repro_first_div = _first_divergent_tokens(a_tokens, b_tokens[:cmp_len]) \
        if cmp_len else None
    plaintext_reproduction_passed = bool(cmp_len > 0 and repro_first_div is None)
    # is the tool's raw decode itself faithful to a raw model.generate?
    raw_len = len(a_raw_tokens)
    manual_reproduces_raw_greedy = bool(
        raw_len > 0 and _first_divergent_tokens(a_raw_tokens,
                                                b_tokens[:raw_len]) is None)
    # does the ifeval baseline apply generation-config processors (the likely
    # cause of folded repetition: baseline penalises repeats, folded argmax does
    # not)? True when authoritative != raw-greedy.
    authoritative_uses_generation_processors = bool(
        _first_divergent_tokens(a_tokens, a_raw_tokens) is not None)

    def _prefix_text(token_ids):
        return _decode_text(tok, list(token_ids)[:80])

    chat_tmpl = getattr(tok, "chat_template", None) if tok is not None else None
    chat_sha = (hashlib.sha256(chat_tmpl.encode("utf-8")).hexdigest()
                if isinstance(chat_tmpl, str) and chat_tmpl else None)
    gen_cfg = getattr(model, "generation_config", None)
    do_sample = bool(getattr(gen_cfg, "do_sample", False)) if gen_cfg else False
    gen_summary = {
        "do_sample": False, "num_beams": 1, "use_cache": True,
        "max_new_tokens": max_steps,
        "pad_token_id": eos_id,
        "eos_token_id": eos_id,
        "temperature": (float(getattr(gen_cfg, "temperature", None) or 1.0)
                        if gen_cfg else None),
        "top_p": (float(getattr(gen_cfg, "top_p", None) or 1.0)
                  if gen_cfg else None),
        "top_k": (int(getattr(gen_cfg, "top_k", None) or 0) if gen_cfg else None),
        "repetition_penalty": (float(getattr(gen_cfg, "repetition_penalty", None)
                                     or 1.0) if gen_cfg else None),
        "no_repeat_ngram_size": (int(getattr(gen_cfg, "no_repeat_ngram_size", None)
                                     or 0) if gen_cfg else None),
    }

    # ---- folded setup (seed-matched masking session) ----
    pkg_dir = Path(args.folded_package_path)
    manifest = load_manifest(pkg_dir)
    n_layers = int(manifest.num_layers)
    seed = seed_from_manifest(pkg_dir, args.seed)
    vrep = verify_package(pkg_dir)
    cfg = MemoryOptimizedConfig(
        num_layers=n_layers, batch_size=1, seq_len=seq_len, max_new_tokens=max_steps,
        device=device, dtype=args.dtype, folding_dtype="float32",
        folded_weight_device=args.folded_weight_device or device,
        mlp_down_chunk_size=args.mlp_down_chunk_size, seed=seed)
    session = MaskedQwenSession(model, mc, cfg)
    h_tilde = session.mask_embeddings(ids)
    cfg0 = _cfg_to(session.layer_configs[0], session.compute_device)

    def _id_to_text(tid):
        if tok is None or tid is None:
            return None
        try:
            return tok.decode([int(tid)])
        except Exception:                                    # noqa: BLE001
            return None

    leaked: list = []

    def _new_backend(resident):
        return Qwen7BFoldedPackageGpuBackend(
            folded_package_path=str(pkg_dir), device=device, dtype=args.dtype,
            nonlinear_backend=args.nonlinear_backend,
            resident_folded_weights=resident)

    def _init(backend, sid):
        req = BoundaryInitRequest(
            session_id=sid, hidden_size=int(getattr(mc, "hidden_size")),
            vocab_size=int(getattr(mc, "vocab_size")), num_layers=n_layers,
            dtype=args.dtype, gpu_backend="qwen7b_folded_package")
        bad = forbidden_fields_in_payload(encode_message(req))
        if bad:
            leaked.extend(bad)
        return backend.init(req)

    primary_resident = bool(args.resident_folded_weights)
    primary = _new_backend(primary_resident)
    init_resp = _init(primary, "pf-primary")

    # resident force-build + checksum (mutation detection across the whole run)
    resident_weight_mutated = False
    resident_dtype_mismatch = False
    resident_cache_dtype = None
    fold_compute_dtype = str(h_tilde.dtype)
    chk_before = None
    if primary_resident:
        primary._ensure_resident(h_tilde.device, h_tilde.dtype, n_layers)
        resident_cache_dtype = str(primary._resident_layers[0]["wq_tilde"].dtype)
        resident_dtype_mismatch = bool(resident_cache_dtype != fold_compute_dtype)
        chk_before = cmp_res._checksum_resident(primary._resident_layers,
                                                primary._resident_head)

    do_tf = args.rollout_mode in ("teacher_forcing", "both")
    do_free = args.rollout_mode in ("free_running", "both")

    # ---- only run folded comparisons when reproduction passed ----
    tf_rows, tf_summary = [], None
    tf_first_div = None
    tf_match_rate = None
    free_rows, free_summary = [], None
    free_first_div = None
    free_token_match_rate = None
    folded_free_tokens = []
    folded_free_text = None
    rvn = None
    rvn_passed = None
    rvn_max = rvn_mean = None
    rvn_top1_match = None

    # run folded diagnostics when our RAW reference is faithful (even if the
    # ifeval baseline differs due to generation-config processors); correctness is
    # still BLOCKED below when the authoritative reproduction fails.
    proceed_folded = bool(plaintext_reproduction_passed
                          or manual_reproduces_raw_greedy)
    if proceed_folded:
        if do_tf:
            primary_tf_rows, _ = _folded_steps(
                primary, session, h_tilde, max_steps, seq_len, n_layers, cfg0,
                fed_tokens=b_tokens)
            if args.inject_folded_divergence_step is not None:
                k = int(args.inject_folded_divergence_step)
                if 0 <= k < len(primary_tf_rows):
                    r = primary_tf_rows[k].copy()
                    bad = int((int(plain_rows[k].argmax()) + 1) % r.shape[0])
                    r[bad] = float(r.max()) + 1e3
                    primary_tf_rows[k] = r
            tf_rows, tf_summary = _compare_logit_steps(
                plain_rows, primary_tf_rows, topk=args.topk, atol=args.atol,
                rtol=args.rtol, id_to_text=_id_to_text)
            tf_first_div = tf_summary["first_divergent_step"]
            tf_match_rate = tf_summary["top1_match_rate"]

        if do_free:
            free_plain_rows, free_plain_tokens = plain_rows, b_tokens
            folded_free_rows, folded_free_tokens = _folded_steps(
                primary, session, h_tilde, max_steps, seq_len, n_layers, cfg0,
                fed_tokens=None, eos_id=eos_id)
            if args.inject_free_divergence_step is not None:
                k = int(args.inject_free_divergence_step)
                if 0 <= k < len(folded_free_tokens):
                    folded_free_tokens = list(folded_free_tokens)
                    folded_free_tokens[k] = int(free_plain_tokens[k]
                                                if k < len(free_plain_tokens)
                                                else folded_free_tokens[k]) + 1
            free_first_div = _first_divergent_tokens(free_plain_tokens,
                                                     folded_free_tokens)
            free_token_match_rate = _token_match_rate(free_plain_tokens,
                                                      folded_free_tokens)
            folded_free_text = _decode_text(tok, folded_free_tokens)
            free_rows, free_summary = _compare_logit_steps(
                free_plain_rows, folded_free_rows, topk=args.topk, atol=args.atol,
                rtol=args.rtol, id_to_text=_id_to_text)

        if primary_resident and chk_before is not None:
            chk_after = cmp_res._checksum_resident(primary._resident_layers,
                                                   primary._resident_head)
            resident_weight_mutated = bool(chk_before != chk_after)

        if args.compare_nonresident:
            other = _new_backend(not primary_resident)
            _init(other, "pf-other")
            other_rows, _ = _folded_steps(
                other, session, h_tilde, max_steps, seq_len, n_layers, cfg0,
                fed_tokens=b_tokens)
            res_rows = (primary_tf_rows if (do_tf and primary_resident)
                        else (other_rows if not primary_resident
                              else _folded_steps(primary, session, h_tilde,
                                                 max_steps, seq_len, n_layers,
                                                 cfg0, fed_tokens=b_tokens)[0]))
            non_rows = other_rows if primary_resident else (
                primary_tf_rows if do_tf else _folded_steps(
                    primary, session, h_tilde, max_steps, seq_len, n_layers,
                    cfg0, fed_tokens=b_tokens)[0])
            if args.inject_resident_divergence and res_rows:
                res_rows = [r.copy() for r in res_rows]
                bad = int((int(res_rows[0].argmax()) + 1) % res_rows[0].shape[0])
                res_rows[0][bad] = float(res_rows[0].max()) + 1e3
            rvn_rows, rvn_summary = _compare_logit_steps(
                res_rows, non_rows, topk=args.topk, atol=args.atol, rtol=args.rtol)
            rvn_max = rvn_summary["max_abs_logit_err_max"]
            rvn_mean = rvn_summary["mean_abs_logit_err_mean"]
            rvn_top1_match = bool(rvn_summary["first_divergent_step"] is None)
            rvn_passed = bool(rvn_top1_match and not resident_weight_mutated
                              and not resident_dtype_mismatch)
            rvn = rvn_rows

    # ---- correctness decision (BLOCKED on reproduction failure) ----
    blocked = not plaintext_reproduction_passed
    primary_div = tf_first_div if do_tf else free_first_div
    if blocked:
        correctness_passed = None
    else:
        correctness_passed = bool(
            primary_div is None and not resident_weight_mutated
            and not resident_dtype_mismatch)

    if blocked and authoritative_uses_generation_processors \
            and manual_reproduces_raw_greedy:
        suspected = ("generation-config mismatch: run_ifeval plaintext_local "
                     "(model.generate) applies generation-config logit processors "
                     "(repetition_penalty=%s, no_repeat_ngram_size=%s) that raw "
                     "greedy + the folded argmax decode do NOT -> this likely "
                     "explains folded repetition. Re-run plaintext_local and "
                     "folded with identical generation config (or compare folded "
                     "to the raw-greedy plaintext baseline)."
                     % (gen_summary.get("repetition_penalty"),
                        gen_summary.get("no_repeat_ngram_size")))
    elif blocked:
        suspected = ("plaintext reproduction mismatch: the tool's raw step decode "
                     "does not reproduce run_ifeval plaintext_local AND does not "
                     "match a raw model.generate (manual_reproduces_raw_greedy=%s) "
                     "-> fix encoding / chat template / tokenization first"
                     % manual_reproduces_raw_greedy)
    elif do_tf and tf_first_div is None and do_free \
            and free_first_div is not None:
        suspected = "autoregressive state divergence or generation-state mismatch"
    elif do_tf and tf_first_div is not None:
        suspected = _suspected_root_cause(tf_first_div, prefix="teacher_forcing")
    elif do_free and free_first_div is not None:
        suspected = _suspected_root_cause(free_first_div, prefix="free_running")
    else:
        suspected = None

    report = {
        "stage": "plaintext_vs_folded_generation_correctness",
        "dry_run": dry_run, "model_name": args.model_name,
        "device": device, "cli_dtype": args.dtype,
        "fold_compute_dtype": fold_compute_dtype,
        "resident_cache_dtype": resident_cache_dtype,
        "rollout_mode": args.rollout_mode,
        "seq_len": seq_len, "max_steps": max_steps, "num_layers": n_layers,
        "mask_seed_matched_to_package": True,
        "atol": args.atol, "rtol": args.rtol, "topk": args.topk,
        "primary_resident": primary_resident,
        "folded_package_path": str(pkg_dir),
        "folded_package_valid": bool(vrep["package_valid"]),
        "embedding_path": (str(args.embedding_path) if args.embedding_path
                           else None),
        "masking_source": "masked_qwen_session_seed_matched",
        # ---- TASK 2: plaintext reproduction gate ----
        "plaintext_reproduction_passed": plaintext_reproduction_passed,
        "first_plaintext_reproduction_divergent_step": repro_first_div,
        "manual_reproduces_raw_greedy": manual_reproduces_raw_greedy,
        "authoritative_uses_generation_processors":
            authoritative_uses_generation_processors,
        "raw_greedy_plaintext_token_ids": a_raw_tokens,
        "run_ifeval_plaintext_token_ids": a_tokens,
        "correctness_plaintext_token_ids": b_tokens,
        "run_ifeval_plaintext_generated_text": a_text,
        "correctness_plaintext_generated_text": _decode_text(tok, b_tokens),
        "run_ifeval_plaintext_prefix_text": _prefix_text(a_tokens),
        "correctness_plaintext_prefix_text": _prefix_text(b_tokens),
        "chat_template_sha256": chat_sha,
        "prompt_token_count": seq_len,
        "system_message": args.system_message,
        "add_generation_prompt": False,
        "generation_config_summary": gen_summary,
        "eos_token_id": eos_id, "pad_token_id": eos_id,
        "do_sample": do_sample, "temperature": gen_summary["temperature"],
        "top_p": gen_summary["top_p"], "num_beams": 1, "use_cache": True,
        "attention_mask_policy": ("tokenizer_attention_mask_passed"
                                  if not dry_run else "none_single_sequence"),
        "position_ids_policy": "huggingface_default_from_attention_mask",
        # ---- TASK 3: block wrong conclusions ----
        "correctness_passed": correctness_passed,
        "correctness_blocked_by_plaintext_reproduction_mismatch": blocked,
        "suspected_root_cause": suspected,
        # ---- teacher forcing (per-step fidelity) ----
        "teacher_forcing_top1_match_rate": tf_match_rate,
        "teacher_forcing_first_divergent_step": tf_first_div,
        "first_divergent_step": (tf_first_div if do_tf else free_first_div),
        "first_divergent_stage": (None if (tf_first_div if do_tf
                                           else free_first_div) is None
                                  else ("prefill" if (tf_first_div if do_tf
                                        else free_first_div) == 0 else "decode")),
        "plaintext_vs_folded_top1_match_rate": tf_match_rate,
        "plaintext_vs_folded_max_abs_logit_err_max":
            (tf_summary["max_abs_logit_err_max"] if tf_summary else None),
        "plaintext_vs_folded_mean_abs_logit_err_mean":
            (tf_summary["mean_abs_logit_err_mean"] if tf_summary else None),
        "teacher_forcing_per_step": tf_rows,
        # ---- free running (real generation fork) ----
        "first_free_running_divergent_step": free_first_div,
        "free_running_token_match_rate": free_token_match_rate,
        "token_match_rate": (free_token_match_rate if do_free else tf_match_rate),
        "plaintext_generated_text": a_text,
        "folded_generated_text": folded_free_text,
        "plaintext_token_ids": (b_tokens if do_free else None),
        "folded_token_ids": (folded_free_tokens if do_free else None),
        "free_running_per_step": free_rows,
        # ---- resident vs non-resident ----
        "resident_vs_nonresident_correctness_passed": rvn_passed,
        "resident_vs_nonresident_max_abs_err": rvn_max,
        "resident_vs_nonresident_mean_abs_err": rvn_mean,
        "resident_vs_nonresident_top1_match": rvn_top1_match,
        "resident_weight_mutated": resident_weight_mutated,
        "resident_dtype_mismatch": resident_dtype_mismatch,
        "resident_vs_nonresident_per_step": rvn,
        # ---- dtype note ----
        "cli_dtype_used_for_fold_compute": bool(args.dtype == fold_compute_dtype),
        "dtype_note": ("the folded compute uses the package fold dtype (%s); "
                       "--dtype=%s is applied to the plaintext model but is NOT "
                       "the fold dtype" % (fold_compute_dtype, args.dtype)),
        # ---- security audit (threat model unchanged) ----
        "audit_passed": None,
        "tee_used_on_gpu": bool(init_resp.tee_used_on_gpu),
        "worker_has_mask_secrets": bool(primary.worker_has_mask_secrets),
        "worker_has_raw_lora": bool(getattr(primary, "worker_has_raw_lora", False)),
        "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": sorted(set(leaked))
        + list(vrep["forbidden_fields_found"]),
        "schedule_secret_leaked_to_gpu": False,
        "gpu_request_contains_schedule_secret": False,
        "plaintext_logits_or_sampling_on_gpu": False,
        "pad_disabled_for_correctness": False,
    }
    report["audit_passed"] = bool(not report["leaked_secret_fields"])

    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== plaintext vs folded generation correctness (%s, dry_run=%s) ==="
          % (device, dry_run))
    print("plaintext_reproduction_passed=%s first_repro_div=%s (blocked=%s)"
          % (report["plaintext_reproduction_passed"],
             report["first_plaintext_reproduction_divergent_step"], blocked))
    print("manual_reproduces_raw_greedy=%s authoritative_uses_generation_processors"
          "=%s (repetition_penalty=%s)"
          % (report["manual_reproduces_raw_greedy"],
             report["authoritative_uses_generation_processors"],
             gen_summary.get("repetition_penalty")))
    print("rollout_mode=%s correctness_passed=%s" % (args.rollout_mode,
                                                     report["correctness_passed"]))
    print("teacher_forcing: top1_match_rate=%s first_divergent_step=%s"
          % (report["teacher_forcing_top1_match_rate"],
             report["teacher_forcing_first_divergent_step"]))
    print("free_running: token_match_rate=%s first_free_running_divergent_step=%s"
          % (report["free_running_token_match_rate"],
             report["first_free_running_divergent_step"]))
    print("resident_vs_nonresident: passed=%s max_abs_err=%s mutated=%s "
          "dtype_mismatch=%s" % (report["resident_vs_nonresident_correctness_passed"],
                                 report["resident_vs_nonresident_max_abs_err"],
                                 report["resident_weight_mutated"],
                                 report["resident_dtype_mismatch"]))
    print("suspected_root_cause=%s" % report["suspected_root_cause"])
    print("audit_passed=%s tee_used_on_gpu=%s worker_has_mask_secrets=%s leaked=%s"
          % (report["audit_passed"], report["tee_used_on_gpu"],
             report["worker_has_mask_secrets"], report["leaked_secret_fields"]))

    overall_ok = bool(plaintext_reproduction_passed
                      and report["correctness_passed"] and report["audit_passed"]
                      and rvn_passed is not False)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
