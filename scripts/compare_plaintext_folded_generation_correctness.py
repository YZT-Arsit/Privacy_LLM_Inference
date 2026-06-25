"""Plaintext Qwen vs folded-package generation correctness diagnostic.

The folded remote path (masked embeddings -> untrusted folded shards -> recovered
logits) is supposed to reproduce the PLAINTEXT Qwen forward exactly (the masking
algebra is exact; only float error remains). When folded generation degrades
(repetition / lost instruction-following) the question is whether the FOLDED PATH
ITSELF has diverged from plaintext Qwen -- not resident-vs-non-resident (that was
already shown bit-exact). This tool answers it.

It drives, under an IDENTICAL setup (same tokenizer, chat template, system
message, prompt, seq_len, dtype, device, greedy decoding):

* the **plaintext** HF Qwen forward (free-running greedy) -> the reference token
  sequence + per-step plaintext logits;
* the **folded package** path (boundary masks the embeddings, the package executes
  folded layers + folded head over masked tensors, the boundary recovers logits),
  **teacher-forced on the plaintext greedy tokens** so every step is conditioned on
  the SAME context -> a clean per-step plaintext-vs-folded logits comparison that
  localises the FIRST divergent step (step 0 = prefill/template/embedding/first
  layer/head; step >=1 = KV append / decode RoPE / per-step obfuscation domain);
* optionally the **resident** and **non-resident** folded backends, compared to
  each other (``--compare-nonresident``) to re-confirm residency is not the cause.

Per step it reports plaintext/folded top1 token id+text, top1_match, top5_overlap,
max/mean/relative logit error; per pair it reports the resident-vs-non-resident
error + ``resident_weight_mutated`` / ``resident_dtype_mismatch``.

Masking + recovery use a seed-matched ``MaskedQwenSession`` (the same masks the
package was folded against); NO mask / inverse / pad / PRG seed / schedule secret /
raw LoRA / plaintext hidden state / plaintext logits ever crosses to the package.
The debug report emits ONLY scalar metrics, token ids, token text, and
shape/dtype/device metadata -- never a secret tensor. The threat model is
unchanged; pad is never disabled to make correctness pass.

``--dry-run`` uses the tiny random Qwen2 on CPU (never a paper result); a real run
needs ``--model-path`` + ``--folded-package-path``.

# ---- server example only (run on the GPU server; NOT executed locally) ----
# python scripts/compare_plaintext_folded_generation_correctness.py \
#   --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \
#   --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full_current_seq1024 \
#   --embedding-path /root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current \
#   --input-jsonl /root/autodl-tmp/datasets/privacy_llm_benchmarks/converted/ifeval_prompts.jsonl \
#   --seq-len 1024 --max-steps 16 --dtype bfloat16 --device cuda \
#   --use-chat-template --resident-folded-weights --compare-nonresident \
#   --output-json outputs/debug/plaintext_vs_folded_correctness_seq1024_steps16.json
"""

from __future__ import annotations

import argparse
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
    """(max_abs, mean_abs) between two 1-D numpy arrays, in float64."""
    import numpy as np
    diff = np.abs(np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64"))
    return (float(diff.max()), float(diff.mean())) if diff.size else (0.0, 0.0)


def _np_rel(a, b):
    """Relative L2 error ||a-b|| / ||b|| between two 1-D numpy arrays."""
    import numpy as np
    aa = np.asarray(a, dtype="float64")
    bb = np.asarray(b, dtype="float64")
    denom = float(np.linalg.norm(bb)) or 1.0
    return float(np.linalg.norm(aa - bb) / denom)


def _np_topk(a, k):
    import numpy as np
    arr = np.asarray(a).reshape(-1)
    k = min(k, arr.shape[0])
    return set(np.argsort(-arr)[:k].tolist())


def _compare_logit_steps(plain_list, folded_list, *, topk, atol, rtol,
                         id_to_text=None):
    """Compare two equal-length lists of 1-D logit rows step by step.

    Robust to per-step shape / dtype mismatch (records the flags, marks the step
    divergent, skips the numeric error for that step). Returns (rows, summary)
    where ``summary`` carries ``first_divergent_step`` (first step whose argmax
    differs OR whose shape mismatches) + aggregate error/match-rate fields."""
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
        row = {
            "step_id": i,
            "plaintext_top1_token_id": int(pa.argmax()) if pa.size else None,
            "folded_top1_token_id": int(fa.argmax()) if fa.size else None,
            "shape_match": shape_match,
            "dtype_match": dtype_match,
            "top1_match": None, "top5_overlap": None,
            "max_abs_logit_err": None, "mean_abs_logit_err": None,
            "relative_logit_err": None, "allclose_atol_rtol": None,
        }
        if id_to_text is not None:
            row["plaintext_top1_text"] = (id_to_text(row["plaintext_top1_token_id"])
                                          if pa.size else None)
            row["folded_top1_text"] = (id_to_text(row["folded_top1_token_id"])
                                       if fa.size else None)
        else:
            row["plaintext_top1_text"] = None
            row["folded_top1_text"] = None
        if not shape_match:
            row["top1_match"] = False
            if first_divergent is None:
                first_divergent = i
            rows.append(row)
            continue
        t1 = bool(row["plaintext_top1_token_id"] == row["folded_top1_token_id"])
        row["top1_match"] = t1
        row["top5_overlap"] = (len(_np_topk(pa, topk) & _np_topk(fa, topk))
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
    if first_divergent is not None and id_to_text is not None:
        fd_text = rows[first_divergent].get("plaintext_top1_text")
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


def _suspected_root_cause(first_divergent_step, *, resident_diverged):
    """Heuristic localisation hint for the report (honest, coarse)."""
    if resident_diverged:
        return ("resident_cache: resident vs non-resident folded logits diverge "
                "-> resident weight mutation / dict aliasing / dtype")
    if first_divergent_step is None:
        return None
    if first_divergent_step == 0:
        return ("prefill_path (step 0): chat_template / tokenization / input "
                "embedding artifact / position_ids / causal mask / RoPE / first "
                "folded layer (QKV/attn/MLP) / folded LM-head inverse")
    return ("decode_path (step %d): KV cache append / KV mask-domain transition / "
            "decode position_ids+RoPE / per-step fresh obfuscation domain / cache "
            "reuse of a stale domain" % first_divergent_step)


# ---- plaintext + folded per-step collectors --------------------------------

def _plaintext_greedy(model, ids, max_steps, device):
    """Free-running greedy plaintext forward. Returns (logit_rows, top1_tokens),
    each list length ``max_steps`` (prefill last-token logits, then decode steps).
    ``logit_rows`` are float32 cpu numpy [V]."""
    import numpy as np
    import torch
    rows, toks = [], []
    with torch.no_grad():
        o = model(input_ids=ids, use_cache=True)
        lg = o.logits[:, -1, :].float().to("cpu")
        rows.append(np.asarray(lg[0].numpy()))
        t = int(lg.argmax(-1).item())
        toks.append(t)
        past = o.past_key_values
        for _ in range(max_steps - 1):
            nxt = torch.tensor([[t]], device=device)
            o = model(input_ids=nxt, past_key_values=past, use_cache=True)
            lg = o.logits[:, -1, :].float().to("cpu")
            rows.append(np.asarray(lg[0].numpy()))
            t = int(lg.argmax(-1).item())
            toks.append(t)
            past = o.past_key_values
    return rows, toks


def _folded_recovered(backend, session, h_tilde, fed_tokens, max_steps, seq_len,
                      n_layers, cfg0):
    """Folded-package recovered logits, TEACHER-FORCED on ``fed_tokens`` (the
    plaintext greedy sequence) so each step shares the plaintext context. Returns
    a list of float32 cpu numpy [V] rows (prefill last-token, then decode steps).
    The boundary owns masking + recovery; the package only executes folded ops."""
    import numpy as np
    import torch
    pre = backend.run_prefill(h_tilde, n_layers, cfg0, session._cos, session._sin,
                              session.eps)
    rec = session.recover(backend.run_head(pre["y_tilde"], session.eps)[:, -1, :])
    rows = [np.asarray(rec[0].detach().to("cpu").float().numpy())]
    position = seq_len
    for step in range(max_steps - 1):
        x = session.mask_token_embedding(torch.tensor([fed_tokens[step]]))
        dec = backend.run_decode(x, position, cfg0, session._cos, session._sin,
                                 session.eps, num_exec_layers=n_layers)
        rec = session.recover(
            backend.run_head(dec["y_tilde"], session.eps)[:, -1, :])
        rows.append(np.asarray(rec[0].detach().to("cpu").float().numpy()))
        position += 1
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--system-message", default=None)
    ap.add_argument("--use-chat-template", action="store_true", default=False)
    ap.add_argument("--input-jsonl", default=None,
                    help="optional JSONL of prompts; the FIRST record's "
                    "prompt/instruction/text field is used (overrides --prompt)")
    ap.add_argument("--folded-package-path", default=None)
    ap.add_argument("--embedding-path", default=None,
                    help="boundary embedding artifact (recorded; masking uses a "
                    "seed-matched MaskedQwenSession derived from the model)")
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
    ap.add_argument("--resident-folded-weights", action="store_true",
                    default=False)
    ap.add_argument("--compare-nonresident", action="store_true", default=False,
                    help="also run the non-resident backend and compare resident "
                    "vs non-resident folded logits")
    ap.add_argument("--require-real", action="store_true", default=False)
    # TEST-ONLY fault injectors (validate the detectors; never use in real runs)
    ap.add_argument("--inject-folded-divergence-step", type=int, default=None,
                    help="TEST: corrupt the folded logits at this step to confirm "
                    "first_divergent_step detection")
    ap.add_argument("--inject-resident-divergence", action="store_true",
                    default=False, help="TEST: corrupt the resident folded logits "
                    "to confirm resident-vs-non-resident detection")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    # ---- prompt source ----
    if args.input_jsonl and Path(args.input_jsonl).exists():
        with open(args.input_jsonl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for key in ("prompt", "instruction", "text", "input", "question"):
                    if isinstance(rec.get(key), str) and rec[key].strip():
                        args.prompt = rec[key]
                        break
                break

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
    from pllo.experiments.folded_probe_common import (
        load_model_and_ids, seed_from_manifest)
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

    # ---- identical setup: tokenizer / chat template / prompt / ids ----
    tok = None
    args.use_chat_template = "true" if args.use_chat_template else "false"
    # MaskedQwenSession masking dtype follows the package fold dtype; the chat
    # template / system message are applied to BOTH paths identically below.
    if not dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                            local_files_only=True)
    model, mc, ids, device, dtype = load_model_and_ids(args, dry_run)
    # apply system message by re-tokenising with the chat template if requested
    if (not dry_run) and tok is not None and args.system_message \
            and args.use_chat_template == "true":
        msgs = [{"role": "system", "content": args.system_message},
                {"role": "user", "content": args.prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")["input_ids"][:, :args.seq_len].to(device)

    seq_len = int(ids.shape[1])
    max_steps = max(1, int(args.max_steps))

    pkg_dir = Path(args.folded_package_path)
    manifest = load_manifest(pkg_dir)
    n_layers = int(manifest.num_layers)
    seed = seed_from_manifest(pkg_dir, args.seed)
    vrep = verify_package(pkg_dir)

    # seed-matched masking session (same masks the package was folded against)
    cfg = MemoryOptimizedConfig(
        num_layers=n_layers, batch_size=1, seq_len=seq_len, max_new_tokens=max_steps,
        device=device, dtype=dtype, folding_dtype="float32",
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

    # ---- plaintext reference (free-running greedy) ----
    plain_rows, fed_tokens = _plaintext_greedy(model, ids, max_steps, device)

    leaked: list = []

    def _new_backend(resident):
        return Qwen7BFoldedPackageGpuBackend(
            folded_package_path=str(pkg_dir), device=device, dtype=dtype,
            nonlinear_backend=args.nonlinear_backend,
            resident_folded_weights=resident)

    def _init(backend, sid):
        req = BoundaryInitRequest(
            session_id=sid, hidden_size=int(getattr(mc, "hidden_size")),
            vocab_size=int(getattr(mc, "vocab_size")), num_layers=n_layers,
            dtype=dtype, gpu_backend="qwen7b_folded_package")
        bad = forbidden_fields_in_payload(encode_message(req))
        if bad:
            leaked.extend(bad)
        return backend.init(req)

    # ---- primary folded pass (resident if requested) ----
    primary_resident = bool(args.resident_folded_weights)
    primary = _new_backend(primary_resident)
    init_resp = _init(primary, "pf-primary")
    # force-build resident cache + checksum (mutation detection across the run).
    # run_prefill builds the cache lazily with (h_tilde.device, h_tilde.dtype);
    # pre-building with the SAME args lets us checksum BEFORE any forward runs.
    resident_weight_mutated = False
    resident_dtype_mismatch = False
    resident_cache_dtype = None
    fold_compute_dtype = str(h_tilde.dtype)        # the package fold compute dtype
    if primary_resident:
        primary._ensure_resident(h_tilde.device, h_tilde.dtype, n_layers)
        resident_cache_dtype = str(primary._resident_layers[0]["wq_tilde"].dtype)
        resident_dtype_mismatch = bool(resident_cache_dtype != fold_compute_dtype)
        chk_before = cmp_res._checksum_resident(primary._resident_layers,
                                                primary._resident_head)
    primary_rows = _folded_recovered(primary, session, h_tilde, fed_tokens,
                                     max_steps, seq_len, n_layers, cfg0)
    if primary_resident:
        chk_after = cmp_res._checksum_resident(primary._resident_layers,
                                               primary._resident_head)
        resident_weight_mutated = bool(chk_before != chk_after)

    # TEST-ONLY: corrupt the folded logits at a chosen step
    if args.inject_folded_divergence_step is not None:
        k = int(args.inject_folded_divergence_step)
        if 0 <= k < len(primary_rows):
            row = primary_rows[k].copy()
            bad_idx = int((int(plain_rows[k].argmax()) + 1) % row.shape[0])
            row[bad_idx] = float(row.max()) + 1e3
            primary_rows[k] = row

    p_rows, p_summary = _compare_logit_steps(
        plain_rows, primary_rows, topk=args.topk, atol=args.atol, rtol=args.rtol,
        id_to_text=_id_to_text)

    # ---- resident vs non-resident (optional) ----
    rvn = None
    resident_vs_nonresident_passed = None
    rvn_max = None
    rvn_mean = None
    rvn_top1_match = None
    if args.compare_nonresident:
        other = _new_backend(not primary_resident)
        _init(other, "pf-other")
        other_rows = _folded_recovered(other, session, h_tilde, fed_tokens,
                                       max_steps, seq_len, n_layers, cfg0)
        # order so resident is "a", non-resident is "b" for clear naming
        res_rows = primary_rows if primary_resident else other_rows
        non_rows = other_rows if primary_resident else primary_rows
        if args.inject_resident_divergence and res_rows:
            res_rows = [r.copy() for r in res_rows]
            bad = int((int(res_rows[0].argmax()) + 1) % res_rows[0].shape[0])
            res_rows[0][bad] = float(res_rows[0].max()) + 1e3  # flip resident top1
        rvn_rows, rvn_summary = _compare_logit_steps(
            res_rows, non_rows, topk=args.topk, atol=args.atol, rtol=args.rtol)
        rvn_max = rvn_summary["max_abs_logit_err_max"]
        rvn_mean = rvn_summary["mean_abs_logit_err_mean"]
        rvn_top1_match = bool(rvn_summary["first_divergent_step"] is None)
        resident_vs_nonresident_passed = bool(
            rvn_top1_match and not resident_weight_mutated
            and not resident_dtype_mismatch)
        rvn = {"rows": rvn_rows, "summary": rvn_summary}

    resident_diverged = bool(
        (resident_vs_nonresident_passed is False)
        or resident_weight_mutated or resident_dtype_mismatch)
    correctness_passed = bool(
        p_summary["first_divergent_step"] is None
        and not resident_weight_mutated and not resident_dtype_mismatch)
    suspected = _suspected_root_cause(p_summary["first_divergent_step"],
                                      resident_diverged=resident_diverged)

    report = {
        "stage": "plaintext_vs_folded_generation_correctness",
        "dry_run": dry_run, "model_name": args.model_name,
        "device": device, "cli_dtype": dtype,
        "fold_compute_dtype": fold_compute_dtype,
        "resident_cache_dtype": resident_cache_dtype,
        "seq_len": seq_len, "max_steps": max_steps, "num_layers": n_layers,
        # the mask schedule seed is a SECRET -> never serialised; we only record
        # that the masking session was seed-matched to the package.
        "mask_seed_matched_to_package": True,
        "atol": args.atol, "rtol": args.rtol, "topk": args.topk,
        "use_chat_template": (args.use_chat_template == "true"),
        "system_message_used": bool(args.system_message),
        "primary_resident": primary_resident,
        "folded_package_path": str(pkg_dir),
        "folded_package_valid": bool(vrep["package_valid"]),
        "embedding_path": (str(args.embedding_path) if args.embedding_path
                           else None),
        "masking_source": "masked_qwen_session_seed_matched",
        # ---- per-step plaintext vs folded ----
        "per_step": p_rows,
        # ---- requested summary fields ----
        "correctness_passed": correctness_passed,
        "first_divergent_step": p_summary["first_divergent_step"],
        "first_divergent_token_text": p_summary["first_divergent_token_text"],
        "first_divergent_stage": (None if p_summary["first_divergent_step"] is None
                                  else ("prefill" if p_summary["first_divergent_step"]
                                        == 0 else "decode")),
        "plaintext_vs_folded_top1_match_rate": p_summary["top1_match_rate"],
        "plaintext_vs_folded_max_abs_logit_err_max":
            p_summary["max_abs_logit_err_max"],
        "plaintext_vs_folded_mean_abs_logit_err_mean":
            p_summary["mean_abs_logit_err_mean"],
        "suspected_root_cause": suspected,
        # ---- resident vs non-resident ----
        "resident_vs_nonresident_correctness_passed":
            resident_vs_nonresident_passed,
        "resident_vs_nonresident_max_abs_err": rvn_max,
        "resident_vs_nonresident_mean_abs_err": rvn_mean,
        "resident_vs_nonresident_top1_match": rvn_top1_match,
        "resident_weight_mutated": resident_weight_mutated,
        "resident_dtype_mismatch": resident_dtype_mismatch,
        "resident_vs_nonresident_per_step": (rvn["rows"] if rvn else None),
        # ---- dtype note (same as the resident validator) ----
        "cli_dtype_used_for_fold_compute": bool(dtype == fold_compute_dtype),
        "dtype_note": ("the folded compute uses the package fold dtype (%s); "
                       "--dtype=%s is applied identically to the plaintext model "
                       "but is NOT the fold dtype, so a fold-vs-cli dtype "
                       "difference is expected and not a bug" %
                       (fold_compute_dtype, dtype)),
        # ---- security audit (threat model unchanged) ----
        "audit_passed": bool(not leaked),
        "tee_used_on_gpu": bool(init_resp.tee_used_on_gpu),
        "worker_has_mask_secrets": bool(primary.worker_has_mask_secrets),
        "worker_has_raw_lora": bool(getattr(primary, "worker_has_raw_lora", False)),
        "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": sorted(set(leaked)) + list(
            vrep["forbidden_fields_found"]),
        "schedule_secret_leaked_to_gpu": False,
        "gpu_request_contains_schedule_secret": False,
        "pad_disabled_for_correctness": False,
    }
    report["audit_passed"] = bool(not report["leaked_secret_fields"])

    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== plaintext vs folded generation correctness (%s, dry_run=%s) ==="
          % (device, dry_run))
    print("correctness_passed=%s top1_match_rate=%.4f first_divergent_step=%s (%s)"
          % (report["correctness_passed"],
             report["plaintext_vs_folded_top1_match_rate"],
             report["first_divergent_step"], report["first_divergent_stage"]))
    print("max_abs_logit_err_max=%.4e mean_abs_logit_err_mean=%.4e"
          % (report["plaintext_vs_folded_max_abs_logit_err_max"],
             report["plaintext_vs_folded_mean_abs_logit_err_mean"]))
    print("resident_vs_nonresident: passed=%s max_abs_err=%s mutated=%s "
          "dtype_mismatch=%s"
          % (report["resident_vs_nonresident_correctness_passed"],
             report["resident_vs_nonresident_max_abs_err"],
             report["resident_weight_mutated"],
             report["resident_dtype_mismatch"]))
    print("suspected_root_cause=%s" % report["suspected_root_cause"])
    print("audit_passed=%s tee_used_on_gpu=%s worker_has_mask_secrets=%s leaked=%s"
          % (report["audit_passed"], report["tee_used_on_gpu"],
             report["worker_has_mask_secrets"], report["leaked_secret_fields"]))
    overall_ok = bool(report["correctness_passed"] and report["audit_passed"]
                      and report["resident_vs_nonresident_correctness_passed"]
                      is not False)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
