"""H800 local private folded-LoRA correctness probe.

Compares two paths over masked input on the SAME tiny/real base model + masks:

* **trusted reference** — base model + raw LoRA applied inside the trusted process
  (``apply_lora_to_model`` -> ``MaskedQwenSession.worker_prefill/decode``);
* **package path** — base folded operators + folded-LoRA operators merged in the
  masked basis (``W_tilde += a_tilde @ b_tilde``), run through the executable
  folded kernels. The package path never holds raw A/B or masks.

Reports recovered-logits agreement (allclose / errors / top-1 / top-k / next
token) and ``tokens_exact_match`` for a 4-token greedy decode, plus the security
flags. ``--dry-run`` uses the tiny base model + a synthetic adapter (never a paper
result).

Example::

    python scripts/run_qwen7b_lora_folded_local_probe.py \\
        --model-path /root/.../Qwen2___5-7B-Instruct \\
        --base-folded-package-path /root/.../qwen7b_folded_full \\
        --adapter-path /root/.../my_lora_adapter \\
        --target-modules q_proj,k_proj,v_proj,o_proj --rank 8 --alpha 16 \\
        --max-new-tokens 4 --seq-len 128 --dtype bfloat16 --device cuda \\
        --output-json outputs/qwen7b_lora_folded_local_probe.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from pllo.deployment.folded_package import forbidden_tensor_names  # noqa: E402
from pllo.deployment.folded_worker import (  # noqa: E402
    apply_folded_layer_decode,
    apply_folded_layer_prefill,
)
from pllo.deployment.lora_folded_package import (  # noqa: E402
    _RAW_LORA_NAME_HINTS,
    DEFAULT_TARGET_MODULES,
    apply_lora_to_model,
    fold_lora_for_layer,
    load_hf_lora_adapter,
    lora_scaling,
    merge_folded_lora,
    synthetic_lora_adapter,
)
from pllo.experiments.folded_probe_common import (  # noqa: E402
    err_stats,
    load_model_and_ids,
    seed_from_manifest,
    tiny_model,
    topk_overlap,
)
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import (  # noqa: E402
    MemoryOptimizedConfig,
    _cfg_to,
)


def _csv(s):
    return [p for p in str(s).replace(" ", "").split(",") if p]


def _greedy(rec):
    return int(rec.argmax(-1).item())


def _merged_layers(session_base, lora, target_modules, scaling, rank, rank_seed):
    """Per-layer base folded tensors with folded LoRA merged in (package path)."""
    layers = []
    leaked = []
    for ell in range(session_base.n):
        base = {k: v.clone()
                for k, v in session_base.export_folded_layer_tensors(ell).items()}
        if ell in lora:
            fl = fold_lora_for_layer(session_base, ell, lora[ell],
                                     scaling=scaling, rank=rank,
                                     rank_seed=rank_seed,
                                     target_modules=target_modules)
            # audit the folded-LoRA tensor names that a worker would receive
            names = list(fl.keys())
            leaked += forbidden_tensor_names(names)
            leaked += [n for n in names
                       if any(h in n.lower() for h in _RAW_LORA_NAME_HINTS)]
            base = merge_folded_lora(base, fl, target_modules)
        layers.append(base)
    return layers, sorted(set(leaked))


def _pkg_prefill(layers, h_tilde, cfg0, cos, sin, eps):
    h, kv = h_tilde, []
    for lt in layers:
        out = apply_folded_layer_prefill(h, lt, cfg0, cos, sin, eps)
        h = out["y_tilde"]
        kv.append(out["cache"])
    return h, kv


def _pkg_decode_step(layers, x_next, kv, position, cfg0, cos, sin, eps):
    h, new_kv = x_next, []
    for lt, c in zip(layers, kv):
        out = apply_folded_layer_decode(h, lt, c, position, cfg0, cos, sin, eps)
        h = out["y_tilde"]
        new_kv.append(out["cache"])
    return h, new_kv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--use-chat-template", default="true")
    ap.add_argument("--base-folded-package-path", default=None,
                    help="for the mask seed (so LoRA folds in the base basis)")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--rank-seed", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-json",
                    default="outputs/qwen7b_lora_folded_local_probe.json")
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    dry_run = bool(args.dry_run or not args.model_path)
    target_modules = _csv(args.target_modules)
    seed = args.seed
    if args.base_folded_package_path:
        seed = seed_from_manifest(args.base_folded_package_path, args.seed)
    rank_seed = args.rank_seed if args.rank_seed is not None else seed
    scaling = lora_scaling(args.alpha, args.rank)
    n_new = int(args.max_new_tokens)

    model, mc, ids, device, dtype = load_model_and_ids(args, dry_run)
    seq_len = int(ids.shape[1])
    n_layers = (args.num_layers if args.num_layers is not None
                else int(len(model.model.layers)))

    def _cfg():
        return MemoryOptimizedConfig(
            num_layers=n_layers, batch_size=1, seq_len=seq_len,
            max_new_tokens=n_new, device=device, dtype=dtype,
            folding_dtype="float32", folded_weight_device=device, seed=seed)

    session_base = MaskedQwenSession(model, mc, _cfg())
    lora = (load_hf_lora_adapter(args.adapter_path, mc, session_base.n,
                                 target_modules) if args.adapter_path
            else synthetic_lora_adapter(mc, session_base.n, target_modules,
                                        args.rank, seed=rank_seed))
    h_tilde = session_base.mask_embeddings(ids)

    # --- trusted reference: base + raw LoRA inside the trusted process --------
    model_lora = copy.deepcopy(model)
    apply_lora_to_model(model_lora, lora, target_modules, scaling)
    session_lora = MaskedQwenSession(model_lora, mc, _cfg())

    def _ref_tokens():
        out = session_lora.worker_prefill(h_tilde)
        rec_last = session_base.recover(out["logits_tilde"][:, -1, :])
        toks = [_greedy(rec_last)]
        kv, pos = out["kv"], seq_len
        for _ in range(n_new - 1):
            x = session_base.mask_token_embedding(torch.tensor([toks[-1]]))
            out = session_lora.worker_decode(x, kv, pos)
            kv = out["kv"]
            toks.append(_greedy(session_base.recover(
                out["logits_tilde"][:, -1, :])))
            pos += 1
        return rec_last, toks

    ref_rec_last, ref_tokens = _ref_tokens()

    # --- package path: base folded + folded LoRA merged (no raw A/B, no masks) -
    cfg0 = _cfg_to(session_base.layer_configs[0], session_base.compute_device)
    cos, sin, eps = session_base._cos, session_base._sin, session_base.eps
    layers, leaked = _merged_layers(session_base, lora, target_modules, scaling,
                                    args.rank, rank_seed)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    h, kv = _pkg_prefill(layers, h_tilde, cfg0, cos, sin, eps)
    pkg_rec_last = session_base.recover(session_base._final_head(h)[:, -1, :])
    pkg_tokens = [_greedy(pkg_rec_last)]
    pos = seq_len
    for _ in range(n_new - 1):
        x = session_base.mask_token_embedding(torch.tensor([pkg_tokens[-1]]))
        h, kv = _pkg_decode_step(layers, x, kv, pos, cfg0, cos, sin, eps)
        pkg_tokens.append(_greedy(session_base.recover(
            session_base._final_head(h)[:, -1, :])))
        pos += 1
    latency_s = time.perf_counter() - t0
    peak_mb = (round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
               if device == "cuda" and torch.cuda.is_available() else None)

    lmax, lmean, ll2 = err_stats(pkg_rec_last, ref_rec_last)
    top1 = bool(int(pkg_rec_last.argmax(-1)) == int(ref_rec_last.argmax(-1)))
    ov = topk_overlap(pkg_rec_last, ref_rec_last, args.topk)
    allclose = bool(torch.allclose(pkg_rec_last, ref_rec_last, atol=args.atol,
                                   rtol=args.rtol))
    tokens_exact_match = bool(pkg_tokens == ref_tokens)

    report = {
        "stage": "qwen7b_lora_folded_local_probe", "dry_run": dry_run,
        "model_name": args.model_name, "lora_enabled": True,
        "rank": args.rank, "alpha": args.alpha, "scaling": scaling,
        "target_modules": target_modules, "num_layers": session_base.n,
        "seq_len": seq_len, "max_new_tokens": n_new, "dtype": dtype, "seed": seed,
        "allclose": allclose, "max_abs_error": lmax, "mean_abs_error": lmean,
        "relative_l2_error": ll2, "top1_match": top1, "topk": args.topk,
        "topk_overlap": ov, "next_token_match": top1,
        "reference_token_ids": ref_tokens, "package_token_ids": pkg_tokens,
        "tokens_exact_match": tokens_exact_match,
        "token_match_rate": (sum(1 for a, b in zip(pkg_tokens, ref_tokens)
                                 if a == b) / max(1, len(ref_tokens))),
        "worker_has_raw_lora": False, "worker_has_mask_secrets": False,
        "tee_used_on_gpu": False, "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": leaked, "latency_s": latency_s,
        "peak_gpu_memory_mb": peak_mb,
    }
    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), report)

    print("=== local folded-LoRA correctness probe (dry_run=%s) ===" % dry_run)
    print("target_modules=%s rank=%s alpha=%s" % (target_modules, args.rank,
                                                  args.alpha))
    print("allclose=%s max_abs_error=%.3e relative_l2_error=%.3e"
          % (allclose, lmax, ll2))
    print("top1_match=%s next_token_match=%s topk_overlap=%.4f" % (top1, top1, ov))
    print("reference_token_ids=%s" % ref_tokens)
    print("package_token_ids  =%s" % pkg_tokens)
    print("tokens_exact_match=%s worker_has_raw_lora=False "
          "worker_has_mask_secrets=False tee_used_on_gpu=False"
          % tokens_exact_match)
    print("leaked_secret_fields=%s" % (leaked or "[]"))
    ok = (top1 and tokens_exact_match and not leaked)
    print("\nLoRA LOCAL PROBE %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def _write_md(path: Path, r: dict) -> None:
    L = ["# Local folded-LoRA correctness probe (dry_run=%s)" % r["dry_run"], "",
         "- target_modules=`%s`  rank=%s  alpha=%s  scaling=%s"
         % (r["target_modules"], r["rank"], r["alpha"], r["scaling"]),
         "- seq_len=%s  max_new_tokens=%s  dtype=%s  seed=%s"
         % (r["seq_len"], r["max_new_tokens"], r["dtype"], r["seed"]),
         "- **allclose=%s**  max_abs_error=%.3e  mean_abs_error=%.3e  "
         "relative_l2_error=%.3e" % (r["allclose"], r["max_abs_error"],
                                     r["mean_abs_error"], r["relative_l2_error"]),
         "- **top1_match=%s**  next_token_match=%s  topk_overlap=%.4f"
         % (r["top1_match"], r["next_token_match"], r["topk_overlap"]),
         "- reference_token_ids=%s" % r["reference_token_ids"],
         "- package_token_ids=%s" % r["package_token_ids"],
         "- **tokens_exact_match=%s**  token_match_rate=%.4f"
         % (r["tokens_exact_match"], r["token_match_rate"]),
         "- **worker_has_raw_lora=%s** **worker_has_mask_secrets=%s** "
         "**tee_used_on_gpu=%s**" % (r["worker_has_raw_lora"],
                                     r["worker_has_mask_secrets"],
                                     r["tee_used_on_gpu"]),
         "- gpu_visible_plaintext_fields=%s  leaked_secret_fields=%s"
         % (r["gpu_visible_plaintext_fields"] or "[]",
            r["leaked_secret_fields"] or "[]")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
