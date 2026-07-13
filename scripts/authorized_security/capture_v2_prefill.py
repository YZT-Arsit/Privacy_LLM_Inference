#!/usr/bin/env python3
"""Collect one reusable V2 prefill corpus from researcher-owned fixtures.

This diagnostic entrypoint does not modify the production worker and does not
contact TDX. It records summaries of tensors the current GPU runtime already
materializes. Raw prompts, references, plaintext tensors and secrets are never
written to the attacker corpus. Diagnostic timing is not paper-latency evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/authorized_security"))
from h800_unified_worker import (  # noqa: E402
    CKPT, PKG, MaskedQwen, PackageNativeLoader, apply_rope,
    compute_root_hash, new_counters, repeat_kv, rmsnorm_core, rope_cos_sin,
)
from view_contracts import View, ViewRecord  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def forward_trace(model: MaskedQwen, input_ids: torch.Tensor, counters: dict) -> dict:
    """Mirror MaskedQwen.forward while retaining only bounded V2 summaries."""
    T = input_ids.shape[0]
    h = F.embedding(input_ids, model.t["embed"])
    counters["mask_domain_transitions"] += 1
    cos, sin = rope_cos_sin(T, model.hd, model.theta, model.dtype)
    cos = cos.to(model.device); sin = sin.to(model.device)
    causal = torch.full((T, T), float("-inf"), device=model.device, dtype=model.dtype).triu(1)
    layers = []
    for layer in range(model.L):
        r = rmsnorm_core(h, model.eps)
        q = model._proj(r, layer, "q_proj", model.t[f"L{layer}.q_proj.b"])
        k = model._proj(r, layer, "k_proj", model.t[f"L{layer}.k_proj.b"])
        v = model._proj(r, layer, "v_proj", model.t[f"L{layer}.v_proj.b"])
        q = q.view(T, model.nh, model.hd).transpose(0, 1)
        k = k.view(T, model.nkv, model.hd).transpose(0, 1)
        v = v.view(T, model.nkv, model.hd).transpose(0, 1)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        k = repeat_kv(k, model.nh // model.nkv)
        v = repeat_kv(v, model.nh // model.nkv)
        scores = (q @ k.transpose(-1, -2)) / (model.hd ** 0.5)
        attn = torch.softmax(scores + causal, dim=-1)
        last_attn = attn[:, -1, :].float()
        entropy = -(last_attn * last_attn.clamp_min(1e-30).log()).sum(-1)
        layers.append({
            "layer": layer,
            "q_last_l2": q[:, -1, :].float().norm(dim=-1).cpu().tolist(),
            "k_last_l2": k[:, -1, :].float().norm(dim=-1).cpu().tolist(),
            "v_last_l2": v[:, -1, :].float().norm(dim=-1).cpu().tolist(),
            "attention_last_entropy": entropy.cpu().tolist(),
            "attention_last_max": last_attn.max(dim=-1).values.cpu().tolist(),
            "attention_last_argmax": last_attn.argmax(dim=-1).cpu().tolist(),
            "true_score_last_mean": scores[:, -1, :].float().mean(dim=-1).cpu().tolist(),
            "true_score_last_std": scores[:, -1, :].float().std(dim=-1).cpu().tolist(),
            "hidden_last_l2_before_attention": float(h[-1].float().norm().cpu()),
        })
        o = (attn @ v).transpose(0, 1).reshape(T, model.H)
        o = model._proj(o, layer, "o_proj")
        h = h + o; counters["mask_domain_transitions"] += 2
        r2 = rmsnorm_core(h, model.eps)
        gate = model._proj(r2, layer, "gate_proj")
        up = model._proj(r2, layer, "up_proj")
        down = model._proj(F.silu(gate) * up, layer, "down_proj")
        h = h + down; counters["mask_domain_transitions"] += 2
    h = rmsnorm_core(h, model.eps)
    logits = h @ model.t["lm_head"].t()
    last = logits[-1].float()
    top = torch.topk(last, k=min(32, last.numel()))
    return {
        "transformed_hidden": {
            "final_last_l2": float(h[-1].float().norm().cpu()),
            "per_layer": [{"layer": x["layer"],
                           "last_l2": x["hidden_last_l2_before_attention"]} for x in layers],
        },
        "masked_q": [{"layer": x["layer"], "last_l2_by_head": x["q_last_l2"]} for x in layers],
        "masked_k": [{"layer": x["layer"], "last_l2_by_head": x["k_last_l2"]} for x in layers],
        "masked_v": [{"layer": x["layer"], "last_l2_by_head": x["v_last_l2"]} for x in layers],
        "attention_scores": [{
            "layer": x["layer"], "last_entropy_by_head": x["attention_last_entropy"],
            "last_max_by_head": x["attention_last_max"],
            "last_argmax_by_head": x["attention_last_argmax"],
            "true_score_last_mean_by_head": x["true_score_last_mean"],
            "true_score_last_std_by_head": x["true_score_last_std"],
        } for x in layers],
        "masked_logits": {
            "mean": float(last.mean().cpu()), "std": float(last.std().cpu()),
            "l2": float(last.norm().cpu()), "min": float(last.min().cpu()),
            "max": float(last.max().cpu()), "top_indices": top.indices.cpu().tolist(),
            "top_values": top.values.cpu().tolist(),
        },
        "tensor_shapes": {
            "sequence_length": int(T), "hidden_size": model.H,
            "attention_heads": model.nh, "masked_logits": [int(T), int(last.numel())],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, choices=["V2"])
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = ap.parse_args()
    view = View(args.view)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    cfg = json.loads((CKPT / "config.json").read_text())
    root = compute_root_hash(PKG)
    loader = PackageNativeLoader(PKG, dev, dtype)
    loader.load_all(cfg["num_hidden_layers"])
    model = MaskedQwen(loader.tensors, cfg, dev, dtype, lora_rank=8, lora_alpha=16)
    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    model.lora = {}
    for key, (a, b) in adapter.items():
        layer, proj = key.split(".")
        model.lora[(int(layer), proj)] = (a.to(dev, dtype), b.to(dev, dtype))
    queries = json.loads(args.queries.read_text())
    if args.max_samples > 0:
        queries = queries[:args.max_samples]
    counters = new_counters(); records = []; t0 = time.time()
    with torch.no_grad():
        for index, query in enumerate(queries):
            trace = forward_trace(model, torch.tensor(query["prompt_ids"], device=dev), counters)
            row = {
                "sample_id": f"e2e_nlg:test:{int(query['sample_id']):06d}",
                "run_id": args.run_id, "task": "e2e_nlg", "split": "test",
                **trace,
                "package_metadata": {
                    "base_package_root_hash": root,
                    "adapter_sha256": sha256(args.adapter),
                    "dtype": args.dtype, "prefill_only": True,
                    "diagnostic_timing_eligible": False,
                },
            }
            records.append(ViewRecord.build(view, row).export())
            if index % 20 == 0:
                print(json.dumps({"captured": index + 1, "total": len(queries)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite V2 corpus: {args.output}")
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    profile = {
        "schema": "authorized_v2_prefill_capture", "version": "1.0",
        "view": "V2", "records": len(records), "run_id": args.run_id,
        "base_package_root_hash": root, "adapter_sha256": sha256(args.adapter),
        "queries_sha256": sha256(args.queries), "output_sha256": sha256(args.output),
        "wall_sec": time.time() - t0, "device": dev, "dtype": args.dtype,
        "victim_runs": 1, "prefill_per_sample": 1, "timing_eligible": False,
        "plaintext_or_tdx_state_written": False, "worker_counters": counters,
    }
    args.output.with_suffix(args.output.suffix + ".profile.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    print(json.dumps(profile))


if __name__ == "__main__":
    main()
