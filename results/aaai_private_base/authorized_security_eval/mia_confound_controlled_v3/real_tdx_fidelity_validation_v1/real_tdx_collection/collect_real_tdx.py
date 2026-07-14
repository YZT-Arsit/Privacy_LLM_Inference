#!/usr/bin/env python3
"""Collect frozen V2 invariants and V0 outputs on the transformed A10 + real TDX path."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/pllo_pb/scripts")
from h800_direct_runner import TDXChannel, TransportError, mac  # noqa: E402
from h800_unified_worker import (  # noqa: E402
    CKPT, PKG, MaskedQwen, PackageNativeLoader, apply_rope, compute_root_hash,
    new_counters, repeat_kv, rmsnorm_core, rope_cos_sin,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reductions(value: torch.Tensor) -> dict[str, float]:
    x = value.detach().float().cpu().numpy().astype(np.float64, copy=False)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max())}


def signed_permute(value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    width = value.shape[-1]
    perm = torch.randperm(width, generator=generator).to(value.device)
    signs = (torch.randint(0, 2, (width,), generator=generator, dtype=torch.int8) * 2 - 1)
    return value.index_select(-1, perm) * signs.to(value.device, value.dtype)


def trace_forward(model: MaskedQwen, ids: torch.Tensor, names: list[str], seed: int,
                  counters: dict) -> tuple[np.ndarray, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed % (2**63 - 1))
    length = ids.shape[0]
    h = F.embedding(ids, model.t["embed"])
    counters["mask_domain_transitions"] += 1
    cos, sin = rope_cos_sin(length, model.hd, model.theta, model.dtype)
    cos, sin = cos.to(model.device), sin.to(model.device)
    causal = torch.full((length, length), float("-inf"), device=model.device,
                        dtype=model.dtype).triu(1)
    values: dict[str, float] = {}
    for layer in range(model.L):
        values[f"hidden.layer_{layer:02d}.last_l2"] = float(
            signed_permute(h[-1].float(), generator).norm().cpu())
        r = rmsnorm_core(h, model.eps)
        q = model._proj(r, layer, "q_proj", model.t[f"L{layer}.q_proj.b"])
        k = model._proj(r, layer, "k_proj", model.t[f"L{layer}.k_proj.b"])
        v = model._proj(r, layer, "v_proj", model.t[f"L{layer}.v_proj.b"])
        q = q.view(length, model.nh, model.hd).transpose(0, 1)
        k = k.view(length, model.nkv, model.hd).transpose(0, 1)
        v = v.view(length, model.nkv, model.hd).transpose(0, 1)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = repeat_kv(k, model.nh // model.nkv)
        v = repeat_kv(v, model.nh // model.nkv)

        width = model.hd
        perm = torch.randperm(width, generator=generator).to(q.device)
        signs = (torch.randint(0, 2, (width,), generator=generator, dtype=torch.int8) * 2 - 1)
        signs = signs.to(q.device, q.dtype)
        q_view = q.index_select(-1, perm) * signs
        k_view = k.index_select(-1, perm) * signs
        v_view = signed_permute(v, generator)
        k_norms = signed_permute(k_view[:, -1, :], generator).float().norm(dim=-1)
        v_norms = v_view[:, -1, :].float().norm(dim=-1)
        for family, tensor in (("k", k_norms), ("v", v_norms)):
            for red, val in reductions(tensor).items():
                values[f"kv.{family}.layer_{layer:02d}.last_l2_by_head.{red}"] = val

        products = q_view[:, -1, None, :].float() * k_view.float()
        last_scores = products.sort(dim=-1).values.sum(dim=-1) / math.sqrt(model.hd)
        probs = torch.softmax(last_scores, dim=-1)
        metrics = {
            "last_entropy_by_head": -(probs * probs.clamp_min(1e-30).log()).sum(-1),
            "last_max_by_head": probs.max(-1).values,
            "true_score_last_mean_by_head": last_scores.mean(-1),
            "true_score_last_std_by_head": last_scores.std(-1),
        }
        for metric, tensor in metrics.items():
            for red, val in reductions(tensor).items():
                values[f"attention.layer_{layer:02d}.{metric}.{red}"] = val

        attn = torch.softmax(last_scores.unsqueeze(1).expand(-1, length, -1) + causal, dim=-1)
        # Recompute the full true-score attention used by the actual forward. The last-row
        # summaries above use an order-stable product reduction; forward semantics remain exact.
        full_scores = (q @ k.transpose(-1, -2)) / math.sqrt(model.hd)
        attn = torch.softmax(full_scores + causal, dim=-1)
        o = (attn @ v).transpose(0, 1).reshape(length, model.H)
        h = h + model._proj(o, layer, "o_proj")
        counters["mask_domain_transitions"] += 2
        r2 = rmsnorm_core(h, model.eps)
        gate = model._proj(r2, layer, "gate_proj")
        up = model._proj(r2, layer, "up_proj")
        h = h + model._proj(F.silu(gate) * up, layer, "down_proj")
        counters["mask_domain_transitions"] += 2

    h = rmsnorm_core(h, model.eps)
    values["hidden.final.last_l2"] = float(signed_permute(h[-1].float(), generator).norm().cpu())
    logits = h @ model.t["lm_head"].t()
    last = logits[-1].float()
    permuted = last.index_select(0, torch.randperm(last.numel(), generator=generator).to(last.device))
    canonical = permuted.sort().values
    top = torch.topk(permuted, k=32).values
    values.update({
        "logit.mean": float(canonical.mean().cpu()),
        "logit.std": float(canonical.std().cpu()),
        "logit.l2": float(canonical.norm().cpu()),
        "logit.min": float(canonical[0].cpu()),
        "logit.max": float(canonical[-1].cpu()),
    })
    for red, val in reductions(top).items():
        values[f"logit.top_values.{red}"] = val
    values["logit.top1_top2_margin"] = float((top[0] - top[1]).cpu())
    missing, extra = [x for x in names if x not in values], [x for x in values if x not in names]
    if missing or extra:
        raise RuntimeError(f"schema mismatch missing={missing[:4]} extra={extra[:4]}")
    row = np.asarray([values[name] for name in names], dtype=np.float64)
    if not np.isfinite(row).all():
        raise RuntimeError("non-finite feature row")
    return row, logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--tdx", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--service-cmd", required=True)
    ap.add_argument("--masked-adapter", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--require-attestation", action="store_true")
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("refusing nonempty collection output")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    session = json.loads(args.session.read_text())
    key, run_id = bytes.fromhex(session["session_key_hex"]), session["run_id"]
    schema = json.loads(args.schema.read_text())
    names = [x["feature_name"] for x in schema["features"] if x["status"] == "ALLOWED"]
    if len(names) != 611 or len(set(names)) != 611:
        raise RuntimeError("validated schema is not 611 unique allowed features")
    queries = [json.loads(x) for x in args.queries.read_text().splitlines() if x.strip()]
    if len(queries) != 250:
        raise RuntimeError("query count is not 250")
    ids = [x["sample_id"] for x in queries]
    id_hash = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if id_hash != "ea399513af254681b1fba253735915936399a2a14e020ede3eb023a7a262baef":
        raise RuntimeError("frozen sample ID hash mismatch")

    cfg = json.loads((CKPT / "config.json").read_text())
    package_root = compute_root_hash(PKG)
    if package_root != "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1":
        raise RuntimeError("transformed package root mismatch")
    device, dtype = torch.device("cuda"), torch.float32
    loader = PackageNativeLoader(PKG, device, dtype)
    loader.load_all(cfg["num_hidden_layers"])
    model = MaskedQwen(loader.tensors, cfg, device, dtype, lora_rank=8, lora_alpha=16)
    masked = torch.load(args.masked_adapter, map_location="cpu", weights_only=False)
    if set(masked) != {f"{l}.{p}" for l in range(24) for p in
                       ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}:
        raise RuntimeError("masked adapter coverage mismatch")
    model.lora = {}
    for name, (a, b) in masked.items():
        layer, proj = name.split(".", 1)
        model.lora[(int(layer), proj)] = (a.to(device, dtype), b.to(device, dtype))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/root/qwen25_05b", local_files_only=True)
    eos, vocab = tokenizer.eos_token_id, int(cfg["vocab_size"])
    channel = TDXChannel(args.key, args.tdx, args.service_cmd)
    hello, _, _ = channel.request({"op": "handshake", "run_id": run_id}, timeout=360.0)
    attestation = hello.get("attestation", {})
    attested = bool(attestation.get("attestation_verified"))
    if args.require_attestation and not (
        attested and attestation.get("reportdata_bound", attested) is not False
        and attestation.get("debug_false", True) is not False
    ):
        raise TransportError(f"attestation gate failed: {attestation}")

    master = os.urandom(32)
    matrix, outputs, commitments = [], [], []
    counters = new_counters()
    t0, decode_calls, token_count, seq = time.time(), 0, 0, 0
    with torch.inference_mode():
        for index, query in enumerate(queries):
            digest = hashlib.sha256(master + query["sample_id"].encode()).hexdigest()
            commitments.append({"sample_id": query["sample_id"], "transform_commitment": digest})
            ids_now = list(query["prompt_ids"])
            row, logits = trace_forward(
                model, torch.tensor(ids_now, device=device), names, int(digest[:16], 16), counters)
            matrix.append(row)
            generated, finish = [], "length"
            for step in range(args.max_new_tokens):
                if step > 0:
                    logits = model.forward(torch.tensor(ids_now, device=device), counters)
                masked_argmax = int(logits[-1].argmax().item())
                payload = json.dumps({"idx": [masked_argmax], "V": vocab}).encode()
                seq += 1
                reply, reply_payload, _ = channel.request({
                    "op": "decode_argmax", "seq": seq, "run_id": run_id,
                    "hmac": mac(key, payload, seq, run_id, "decode_argmax")}, payload)
                if reply.get("hmac") != mac(key, reply_payload, reply["seq"], run_id, "decode_ack"):
                    raise TransportError("decode acknowledgement HMAC mismatch")
                token = int(json.loads(reply_payload.decode())["tokens"][0])
                decode_calls += 1
                if token == eos:
                    finish = "eos"
                    break
                generated.append(token)
                ids_now.append(token)
            token_count += len(generated)
            outputs.append({"sample_id": query["sample_id"],
                            "generated_text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
                            "token_ids": generated, "finish_reason": finish})
            if index % 10 == 0 or index == len(queries) - 1:
                print(json.dumps({"captured": index + 1, "total": len(queries),
                                  "decode_calls": decode_calls,
                                  "wall_sec": round(time.time() - t0, 1)}), flush=True)
    channel.close()
    matrix_np = np.stack(matrix)

    with (args.output_dir / "real_tdx_v2_features.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(matrix_np.tolist())
    (args.output_dir / "real_tdx_v0_outputs.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in outputs))
    (args.output_dir / "sample_index.json").write_text(json.dumps(ids, indent=2) + "\n")
    (args.output_dir / "transform_commitments.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in commitments))
    manifest = {
        "schema": "real_tdx_v2_collection_v1", "status": "PASS",
        "collection_mode": "REAL_TDX_BACKED_VIEW",
        "scope": "frozen 250-sample fidelity-validation subset",
        "records": len(ids), "feature_columns": len(names), "dtype": "fp32",
        "batch_size": 1, "padding": "none_batch1", "model_mode": "inference",
        "sample_ids_sha256": id_hash, "schema_sha256": sha256(args.schema),
        "masked_adapter_sha256": sha256(args.masked_adapter),
        "transformed_package_root_hash": package_root,
        "attestation_verified": attested, "attestation": attestation,
        "tdx_decode_calls": decode_calls, "generated_tokens": token_count,
        "all_finite": bool(np.isfinite(matrix_np).all()), "missing_rows": 0,
        "duplicate_rows": len(ids) - len(set(ids)), "labels_collected": False,
        "loss_collected": False, "gradients_collected": False,
        "runtime_metadata_in_feature_matrix": False, "silent_fallbacks": 0,
        "worker_counters": counters, "wall_sec": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(0),
    }
    (args.output_dir / "collection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("status", "records", "feature_columns",
                                               "attestation_verified", "tdx_decode_calls", "wall_sec")}))


if __name__ == "__main__":
    main()
