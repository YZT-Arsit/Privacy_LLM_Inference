"""G2 protected generation — autoregressive greedy decode on the REAL A10 with a REAL TDX argmax
boundary. Runs on the A10 (untrusted GPU). Loads the immutable transformed package (root bfd578b8)
and the trained MASKED LoRA adapter (.adapter.pt, masked runtime domain) into MaskedQwen. The GPU
NEVER materializes plaintext base weights, plaintext A/B, the vocab permutation, or plaintext logits.

Per generated token: the GPU runs the masked forward, takes the last-position masked-logit argmax
index j (a scalar), and asks the TDX enclave to map it through the SECRET inverse vocab permutation
(decode_argmax op) -> true token id. Because the vocab mask is a pure permutation,
perm_inv[argmax(masked)] == argmax(plaintext_logits). One trusted scalar round-trip per token;
attestation-gated + HMAC + monotonic seq. The returned tokens are the public generated output.

Fresh-process safe: this is a standalone process; nothing from any training process is imported.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h800_unified_worker import PackageNativeLoader, MaskedQwen, compute_root_hash, new_counters, PKG, CKPT  # noqa: E402
from h800_direct_runner import TDXChannel, mac, TransportError  # noqa: E402


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def rep_bigram(toks):
    if len(toks) < 2: return 0.0
    bg = list(zip(toks, toks[1:]))
    return round(1.0 - len(set(bg)) / max(1, len(bg)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)       # {session_key_hex, run_id}
    ap.add_argument("--tdx", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--service-cmd", required=True)
    ap.add_argument("--adapter", default="")          # .adapter.pt (masked runtime A/B); empty = base (G0)
    ap.add_argument("--adapter-package", default="")   # PHASE 7: verified transformed-adapter package dir
    ap.add_argument("--expect-base-root", default=""); ap.add_argument("--expect-model-cfg", default="")
    ap.add_argument("--expect-targets", default=""); ap.add_argument("--expect-rank", type=int, default=8)
    ap.add_argument("--expect-transform-version", default="private_base_fold_v1.0")
    ap.add_argument("--min-optimizer-state-version", type=int, default=0)
    ap.add_argument("--lora-rank", type=int, default=8); ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--tok", default="/root/genlora_tok")
    ap.add_argument("--gen-in", required=True); ap.add_argument("--gen-out", required=True)
    ap.add_argument("--gen-max", type=int, default=0); ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--require-attestation", action="store_true")
    ap.add_argument("--cell", default="G2_protected")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    CDT = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    sess = json.loads(Path(a.session).read_text())
    key = bytes.fromhex(sess["session_key_hex"]); run_id = sess["run_id"]
    cfg = json.loads((CKPT / "config.json").read_text())
    root = compute_root_hash(PKG)
    counters = new_counters()

    loader = PackageNativeLoader(PKG, dev, CDT); loader.load_all(cfg["num_hidden_layers"])
    model = MaskedQwen(loader.tensors, cfg, dev, CDT, lora_rank=a.lora_rank, lora_alpha=a.lora_alpha)
    adapter_hash = "none"; adapter_source = "base_no_adapter"
    if a.adapter_package:
        # PHASE 7 fresh-process handoff: load ONLY the verified transformed-adapter package. Fail-closed
        # loader checks manifest/tensor hashes, base-package root, model cfg, targets, rank, transform
        # version, staleness, and refuses plaintext/optimizer tensors. NEVER touches a training process.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from adapter_handoff import load_adapter_for_generation
        res = load_adapter_for_generation(
            a.adapter_package, expected_base_root_hash=(a.expect_base_root or root),
            expected_model_config_hash=a.expect_model_cfg, expected_targets=a.expect_targets.split(","),
            expected_rank=a.expect_rank, expected_transform_version=a.expect_transform_version,
            min_optimizer_state_version=a.min_optimizer_state_version)
        model.lora = {}
        for name, t in res["tensors"].items():
            side, l, proj = name.split("."); l = int(l)
            lora_key = (l, proj); AB = list(model.lora.get(lora_key, (None, None)))
            AB[0 if side == "A" else 1] = t.to(dev, CDT); model.lora[lora_key] = tuple(AB)
        adapter_hash = sha16((Path(a.adapter_package) / "adapter_tensors.safetensors").read_bytes())
        adapter_source = f"verified_package:{a.adapter_package}"
        print(f"[protected_gen] loaded VERIFIED package {a.adapter_package} ({len(model.lora)} factors)", flush=True)
    elif a.adapter:
        adp = torch.load(a.adapter, map_location="cpu", weights_only=False)
        adapter_hash = sha16(Path(a.adapter).read_bytes())
        model.lora = {}
        for k, (A, B) in adp.items():
            l, proj = k.split("."); l = int(l)
            model.lora[(l, proj)] = (A.to(dev, CDT), B.to(dev, CDT))
        adapter_source = f"raw_adapter_pt:{a.adapter}"
        print(f"[protected_gen] loaded adapter {a.adapter} ({len(model.lora)} factors) sha16 {adapter_hash}", flush=True)

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.tok); eos = tk.eos_token_id
    V = cfg["vocab_size"]

    ch = TDXChannel(a.key, a.tdx, a.service_cmd)
    # handshake -> attestation evidence (once)
    hh, _, _ = ch.request({"op": "handshake", "run_id": run_id})
    att = hh.get("attestation", {})
    att_verified = bool(att.get("attestation_verified"))
    if a.require_attestation and not (att_verified and att.get("reportdata_bound", att_verified) is not False
                                      and att.get("debug_false", True) is not False):
        raise TransportError(f"attestation_gate: protected generation refused (verified={att_verified}); "
                             "no decode, fail-closed")

    prompts = json.loads(Path(a.gen_in).read_text())
    if a.gen_max > 0: prompts = prompts[:a.gen_max]
    seq = 0; out_lines = []; t0 = time.time(); n_tokens = 0; tdx_calls = 0; tdx_wait = 0.0
    decode_cfg = {"strategy": "greedy", "max_new_tokens": a.max_new, "eos": eos,
                  "boundary": "tdx_decode_argmax_scalar", "dtype": a.dtype}
    for jx, p in enumerate(prompts):
        ids = list(p["prompt_ids"]); new = []; fr = "length"
        for _ in range(a.max_new):
            x = torch.tensor(ids, device=dev)
            with torch.no_grad():
                logits_masked = model.forward(x, counters)      # (T,V) vocab-permuted, masked
            j = int(logits_masked[-1].argmax().item())          # masked-domain argmax (scalar)
            payload = json.dumps({"idx": [j], "V": V}).encode()
            seq += 1
            tw = time.time()
            rh, rpl, _ = ch.request({"op": "decode_argmax", "seq": seq, "run_id": run_id,
                                     "hmac": mac(key, payload, seq, run_id, "decode_argmax")}, payload)
            tdx_wait += time.time() - tw; tdx_calls += 1
            if rh.get("hmac") != mac(key, rpl, rh["seq"], run_id, "decode_ack"):
                raise TransportError("decode_ack hmac mismatch")
            tok = json.loads(rpl.decode())["tokens"][0]
            if tok == eos:
                fr = "eos"; break
            new.append(tok); ids.append(tok)
        text = tk.decode(new, skip_special_tokens=True).strip(); n_tokens += len(new)
        out_lines.append({"sample_id": p["sample_id"], "input_hash": sha16(json.dumps(p["prompt_ids"]).encode()),
                          "meaning_representation": p.get("meaning_representation", ""),
                          "references": p.get("references", []), "generated_text": text, "token_ids": new,
                          "finish_reason": fr, "generated_token_count": len(new),
                          "invalid_output": len(text) == 0, "repetition_bigram_frac": rep_bigram(new),
                          "adapter_hash": adapter_hash, "base_package_root_hash": root, "decoding": decode_cfg})
        if jx % 20 == 0:
            print(f"  gen {jx}/{len(prompts)} tdx_calls {tdx_calls} {time.time()-t0:.0f}s", flush=True)
    ch.close()
    Path(a.gen_out).write_text("".join(json.dumps(r) + "\n" for r in out_lines))
    prof = {"mode": "protected_generate", "cell": a.cell, "n": len(out_lines), "adapter": a.adapter,
            "adapter_source": adapter_source, "adapter_package": a.adapter_package,
            "adapter_hash": adapter_hash, "base_package_root_hash": root, "dtype": a.dtype,
            "decoding": decode_cfg, "total_new_tokens": n_tokens, "tdx_decode_calls": tdx_calls,
            "tdx_wait_sec": round(tdx_wait, 1), "wall_sec": round(time.time()-t0, 1),
            "tokens_per_sec": round(n_tokens / max(1e-9, time.time()-t0), 2),
            "attestation_verified": att_verified, "attestation": att,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "worker_counters": counters,
            "security_note": ("GPU held masked base+adapter, masked hidden states, true attention scores, "
                              "masked logits; NEVER plaintext base/A/B/DeltaW, vocab perm, or plaintext logits. "
                              "Per token: 1 scalar argmax index -> TDX -> true token (public output).")}
    Path(a.gen_out + ".profile.json").write_text(json.dumps(prof, indent=2))
    print("[protected_gen] done", json.dumps({k: prof[k] for k in
          ["cell", "n", "total_new_tokens", "tdx_decode_calls", "wall_sec", "tokens_per_sec", "attestation_verified"]}))


if __name__ == "__main__":
    main()
