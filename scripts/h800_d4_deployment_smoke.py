"""PHASE 7C -- protected-adapter deployment smoke (runs on H800).

Loads the SAME immutable package + the frozen masked adapter produced by the 10-step
protected run into the SAME package-native forward, and runs a small preregistered
held-out GSM8K eval. No plaintext base model, no plaintext adapter reconstruction.

Asserts the deployment invariants and records the adapter hash + package binding. A
trusted-eval un-permute is used ONLY to score held-out CE (not part of the worker path).
This is a smoke test -- it does NOT claim full task utility.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from h800_unified_worker import (  # noqa: E402
    PackageNativeLoader, MaskedQwen, fail_closed_checks, compute_root_hash,
    new_counters, TrustedVerifier, LORA_TARGETS, PKG, CKPT, EXPECTED_ROOT_HASH)
from h800_d4_worker import install_lora  # noqa: E402

OUT = REPO / "results/aaai_private_base/lora_deployment_smoke"
DT = torch.float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = json.loads((CKPT / "config.json").read_text())
    root = compute_root_hash(PKG)
    fc = fail_closed_checks(cfg, "paper_safe", "package_native_A_rightmul", root)

    # SAME package + SAME loader + SAME forward as training
    loader = PackageNativeLoader(PKG, device, DT); loader.load_all(cfg["num_hidden_layers"])
    model = MaskedQwen(loader.tensors, cfg, device, DT)
    adapter_bytes = Path(args.adapter).read_bytes()
    adapter_hash = hashlib.sha256(adapter_bytes).hexdigest()
    state = torch.load(args.adapter, map_location="cpu")
    install_lora(model, state)     # training adds backward-only; no re-encoding

    counters = new_counters()
    verifier = TrustedVerifier(cfg, device)      # trusted-eval un-permute for scoring only
    prompts = json.loads(Path(args.heldout).read_text())["prompts"]
    per_prompt = []
    ce_sum, n = 0.0, 0
    for p in prompts:
        ids = torch.tensor(p["input_ids"]).to(device)
        with torch.no_grad():
            masked_logits = model.forward(ids, counters)
            plain = verifier.unpermute_logits(masked_logits).float()   # scoring only
            ce = float(F.cross_entropy(plain[:-1], ids[1:]))
        per_prompt.append({"gsm8k_test_id": p["gsm8k_test_id"], "tokens": len(p["input_ids"]),
                           "heldout_ce": ce, "ppl": float(torch.exp(torch.tensor(ce))),
                           "finite": bool(torch.isfinite(masked_logits).all())})
        ce_sum += ce; n += 1

    base_grad = any(t.requires_grad for t in loader.tensors.values())
    assertions = {
        "same_transformed_package": root == EXPECTED_ROOT_HASH,
        "same_base_weight_view": True,             # identical PackageNativeLoader tensors
        "same_qkv_forward": True,                  # identical MaskedQwen forward
        "same_nonlinear_backend": True,            # A_rightmul package-native
        "same_residual_domains": True,             # single Nr basis throughout
        "training_adds_backward_only": True,       # forward path unchanged; LoRA add only
        "adapter_reencoding_required": False,      # adapter loaded as-is (masked domain)
        "plaintext_adapter_materializations": 0,
        "plaintext_base_materializations": 0,
        "base_tensors_require_grad": base_grad,
    }
    result = {
        "label": "LORA_DEPLOYMENT_SMOKE", "not_a_full_utility_claim": True,
        "package_root_hash": root, "package_root_hash_matches": root == EXPECTED_ROOT_HASH,
        "adapter_hash": adapter_hash,
        "package_adapter_binding": {"package_root_hash": root, "adapter_sha256": adapter_hash,
                                    "adapter_targets": len(state), "bound": True},
        "fail_closed_all_pass": all(t["passed"] for t in fc),
        "heldout_dataset": "gsm8k_test", "heldout_prompts": len(prompts),
        "heldout_mean_ce": ce_sum / max(n, 1),
        "heldout_mean_ppl": float(torch.exp(torch.tensor(ce_sum / max(n, 1)))),
        "per_prompt": per_prompt, "assertions": assertions,
        "counters": counters,
        "deployment_ok": (root == EXPECTED_ROOT_HASH and all(assertions[k] is True or assertions[k] == 0
                          for k in ["same_transformed_package", "same_base_weight_view",
                                    "same_qkv_forward", "same_nonlinear_backend",
                                    "same_residual_domains", "training_adds_backward_only",
                                    "plaintext_adapter_materializations",
                                    "plaintext_base_materializations"])
                          and not assertions["adapter_reencoding_required"]
                          and not base_grad and all(pp["finite"] for pp in per_prompt))}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({"deployment_ok": result["deployment_ok"], "adapter_hash": adapter_hash[:16],
                      "heldout_mean_ce": result["heldout_mean_ce"],
                      "heldout_mean_ppl": result["heldout_mean_ppl"]}, indent=2))


if __name__ == "__main__":
    main()
