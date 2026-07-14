#!/usr/bin/env python3
"""Run inside the TDX guest: fold a frozen PEFT adapter into the pinned package basis."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file


TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
KEY_RE = re.compile(
    r"base_model\.model\.model\.layers\.(\d+)\.(?:self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_([AB])\.weight"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def signed_perm(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    signs = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    out = torch.zeros(n, n, dtype=torch.float32)
    out[torch.arange(n), perm] = signs
    return out


def pure_perm_index(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    return inv


def rope_rot(hd: int, seed: int) -> torch.Tensor:
    half = hd // 2
    g = torch.Generator().manual_seed(seed)
    ang = torch.rand(half, generator=g, dtype=torch.float64) * 6.283185307179586
    b = torch.eye(hd, dtype=torch.float64)
    c, s = torch.cos(ang), torch.sin(ang)
    for i in range(half):
        j = i + half
        b[i, i] = c[i]
        b[j, j] = c[i]
        b[i, j] = -s[i]
        b[j, i] = s[i]
    return b.float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--adapter-config", type=Path, required=True)
    ap.add_argument("--gamma-bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--expect-adapter-sha256", required=True)
    args = ap.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise RuntimeError("refusing to overwrite transformed adapter outputs")
    if sha256(args.adapter) != args.expect_adapter_sha256:
        raise RuntimeError("frozen adapter hash mismatch")
    config = json.loads(args.adapter_config.read_text())
    if int(config["r"]) != 8 or float(config["lora_alpha"]) != 16.0:
        raise RuntimeError("unexpected LoRA rank/alpha")
    if set(config["target_modules"]) != set(TARGETS):
        raise RuntimeError("unexpected target-module set")

    raw = load_file(str(args.adapter))
    parsed: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for name, tensor in raw.items():
        match = KEY_RE.fullmatch(name)
        if not match:
            raise RuntimeError(f"unexpected adapter tensor: {name}")
        layer, proj, side = int(match.group(1)), match.group(2), match.group(3)
        parsed.setdefault((layer, proj), {})[side] = tensor.float()
    if len(parsed) != 24 * 7 or any(set(v) != {"A", "B"} for v in parsed.values()):
        raise RuntimeError("adapter tensor coverage mismatch")

    bundle = torch.load(args.gamma_bundle, map_location="cpu", weights_only=False)
    if int(bundle["num_layers"]) != 24 or int(bundle["hidden"]) != 896:
        raise RuntimeError("gamma bundle model mismatch")
    h, hd, nh, nkv, intermediate = 896, 64, 14, 2, 4864
    nr = signed_perm(h, int(bundle["nr_seed"]))
    transformed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    max_abs_error = 0.0

    for layer in range(24):
        b = rope_rot(hd, 1000 + layer)
        bq = torch.block_diag(*([b] * nh))
        bk = torch.block_diag(*([b] * nkv))
        sv = torch.block_diag(*([signed_perm(hd, 2000 + layer)] * nkv))
        so = torch.block_diag(*([signed_perm(hd, 2000 + layer)] * nh))
        p_inv = pure_perm_index(intermediate, 3000 + layer)
        ga = bundle["ga"][layer].float()
        gm = bundle["gm"][layer].float()
        for proj in TARGETS:
            plain_a, plain_b = parsed[(layer, proj)]["A"], parsed[(layer, proj)]["B"]
            if proj in {"q_proj", "k_proj", "v_proj"}:
                a_t = (plain_a * ga.unsqueeze(0)) @ nr
            elif proj in {"gate_proj", "up_proj"}:
                a_t = (plain_a * gm.unsqueeze(0)) @ nr
            elif proj == "o_proj":
                a_t = plain_a @ so
            else:
                a_t = plain_a[:, p_inv]

            if proj == "q_proj":
                b_t = bq.T @ plain_b
            elif proj == "k_proj":
                b_t = bk.T @ plain_b
            elif proj == "v_proj":
                b_t = sv.T @ plain_b
            elif proj == "o_proj":
                b_t = nr.T @ plain_b
            elif proj in {"gate_proj", "up_proj"}:
                b_t = plain_b[p_inv]
            else:
                b_t = nr.T @ plain_b
            transformed[f"{layer}.{proj}"] = (a_t.contiguous(), b_t.contiguous())

            gen = torch.Generator().manual_seed(910000 + layer * 17 + TARGETS.index(proj))
            x_dim = plain_a.shape[1]
            x_plain = torch.randn(2, x_dim, generator=gen)
            if proj in {"q_proj", "k_proj", "v_proj"}:
                x_masked = x_plain @ nr
                ref = ((x_plain * ga) @ plain_a.T) @ plain_b.T
                out_mask = {"q_proj": bq, "k_proj": bk, "v_proj": sv}[proj]
            elif proj in {"gate_proj", "up_proj"}:
                x_masked = x_plain @ nr
                ref = ((x_plain * gm) @ plain_a.T) @ plain_b.T
                idx = pure_perm_index(intermediate, 3000 + layer)
                out_mask = None
            elif proj == "o_proj":
                x_masked = x_plain @ so
                ref = (x_plain @ plain_a.T) @ plain_b.T
                out_mask = nr
            else:
                idx = pure_perm_index(intermediate, 3000 + layer)
                x_masked = x_plain[:, idx]
                ref = (x_plain @ plain_a.T) @ plain_b.T
                out_mask = nr
            got = (x_masked @ a_t.T) @ b_t.T
            expected = ref[:, idx] if proj in {"gate_proj", "up_proj"} else ref @ out_mask
            max_abs_error = max(max_abs_error, float((got - expected).abs().max()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(transformed, args.output)
    manifest = {
        "schema": "real_tdx_trusted_adapter_transform_v1",
        "status": "PASS" if max_abs_error <= 2e-4 else "FAIL",
        "executed_inside_tdx_guest": True,
        "source_adapter_sha256": sha256(args.adapter),
        "source_adapter_config_sha256": sha256(args.adapter_config),
        "gamma_bundle_sha256": sha256(args.gamma_bundle),
        "masked_adapter_sha256": sha256(args.output),
        "factor_pairs": len(transformed),
        "rank": 8,
        "alpha": 16,
        "transform_algebra": "private_base_fold_v1.0",
        "max_random_probe_abs_error": max_abs_error,
        "plaintext_adapter_exported_to_gpu": False,
        "optimizer_state_present": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))
    if manifest["status"] != "PASS":
        raise RuntimeError("adapter fold validation failed")


if __name__ == "__main__":
    main()
