#!/usr/bin/env python3
"""CPU regression for the protected target-set ablation protocol.

Exercises only researcher-owned fixtures.  It verifies that T1--T4 initialize and
update exactly their requested trusted factors, while the frozen all-seven default
continues to require the original complete state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from tdx_adamw_protocol import TrustedAdamW  # noqa: E402

BUNDLE = REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"
CFG = {"num_attention_heads": 14, "num_key_value_heads": 2,
       "hidden_size": 896, "intermediate_size": 4864}
PROFILES = {
    "T1": ("q_proj", "v_proj"),
    "T2": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "T3": ("gate_proj", "up_proj", "down_proj"),
    "T4": ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"),
}


def exercise(gb: dict, name: str, targets: tuple[str, ...]) -> dict:
    tw = TrustedAdamW(gb, CFG, lr=2e-4, state_dtype=torch.float32,
                      active_targets=targets)
    tw.set_binding(run_id=f"subset-{name}", package_root_hash="fixture",
                   adapter_id=f"adapter-{name}")
    gen = torch.Generator().manual_seed(100 + len(targets))
    for key, (fold, _, _) in tw.tinA.items():
        plain = torch.randn(8, fold.shape[0], generator=gen)
        tw.init_factor(*key, A_tilde=plain @ fold)
    for key, tout in tw.toutB.items():
        plain = torch.randn(tout.shape[0], 8, generator=gen)
        tw.init_factor(*key, B_tilde=tout.T @ plain)
    assert tw.state_present(), f"{name}: requested trusted state incomplete"
    grad_a = {f"{l}.{p}": torch.zeros_like(state[0])
              for (l, p), state in tw.stateA.items()}
    grad_b = {f"{l}.{p}": torch.zeros_like(state[0])
              for (l, p), state in tw.stateB.items()}
    out_a, out_b, missing = tw.step(grad_a, grad_b)
    assert missing == 0 and tw.version == 1
    assert len(out_a) == len(tw.stateA) and len(out_b) == len(tw.stateB)

    blob = tw.checkpoint(b"s" * 32)
    expected = {**tw.binding, "version": tw.version,
                "checkpoint_sequence": tw.checkpoint_seq}
    restored = TrustedAdamW(gb, CFG, lr=2e-4, state_dtype=torch.float32)
    restored.restore(blob, b"s" * 32, expected)
    assert restored.state_present()
    assert set(restored.trusted_A) == set(tw.trusted_A)
    assert set(restored.trusted_B) == set(tw.trusted_B)
    return {"targets": list(targets), "trusted_A_factors": len(tw.stateA),
            "trusted_B_factors": len(tw.stateB), "missing": missing,
            "checkpoint_restore": True}


def main() -> None:
    gb = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    results = {name: exercise(gb, name, targets)
               for name, targets in PROFILES.items()}
    print(json.dumps({"PASS": True, "profiles": results}, indent=2))


if __name__ == "__main__":
    main()
