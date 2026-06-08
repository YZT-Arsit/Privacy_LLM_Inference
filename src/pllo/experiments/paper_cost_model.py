"""Stage 7.7f -- Paper-ready complexity and cost model.

Symbolic + tiny-config + real-config estimates of online boundary
transfers, trusted compute, accelerator compute, preprocessing,
and mask / pad / KV / LM-head storage for every protocol mode the
project supports. No real wall-clock measurement is performed.

Symbols:
    L           = number of decoder layers
    d           = hidden dimension
    h           = num query heads
    h_kv        = num KV heads     (group_size = h / h_kv)
    d_h         = head dimension   (d = h * d_h, typically)
    s           = prompt length
    s_total     = past_len + s_new (KV cache total length)
    V           = vocab size
    r           = LoRA true rank
    block_size  = paged-cache block size
    b           = LM-head block size
    bytes_dtype = bytes per scalar
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperCostModelConfig:
    bytes_dtype: int = 8
    # Tiny config (matches the modern decoder used elsewhere).
    tiny_L: int = 1
    tiny_d: int = 64
    tiny_h: int = 4
    tiny_h_kv: int = 2
    tiny_d_h: int = 16
    tiny_s: int = 6
    tiny_V: int = 97
    tiny_r: int = 4
    tiny_block_size: int = 4   # paged KV
    tiny_lm_b: int = 64        # LM-head block size
    # Real reference config (LLaMA-2-7B-ish).
    real_L: int = 32
    real_d: int = 4096
    real_h: int = 32
    real_h_kv: int = 32
    real_d_h: int = 128
    real_s: int = 1024
    real_V: int = 32000
    real_r: int = 16
    real_block_size: int = 16
    real_lm_b: int = 1024


MODES: Tuple[str, ...] = (
    "baseline_plain",
    "padded_correctness_trusted_fallback",
    "low_interaction_sequence_norm_exact_visible_attention",
    "low_interaction_token_norm_exact_visible_attention",
    "trusted_softmax_attention",
    "rope_safe_pre_mask",
    "lora_enabled",
    "paged_kv",
    "scalable_lm_head_permutation",
    "scalable_lm_head_block",
)


# ---------------------------------------------------------------------------
# Symbolic formulas (returned as plain strings)
# ---------------------------------------------------------------------------


SYMBOLIC_FORMULAS: Dict[str, str] = {
    "linear_padded_transform": (
        "Linear ``Y = X W`` becomes ``X_pad = (X - T) M``, "
        "``Y_tilde = X_pad W_tilde + C_linear`` with "
        "``W_tilde = M^{-1} W N_out``, ``C_linear = T W N_out``. "
        "Trusted recovery: ``Y = Y_tilde N_out^{-1}``. "
        "Per-call FLOPs: O(B S d_in d_out) for the linear + "
        "O(B S d_in) for the (X-T) M transition. "
        "Compile FLOPs: O(d_in d_out d_out) for W_tilde and "
        "O(B S d_in d_out) for C_linear."
    ),
    "rmsnorm_sequence_chunk_token": (
        "RMSNorm core preserves orthogonal right-action: "
        "``RMSNormCore(H Q) = RMSNormCore(H) Q``. "
        "sequence mode: 1 Q per layer per call (size [d, d]). "
        "chunk(k) mode: ceil(S/k) Q per layer per call. "
        "token mode: S Q per layer per call. "
        "Per-call extra trusted FLOPs in chunk/token modes: "
        "O(num_chunks d^3) sampling + O(B S d^2) per-row transitions."
    ),
    "rope_safe_block_rotation": (
        "Per-head right mask B_Q (B_K) is a block-diagonal 2D rotation "
        "in each RoPE pair (d_h/2 angles). "
        "Storage per head: d_h^2 (block-diagonal pattern, but stored "
        "as full d_h x d_h). "
        "Folded into q/k weights once per session: no per-call extra."
    ),
    "gqa_head_masks": (
        "N_K[h_kv], N_V[h_kv] orthogonal d_h x d_h per KV head; "
        "N_Q[h] = N_K[h // group_size]^{-T} per Q head (derived). "
        "Storage: 2 * h_kv * d_h^2 per layer."
    ),
    "kv_cache_append": (
        "Append per token per layer per KV head: "
        "K_tilde_new = K_new @ N_K, V_tilde_new = V_new @ N_V. "
        "FLOPs per token: 2 * L * h_kv * d_h^2."
    ),
    "lm_head_dense_permutation_block": (
        "Dense: z_tilde = z @ N_vocab, recovery z = z_tilde @ N_vocab^{-1}. "
        "Memory O(V^2), FLOPs O(B S V^2). "
        "Permutation: z_tilde[..., i] = z[..., perm[i]]. "
        "Memory O(V), FLOPs O(B S V). "
        "Block: per-block matmul, memory O(V b), FLOPs O(B S V b)."
    ),
    "trusted_softmax_extra_boundary_cost": (
        "Per layer per call: ship Q_tilde, K_cache_tilde, V_cache_tilde "
        "to TEE (bytes ~ B (s_new + s_total) (h + 2 h_kv) d_h * bytes), "
        "TEE returns attn_out_tilde (bytes ~ B s_new h d_h * bytes). "
        "Round trips per decode step: 1 + L."
    ),
    "lora_rank_r_overhead": (
        "Per LoRA-augmented linear: extra matmul X_pad A_tilde B_tilde "
        "with A_tilde [d_in, r_pad], B_tilde [r_pad, d_out]. "
        "Per-call FLOPs: O(B S d_in r_pad + B S r_pad d_out). "
        "Compile: O(d_in r_pad d_out)."
    ),
}


# ---------------------------------------------------------------------------
# Numeric estimation per mode
# ---------------------------------------------------------------------------


def _estimate_mode(
    mode: str,
    *,
    L: int, d: int, h: int, h_kv: int, d_h: int, s: int, V: int,
    r: int, block_size: int, lm_b: int, bytes_dtype: int,
    s_total_factor: int = 2,  # assume past_len ~= s -> s_total = 2s
) -> Dict[str, Any]:
    s_total = max(1, s_total_factor * s)
    base_linear_FLOPs_per_layer = (
        # qkv: B*S * d * (h + 2 h_kv) * d_h
        s * d * (h + 2 * h_kv) * d_h
        # o_proj: B*S * h*d_h * d
        + s * h * d_h * d
        # mlp up + gate: 2 * B*S * d * (intermediate ~ 4 * d)
        + 2 * s * d * 4 * d
        # mlp down
        + s * 4 * d * d
    )
    base_kv_append = 2 * L * h_kv * d_h * d_h
    # Mask storage (sequence Q per layer + N_K/N_V per (layer, head)).
    mask_storage_sequence = L * d * d + L * 2 * h_kv * d_h * d_h
    mask_storage_token = L * s * d * d + L * 2 * h_kv * d_h * d_h
    mask_storage_chunk = L * (s + block_size - 1) // block_size * d * d \
        + L * 2 * h_kv * d_h * d_h
    pad_storage = L * s * d * bytes_dtype

    if mode == "baseline_plain":
        return {
            "mode": mode,
            "round_trips_per_decode_step": 0,
            "intermediate_tee_reentry": False,
            "trusted_compute_ops": 0,
            "accelerator_compute_ops": L * base_linear_FLOPs_per_layer,
            "preprocessing_ops": 0,
            "mask_storage_bytes": 0,
            "pad_compensation_storage_bytes": 0,
            "kv_cache_overhead_bytes": 0,
            "lm_head_mask_overhead_bytes": 0,
            "asymptotic_complexity": "O(L d^2)",
            "notes": "No masks, no pads, no TEE.",
        }
    if mode == "padded_correctness_trusted_fallback":
        return {
            "mode": mode,
            "round_trips_per_decode_step": "O(L)",
            "intermediate_tee_reentry": True,
            "trusted_compute_ops": L * base_linear_FLOPs_per_layer // 2,
            "accelerator_compute_ops": L * base_linear_FLOPs_per_layer,
            "preprocessing_ops": L * d * d * d,
            "mask_storage_bytes": mask_storage_sequence * bytes_dtype,
            "pad_compensation_storage_bytes": pad_storage,
            "kv_cache_overhead_bytes": base_kv_append * bytes_dtype,
            "lm_head_mask_overhead_bytes": V * V * bytes_dtype,
            "asymptotic_complexity": "O(L d^2 + L d^2 nonlinear-fallback)",
            "notes": (
                "Pad-only path with trusted fallback at every "
                "nonlinear core. Many TEE re-entries."
            ),
        }
    if mode == "low_interaction_sequence_norm_exact_visible_attention":
        return {
            "mode": mode,
            "round_trips_per_decode_step": 1,
            "intermediate_tee_reentry": False,
            "trusted_compute_ops": 0,
            "accelerator_compute_ops":
                L * base_linear_FLOPs_per_layer + base_kv_append + V * d,
            "preprocessing_ops": L * d * d * d + L * 2 * h_kv * d_h * d_h * d_h,
            "mask_storage_bytes": mask_storage_sequence * bytes_dtype,
            "pad_compensation_storage_bytes": pad_storage,
            "kv_cache_overhead_bytes": base_kv_append * bytes_dtype,
            "lm_head_mask_overhead_bytes": V * V * bytes_dtype,
            "asymptotic_complexity": "O(L d^2) per call",
            "notes": "Main protocol; one TEE-accelerator round trip per step.",
        }
    if mode == "low_interaction_token_norm_exact_visible_attention":
        return {
            "mode": mode,
            "round_trips_per_decode_step": 1,
            "intermediate_tee_reentry": False,
            "trusted_compute_ops": L * s * d * d * d,
            "accelerator_compute_ops": L * s * d * d * d
                                       + L * base_linear_FLOPs_per_layer,
            "preprocessing_ops": L * s * d * d * d,
            "mask_storage_bytes": mask_storage_token * bytes_dtype,
            "pad_compensation_storage_bytes": pad_storage,
            "kv_cache_overhead_bytes": base_kv_append * bytes_dtype,
            "lm_head_mask_overhead_bytes": V * V * bytes_dtype,
            "asymptotic_complexity": "O(L s d^3) per call (per-row Q)",
            "notes": (
                "Token-wise Q: full Gram off-diagonal disrupted; "
                "per-row transition matmuls multiply by S."
            ),
        }
    if mode == "trusted_softmax_attention":
        ts_extra_bytes = (
            s * (h + 2 * h_kv) * d_h * bytes_dtype  # ship Q/K/V tilde
            + s * h * d_h * bytes_dtype             # receive attn_out_tilde
        )
        return {
            "mode": mode,
            "round_trips_per_decode_step": 1 + L,
            "intermediate_tee_reentry": True,
            "trusted_compute_ops":
                L * (s * s_total * h * d_h + s * s_total * h),  # softmax
            "accelerator_compute_ops": L * base_linear_FLOPs_per_layer,
            "preprocessing_ops": L * d * d * d,
            "mask_storage_bytes": mask_storage_sequence * bytes_dtype,
            "pad_compensation_storage_bytes": pad_storage,
            "kv_cache_overhead_bytes": base_kv_append * bytes_dtype,
            "lm_head_mask_overhead_bytes": V * V * bytes_dtype,
            "extra_boundary_bytes_per_layer_per_step": ts_extra_bytes,
            "asymptotic_complexity": "O(L d^2 + L s^2 h d_h)",
            "notes": (
                "Exact attention hidden from accelerator transcript "
                "at the cost of L extra TEE round trips per step."
            ),
        }
    if mode == "rope_safe_pre_mask":
        return {
            "mode": mode,
            "round_trips_per_decode_step": 1,
            "intermediate_tee_reentry": False,
            "trusted_compute_ops": 0,
            "accelerator_compute_ops": L * base_linear_FLOPs_per_layer,
            "preprocessing_ops": L * d * d * d
                                 + L * (h_kv * (d_h * d_h)),
            "mask_storage_bytes": mask_storage_sequence * bytes_dtype
                                  + L * h * d_h * d_h * bytes_dtype,
            "pad_compensation_storage_bytes": pad_storage,
            "kv_cache_overhead_bytes": base_kv_append * bytes_dtype,
            "lm_head_mask_overhead_bytes": V * V * bytes_dtype,
            "asymptotic_complexity": "O(L d^2) per call",
            "notes": "Block-diagonal B_Q/B_K folded into q/k weights.",
        }
    if mode == "lora_enabled":
        # Per-layer LoRA extra matmul on every linear (7 linears typical).
        n_lora_sites = 7
        lora_extra = n_lora_sites * (s * d * r + s * r * d)
        return {
            "mode": mode,
            "round_trips_per_decode_step": 1,
            "intermediate_tee_reentry": False,
            "trusted_compute_ops": 0,
            "accelerator_compute_ops": L * (
                base_linear_FLOPs_per_layer + lora_extra
            ),
            "preprocessing_ops": L * n_lora_sites * d * r * d,
            "mask_storage_bytes": mask_storage_sequence * bytes_dtype
                                  + L * n_lora_sites * (d * r + r * d) * bytes_dtype,
            "pad_compensation_storage_bytes": pad_storage,
            "kv_cache_overhead_bytes": base_kv_append * bytes_dtype,
            "lm_head_mask_overhead_bytes": V * V * bytes_dtype,
            "asymptotic_complexity": "O(L (d^2 + d r))",
            "notes": "Forward-only LoRA; rank-padded inner dimension r.",
        }
    if mode == "paged_kv":
        return {
            "mode": mode,
            "round_trips_per_decode_step": 1,
            "intermediate_tee_reentry": False,
            "trusted_compute_ops": 0,
            "accelerator_compute_ops": L * base_linear_FLOPs_per_layer
                                       + base_kv_append,
            "preprocessing_ops": 0,
            "mask_storage_bytes": mask_storage_sequence * bytes_dtype,
            "pad_compensation_storage_bytes": pad_storage,
            "kv_cache_overhead_bytes": (
                # Same invariant K_tilde / V_tilde, but stored as paged
                # blocks; per-session block-table + L blocks per head.
                base_kv_append * bytes_dtype
                + L * h_kv * ((s + block_size - 1) // block_size) * 8
            ),
            "lm_head_mask_overhead_bytes": V * V * bytes_dtype,
            "asymptotic_complexity": "O(L d^2)",
            "notes": "Block-table indexing adds O(s/block_size) per (L, head).",
        }
    if mode == "scalable_lm_head_permutation":
        return {
            "mode": mode,
            "round_trips_per_decode_step": 1,
            "intermediate_tee_reentry": False,
            "trusted_compute_ops": s * V,  # inverse-perm index_select
            "accelerator_compute_ops": s * V,
            "preprocessing_ops": V,
            "mask_storage_bytes": V * 8,
            "pad_compensation_storage_bytes": pad_storage,
            "kv_cache_overhead_bytes": base_kv_append * bytes_dtype,
            "lm_head_mask_overhead_bytes": V * 8,
            "asymptotic_complexity": "O(V) lm-head, O(L d^2) rest",
            "notes": "Logit multiset is observable; index alignment hidden.",
        }
    if mode == "scalable_lm_head_block":
        num_blocks = (V + lm_b - 1) // lm_b
        return {
            "mode": mode,
            "round_trips_per_decode_step": 1,
            "intermediate_tee_reentry": False,
            "trusted_compute_ops": s * num_blocks * lm_b * lm_b,
            "accelerator_compute_ops": s * num_blocks * lm_b * lm_b,
            "preprocessing_ops": num_blocks * lm_b ** 3,
            "mask_storage_bytes": num_blocks * lm_b * lm_b * bytes_dtype,
            "pad_compensation_storage_bytes": pad_storage,
            "kv_cache_overhead_bytes": base_kv_append * bytes_dtype,
            "lm_head_mask_overhead_bytes": num_blocks * lm_b * lm_b * bytes_dtype,
            "asymptotic_complexity": "O(V b) lm-head, O(L d^2) rest",
            "notes": "Block-membership of each vocab index observable.",
        }
    raise ValueError(f"unknown mode {mode!r}")


def _measure_tiny(cfg: PaperCostModelConfig) -> Dict[str, Any]:
    return {
        m: _estimate_mode(
            m,
            L=cfg.tiny_L, d=cfg.tiny_d, h=cfg.tiny_h, h_kv=cfg.tiny_h_kv,
            d_h=cfg.tiny_d_h, s=cfg.tiny_s, V=cfg.tiny_V,
            r=cfg.tiny_r, block_size=cfg.tiny_block_size, lm_b=cfg.tiny_lm_b,
            bytes_dtype=cfg.bytes_dtype,
        )
        for m in MODES
    }


def _measure_real(cfg: PaperCostModelConfig) -> Dict[str, Any]:
    return {
        m: _estimate_mode(
            m,
            L=cfg.real_L, d=cfg.real_d, h=cfg.real_h, h_kv=cfg.real_h_kv,
            d_h=cfg.real_d_h, s=cfg.real_s, V=cfg.real_V,
            r=cfg.real_r, block_size=cfg.real_block_size, lm_b=cfg.real_lm_b,
            bytes_dtype=cfg.bytes_dtype,
        )
        for m in MODES
    }


def run_paper_cost_model(
    *, cfg: Optional[PaperCostModelConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = PaperCostModelConfig()
    tiny = _measure_tiny(cfg)
    real = _measure_real(cfg)
    report = {
        "status": "ok",
        "stage": "7.7f",
        "main_mode": "paper_cost_model",
        "device": "cpu",
        "dtype": f"bytes={cfg.bytes_dtype}",
        "config_tiny": {
            "L": cfg.tiny_L, "d": cfg.tiny_d, "h": cfg.tiny_h,
            "h_kv": cfg.tiny_h_kv, "d_h": cfg.tiny_d_h, "s": cfg.tiny_s,
            "V": cfg.tiny_V, "r": cfg.tiny_r,
            "block_size": cfg.tiny_block_size, "lm_b": cfg.tiny_lm_b,
        },
        "config_real": {
            "L": cfg.real_L, "d": cfg.real_d, "h": cfg.real_h,
            "h_kv": cfg.real_h_kv, "d_h": cfg.real_d_h, "s": cfg.real_s,
            "V": cfg.real_V, "r": cfg.real_r,
            "block_size": cfg.real_block_size, "lm_b": cfg.real_lm_b,
        },
        "symbolic_formulas": SYMBOLIC_FORMULAS,
        "modes_evaluated": list(MODES),
        "tiny_config_counts": tiny,
        "real_config_estimates": real,
        "real_gpu_wall_clock_measured": False,
        "real_tee_wall_clock_measured": False,
        "limitations": [
            "CPU local emulation only; no real wall-clock measurement.",
            "All numbers are FLOP / byte estimates, NOT real timings.",
            "Real GPU / TEE deployment cost is not modelled.",
            "Memory-bandwidth, kernel launch overhead, network "
            "round-trip latency are NOT modelled.",
            "LoRA training (backward pass) is NOT modelled.",
            "Not formal cryptographic / semantic / differential-"
            "privacy security.",
        ],
        "paper_safe_wording": (
            "We provide symbolic and tiny / real-config FLOP and "
            "storage estimates for every protocol mode. These are "
            "complexity-model evidence only; no real GPU / TEE "
            "wall-clock is measured."
        ),
        "unsafe_wording_to_avoid": [
            "Measured real GPU/TEE performance.",
            "Wall-clock latency benchmark.",
            "Throughput benchmark.",
            "This is formal cryptographic security.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Paper Complexity & Cost Model")
    w()
    w(
        "_Stage 7.7f: symbolic and tiny / real-config estimates of "
        "FLOPs, transfers, and storage for every protocol mode. No "
        "real wall-clock measurement._"
    )
    w()
    w("## Symbolic Formulas")
    w()
    for name, body in report["symbolic_formulas"].items():
        w(f"### {name.replace('_', ' ').title()}")
        w()
        w("```")
        w(body)
        w("```")
        w()

    for label, key in (
        ("Tiny Config (synthetic decoder)", "tiny_config_counts"),
        ("Real Config Estimates (LLaMA-7B-ish)", "real_config_estimates"),
    ):
        cfg = report["config_tiny" if "Tiny" in label else "config_real"]
        w(f"## {label}")
        w()
        w(
            "| Param | "
            + " | ".join(str(k) for k in cfg.keys()) + " |"
        )
        w("|---" * (1 + len(cfg)) + "|")
        w(
            "| value | "
            + " | ".join(str(v) for v in cfg.values()) + " |"
        )
        w()
        w(
            "| Mode | round_trips | "
            "intermediate_tee_reentry | trusted_ops | "
            "accel_ops | mask_storage_bytes | "
            "lm_head_mask_overhead_bytes | asymptotic |"
        )
        w("|---|---|---|---|---|---|---|---|")
        for m in report["modes_evaluated"]:
            r = report[key][m]
            w(
                f"| `{m}` | {r['round_trips_per_decode_step']} | "
                f"{r['intermediate_tee_reentry']} | "
                f"{r['trusted_compute_ops']} | "
                f"{r['accelerator_compute_ops']} | "
                f"{r['mask_storage_bytes']} | "
                f"{r['lm_head_mask_overhead_bytes']} | "
                f"{r['asymptotic_complexity']} |"
            )
        w()

    w("## Notes Per Mode")
    w()
    for m in report["modes_evaluated"]:
        note = report["real_config_estimates"][m].get("notes", "")
        w(f"- `{m}` — {note}")
    w()

    w("## Limitations")
    w()
    for x in report["limitations"]:
        w(f"- {x}")
    w()
    w("## Paper-Safe Wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()
    w("## Unsafe Wording to Avoid")
    w()
    for x in report["unsafe_wording_to_avoid"]:
        w(f"- {x}")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: Dict[str, Any], *, outputs_dir: Path,
    json_filename: str = "paper_cost_model.json",
    md_filename: str = "paper_cost_model.md",
) -> Tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = outputs_dir / json_filename
    md_path = outputs_dir / md_filename
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "MODES",
    "PaperCostModelConfig",
    "render_markdown",
    "run_paper_cost_model",
    "write_reports",
]
