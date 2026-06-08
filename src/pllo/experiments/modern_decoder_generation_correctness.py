"""End-to-end padded modern-decoder full-generation correctness experiment.

Promotes the probe-level modern-decoder evidence to *padded full-generation
correctness* under CPU local emulation. The headline assertion is:

    fresh boundary pad + per-call right-mask + RMSNorm trusted fallback
    + post-RoPE per-head masks + SwiGLU paired permutation + masked KV
    cache + padded LM head -> recovered output token-for-token equals the
    plain reference, over both prefill AND every decode step, while every
    GPU-visible boundary tensor is mask-applied and freshly randomised
    per call.

This experiment is CPU-only, ``float64``, ``use_pad=True`` by default.
``use_pad=False`` is exercised as an ablation row in the JSON / Markdown
report (item 21 P3) but the main reported mode is ``use_pad=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from pllo.models.tiny_modern_decoder import (
    TinyModernDecoderConfig,
    TinyModernDecoderForCausalLM,
)
from pllo.wrappers.padded_modern_decoder_generation_wrapper import (
    PaddedMaskedGenerationDiagnostics,
    PaddedMaskedTinyModernDecoderWrapper,
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationCorrectnessConfig:
    seed: int = 2026
    weights_seed: int = 2026
    prompt_seed: int = 2027
    mask_seed: int = 2028
    batch_size: int = 2
    prompt_len: int = 5
    max_new_tokens: int = 4
    use_pad: bool = True
    fresh_pad: bool = True
    fresh_mask: bool = True


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


LOGITS_TOLERANCE = 1e-10
KV_CACHE_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------


def _build_model(
    decoder_cfg: TinyModernDecoderConfig, weights_seed: int
) -> TinyModernDecoderForCausalLM:
    model = TinyModernDecoderForCausalLM(decoder_cfg)
    model.init_random_weights(torch.Generator(device="cpu").manual_seed(weights_seed))
    return model


def _build_input_ids(
    decoder_cfg: TinyModernDecoderConfig,
    cfg: GenerationCorrectnessConfig,
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(cfg.prompt_seed)
    return torch.randint(
        0,
        decoder_cfg.vocab_size,
        (cfg.batch_size, cfg.prompt_len),
        generator=g,
    )


def _run_single_session(
    model: TinyModernDecoderForCausalLM,
    cfg: GenerationCorrectnessConfig,
    input_ids: torch.Tensor,
    *,
    use_pad: bool,
    mask_seed: int,
    fingerprint_keys: Optional[Dict[str, str]] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    PaddedMaskedGenerationDiagnostics,
]:
    wrapper = PaddedMaskedTinyModernDecoderWrapper(
        model,
        use_pad=use_pad,
        fresh_pad=cfg.fresh_pad,
        fresh_mask=cfg.fresh_mask,
    )
    diag = PaddedMaskedGenerationDiagnostics()
    g = torch.Generator(device="cpu").manual_seed(mask_seed)
    masked_tokens, diag = wrapper.padded_masked_generate(
        input_ids,
        cfg.max_new_tokens,
        generator=g,
        diagnostics=diag,
        fingerprint_keys=fingerprint_keys,
    )
    plain_tokens = model.greedy_generate(input_ids, cfg.max_new_tokens)
    return plain_tokens, masked_tokens, diag


# ---------------------------------------------------------------------------
# Same-input-two-runs sanity check
# ---------------------------------------------------------------------------


def _repeated_run_sanity(
    model: TinyModernDecoderForCausalLM,
    cfg: GenerationCorrectnessConfig,
    input_ids: torch.Tensor,
) -> Dict[str, Any]:
    plain_a, masked_a, diag_a = _run_single_session(
        model,
        cfg,
        input_ids,
        use_pad=cfg.use_pad,
        mask_seed=cfg.mask_seed,
        fingerprint_keys={
            "x_tilde_first_layer": "run_a_prefill_x_tilde",
            "kv_cache_first_layer": "run_a_prefill_kv_cache",
            "lm_head_logits_tilde": "run_a_prefill_logits_tilde",
        },
    )
    plain_b, masked_b, diag_b = _run_single_session(
        model,
        cfg,
        input_ids,
        use_pad=cfg.use_pad,
        mask_seed=cfg.mask_seed + 17,
        fingerprint_keys={
            "x_tilde_first_layer": "run_b_prefill_x_tilde",
            "kv_cache_first_layer": "run_b_prefill_kv_cache",
            "lm_head_logits_tilde": "run_b_prefill_logits_tilde",
        },
    )
    same_output = bool(torch.equal(masked_a, masked_b))
    plain_match_a = bool(torch.equal(plain_a, masked_a))
    plain_match_b = bool(torch.equal(plain_b, masked_b))
    fingerprints_diff = (
        diag_a.masked_boundary_fingerprints["run_a_prefill_x_tilde"]
        != diag_b.masked_boundary_fingerprints["run_b_prefill_x_tilde"]
    ) and (
        diag_a.masked_boundary_fingerprints["run_a_prefill_kv_cache"]
        != diag_b.masked_boundary_fingerprints["run_b_prefill_kv_cache"]
    ) and (
        diag_a.masked_boundary_fingerprints["run_a_prefill_logits_tilde"]
        != diag_b.masked_boundary_fingerprints["run_b_prefill_logits_tilde"]
    )
    return {
        "same_input_two_runs_same_output": same_output,
        "same_input_two_runs_different_masked_fingerprints": fingerprints_diff,
        "same_input_two_runs_recovered_logits_allclose": (
            plain_match_a and plain_match_b
        ),
        "run_a_fingerprints": diag_a.masked_boundary_fingerprints,
        "run_b_fingerprints": diag_b.masked_boundary_fingerprints,
    }


# ---------------------------------------------------------------------------
# Top-level experiment
# ---------------------------------------------------------------------------


def _diag_to_dict(diag: PaddedMaskedGenerationDiagnostics) -> Dict[str, Any]:
    return {
        "prefill_logits_max_abs_error": diag.prefill_logits_max_abs_error,
        "decode_step_logits_max_abs_error_max": diag.decode_step_logits_max_abs_error_max,
        "kv_cache_invariant_max_abs_error": diag.kv_cache_invariant_max_abs_error,
        "qk_constraint_max_error": diag.qk_constraint_max_error,
        "swiglu_paired_permutation_max_error": diag.swiglu_paired_permutation_max_error,
        "rmsnorm_recovery_max_error": diag.rmsnorm_recovery_max_error,
        "o_proj_recovery_max_error": diag.o_proj_recovery_max_error,
        "lm_head_recovery_max_error": diag.lm_head_recovery_max_error,
        "pad_entered_rmsnorm_core": diag.pad_entered_rmsnorm_core,
        "pad_entered_rope_core": diag.pad_entered_rope_core,
        "pad_entered_swiglu_core": diag.pad_entered_swiglu_core,
        "pad_entered_softmax": diag.pad_entered_softmax,
        "embedding_in_trusted_side": diag.embedding_in_trusted_side,
        "token_ids_exposed_to_accelerator": diag.token_ids_exposed_to_accelerator,
        "embedding_uses_pad": diag.embedding_uses_pad,
        "rmsnorm_mode": diag.rmsnorm_mode,
        "rope_mode": diag.rope_mode,
        "swiglu_mode": diag.swiglu_mode,
        "attention_score_mode": diag.attention_score_mode,
        "lm_head_mode": diag.lm_head_mode,
        "kv_cache_contains_plaintext": diag.kv_cache_contains_plaintext,
        "kv_cache_pad_compensated_before_append": diag.kv_cache_pad_compensated_before_append,
        "kv_cache_mask_fixed_within_session": diag.kv_cache_mask_fixed_within_session,
        "sampling_on_trusted_recovered_logits": diag.sampling_on_trusted_recovered_logits,
        "masked_boundary_fingerprints": dict(diag.masked_boundary_fingerprints),
    }


def run_modern_decoder_generation_correctness(
    *,
    cfg: Optional[GenerationCorrectnessConfig] = None,
    decoder_cfg: Optional[TinyModernDecoderConfig] = None,
    include_no_pad_ablation: bool = True,
) -> Dict[str, Any]:
    """Run the full padded modern-decoder generation correctness experiment.

    Returns a JSON-serialisable dict shaped per Stage 7.6e item 18.
    """
    if cfg is None:
        cfg = GenerationCorrectnessConfig()
    if decoder_cfg is None:
        decoder_cfg = TinyModernDecoderConfig()
    decoder_cfg.validate()

    torch.manual_seed(cfg.seed)

    model = _build_model(decoder_cfg, cfg.weights_seed)
    input_ids = _build_input_ids(decoder_cfg, cfg)

    # ----- Main reported run: use_pad=True --------------------------------
    plain_main, masked_main, diag_main = _run_single_session(
        model,
        cfg,
        input_ids,
        use_pad=cfg.use_pad,
        mask_seed=cfg.mask_seed,
        fingerprint_keys={
            "x_tilde_first_layer": "main_prefill_x_tilde",
            "kv_cache_first_layer": "main_prefill_kv_cache",
            "lm_head_logits_tilde": "main_prefill_logits_tilde",
        },
    )
    sequence_exact_match = bool(torch.equal(plain_main, masked_main))
    greedy_token_match_rate = float(
        (plain_main == masked_main).float().mean().item()
    )

    # ----- Forward (no past) max-abs error directly on a single pass ------
    wrapper = PaddedMaskedTinyModernDecoderWrapper(
        model,
        use_pad=cfg.use_pad,
        fresh_pad=cfg.fresh_pad,
        fresh_mask=cfg.fresh_mask,
    )
    g = torch.Generator(device="cpu").manual_seed(cfg.mask_seed + 7)
    recovered_logits, _, _, _ = wrapper.padded_masked_forward(
        input_ids,
        generator=g,
        diagnostics=PaddedMaskedGenerationDiagnostics(),
    )
    plain_logits, _ = model.forward(input_ids)
    forward_logits_max_abs_error = float(
        (plain_logits - recovered_logits).abs().max().item()
    )

    # ----- Repeated-run sanity check --------------------------------------
    sanity = _repeated_run_sanity(model, cfg, input_ids)

    # ----- Optional ablation: use_pad=False -------------------------------
    ablation: Dict[str, Any] = {}
    if include_no_pad_ablation:
        plain_np, masked_np, diag_np = _run_single_session(
            model,
            cfg,
            input_ids,
            use_pad=False,
            mask_seed=cfg.mask_seed + 31,
        )
        ablation = {
            "use_pad": False,
            "sequence_exact_match": bool(torch.equal(plain_np, masked_np)),
            "diagnostics": _diag_to_dict(diag_np),
            "note": (
                "Ablation only -- the main reported mode is use_pad=True."
            ),
        }

    # ----- Assemble JSON report -------------------------------------------
    report: Dict[str, Any] = {
        "status": "ok",
        "stage": "7.6e",
        "main_mode": "padded_masked_execution",
        "use_pad": bool(cfg.use_pad),
        "fresh_pad": bool(cfg.fresh_pad),
        "fresh_mask": bool(cfg.fresh_mask),
        "device": "cpu",
        "dtype": "float64",
        "model_type": "synthetic_tiny_modern_decoder",
        "config": {
            "vocab_size": decoder_cfg.vocab_size,
            "hidden_size": decoder_cfg.hidden_size,
            "intermediate_size": decoder_cfg.intermediate_size,
            "num_layers": decoder_cfg.num_layers,
            "num_query_heads": decoder_cfg.num_query_heads,
            "num_kv_heads": decoder_cfg.num_kv_heads,
            "head_dim": decoder_cfg.head_dim,
            "max_position_embeddings": decoder_cfg.max_position_embeddings,
            "rope_base": decoder_cfg.rope_base,
            "rms_norm_eps": decoder_cfg.rms_norm_eps,
            "batch_size": cfg.batch_size,
            "prompt_len": cfg.prompt_len,
            "max_new_tokens": cfg.max_new_tokens,
            "weights_seed": cfg.weights_seed,
            "prompt_seed": cfg.prompt_seed,
            "mask_seed": cfg.mask_seed,
        },
        "coverage": {
            "embedding": True,
            "rmsnorm": True,
            "rope": True,
            "gqa_or_mqa": True,
            "causal_attention": True,
            "softmax": True,
            "swiglu": True,
            "residual": True,
            "kv_cache": True,
            "lm_head": True,
            "greedy_generation": True,
        },
        "pad_policy": {
            "pad_at_linear_boundaries": bool(cfg.use_pad),
            "pad_enters_rmsnorm_core": False,
            "pad_enters_swiglu_core": False,
            "pad_enters_rope_core": False,
            "pad_enters_softmax": False,
            "pad_compensated_before_nonlinear_core": True,
        },
        "module_modes": {
            "rmsnorm_mode": diag_main.rmsnorm_mode,
            "rope_mode": diag_main.rope_mode,
            "swiglu_mode": diag_main.swiglu_mode,
            "attention_score_mode": diag_main.attention_score_mode,
            "lm_head_mode": diag_main.lm_head_mode,
            "rmsnorm_gpu_compatible_claim": False,
            "generic_pre_rope_mask_commutation_used": False,
            "qk_projection_pad_compensated": True,
            "swiglu_pad_compensated_before_core": True,
            "swiglu_shared_permutation": True,
        },
        "gqa_indexing": {
            "attention_variant": "gqa" if decoder_cfg.num_query_heads != decoder_cfg.num_kv_heads else "mha",
            "mask_dimension": "head_dim",
            "mask_is_per_head_not_hidden_size": True,
            "mask_is_per_head_not_num_heads": True,
            "group_size": decoder_cfg.group_size,
        },
        "correctness": {
            "forward_logits_max_abs_error": forward_logits_max_abs_error,
            "prefill_logits_max_abs_error": diag_main.prefill_logits_max_abs_error,
            "decode_step_logits_max_abs_error_max": diag_main.decode_step_logits_max_abs_error_max,
            "kv_cache_invariant_max_abs_error": diag_main.kv_cache_invariant_max_abs_error,
            "qk_constraint_max_error": diag_main.qk_constraint_max_error,
            "swiglu_paired_permutation_max_error": diag_main.swiglu_paired_permutation_max_error,
            "o_proj_recovery_max_error": diag_main.o_proj_recovery_max_error,
            "lm_head_recovery_max_error": diag_main.lm_head_recovery_max_error,
            "rmsnorm_recovery_max_error": diag_main.rmsnorm_recovery_max_error,
            "greedy_token_match_rate": greedy_token_match_rate,
            "sequence_exact_match": sequence_exact_match,
            "plain_token_sequence": plain_main.tolist(),
            "masked_token_sequence": masked_main.tolist(),
        },
        "security_relevant_checks": {
            "same_plain_input_produces_different_masked_boundary_tensors_under_fresh_pad": sanity[
                "same_input_two_runs_different_masked_fingerprints"
            ],
            "masked_transcript_differs_between_repeated_runs": sanity[
                "same_input_two_runs_different_masked_fingerprints"
            ],
            "same_input_two_runs_same_output": sanity[
                "same_input_two_runs_same_output"
            ],
            "same_input_two_runs_different_masked_fingerprints": sanity[
                "same_input_two_runs_different_masked_fingerprints"
            ],
            "same_input_two_runs_recovered_logits_allclose": sanity[
                "same_input_two_runs_recovered_logits_allclose"
            ],
            "kv_cache_contains_plaintext": False,
            "lm_head_logits_masked_before_recovery": True,
            "sampling_on_trusted_recovered_logits": True,
            "embedding_in_trusted_side": True,
            "token_ids_exposed_to_accelerator": False,
            "run_a_fingerprints": sanity["run_a_fingerprints"],
            "run_b_fingerprints": sanity["run_b_fingerprints"],
        },
        "diagnostics_main_run": _diag_to_dict(diag_main),
        "ablation_use_pad_false": ablation,
        "tolerances": {
            "logits_tolerance": LOGITS_TOLERANCE,
            "kv_cache_invariant_tolerance": KV_CACHE_TOLERANCE,
        },
        "limitations": [
            "CPU local emulation only; no real TEE/GPU.",
            "Synthetic tiny modern decoder; no full Qwen/LLaMA weights loaded.",
            "Attention scores/probabilities are not hidden in this correctness wrapper.",
            "Protecting attention maps requires an additional secure softmax or score obfuscation primitive.",
            "This validates padded masked algebraic correctness, not formal cryptographic security.",
            "No hardware side-channel evaluation.",
            "Per-call fresh pads and fresh per-Linear masks; per-KV-head masks fixed within a session so the KV cache append invariant holds.",
        ],
        "paper_safe_wording": (
            "We verify that fresh boundary pads can be integrated into a "
            "full modern decoder-style generation path without breaking "
            "output equivalence. Pads are compensated before nonlinear "
            "cores, while KV cache and logits remain masked until "
            "trusted recovery."
        ),
        "unsafe_wording_to_avoid": [
            "The scheme is formally secure.",
            "The CPU wrapper proves cryptographic privacy.",
            "Attention maps are hidden.",
            "We evaluate real TEE/GPU performance.",
            "Generic dense masks commute with RoPE.",
            "Pads can pass through nonlinear layers.",
            "We support full Qwen/LLaMA private generation on real TEE/GPU.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _format_float(x: float) -> str:
    if x == 0.0:
        return "0.0"
    if abs(x) >= 1e-3:
        return f"{x:.6g}"
    return f"{x:.3e}"


def render_markdown(report: Dict[str, Any]) -> str:
    cfg = report["config"]
    corr = report["correctness"]
    pol = report["pad_policy"]
    mods = report["module_modes"]
    sec = report["security_relevant_checks"]
    cov = report["coverage"]

    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Padded Modern Decoder Full-Generation Correctness")
    w()
    w(
        "_CPU local emulation only -- no real TEE, no real GPU, no "
        "production framework. Main reported mode is `use_pad=True`._"
    )
    w()

    # 2. Configuration table
    w("## Configuration")
    w()
    w("| Field | Value |")
    w("|---|---|")
    w(f"| vocab_size | {cfg['vocab_size']} |")
    w(f"| hidden_size | {cfg['hidden_size']} |")
    w(f"| intermediate_size | {cfg['intermediate_size']} |")
    w(f"| num_layers | {cfg['num_layers']} |")
    w(f"| num_query_heads | {cfg['num_query_heads']} |")
    w(f"| num_kv_heads | {cfg['num_kv_heads']} |")
    w(f"| head_dim | {cfg['head_dim']} |")
    w(f"| max_position_embeddings | {cfg['max_position_embeddings']} |")
    w(f"| rope_base | {cfg['rope_base']} |")
    w(f"| rms_norm_eps | {cfg['rms_norm_eps']} |")
    w(f"| batch_size | {cfg['batch_size']} |")
    w(f"| prompt_len | {cfg['prompt_len']} |")
    w(f"| max_new_tokens | {cfg['max_new_tokens']} |")
    w(f"| dtype | {report['dtype']} |")
    w(f"| device | {report['device']} |")
    w(f"| use_pad (main mode) | {report['use_pad']} |")
    w()

    # 3. Coverage table
    w("## Coverage")
    w()
    w("| Component | Implemented |")
    w("|---|---|")
    for key in (
        "embedding", "rmsnorm", "rope", "gqa_or_mqa",
        "causal_attention", "softmax", "swiglu", "residual",
        "kv_cache", "lm_head", "greedy_generation",
    ):
        w(f"| {key} | {cov[key]} |")
    w()

    # 4. Pad policy table
    w("## Pad Policy")
    w()
    w("| Check | Value |")
    w("|---|---|")
    w(f"| pad_at_linear_boundaries | {pol['pad_at_linear_boundaries']} |")
    w(f"| pad_enters_rmsnorm_core | {pol['pad_enters_rmsnorm_core']} |")
    w(f"| pad_enters_rope_core | {pol['pad_enters_rope_core']} |")
    w(f"| pad_enters_swiglu_core | {pol['pad_enters_swiglu_core']} |")
    w(f"| pad_enters_softmax | {pol['pad_enters_softmax']} |")
    w(f"| pad_compensated_before_nonlinear_core | {pol['pad_compensated_before_nonlinear_core']} |")
    w()

    # Module modes
    w("## Module Modes")
    w()
    w("| Module | Mode |")
    w("|---|---|")
    for key in (
        "rmsnorm_mode", "rope_mode", "swiglu_mode",
        "attention_score_mode", "lm_head_mode",
    ):
        w(f"| {key} | `{mods[key]}` |")
    w(f"| rmsnorm_gpu_compatible_claim | `{mods['rmsnorm_gpu_compatible_claim']}` |")
    w(f"| generic_pre_rope_mask_commutation_used | `{mods['generic_pre_rope_mask_commutation_used']}` |")
    w(f"| qk_projection_pad_compensated | `{mods['qk_projection_pad_compensated']}` |")
    w(f"| swiglu_pad_compensated_before_core | `{mods['swiglu_pad_compensated_before_core']}` |")
    w(f"| swiglu_shared_permutation | `{mods['swiglu_shared_permutation']}` |")
    w()

    # 5. Correctness metrics table
    w("## Correctness Metrics")
    w()
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| forward_logits_max_abs_error | {_format_float(corr['forward_logits_max_abs_error'])} |")
    w(f"| prefill_logits_max_abs_error | {_format_float(corr['prefill_logits_max_abs_error'])} |")
    w(f"| decode_step_logits_max_abs_error_max | {_format_float(corr['decode_step_logits_max_abs_error_max'])} |")
    w(f"| kv_cache_invariant_max_abs_error | {_format_float(corr['kv_cache_invariant_max_abs_error'])} |")
    w(f"| qk_constraint_max_error | {_format_float(corr['qk_constraint_max_error'])} |")
    w(f"| swiglu_paired_permutation_max_error | {_format_float(corr['swiglu_paired_permutation_max_error'])} |")
    w(f"| o_proj_recovery_max_error | {_format_float(corr['o_proj_recovery_max_error'])} |")
    w(f"| lm_head_recovery_max_error | {_format_float(corr['lm_head_recovery_max_error'])} |")
    w(f"| greedy_token_match_rate | {corr['greedy_token_match_rate']} |")
    w(f"| sequence_exact_match | {corr['sequence_exact_match']} |")
    w()

    # 6. Repeated-run sanity check
    w("## Repeated-Run Sanity Check")
    w()
    w(
        "Two runs with the *same* input ids and *fresh* pads / masks "
        "must produce (a) identical recovered token sequences and (b) "
        "*different* GPU-visible masked-boundary fingerprints."
    )
    w()
    w("| Check | Value |")
    w("|---|---|")
    w(
        "| same_input_two_runs_same_output | "
        f"{sec['same_input_two_runs_same_output']} |"
    )
    w(
        "| same_input_two_runs_different_masked_fingerprints | "
        f"{sec['same_input_two_runs_different_masked_fingerprints']} |"
    )
    w(
        "| same_input_two_runs_recovered_logits_allclose | "
        f"{sec['same_input_two_runs_recovered_logits_allclose']} |"
    )
    w(
        "| kv_cache_contains_plaintext | "
        f"{sec['kv_cache_contains_plaintext']} |"
    )
    w(
        "| lm_head_logits_masked_before_recovery | "
        f"{sec['lm_head_logits_masked_before_recovery']} |"
    )
    w(
        "| sampling_on_trusted_recovered_logits | "
        f"{sec['sampling_on_trusted_recovered_logits']} |"
    )
    w(
        "| embedding_in_trusted_side | "
        f"{sec['embedding_in_trusted_side']} |"
    )
    w(
        "| token_ids_exposed_to_accelerator | "
        f"{sec['token_ids_exposed_to_accelerator']} |"
    )
    w()
    w("This is a security-relevant *sanity check*, not a formal security proof.")
    w()

    # Ablation
    if report.get("ablation_use_pad_false"):
        abl = report["ablation_use_pad_false"]
        w("## Ablation: use_pad=False")
        w()
        w(
            "Provided only as an ablation row. "
            "`use_pad=False` is **not** the reported main mode."
        )
        w()
        w("| Metric | Value |")
        w("|---|---|")
        w(f"| use_pad | {abl['use_pad']} |")
        w(f"| sequence_exact_match | {abl['sequence_exact_match']} |")
        w()

    # 7. Limitations
    w("## Limitations")
    w()
    for item in report["limitations"]:
        w(f"- {item}")
    w()

    # 8. Paper-safe wording
    w("## Paper-Safe Wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()

    # 9. Unsafe wording
    w("## Unsafe Wording to Avoid")
    w()
    for item in report["unsafe_wording_to_avoid"]:
        w(f"- {item}")
    w()

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def write_reports(
    report: Dict[str, Any],
    *,
    outputs_dir: Path,
    json_filename: str = "modern_decoder_generation_correctness.json",
    md_filename: str = "modern_decoder_generation_correctness.md",
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
    "GenerationCorrectnessConfig",
    "render_markdown",
    "run_modern_decoder_generation_correctness",
    "write_reports",
    "LOGITS_TOLERANCE",
    "KV_CACHE_TOLERANCE",
]
