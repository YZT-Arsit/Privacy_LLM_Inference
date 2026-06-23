"""Stage 8.2 -- real ModelScope checkpoint GPU experiment.

Runs bounded, real-model experiments (Qwen2.5 0.5B -> 7B) using ModelScope
checkpoints (never Hugging Face remote download). For each checkpoint it
evaluates three paths over a short prefill + bounded greedy decode horizon:

1. plain HF/ModelScope ``generate`` baseline (diagnostic);
2. our extracted-weight plaintext reference (Stage 6.9 path, possibly a
   partial-layer stack when ``max_layers`` < total);
3. the masked runtime (trusted embedding boundary -> masked decoder -> masked
   logits -> trusted recovery + greedy), with a *simulated* TEE only.

Real checkpoints use bf16/fp16 (never float64) and scalable, RMSNorm-compatible
residual masks (``signed_permutation`` by default; shared residual mask by
default). All of these are honestly weaker than dense orthogonal / per-layer
masking. No semantic, cryptographic, or formal security is claimed; attention
scores remain GPU-visible. Reports are compact (no tensor dumps).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from pllo.hf_wrappers.hf_causal_lm_skeleton import (
    NEGATIVE_CONTROLS,
    HFCausalLMSkeletonConfig,
    extract_hf_causal_lm_skeleton_weights,
    generate_hf_causal_lm_masks,
    has_transformers,
    hf_causal_lm_masked_greedy_decode,
    hf_causal_lm_plain_decode,
    hf_causal_lm_plain_prefill,
)

__all__ = [
    "ModelScopeRealCheckpointProbeConfig",
    "REQUIRED_STATEMENT",
    "build_attention_mask",
    "build_caveats",
    "build_probe_inputs",
    "has_modelscope",
    "load_prompts",
    "load_modelscope_checkpoint",
    "resolve_dtype",
    "run_modelscope_real_checkpoint_probe",
]


_DTYPE_BYTES = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4,
                torch.float64: 8}

DEFAULT_PROMPT = "Hello, please answer briefly:"

REQUIRED_STATEMENT = (
    "This stage runs bounded real ModelScope checkpoints through a masked "
    "runtime with a simulated TEE only. It does not use real TEE hardware, "
    "does not validate production generation, and claims no semantic, "
    "cryptographic, or formal security."
)


@dataclass
class ModelScopeRealCheckpointProbeConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    cache_dir: str = "/root/modelscope_cache"
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = 1
    prefill_seq_len: int = 16
    decode_steps: int = 8
    max_layers: int | str = 1
    mask_mode: str = "signed_permutation"
    residual_mask_strategy: str = "shared"
    block_size: int = 64
    allow_dense_large_mask: bool = False
    # Mixed-precision knobs (Stage 8.2 bf16). ``dtype`` governs the model load
    # + HF baseline; the masked correctness pipeline folds/recovers/compares in
    # the higher precisions below to avoid bf16 inverse/scaling drift.
    folding_dtype: str = "float32"
    folded_weight_runtime_dtype: str = "float32"
    recovery_dtype: str = "float32"
    compare_dtype: str = "float32"
    run_hf_baseline: bool = True
    run_extracted_plain: bool = True
    run_masked_runtime: bool = True
    compare_hf_forward: bool = False
    write_tensor_dumps: bool = False
    max_report_mb: int = 10
    seed: int = 2035
    # Real-input + audit knobs (Stage 8.2 audit). ``prompt_file`` (JSONL of
    # ``{"id","prompt"}``) takes precedence over ``prompt`` (a single literal
    # prompt); both are tokenized with the real ModelScope checkpoint tokenizer.
    # With neither set the probe falls back to synthetic tokens.
    prompt: str | None = None
    prompt_file: str | None = None
    include_prompts_in_report: bool = False
    negative_control: str = "none"  # none|wrong_vocab_recovery|plaintext_weights_on_masked_hidden


# ---------------------------------------------------------------------------
# Optional dependency + environment helpers
# ---------------------------------------------------------------------------


def has_modelscope() -> bool:
    return importlib.util.find_spec("modelscope") is not None


def resolve_dtype(name: str, device: str) -> torch.dtype:
    """bf16 preferred (if supported), else fp16; fp32 only when explicitly
    requested or on CPU. Never float64 for real checkpoints."""
    name = (name or "").lower()
    if name in ("float32", "fp32", "f32"):
        return torch.float32
    if name in ("float16", "fp16", "half", "f16"):
        return torch.float16
    # bfloat16 path (default)
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    # CPU: bf16 matmul is poorly supported -> fall back to fp32 diagnostics.
    return torch.float32


def _nvidia_smi() -> dict[str, Any] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for ln in out.stdout.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 4:
            rows.append({"name": parts[0], "memory_total_mb": parts[1],
                         "memory_used_mb": parts[2], "memory_free_mb": parts[3]})
    return rows or None


def _collect_environment(device: str) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    env: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_requested": device,
        "has_transformers": has_transformers(),
        "has_modelscope": has_modelscope(),
        "nvidia_smi": _nvidia_smi(),
    }
    if cuda:
        try:
            env["device_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            env["vram_free_mb"] = round(free / 2 ** 20, 1)
            env["vram_total_mb"] = round(total / 2 ** 20, 1)
            env["bf16_supported"] = torch.cuda.is_bf16_supported()
        except Exception:
            pass
    return env


def _reset_peak(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_mem(device: str) -> dict[str, Any] | None:
    if device != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 2 ** 20, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / 2 ** 20, 2),
        "max_allocated_mb": round(
            torch.cuda.max_memory_allocated() / 2 ** 20, 2),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20, 2),
    }


# ---------------------------------------------------------------------------
# ModelScope loading (no HF remote download)
# ---------------------------------------------------------------------------


def load_modelscope_checkpoint(
    model_id: str, cache_dir: str, dtype: torch.dtype, device: str,
) -> dict[str, Any]:
    """Download (ModelScope only) + locally load a checkpoint.

    Returns ``{"status": "ok", "model", "tokenizer", "local_path"}`` or a
    clean ``skipped_*`` status. Never uses Hugging Face remote download.
    """
    if not has_transformers():
        return {"status": "skipped_transformers_unavailable"}
    if not has_modelscope():
        return {"status": "skipped_modelscope_unavailable"}
    try:
        from modelscope import snapshot_download
        local_path = snapshot_download(model_id, cache_dir=cache_dir)
    except Exception as exc:  # network / auth / disk
        return {"status": "skipped_modelscope_download_failed",
                "reason": f"{type(exc).__name__}: {exc}"}
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            local_path, local_files_only=True, dtype=dtype,
            trust_remote_code=True, device_map=None)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            local_path, local_files_only=True, trust_remote_code=True)
    except Exception as exc:
        return {"status": "skipped_model_load_failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "local_path": local_path}
    moved_to_cuda = False
    if device == "cuda" and torch.cuda.is_available():
        try:
            model.to("cuda")
            moved_to_cuda = True
        except RuntimeError as exc:  # OOM on move
            return {"status": "skipped_oom_on_model_move",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "local_path": local_path}
    return {"status": "ok", "model": model, "tokenizer": tokenizer,
            "local_path": local_path, "moved_to_cuda": moved_to_cuda}


# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------


def build_caveats(config: ModelScopeRealCheckpointProbeConfig,
                  partial_layers: bool) -> list[str]:
    caveats = [
        "Simulated TEE only -- no real TEE hardware is used.",
        f"mask_mode={config.mask_mode}: scalable + RMSNorm-compatible but "
        "weaker than dense orthogonal masking.",
        f"residual_mask_strategy={config.residual_mask_strategy}: a shared "
        "residual mask is weaker than per-layer residual masks.",
        "Attention scores / probabilities remain GPU-visible.",
        "No semantic, cryptographic, or formal security is claimed.",
        "Extracted-weight reference (adjacent-pair RoPE); HF forward parity "
        "is diagnostic only (RoPE/cache conventions differ).",
    ]
    if partial_layers:
        caveats.append(
            "max_layers < total: the extracted/masked stack is a PARTIAL "
            "model; HF baseline runs the full model, so HF tokens need not "
            "match the partial-stack tokens.")
    return caveats


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def _dtype_bytes(dtype: torch.dtype) -> int:
    return _DTYPE_BYTES.get(dtype, 4)


def _boundary_accounting(
    config: ModelScopeRealCheckpointProbeConfig, hidden: int, vocab: int,
    dtype: torch.dtype, n_layers: int, shared: bool,
) -> dict[str, Any]:
    b = config.batch_size
    t = config.prefill_seq_len
    d = config.decode_steps
    nb = _dtype_bytes(dtype)
    tee_to_gpu = (b * t * hidden + d * b * hidden) * nb          # masked embeds
    gpu_to_tee = (1 + d) * b * vocab * nb                        # masked logits
    handoff_gemms = 0 if shared else (n_layers - 1) * (t > 0) + \
        (0 if shared else (n_layers - 1) * d)
    handoff_flops = 0 if shared else (
        (n_layers - 1) * 2 * b * t * hidden * hidden
        + (n_layers - 1) * d * 2 * b * hidden * hidden)
    recovery_flops = (1 + d) * b * vocab * 2                     # perm + scale
    return {
        "boundary_calls": 2 + 2 * d,
        "tee_to_gpu_bytes": int(tee_to_gpu),
        "gpu_to_tee_bytes": int(gpu_to_tee),
        "tee_to_gpu_mb": round(tee_to_gpu / 2 ** 20, 4),
        "gpu_to_tee_mb": round(gpu_to_tee / 2 ** 20, 4),
        "handoff_gemm_count": int(handoff_gemms),
        "handoff_gemm_flops": int(handoff_flops),
        "logits_recovery_flops": int(recovery_flops),
        "dtype_bytes": nb,
    }


def build_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Explicit all-ones attention mask for non-padded synthetic input_ids.

    The synthetic prompt has no padding, and Qwen's pad_token == eos_token, so
    transformers cannot infer the mask and warns. We pass an explicit mask of
    ones (long) on the same device, matching the no-padding assumption used by
    the extracted-plaintext and masked pipelines (pure causal attention, no
    padding positions)."""
    return torch.ones_like(input_ids, dtype=torch.long,
                           device=input_ids.device)


def _hf_baseline(model: Any, tokenizer: Any,
                 config: ModelScopeRealCheckpointProbeConfig,
                 device: str) -> dict[str, Any]:
    enc = tokenizer(DEFAULT_PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"]
    if device == "cuda" and torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    # Always pass an explicit all-ones attention mask (no padding).
    attention_mask = build_attention_mask(input_ids)
    _reset_peak(device)
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=config.decode_steps, do_sample=False,
            num_beams=1, use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None))
    _sync(device)
    latency = time.perf_counter() - t0
    new_ids = out[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {
        "prompt_token_len": int(input_ids.shape[1]),
        "new_token_ids": new_ids,
        "generated_text_head": text[:100],
        "latency_s": round(latency, 4),
        "tokens_per_s": round(config.decode_steps / latency, 3)
        if latency > 0 else None,
        "peak_cuda_memory": _cuda_mem(device),
    }


def load_prompts(
    config: ModelScopeRealCheckpointProbeConfig,
) -> tuple[list[dict[str, str]], str]:
    """Resolve the prompt source.

    Returns ``(prompts, input_source)`` where ``prompts`` is a list of
    ``{"id", "prompt"}`` dicts and ``input_source`` is one of
    ``"prompt_file"`` / ``"literal_prompt"`` / ``"synthetic_tokens"``.
    For ``synthetic_tokens`` the list is empty (no real text)."""
    if config.prompt_file:
        prompts: list[dict[str, str]] = []
        path = Path(config.prompt_file)
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prompts.append({
                    "id": str(rec.get("id", f"prompt_{i}")),
                    "prompt": str(rec["prompt"]),
                })
        if not prompts:
            raise ValueError(f"prompt-file {config.prompt_file!r} had no rows")
        return prompts, "prompt_file"
    if config.prompt:
        return [{"id": "literal_0", "prompt": config.prompt}], "literal_prompt"
    return [], "synthetic_tokens"


def _tokenize_pad_truncate(
    tokenizer: Any, prompts: list[dict[str, str]], t: int,
    pad_id: int, vocab: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Tokenize each prompt with the real tokenizer, then truncate/pad to ``t``.

    Returns ``(padded_ids, raw_ids)`` where ``raw_ids`` is the pre-pad/truncate
    tokenization (for length stats) and ``padded_ids`` is exactly length ``t``."""
    raw_ids: list[list[int]] = []
    padded_ids: list[list[int]] = []
    for rec in prompts:
        ids = list(tokenizer(rec["prompt"])["input_ids"])
        raw_ids.append(ids)
        cut = ids[:t]
        if len(cut) < t:
            cut = cut + [pad_id] * (t - len(cut))
        padded_ids.append([int(x) % vocab for x in cut])
    return padded_ids, raw_ids


def _length_stats(raw_ids: list[list[int]]) -> dict[str, Any]:
    lens = [len(x) for x in raw_ids]
    if not lens:
        return {"count": 0}
    return {
        "count": len(lens), "min": min(lens), "max": max(lens),
        "mean": round(sum(lens) / len(lens), 3),
    }


def build_probe_inputs(
    tokenizer: Any, vocab: int,
    config: ModelScopeRealCheckpointProbeConfig, device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build ``input_ids`` [N, prefill_seq_len] plus an input-audit dict.

    ``prompt_file`` / ``prompt`` modes tokenize real text with the real
    checkpoint tokenizer, pass an explicit all-ones attention mask, and
    truncate/pad to ``prefill_seq_len``. Token *content* does not affect the
    masked-vs-plain correctness invariant, but real prompts make the probe a
    real-input experiment rather than a toy random-token one."""
    t = config.prefill_seq_len
    prompts, input_source = load_prompts(config)
    pad_id = 0
    if tokenizer is not None:
        pad_id = (getattr(tokenizer, "pad_token_id", None)
                  or getattr(tokenizer, "eos_token_id", None) or 0)
    pad_id = int(pad_id) % max(vocab, 1)

    meta: dict[str, Any] = {"input_source": input_source}
    if input_source == "synthetic_tokens" or tokenizer is None:
        ids: list[int] = []
        if tokenizer is not None:
            try:
                ids = list(tokenizer(DEFAULT_PROMPT)["input_ids"])
            except Exception:
                ids = []
        raw_len = len(ids)
        if len(ids) >= t:
            ids = ids[:t]
        else:
            g = torch.Generator().manual_seed(config.seed)
            pad = torch.randint(0, vocab, (t - len(ids),),
                                generator=g).tolist()
            ids = ids + pad
        rows = [[int(x) % vocab for x in ids[:t]]]
        meta.update({
            "input_source": "synthetic_tokens",
            "prompt_count": 1,
            "tokenizer_used": False,
            "synthetic_seed_prompt_token_len": raw_len,
            "tokenized_length_stats": {"count": 1, "min": raw_len,
                                       "max": raw_len, "mean": float(raw_len)},
            "prompt_ids": rows,
        })
    else:
        rows, raw_ids = _tokenize_pad_truncate(
            tokenizer, prompts, t, pad_id, vocab)
        meta.update({
            "prompt_count": len(prompts),
            "tokenizer_used": True,
            "prompt_pad_token_id": pad_id,
            "tokenized_length_stats": _length_stats(raw_ids),
            "prompt_ids": [r[:t] for r in raw_ids],  # raw (pre-pad) token ids
        })
        if config.include_prompts_in_report:
            meta["prompts"] = prompts
            meta["prompt_id_order"] = [p["id"] for p in prompts]
        else:
            meta["prompt_id_order"] = [p["id"] for p in prompts]

    out = torch.tensor(rows, dtype=torch.long)
    attention_mask = torch.ones_like(out, dtype=torch.long)
    if device == "cuda" and torch.cuda.is_available():
        out = out.to("cuda")
        attention_mask = attention_mask.to("cuda")
    meta["input_ids_shape"] = list(out.shape)
    meta["attention_mask_shape"] = list(attention_mask.shape)
    meta["attention_mask_explicit"] = True
    meta["attention_mask_all_ones"] = bool(
        torch.equal(attention_mask, torch.ones_like(attention_mask)))
    return out, meta


def _build_audit(
    config: ModelScopeRealCheckpointProbeConfig, model_config: Any,
    dtype_names: dict[str, str], local_path: str, total_layers: int,
    layers_executed: int, extract_meta: dict[str, Any],
    mask_meta: dict[str, Any], input_meta: dict[str, Any],
    masked_ran: bool, tokenizer_used: bool,
) -> dict[str, Any]:
    """Self-describing audit block: enough metadata to confirm a real
    ModelScope checkpoint, a real masked/folded operator path, real inputs,
    and the negative-control posture -- without any tensor dumps."""
    def _cfg(name: str, default: Any = None) -> Any:
        return getattr(model_config, name, default)

    mask_mode = mask_meta.get("mask_mode", config.mask_mode)
    audit: dict[str, Any] = {
        # provenance
        "model_id": config.model_id,
        "local_checkpoint_path": local_path,
        "checkpoint_source": "ModelScope",
        "hf_remote_download_used": False,
        "tokenizer_used": bool(tokenizer_used),
        # inputs
        "input_source": input_meta.get("input_source"),
        "prompt_count": input_meta.get("prompt_count"),
        "prefill_seq_len": config.prefill_seq_len,
        "decode_steps": config.decode_steps,
        "input_ids_shape": input_meta.get("input_ids_shape"),
        "attention_mask_explicit": bool(
            input_meta.get("attention_mask_explicit", False)),
        "attention_mask_shape": input_meta.get("attention_mask_shape"),
        "attention_mask_all_ones": input_meta.get("attention_mask_all_ones"),
        "tokenized_length_stats": input_meta.get("tokenized_length_stats"),
        # model shape
        "num_hidden_layers_total": total_layers,
        "max_layers_executed": layers_executed,
        "hidden_size": int(_cfg("hidden_size", extract_meta.get("hidden_size",
                                                                0))),
        "intermediate_size": int(_cfg("intermediate_size", 0)),
        "num_attention_heads": int(_cfg("num_attention_heads", 0)),
        "num_key_value_heads": int(_cfg("num_key_value_heads",
                                        _cfg("num_attention_heads", 0))),
        "vocab_size": int(_cfg("vocab_size", extract_meta.get("vocab_size",
                                                              0))),
        # dtypes
        "dtype": dtype_names["model"],
        "folding_dtype": dtype_names["folding"],
        "folded_weight_runtime_dtype": dtype_names["folded_weight_runtime"],
        "recovery_dtype": dtype_names["recovery"],
        "compare_dtype": dtype_names["compare"],
        # masking
        "mask_mode": mask_mode,
        "residual_mask_strategy": mask_meta.get("residual_mask_strategy",
                                                config.residual_mask_strategy),
        # operator-path evidence: True only when the masked runtime actually ran
        "used_extracted_weights": bool(masked_ran),
        "used_masked_embedding_boundary": bool(masked_ran),
        "used_masked_decoder_blocks": bool(masked_ran and layers_executed > 0),
        "used_folded_qkv": bool(masked_ran and layers_executed > 0),
        "used_folded_mlp": bool(masked_ran and layers_executed > 0),
        "used_masked_kv_cache": bool(masked_ran and config.decode_steps > 0),
        "used_masked_lm_head": bool(masked_ran),
        "used_vocab_mask_recovery": bool(
            masked_ran and config.negative_control != "wrong_vocab_recovery"),
        "hf_baseline_only": False,
        "simulated_tee_only": True,
    }
    if mask_mode == "block_orthogonal":
        audit["block_size"] = config.block_size
    return audit


def run_modelscope_real_checkpoint_probe(
    config: ModelScopeRealCheckpointProbeConfig,
) -> dict[str, Any]:
    """Run the Stage 8.2 real-checkpoint probe. Compact report; no tensors."""
    if config.negative_control not in NEGATIVE_CONTROLS:
        raise ValueError(
            f"unknown negative_control {config.negative_control!r}; expected "
            f"one of {NEGATIVE_CONTROLS}")
    torch.manual_seed(config.seed)
    device = config.device
    env = _collect_environment(device)
    model_dtype = resolve_dtype(config.dtype, device)          # load + forward
    folding_dtype = resolve_dtype(config.folding_dtype, device)
    runtime_dtype = resolve_dtype(config.folded_weight_runtime_dtype, device)
    recovery_dtype = resolve_dtype(config.recovery_dtype, device)

    def _name(dt: torch.dtype) -> str:
        return str(dt).replace("torch.", "")

    base = {
        "stage": "8.2_modelscope_real_checkpoint",
        "config": asdict(config),
        "resolved_dtypes": {
            "model": _name(model_dtype), "folding": _name(folding_dtype),
            "folded_weight_runtime": _name(runtime_dtype),
            "recovery": _name(recovery_dtype),
            "compare": _name(resolve_dtype(config.compare_dtype, device)),
        },
        "resolved_dtype": _name(model_dtype),  # back-compat
        "environment": env,
        "attention_mask_explicit": True,
        "no_padding_assumption": True,
        "required_statement": REQUIRED_STATEMENT,
    }

    loaded = load_modelscope_checkpoint(config.model_id, config.cache_dir,
                                        model_dtype, device)
    if loaded["status"] != "ok":
        return {**base, "status": loaded["status"],
                "reason": loaded.get("reason"),
                "local_path": loaded.get("local_path"),
                "caveats": build_caveats(config, partial_layers=False)}

    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    model_config = model.config
    total_layers = int(getattr(model_config, "num_hidden_layers", 0))
    max_layers = (None if str(config.max_layers) == "all"
                  else int(config.max_layers))
    partial = max_layers is not None and max_layers < total_layers

    report: dict[str, Any] = {
        **base,
        "status": "ok",
        "model_id": config.model_id,
        "local_path": loaded["local_path"],
        "model_type": str(getattr(model_config, "model_type", "unknown")),
        "total_layers": total_layers,
        "max_layers": "all" if max_layers is None else max_layers,
        "partial_layer_diagnostic": partial,
        "hidden_size": int(getattr(model_config, "hidden_size", 0)),
        "vocab_size": int(getattr(model_config, "vocab_size", 0)),
        "tokenizer_loaded": tokenizer is not None,
    }

    # 1. HF baseline (full model) ----------------------------------------
    if config.run_hf_baseline:
        try:
            report["hf_baseline"] = _hf_baseline(model, tokenizer, config,
                                                 device)
        except RuntimeError as exc:
            report["hf_baseline"] = {"status": "failed",
                                     "reason": f"{type(exc).__name__}: {exc}"}
            if "out of memory" in str(exc).lower():
                report["status"] = "stopped_oom_hf_baseline"
                report["caveats"] = build_caveats(config, partial)
                return report

    # 2/3. Extract weights + masked correctness --------------------------
    # The masked correctness pipeline folds/runs at `folding_dtype` and the
    # plain reference too; only the optional runtime cast drops to a lower
    # precision, and recovery/compare run at `recovery_dtype`.
    folded_rt = runtime_dtype if runtime_dtype != folding_dtype else None
    skel_cfg = HFCausalLMSkeletonConfig(
        model_family=report["model_type"], prefill_seq_len=config.prefill_seq_len,
        decode_steps=config.decode_steps,
        max_layers=max_layers, dtype=folding_dtype, device=device,
        seed=config.seed, mask_mode=config.mask_mode,
        residual_mask_strategy=config.residual_mask_strategy,
        mask_block_size=config.block_size,
        allow_dense_large_mask=config.allow_dense_large_mask,
        folded_runtime_dtype=folded_rt, recovery_dtype=recovery_dtype,
        negative_control=config.negative_control)

    try:
        _reset_peak(device)
        weights, layer_configs, extract_meta = \
            extract_hf_causal_lm_skeleton_weights(
                model, model_config, max_layers=max_layers, dtype=folding_dtype,
                device=device)
        report["extraction"] = {
            "num_layers_extracted": extract_meta["num_layers_extracted"],
            "hidden_size": extract_meta["hidden_size"],
            "vocab_size": extract_meta["vocab_size"],
            "tie_word_embeddings": extract_meta["tie_word_embeddings"],
            "peak_cuda_memory": _cuda_mem(device),
        }
    except RuntimeError as exc:
        report["status"] = ("stopped_oom_extraction"
                            if "out of memory" in str(exc).lower()
                            else "failed_extraction")
        report["reason"] = f"{type(exc).__name__}: {exc}"
        report["caveats"] = build_caveats(config, partial)
        return report

    hidden = extract_meta["hidden_size"]
    vocab = extract_meta["vocab_size"]
    n_extracted = extract_meta["num_layers_extracted"]
    masks = generate_hf_causal_lm_masks(weights, layer_configs, skel_cfg)
    report["mask"] = {
        "mask_mode": masks.metadata["mask_mode"],
        "residual_mask_strategy": masks.metadata["residual_mask_strategy"],
        "shared_residual_mask": masks.shared_residual_mask,
        "mask_security_note": masks.metadata["mask_security_note"],
        "handoff_skip_term_needs_gemm":
            masks.metadata["handoff_skip_term_needs_gemm"],
    }

    input_ids, input_meta = build_probe_inputs(tokenizer, vocab, config, device)
    report["input"] = input_meta

    # extracted plaintext reference latency (plain-only) -----------------
    if config.run_extracted_plain:
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            plain = hf_causal_lm_plain_prefill(input_ids, weights,
                                               layer_configs, masks, skel_cfg)
            caches = plain["caches_plain"]
            nxt = plain["next_token_plain"]
            cos, sin = plain["cos"], plain["sin"]
            ex_tokens = [nxt]
            for step in range(config.decode_steps):
                dec = hf_causal_lm_plain_decode(
                    nxt, caches, weights, layer_configs, masks, skel_cfg,
                    position=config.prefill_seq_len + step, cos=cos, sin=sin)
                caches = dec["caches_plain"]
                nxt = dec["next_token_plain"]
                ex_tokens.append(nxt)
        _sync(device)
        report["extracted_plain"] = {
            "latency_s": round(time.perf_counter() - t0, 4),
            "num_tokens": len(ex_tokens),
            "peak_cuda_memory": _cuda_mem(device),
        }

    # masked runtime correctness (includes plain reference recompute) ----
    if config.run_masked_runtime:
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            res = hf_causal_lm_masked_greedy_decode(
                input_ids, weights, layer_configs, masks, skel_cfg)
        _sync(device)
        masked_latency = time.perf_counter() - t0
        pre = res["prefill_metrics"]
        report["masked_runtime"] = {
            "latency_s_with_reference": round(masked_latency, 4),
            "note": "latency includes the extracted-plain reference recompute "
                    "(diagnostic); not a masked-only timing.",
            "token_match_rate_vs_extracted": res["token_match_rate"],
            "greedy_token_match_prefill": pre["greedy_token_match_rate"],
            "final_hidden_max_abs_error": pre["final_hidden_max_abs_error"],
            "masked_logits_max_abs_error": pre["masked_logits_max_abs_error"],
            "recovered_logits_max_abs_error":
                pre["recovered_logits_max_abs_error"],
            "decode_steps": len(res["decode_step_metrics"]),
            "peak_cuda_memory": _cuda_mem(device),
        }
        report["bf16_diagnostics"] = pre.get("diagnostics", {})

    report["boundary"] = _boundary_accounting(
        config, hidden, vocab, model_dtype, n_extracted,
        masks.shared_residual_mask)

    # Audit section -------------------------------------------------------
    masked_ran = "masked_runtime" in report
    report["audit"] = _build_audit(
        config, model_config,
        {"model": _name(model_dtype), "folding": _name(folding_dtype),
         "folded_weight_runtime": _name(runtime_dtype),
         "recovery": _name(recovery_dtype),
         "compare": _name(resolve_dtype(config.compare_dtype, device))},
        loaded["local_path"], total_layers, n_extracted, extract_meta,
        masks.metadata, input_meta, masked_ran, tokenizer is not None)

    # Negative-control verdict -------------------------------------------
    expected_to_match = config.negative_control == "none"
    token_match = (report.get("masked_runtime", {})
                   .get("token_match_rate_vs_extracted"))
    recovered_err = (report.get("masked_runtime", {})
                     .get("recovered_logits_max_abs_error"))
    observed_match = token_match == 1.0 if token_match is not None else None
    if observed_match is None:
        nc_passed = None
    else:
        nc_passed = observed_match == expected_to_match
    report["negative_control"] = config.negative_control
    report["expected_to_match"] = expected_to_match
    report["negative_control_passed"] = nc_passed
    report["negative_control_detail"] = {
        "observed_token_match": observed_match,
        "token_match_rate_vs_extracted": token_match,
        "recovered_logits_max_abs_error": recovered_err,
    }

    report["caveats"] = build_caveats(config, partial)
    if config.negative_control != "none":
        report["caveats"].append(
            f"negative_control={config.negative_control}: this run "
            "INTENTIONALLY breaks the masked recovery path; mismatch is the "
            "expected (passing) outcome.")

    # free GPU memory promptly
    del model, weights
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report
