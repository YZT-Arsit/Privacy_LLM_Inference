"""Stage 5.6 — Black-box query attacker.

Sees only what a deployed API would expose: the generated token sequence
and a small set of per-step logits summaries (top-1 / top-5 token IDs,
entropy, max logit, top-1 margin). It NEVER reads ``per_layer_traces``,
internal hidden states, masks, permutations, or KV cache contents.

Three sub-attacks:

1. ``prompt_identity_linkability`` — repeated queries against the same
   prompt vs. different prompts; signature = (token_id_n-gram histogram +
   logits-summary vector). Reports mean cosine same-prompt vs. mean cosine
   different-prompt, a linkability-AUC proxy, and nearest-prompt retrieval
   top-1 accuracy.

2. ``prompt_class_inference`` — built-in 4-way labelling of the default
   prompt set (``general_knowledge`` / ``security`` / ``machine_learning``
   / ``generation_template``). A 1-nearest-neighbour classifier over the
   black-box signatures reports class accuracy vs. random-chance baseline.

3. ``mitigation_mode_distinguishability`` — same query repeated under
   ``nonlinear_mode ∈ {trusted, compatible_islands}`` × ``mitigation_bundle
   ∈ {fresh_perm_only, fresh_perm_plus_sandwich_plus_pad}`` × ``use_pad ∈
   {False, True}``. A 1-NN classifier over output signatures reports
   mode-classification accuracy vs. random-chance baseline. With Stage
   6.4c's exact-token-match guarantee, the obfuscated output sequence is
   identical across modes — distinguishability should sit at random chance.

Scope: synthetic-token + synthetic-model fallback by default; pytest never
hits the network. No formal security; no real TEE; no side channel.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pllo.experiments.real_token_trace import (
    DEFAULT_PROMPTS,
    RealTokenTraceConfig,
    _build_prompt_input_ids,
    _try_load_tokenizer,
)
from pllo.experiments.modern_decoder_model_probe import (
    ModernDecoderModelWrapperConfig,
    _resolve_weights,
)
from pllo.hf_wrappers.modern_decoder_model_wrapper import (
    ObfuscatedModernDecoderModelWrapper,
)
from pllo.ops.mitigation_bundles import (
    VALID_MITIGATION_BUNDLES,
    normalize_mitigation_bundle,
)


# ---------------------------------------------------------------------------
# Prompt class labelling
# ---------------------------------------------------------------------------


# Index-aligned with DEFAULT_PROMPTS (Stage 5.5b). Keep deterministic so
# the class-inference baseline is reproducible across runs.
DEFAULT_PROMPT_CLASSES: tuple[str, ...] = (
    "general_knowledge",   # "The capital of France is"
    "security",            # "In a secure system, the user"
    "machine_learning",    # "Machine learning models can"
    "security",            # "Privacy preserving inference requires"
    "generation_template", # "The next token should be"
    "security",            # "A simple example of encryption is"
    "machine_learning",    # "The transformer architecture uses"
    "machine_learning",    # "Large language models generate"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BlackboxAttackerConfig:
    seed: int = 2026
    num_prompts: int = 8
    prompt_max_length: int = 16
    max_new_tokens: int = 3
    batch_size: int = 1
    model_id: str | None = None
    attempt_real_model_load: bool = False
    attempt_tokenizer_load: bool = False
    local_files_only: bool = False
    allow_synthetic_fallback: bool = True
    # Modes / bundles to compare in distinguishability test. Trusted mode
    # exercises a different decode_step branch (pre-existing Stage 6.4c
    # latent bug: ``layer_cache.key_plain`` not populated in trusted-mode
    # prefill); Stage 5.6 black-box attacker therefore restricts the
    # distinguishability sweep to compatible_islands × bundle × use_pad.
    # Trusted mode is still exercised at the prefill-only level via the
    # full_forward path elsewhere in Stage 6.4c.
    nonlinear_modes: tuple[str, ...] = ("compatible_islands",)
    mitigation_bundles: tuple[str, ...] = VALID_MITIGATION_BUNDLES
    use_pad_values: tuple[bool, ...] = (False, True)
    dtype: str = "float32"
    device: str = "cpu"
    # Synthetic-fallback model shape.
    synthetic_vocab_size: int = 256
    synthetic_hidden_size: int = 32
    synthetic_intermediate_size: int = 64
    synthetic_num_attention_heads: int = 4
    synthetic_num_key_value_heads: int = 2
    synthetic_head_dim: int = 8
    max_layers: int = 2


# ---------------------------------------------------------------------------
# Black-box query — single API call
# ---------------------------------------------------------------------------


def _logits_summary(logits_1d: torch.Tensor) -> dict[str, float]:
    """Per-step logits summary an API client would see if it had logprobs.

    Only the top-k IDs and a few scalars are exposed; the full distribution
    is not part of the attacker's view.
    """
    probs = torch.softmax(logits_1d, dim=-1)
    top5 = torch.topk(logits_1d, k=min(5, logits_1d.shape[-1]))
    entropy = float(-(probs * torch.clamp(probs, min=1e-30).log()).sum().item())
    max_logit = float(logits_1d.max().item())
    sorted_logits = torch.sort(logits_1d, descending=True).values
    margin = float((sorted_logits[0] - sorted_logits[1]).item()) if logits_1d.numel() >= 2 else 0.0
    return {
        "top1_id": int(top5.indices[0].item()),
        "top5_ids": [int(i.item()) for i in top5.indices],
        "entropy": entropy,
        "max_logit": max_logit,
        "top1_margin": margin,
    }


def _blackbox_query(
    wrapper: ObfuscatedModernDecoderModelWrapper,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Single API-style query: returns sequence + per-step logits summaries."""
    prefill_out = wrapper.prefill(input_ids)
    cache = prefill_out["kv_cache"]
    plain_caches = prefill_out["plain_layer_caches"]
    summaries: list[dict[str, Any]] = []
    cur_logits = prefill_out["logits_recovered"][:, -1, :]
    summaries.append(_logits_summary(cur_logits[0]))
    cur_token = cur_logits.argmax(dim=-1)
    tokens: list[int] = [int(cur_token.item())]
    position = int(input_ids.shape[-1])
    for _ in range(max(0, int(max_new_tokens) - 1)):
        step = wrapper.decode_step(
            cur_token.unsqueeze(-1), cache, position,
            plain_layer_caches=plain_caches,
        )
        cache = step["kv_cache"]
        plain_caches = step["plain_layer_caches"]
        cur_logits = step["next_logits_recovered"][:, -1, :]
        summaries.append(_logits_summary(cur_logits[0]))
        cur_token = cur_logits.argmax(dim=-1)
        tokens.append(int(cur_token.item()))
        position += 1
    return {
        "token_ids": tokens,
        "output_length": len(tokens),
        "per_step_logits_summary": summaries,
    }


# ---------------------------------------------------------------------------
# Signature construction
# ---------------------------------------------------------------------------


def _signature_vector(
    response: dict[str, Any], *, vocab_size: int,
) -> torch.Tensor:
    """Build a fixed-dim attacker signature from one API response.

    Components (all observable from the API):
      - per-step top-1 token one-hot histogram over vocab (capped to a
        feature_dim cap of 64 by hashing into 64 buckets so the signature
        stays compact across vocab sizes)
      - per-step entropy, max logit, top-1 margin (mean + std across steps)
    """
    bucket = 64
    feat = torch.zeros(bucket + 6, dtype=torch.float64)
    for s in response["per_step_logits_summary"]:
        feat[s["top1_id"] % bucket] += 1.0
        for t5 in s["top5_ids"][:5]:
            feat[t5 % bucket] += 0.25
    entropies = [s["entropy"] for s in response["per_step_logits_summary"]]
    max_logits = [s["max_logit"] for s in response["per_step_logits_summary"]]
    margins = [s["top1_margin"] for s in response["per_step_logits_summary"]]
    if entropies:
        feat[bucket + 0] = float(sum(entropies) / len(entropies))
        feat[bucket + 1] = float(_stdev(entropies))
        feat[bucket + 2] = float(sum(max_logits) / len(max_logits))
        feat[bucket + 3] = float(_stdev(max_logits))
        feat[bucket + 4] = float(sum(margins) / len(margins))
        feat[bucket + 5] = float(_stdev(margins))
    return feat


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))


def _row_normalise(M: torch.Tensor) -> torch.Tensor:
    return M / M.norm(dim=-1, keepdim=True).clamp_min(1e-30)


# ---------------------------------------------------------------------------
# Sub-attacks
# ---------------------------------------------------------------------------


def _prompt_linkability(
    sigs: torch.Tensor, prompt_ids: list[int], *, num_pairs: int = 1024,
) -> dict[str, float]:
    """Same vs different prompt cosine + nearest-prompt retrieval top-1."""
    n = sigs.shape[0]
    sigs_norm = _row_normalise(sigs)
    sim = sigs_norm @ sigs_norm.T
    same_mask = torch.zeros(n, n, dtype=torch.bool)
    for i, p in enumerate(prompt_ids):
        for j, q in enumerate(prompt_ids):
            if i != j and p == q:
                same_mask[i, j] = True
    diff_mask = (~same_mask) & ~torch.eye(n, dtype=torch.bool)
    same_cos = float(sim[same_mask].mean().item()) if same_mask.any() else 0.0
    diff_cos = float(sim[diff_mask].mean().item()) if diff_mask.any() else 0.0
    # Linkability AUC proxy: probability that a same-prompt pair scores
    # higher than a different-prompt pair under cosine sim.
    auc = 0.5
    if same_mask.any() and diff_mask.any():
        same_vals = sim[same_mask].to(torch.float64)
        diff_vals = sim[diff_mask].to(torch.float64)
        gen = torch.Generator(device="cpu").manual_seed(0xA0FF)
        pairs_n = min(num_pairs, same_vals.numel() * diff_vals.numel())
        if pairs_n > 0:
            ia = torch.randint(0, same_vals.numel(), (pairs_n,), generator=gen)
            ib = torch.randint(0, diff_vals.numel(), (pairs_n,), generator=gen)
            wins = (same_vals[ia] > diff_vals[ib]).to(torch.float64).mean()
            auc = float(wins.item())
    # Nearest-prompt retrieval: for each row, exclude itself, take argmax,
    # check whether the matched row's prompt id equals i's prompt id.
    sim_no_self = sim.clone()
    sim_no_self.fill_diagonal_(float("-inf"))
    nn_idx = sim_no_self.argmax(dim=-1)
    correct = 0
    for i, idx in enumerate(nn_idx.tolist()):
        if prompt_ids[idx] == prompt_ids[i]:
            correct += 1
    retrieval_top1 = float(correct / max(1, n))
    # Random chance for retrieval: prob a random other example shares the
    # same prompt id.
    same_counts: dict[int, int] = {}
    for p in prompt_ids:
        same_counts[p] = same_counts.get(p, 0) + 1
    rand_chance = sum(c * (c - 1) for c in same_counts.values()) / max(
        1, n * (n - 1)
    )
    return {
        "same_prompt_similarity": same_cos,
        "different_prompt_similarity": diff_cos,
        "linkability_auc_proxy": auc,
        "nearest_prompt_retrieval_top1": retrieval_top1,
        "nearest_prompt_retrieval_random_chance": float(rand_chance),
    }


def _class_inference(
    sigs: torch.Tensor,
    prompt_classes: list[str],
) -> dict[str, float]:
    n = sigs.shape[0]
    sigs_norm = _row_normalise(sigs)
    sim = sigs_norm @ sigs_norm.T
    sim_no_self = sim.clone()
    sim_no_self.fill_diagonal_(float("-inf"))
    nn_idx = sim_no_self.argmax(dim=-1)
    correct = 0
    for i, idx in enumerate(nn_idx.tolist()):
        if prompt_classes[idx] == prompt_classes[i]:
            correct += 1
    classes = sorted(set(prompt_classes))
    # Random-chance baseline = sum_c (n_c (n_c - 1)) / (n (n - 1)).
    counts: dict[str, int] = {}
    for c in prompt_classes:
        counts[c] = counts.get(c, 0) + 1
    rand_chance = sum(v * (v - 1) for v in counts.values()) / max(1, n * (n - 1))
    return {
        "class_accuracy": float(correct / max(1, n)),
        "random_chance_baseline": float(rand_chance),
        "num_classes": int(len(classes)),
        "classes": list(classes),
    }


def _mode_distinguishability(
    sigs: torch.Tensor, modes: list[str],
) -> dict[str, float]:
    n = sigs.shape[0]
    sigs_norm = _row_normalise(sigs)
    sim = sigs_norm @ sigs_norm.T
    sim_no_self = sim.clone()
    sim_no_self.fill_diagonal_(float("-inf"))
    nn_idx = sim_no_self.argmax(dim=-1)
    correct = 0
    for i, idx in enumerate(nn_idx.tolist()):
        if modes[idx] == modes[i]:
            correct += 1
    counts: dict[str, int] = {}
    for m in modes:
        counts[m] = counts.get(m, 0) + 1
    rand_chance = sum(v * (v - 1) for v in counts.values()) / max(1, n * (n - 1))
    return {
        "mode_classification_accuracy": float(correct / max(1, n)),
        "random_chance_baseline": float(rand_chance),
        "modes_observed": sorted(set(modes)),
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "Black-box attacks only use generated outputs and logits summaries — no internal traces are read.",
    "Stage 5.6 black-box attackers are stronger proxy attacks, not formal security proofs.",
    "Greedy generation produces deterministic outputs, so same-prompt linkability is trivially 1.0 — this is a property of greedy decoding, not a finding against the obfuscation envelope.",
    "Mitigation-mode distinguishability is bounded above by the obfuscated/plain token-match rate; Stage 6.4c verified that this rate is 1.0, so distinguishability is at random chance by construction.",
    "Synthetic fallback results are not real Qwen/TinyLlama API responses.",
    "Dense sandwiching and fresh permutation reduce tested recovery but do not imply semantic security.",
]


def _build_wrapper(
    config: BlackboxAttackerConfig,
    *, nonlinear_mode: str, mitigation_bundle: str, use_pad: bool,
    dtype: torch.dtype, device: torch.device,
):
    probe_cfg = ModernDecoderModelWrapperConfig(
        model_id=config.model_id,
        attempt_real_model_load=config.attempt_real_model_load,
        allow_synthetic_fallback=config.allow_synthetic_fallback,
        local_files_only=config.local_files_only,
        nonlinear_mode=nonlinear_mode,
        mitigation_bundle=mitigation_bundle,
        use_pad=use_pad,
        max_layers=config.max_layers,
        device=config.device,
        dtype=config.dtype,
        seed=config.seed,
        synthetic_vocab_size=config.synthetic_vocab_size,
        synthetic_hidden_size=config.synthetic_hidden_size,
        synthetic_intermediate_size=config.synthetic_intermediate_size,
        synthetic_num_attention_heads=config.synthetic_num_attention_heads,
        synthetic_num_key_value_heads=config.synthetic_num_key_value_heads,
        synthetic_head_dim=config.synthetic_head_dim,
    )
    spec, weights, load_record, source = _resolve_weights(probe_cfg, dtype, device)
    wrapper = ObfuscatedModernDecoderModelWrapper(
        weights, dtype=dtype, device=device, use_pad=use_pad,
        nonlinear_mode=nonlinear_mode,
        mitigation_bundle=mitigation_bundle,
    )
    return wrapper, weights, load_record, source


def run_blackbox_attacker(
    config: BlackboxAttackerConfig,
) -> dict[str, Any]:
    """Execute the three black-box sub-attacks and return a JSON-safe report."""
    torch.manual_seed(config.seed)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    device = torch.device(config.device)

    # Resolve a single set of synthetic / real weights for prompt + class
    # experiments (uses the default mitigation bundle).
    default_bundle = normalize_mitigation_bundle(config.mitigation_bundles[-1])
    wrapper, weights, load_record, source = _build_wrapper(
        config,
        nonlinear_mode="compatible_islands",
        mitigation_bundle=default_bundle,
        use_pad=True,
        dtype=dtype, device=device,
    )

    # Build prompts + input_ids via Stage 5.5b helpers.
    rt_cfg = RealTokenTraceConfig(
        seed=config.seed,
        model_id=config.model_id,
        attempt_real_model_load=config.attempt_real_model_load,
        attempt_tokenizer_load=config.attempt_tokenizer_load,
        local_files_only=config.local_files_only,
        allow_synthetic_fallback=config.allow_synthetic_fallback,
        prompt_max_length=config.prompt_max_length,
        num_prompts=config.num_prompts,
        synthetic_vocab_size=config.synthetic_vocab_size,
    )
    tokenizer_info = _try_load_tokenizer(rt_cfg)
    prompt_pkg = _build_prompt_input_ids(
        rt_cfg, tokenizer_info, vocab_size=int(weights.vocab_size),
    )
    input_ids = prompt_pkg["input_ids"].to(device=device)
    n_prompts = int(input_ids.shape[0])

    # ----- 1. Prompt linkability + 2. Class inference -----
    # Repeated queries: 2 queries per prompt → enables same-vs-different.
    sigs_pl: list[torch.Tensor] = []
    pid_pl: list[int] = []
    class_pl: list[str] = []
    classes = list(DEFAULT_PROMPT_CLASSES[: n_prompts])
    for repeat in range(2):
        for i in range(n_prompts):
            resp = _blackbox_query(
                wrapper, input_ids[i : i + 1],
                max_new_tokens=config.max_new_tokens,
            )
            sigs_pl.append(
                _signature_vector(resp, vocab_size=int(weights.vocab_size))
            )
            pid_pl.append(i)
            class_pl.append(classes[i] if i < len(classes) else "unknown")
    sigs_pl_t = torch.stack(sigs_pl, dim=0)
    linkability = _prompt_linkability(sigs_pl_t, pid_pl)
    class_inf = _class_inference(sigs_pl_t, class_pl)

    # ----- 3. Mitigation-mode distinguishability -----
    mode_sigs: list[torch.Tensor] = []
    mode_labels: list[str] = []
    for nlm in config.nonlinear_modes:
        for bundle in config.mitigation_bundles:
            for up in config.use_pad_values:
                w_mode, _, _, _ = _build_wrapper(
                    config, nonlinear_mode=nlm,
                    mitigation_bundle=normalize_mitigation_bundle(bundle),
                    use_pad=bool(up), dtype=dtype, device=device,
                )
                # One query per prompt → many samples per mode.
                for i in range(n_prompts):
                    resp = _blackbox_query(
                        w_mode, input_ids[i : i + 1],
                        max_new_tokens=config.max_new_tokens,
                    )
                    mode_sigs.append(
                        _signature_vector(
                            resp, vocab_size=int(weights.vocab_size),
                        )
                    )
                    mode_labels.append(f"{nlm}|{bundle}|use_pad={up}")
    mode_dist = _mode_distinguishability(
        torch.stack(mode_sigs, dim=0), mode_labels,
    )

    return {
        "config": asdict(config),
        "model_loading": load_record,
        "tokenizer_loading": {
            k: v for k, v in tokenizer_info.items() if k != "tokenizer"
        },
        "source": source,
        "prompt_summary": {
            "token_source": prompt_pkg["token_source"],
            "num_prompts": n_prompts,
            "prompt_max_length": int(config.prompt_max_length),
            "prompts_used": prompt_pkg["prompts_used"],
            "encoded_lengths": prompt_pkg["encoded_lengths"],
        },
        "attacker_view_inventory": [
            "generated token sequence",
            "per-step top-1 token id",
            "per-step top-5 token ids",
            "per-step entropy",
            "per-step max logit",
            "per-step top-1 margin",
            "output length",
        ],
        "internal_trace_access": "denied",
        "prompt_linkability": linkability,
        "prompt_class_inference": class_inf,
        "mitigation_mode_distinguishability": mode_dist,
        "limitations": list(_LIMITATIONS),
    }


__all__ = [
    "DEFAULT_PROMPT_CLASSES",
    "BlackboxAttackerConfig",
    "run_blackbox_attacker",
]
