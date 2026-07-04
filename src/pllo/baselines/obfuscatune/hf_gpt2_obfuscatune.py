"""HF GPT-2 wrapper for ObfuscaTune (block-level + full-model, eval only).

The obfuscation is applied to the six high-parameter linears of a GPT-2 block
(the fused ``c_attn`` split into Wq/Wk/Wv, the attention ``c_proj``, and the MLP
``c_fc`` / ``c_proj``). Everything else -- token/positional embeddings, both
LayerNorms, the causal softmax, the activation and the LM head -- runs in the
simulated TEE on de-obfuscated values, matching the paper's TEE/outside split.

No network access: build a tiny random model via :func:`make_tiny_gpt2_config`
(no download) or load a local checkpoint with ``local_files_only=True``. The
plaintext reference is always the module's own ``forward`` (we route matmuls
through obfuscation but keep GPT-2's exact attention math), so with orthogonal
matrices the wrapper reproduces the reference to floating-point tolerance.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import ObfuscaTuneConfig
from .metrics import argmax_match_rate, correctness_metrics, nan_or_inf_count
from .modules import ObfuscaTuneLinearSimulator, PhaseMetrics


def _require_transformers():
    try:
        import transformers  # noqa: F401
        from transformers import GPT2Config, GPT2LMHeadModel
        from transformers.models.gpt2.modeling_gpt2 import Conv1D  # noqa: F401
        return GPT2Config, GPT2LMHeadModel
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "the ObfuscaTune GPT-2 wrapper requires the 'transformers' package; "
            "install the optional 'hf' extra. No model is downloaded: use "
            "make_tiny_gpt2_config() or a local --model-name-or-path."
        ) from exc


def make_tiny_gpt2_config(
    *,
    vocab_size: int = 64,
    n_positions: int = 32,
    n_embd: int = 32,
    n_layer: int = 3,
    n_head: int = 4,
    n_inner: int | None = None,
):
    """Return a tiny random ``GPT2Config`` (no download, deterministic dims)."""
    GPT2Config, _ = _require_transformers()
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner or 4 * n_embd,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )


def load_gpt2(
    *,
    model_name_or_path: str | None = None,
    tiny_random_config: Any | None = None,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
):
    """Load a GPT-2 LM head model in eval mode.

    Exactly one of ``model_name_or_path`` (a *local* path) or
    ``tiny_random_config`` must be given. Never downloads.
    """
    GPT2Config, GPT2LMHeadModel = _require_transformers()
    if (model_name_or_path is None) == (tiny_random_config is None):
        raise ValueError(
            "provide exactly one of model_name_or_path (local path) or "
            "tiny_random_config"
        )
    if tiny_random_config is not None:
        torch.manual_seed(seed)
        model = GPT2LMHeadModel(tiny_random_config)
    else:
        model = GPT2LMHeadModel.from_pretrained(
            model_name_or_path, local_files_only=True
        )
    return model.to(dtype).eval()


# ---------------------------------------------------------------------------
# Block-level obfuscated forward
# ---------------------------------------------------------------------------
def _split_c_attn(c_attn, dtype):
    """Split fused GPT-2 c_attn (Conv1D (d,3d)) into (Wq,Wk,Wv, bq,bk,bv).

    Returns lightweight Conv1D-like shims exposing ``.weight`` (d,d) and
    ``.bias`` (d,) so the linear simulator can consume them directly.
    """
    from transformers.models.gpt2.modeling_gpt2 import Conv1D

    w = c_attn.weight.detach().to(dtype)          # (d, 3d)
    b = c_attn.bias.detach().to(dtype)            # (3d,)
    d = w.shape[0]
    shims = []
    for i in range(3):
        sh = Conv1D(d, d)                          # (nf=d, nx=d) -> weight (d,d)
        with torch.no_grad():
            sh.weight.copy_(w[:, i * d:(i + 1) * d])
            sh.bias.copy_(b[i * d:(i + 1) * d])
        shims.append(sh)
    return shims  # [Wq_mod, Wk_mod, Wv_mod]


def gpt2_block_obfuscated_forward(
    block,
    hidden: torch.Tensor,
    config: ObfuscaTuneConfig,
) -> tuple[torch.Tensor, dict[str, Any], PhaseMetrics]:
    """Run one HF GPT-2 block under ObfuscaTune. ``hidden``: (S, d).

    Reproduces GPT-2's exact causal attention and MLP, but routes the six
    linears through the obfuscated primitive. All non-linearities (both
    LayerNorms, softmax, activation) run in the simulated TEE.
    """
    dtype = config.torch_dtype()
    hidden = hidden.to(dtype)
    attn = block.attn
    n_head = int(attn.num_heads)
    S, d = hidden.shape
    hd = d // n_head
    metrics = PhaseMetrics()
    exposed: list[str] = []

    # --- attention ---
    ln1 = block.ln_1.to(dtype)
    h = ln1(hidden)                                        # TEE: LayerNorm
    metrics.boundary_calls += 1
    qmod, kmod, vmod = _split_c_attn(attn.c_attn, dtype)
    q, mq = ObfuscaTuneLinearSimulator(qmod, config, direction="input", seed=config.seed).forward(h)
    k, mk = ObfuscaTuneLinearSimulator(kmod, config, direction="input", seed=config.seed + 1).forward(h)
    v, mv = ObfuscaTuneLinearSimulator(vmod, config, direction="input", seed=config.seed + 2).forward(h)
    for mm in (mq, mk, mv):
        metrics.merge(mm)
    exposed += ["Q_plaintext", "K_plaintext", "V_plaintext"]
    qh = q.view(S, n_head, hd).transpose(0, 1)             # (nh, S, hd)
    kh = k.view(S, n_head, hd).transpose(0, 1)
    vh = v.view(S, n_head, hd).transpose(0, 1)
    scores = qh @ kh.transpose(-1, -2) / (hd ** 0.5)       # GPT-2 scaling
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=hidden.device))
    scores = scores.masked_fill(~causal, float("-inf"))
    att = torch.softmax(scores, dim=-1)                    # TEE: softmax
    metrics.boundary_calls += 1
    ctx = (att @ vh).transpose(0, 1).reshape(S, d)
    o, mo = ObfuscaTuneLinearSimulator(
        attn.c_proj, config, direction="output", seed=config.seed + 3).forward(ctx)
    metrics.merge(mo)
    hidden = hidden + o

    # --- MLP ---
    ln2 = block.ln_2.to(dtype)
    h2 = ln2(hidden)                                       # TEE: LayerNorm
    metrics.boundary_calls += 1
    inter, m1 = ObfuscaTuneLinearSimulator(
        block.mlp.c_fc, config, direction="input", seed=config.seed + 4).forward(h2)
    metrics.merge(m1)
    exposed += ["mlp_intermediate_plaintext"]
    inter = block.mlp.act(inter)                           # TEE: exact NewGELU
    metrics.boundary_calls += 1
    out, m2 = ObfuscaTuneLinearSimulator(
        block.mlp.c_proj, config, direction="output", seed=config.seed + 5).forward(inter)
    metrics.merge(m2)
    hidden = hidden + out

    audit = {
        "exposed_plaintext_tensors": exposed,
        "tee_nonlinear_ops": ["layernorm", "softmax", "layernorm", "gelu"],
    }
    return hidden, audit, metrics


# ---------------------------------------------------------------------------
# Full-model obfuscated logits
# ---------------------------------------------------------------------------
def gpt2_obfuscated_logits(
    model,
    input_ids: torch.Tensor,
    config: ObfuscaTuneConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Full GPT-2 forward under ObfuscaTune for a single sequence.

    ``input_ids``: (S,) or (1, S). Embeddings, final LayerNorm and the LM head
    run in the simulated TEE; every block is obfuscated. Returns
    ``(logits (S, vocab), audit)``.
    """
    dtype = config.torch_dtype()
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    S = input_ids.shape[0]
    tr = model.transformer
    device = model.transformer.wte.weight.device
    pos = torch.arange(S, device=device)
    hidden = tr.wte(input_ids).to(dtype) + tr.wpe(pos).to(dtype)   # TEE: embeddings
    total = PhaseMetrics()
    exposed: set[str] = set()
    for block in tr.h:
        hidden, audit, m = gpt2_block_obfuscated_forward(block, hidden, config)
        total.merge(m)
        exposed.update(audit["exposed_plaintext_tensors"])
    hidden = tr.ln_f.to(dtype)(hidden)                            # TEE: final norm
    logits = hidden @ model.lm_head.weight.to(dtype).transpose(0, 1)  # TEE: LM head
    audit = {
        "exposed_plaintext_tensors": sorted(exposed),
        "phase_metrics": total.to_dict(),
        "n_layers": len(tr.h),
    }
    return logits, audit


def gpt2_plain_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Reference plaintext logits from the model's own forward (no protection)."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits[0]


def run_mode(
    model,
    input_ids: torch.Tensor,
    mode: str,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    condition_number: float = 1.0,
) -> dict[str, Any]:
    """Run a full-model forward under one mode and return logits + metrics.

    Modes: ``unprotected``, ``obfuscatune_orthogonal``, ``obfuscatune_random``,
    ``obfuscatune_cond_sweep`` (uses ``condition_number``).
    """
    dtype_str = {torch.float64: "float64", torch.float32: "float32"}.get(dtype, str(dtype))
    plain = gpt2_plain_logits(model, input_ids)
    if mode == "unprotected":
        return {
            "mode": mode,
            "logits": plain,
            "correctness": correctness_metrics(plain, plain),
            "argmax_match_rate": 1.0,
            "nan_or_inf_count": nan_or_inf_count(plain),
            "audit": {"exposed_plaintext_tensors": [], "n_layers": len(model.transformer.h)},
        }
    if mode == "obfuscatune_orthogonal":
        cfg = ObfuscaTuneConfig(dtype=dtype_str, seed=seed, random_matrix_type="orthogonal")
    elif mode == "obfuscatune_random":
        cfg = ObfuscaTuneConfig(dtype=dtype_str, seed=seed, random_matrix_type="random")
    elif mode == "obfuscatune_cond_sweep":
        cfg = ObfuscaTuneConfig(dtype=dtype_str, seed=seed, random_matrix_type="cond",
                                condition_number=condition_number)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    logits, audit = gpt2_obfuscated_logits(model, input_ids, cfg)
    return {
        "mode": mode,
        "logits": logits,
        "correctness": correctness_metrics(logits, plain),
        "argmax_match_rate": argmax_match_rate(logits, plain),
        "nan_or_inf_count": nan_or_inf_count(logits),
        "condition_number": condition_number if mode == "obfuscatune_cond_sweep" else (
            1.0 if mode == "obfuscatune_orthogonal" else None),
        "audit": audit,
    }


__all__ = [
    "make_tiny_gpt2_config",
    "load_gpt2",
    "gpt2_block_obfuscated_forward",
    "gpt2_obfuscated_logits",
    "gpt2_plain_logits",
    "run_mode",
]
