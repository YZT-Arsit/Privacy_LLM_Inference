"""CONJFORMER baseline (Yukhimchuk et al., "Privacy from Symmetry: Orthogonally
Equivariant Transformers for LLM Inference", arXiv:2606.16461, 2026).

CONJFORMER makes a transformer *exactly* O(d)-equivariant so the client can
rotate its input embeddings by a secret orthogonal U and the server runs the
whole forward pass in the rotated basis, never seeing an unrotated hidden state.

Two ingredients (paper §2.2, App. A.1, App. B):

  1. ARCHITECTURE CHANGE: replace every RMSNorm's per-channel learnable gain
     vector with a single learnable SCALAR gain. Only then does normalization
     commute with an orthogonal U:  RMSNorm_scalar(X U^T) = RMSNorm_scalar(X) U^T
     (Lemma 2.2). This is the one architectural modification, and it is the
     source of CONJFORMER's utility cost on an off-the-shelf model: the trained
     per-channel gains carry information that a scalar cannot, so a pretrained
     model must be RETROFITTED + fine-tuned to recover quality (paper D.3).

  2. WEIGHT CONJUGATION (App. A.1 + B.3): sample secret orthogonal matrices
     - U       (hidden x hidden)         : global, shared by all layers
     - O_b     (per head, block-diag)    : rotates Q/K; MUST commute with RoPE
     - R_b     (per head, block-diag)    : rotates V (unconstrained orthogonal)
     - P_b     (intermediate permutation): commutes with the SwiGLU nonlinearity
     and conjugate the (fine-tuned) weights W~ into server weights W^:

         W^_Q  = O_b W~_Q  U^T,   b^_Q = O_b b~_Q
         W^_K  = O_b W~_K  U^T,   b^_K = O_b b~_K
         W^_V  = R_b W~_V  U^T,   b^_V = R_b b~_V
         W^_O  = U  W~_O   R_b^T
         W^_gate = P_b^T W~_gate U^T,  W^_up = P_b^T W~_up U^T,  W^_down = U W~_down P_b

     giving block-level functional equivalence  F^_b(X U^T) = F_b(X) U^T
     (Lemma 2.1). Attention logits are preserved (Q^ K^T = Q K^T) because O_b is
     orthogonal and (crucially, on RoPE models) commutes with the rotary op.

RoPE compatibility (App. B.2). HF applies RoPE as `rotate_half`, pairing dim i
with dim i+d_h/2 and rotating that 2-plane by theta_i(pos). For O_b to preserve
attention logits AFTER RoPE, O_b must commute with every such plane rotation.
The paper's sufficient choice is O_b block-diagonal with aligned 2x2 SO(2)
blocks; we build exactly that, on the SAME (i, i+d_h/2) planes HF uses, so O_b
commutes with HF RoPE by construction. R_b (on V, which never sees RoPE) stays a
full per-head orthogonal for maximal obfuscation.

GQA (App. "Grouped-query attention"). Qwen2.5 has n_kv < n_q. The rotation
assigned to a KV head is reused for all query heads that share it: O_b/R_b are
sampled per KV head and repeated across the corresponding query-head group.

What this module implements directly from the paper: scalar-gain RMSNorm, the
full blockwise conjugation for a Qwen2 / Llama-style SwiGLU+RoPE+GQA decoder, the
RoPE-compatible O_b construction, and a numerical equivariance verifier
(server-on-rotated == retrofitted-model-on-plain to floating point). Secret
matrices are never serialised. The scheme is NOT cryptographic / DP (paper
"Scope and limitations"): norms, pairwise distances, repeated-token patterns and
attention logits remain visible to the server.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from pllo.baselines.baseline_protocol import BaselineSelfDeclaration

_DECLARE = BaselineSelfDeclaration(
    name="conjformer",
    paper="Privacy from Symmetry: Orthogonally Equivariant Transformers for LLM Inference (Yukhimchuk et al., arXiv:2606.16461, 2026)",
    exact_primitive_implemented=True,
    full_system_reproduced=False,
    requires_crypto_library=False,
    supports_static_forward=True,
    supports_decoder_generation=True,
    supports_kv_cache_append=True,
    supports_lora_training=False,
    notes=(
        "Scalar-gain RMSNorm + blockwise orthogonal weight conjugation for a"
        " Qwen2/Llama SwiGLU+RoPE+GQA decoder, verified exactly equivariant"
        " (server-on-rotated == retrofit-model-on-plain to float). The paper"
        " only reports perplexity on GPT-2 / Llama-3.2-1B; no instruction or"
        " task-accuracy is measured, and the scalar-RMSNorm retrofit requires"
        " fine-tuning to recover quality on a pretrained model."
    ),
)


# ---------------------------------------------------------------------------
# Architecture change: scalar-gain RMSNorm (the only structural modification)
# ---------------------------------------------------------------------------
class ScalarRMSNorm(nn.Module):
    """RMSNorm with a single learnable SCALAR gain (paper Lemma 2.2).

    RMSNorm_eps(x) = gamma * x / sqrt(mean(x^2) + eps), gamma in R. Exactly
    O(d)-equivariant because the l2-norm is orthogonally invariant.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, gamma: float = 1.0):
        super().__init__()
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.tensor(float(gamma)))  # scalar

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * h.to(input_dtype))

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, scalar_gain, eps={self.variance_epsilon}"


def retrofit_scalar_rmsnorm(model: nn.Module, init: str = "mean") -> int:
    """Replace every per-channel Qwen2/Llama RMSNorm with ScalarRMSNorm.

    init='mean' sets the scalar to the mean of the original per-channel gain
    (paper D.3: "initialize the scalar gain to match the average scale of the
    original normalization"). Returns the number of layers replaced. Mutates in
    place; this is the RETROFIT step (a pretrained per-channel model made
    equivariant, before any fine-tuning).
    """
    n = 0
    for parent in model.modules():
        for attr, child in list(parent.named_children()):
            cls = child.__class__.__name__
            if "RMSNorm" in cls and isinstance(getattr(child, "weight", None), nn.Parameter) \
                    and child.weight.dim() == 1:
                w = child.weight.data
                hs = w.numel()
                eps = getattr(child, "variance_epsilon", 1e-6)
                if init == "mean":
                    gamma = float(w.mean().item())
                elif init == "one":
                    gamma = 1.0
                else:
                    raise ValueError(f"unknown init {init}")
                new = ScalarRMSNorm(hs, eps=eps, gamma=gamma).to(w.device, w.dtype)
                setattr(parent, attr, new)
                n += 1
    return n


# ---------------------------------------------------------------------------
# Secret orthogonal-matrix construction
# ---------------------------------------------------------------------------
def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _full_orthogonal(d: int, g: torch.Generator) -> torch.Tensor:
    a = torch.randn(d, d, dtype=torch.float64, generator=g)
    q, r = torch.linalg.qr(a)
    # make the QR sign-deterministic (q @ diag(sign(diag(r))))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q


def _rope_compatible_head(head_dim: int, g: torch.Generator) -> torch.Tensor:
    """Per-head orthogonal that commutes with HF `rotate_half` RoPE.

    Rotates each (i, i+h2) plane by a random angle phi_i (h2=head_dim/2). Same
    plane structure/convention as HF RoPE, so it commutes with RoPE at every
    position (two SO(2) rotations in the same plane commute). Preserves
    Q^ K^T = Q K^T post-RoPE.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE-compatible O_b"
    h2 = head_dim // 2
    ang = torch.rand(h2, dtype=torch.float64, generator=g) * (2.0 * math.pi)
    c, s = torch.cos(ang), torch.sin(ang)
    O = torch.eye(head_dim, dtype=torch.float64)
    idx = torch.arange(h2)
    O[idx, idx] = c
    O[idx, idx + h2] = -s
    O[idx + h2, idx] = s
    O[idx + h2, idx + h2] = c
    return O


def _permutation_matrix(n: int, g: torch.Generator) -> torch.Tensor:
    perm = torch.randperm(n, generator=g)
    P = torch.zeros(n, n, dtype=torch.float64)
    P[torch.arange(n), perm] = 1.0
    return P


@dataclass
class ConjFormerSecrets:
    """Client-held secret basis. NEVER serialised / sent to the server."""

    U: torch.Tensor                       # [hidden, hidden] global
    O_kv: list[torch.Tensor]              # per layer: [n_kv][head_dim, head_dim] (Q/K, RoPE-compat)
    R_kv: list[torch.Tensor]              # per layer: [n_kv][head_dim, head_dim] (V, full orth)
    P: list[torch.Tensor]                 # per layer: [inter, inter] permutation
    meta: dict[str, Any] = field(default_factory=dict)


def sample_secrets(cfg, seed: int = 0, rope_compatible: bool = True) -> ConjFormerSecrets:
    """Sample U (global) and per-layer O_b, R_b, P_b for a Qwen2/Llama config."""
    g = _gen(seed)
    hidden = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or hidden // cfg.num_attention_heads
    inter = cfg.intermediate_size

    U = _full_orthogonal(hidden, g)
    O_kv, R_kv, P = [], [], []
    for _ in range(n_layers):
        if rope_compatible:
            O_kv.append([_rope_compatible_head(head_dim, g) for _ in range(n_kv)])
        else:
            O_kv.append([_full_orthogonal(head_dim, g) for _ in range(n_kv)])
        R_kv.append([_full_orthogonal(head_dim, g) for _ in range(n_kv)])
        P.append(_permutation_matrix(inter, g))
    return ConjFormerSecrets(
        U=U, O_kv=O_kv, R_kv=R_kv, P=P,
        meta={
            "hidden": hidden, "n_layers": n_layers, "n_kv": n_kv,
            "n_q": cfg.num_attention_heads, "head_dim": head_dim,
            "intermediate": inter, "rope_compatible_O": rope_compatible,
        },
    )


def _block_diag_heads(blocks: list[torch.Tensor]) -> torch.Tensor:
    return torch.block_diag(*blocks)


# ---------------------------------------------------------------------------
# Weight conjugation (App. A.1 + B.3), GQA-aware
# ---------------------------------------------------------------------------
def _mm(*mats: torch.Tensor) -> torch.Tensor:
    """Chain matmul in float64 for a clean orthogonal conjugation."""
    out = mats[0].double()
    for m in mats[1:]:
        out = out @ m.double()
    return out


def conjugate_qwen2(model: nn.Module, secrets: ConjFormerSecrets) -> nn.Module:
    """Conjugate a (retrofitted, scalar-RMSNorm) Qwen2/Llama model IN PLACE into
    the server-side model. The caller passes the model that plays the role of
    the fine-tuned equivariant model W~; this returns the server model W^.

    Assumes model.model.layers[*].self_attn.{q,k,v,o}_proj and .mlp.{gate,up,down}_proj
    (Qwen2 has q/k/v bias, o/mlp none). Conjugation math runs in float64 then is
    cast back to each weight's original dtype (orthogonal => numerically benign).
    """
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    n_rep = n_q // n_kv
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    dev = model.model.layers[0].self_attn.q_proj.weight.device
    U = secrets.U.to(dev)                 # conjugation runs in fp64 on the model's device

    layers = model.model.layers
    for li, layer in enumerate(layers):
        O_kv = secrets.O_kv[li]           # [n_kv] head blocks (Q/K)
        R_kv = secrets.R_kv[li]           # [n_kv] head blocks (V)
        P = secrets.P[li].to(dev)         # [inter, inter]

        # per-query-head expansion (GQA: query head qh uses kv head qh // n_rep)
        O_q = _block_diag_heads([O_kv[qh // n_rep] for qh in range(n_q)]).to(dev)  # [hidden, hidden]
        O_k = _block_diag_heads(O_kv).to(dev)                                      # [n_kv*hd, n_kv*hd]
        R_v = _block_diag_heads(R_kv).to(dev)                                      # [n_kv*hd, n_kv*hd]
        R_o = _block_diag_heads([R_kv[qh // n_rep] for qh in range(n_q)]).to(dev)  # [hidden, hidden]

        sa = layer.self_attn
        mlp = layer.mlp

        def set_w(mod, new):
            mod.weight.data = new.to(mod.weight.dtype).contiguous()

        def set_b(mod, new):
            if mod.bias is not None:
                mod.bias.data = new.to(mod.bias.dtype).contiguous()

        # W^_Q = O_q W~_Q U^T ; b^_Q = O_q b~_Q         (HF weight = paper W)
        set_w(sa.q_proj, _mm(O_q, sa.q_proj.weight.data, U.T))
        if sa.q_proj.bias is not None:
            set_b(sa.q_proj, _mm(O_q, sa.q_proj.bias.data.unsqueeze(-1)).squeeze(-1))
        # W^_K = O_k W~_K U^T ; b^_K = O_k b~_K
        set_w(sa.k_proj, _mm(O_k, sa.k_proj.weight.data, U.T))
        if sa.k_proj.bias is not None:
            set_b(sa.k_proj, _mm(O_k, sa.k_proj.bias.data.unsqueeze(-1)).squeeze(-1))
        # W^_V = R_v W~_V U^T ; b^_V = R_v b~_V
        set_w(sa.v_proj, _mm(R_v, sa.v_proj.weight.data, U.T))
        if sa.v_proj.bias is not None:
            set_b(sa.v_proj, _mm(R_v, sa.v_proj.bias.data.unsqueeze(-1)).squeeze(-1))
        # W^_O = U W~_O R_o^T   (no bias)
        set_w(sa.o_proj, _mm(U, sa.o_proj.weight.data, R_o.T))

        # SwiGLU: W^_gate = P^T W~_gate U^T, W^_up = P^T W~_up U^T, W^_down = U W~_down P
        set_w(mlp.gate_proj, _mm(P.T, mlp.gate_proj.weight.data, U.T))
        set_w(mlp.up_proj, _mm(P.T, mlp.up_proj.weight.data, U.T))
        set_w(mlp.down_proj, _mm(U, mlp.down_proj.weight.data, P))

    return model


# ---------------------------------------------------------------------------
# Client-side rotation helpers
# ---------------------------------------------------------------------------
def rotate_in(x: torch.Tensor, secrets: ConjFormerSecrets, hp: bool = False) -> torch.Tensor:
    """Client: X -> X U^T (obfuscated embeddings sent to the server).

    Runtime rotation runs in x's dtype (one cheap d x d matmul per step); the
    orthogonal U keeps it numerically benign. hp=True forces fp64 (used only by
    the equivariance verifier to separate design error from bf16 round-off)."""
    U = secrets.U.to(x.device)
    if hp:
        return (x.double() @ U.T).to(x.dtype)
    return x @ U.to(x.dtype).T


def rotate_out(x_hat: torch.Tensor, secrets: ConjFormerSecrets, hp: bool = False) -> torch.Tensor:
    """Client: X U^T -> X (de-rotate the server's returned hidden state)."""
    U = secrets.U.to(x_hat.device)
    if hp:
        return (x_hat.double() @ U).to(x_hat.dtype)
    return x_hat @ U.to(x_hat.dtype)


# ---------------------------------------------------------------------------
# Numerical equivariance verifier == proof of correct reproduction
# ---------------------------------------------------------------------------
@torch.no_grad()
def verify_equivariance(
    retrofit_model: nn.Module,
    server_model: nn.Module,
    secrets: ConjFormerSecrets,
    input_ids: torch.Tensor,
) -> dict[str, Any]:
    """Assert F^_body(embed U^T) == F_body(embed) U^T to floating point.

    Runs the retrofit (scalar-RMSNorm, unconjugated) model normally, runs the
    server (conjugated) model on rotated embeddings, de-rotates, and compares the
    final normed hidden states + the client LM-head logits. Both models must be
    the SAME retrofit model (server_model is its conjugated copy).
    """
    retrofit_model.eval()
    server_model.eval()

    # reference: retrofit model, plain forward, capture final-norm hidden
    ref_hidden = _run_body(retrofit_model, input_ids, rotate_emb=None)
    # server: conjugated model on rotated embeddings (hp rotation to isolate the
    # design's equivariance from bf16 round-off in the rotation matmul itself)
    srv_hidden_rot = _run_body(server_model, input_ids, rotate_emb=secrets)
    srv_hidden = rotate_out(srv_hidden_rot, secrets, hp=True)

    dh = (srv_hidden.double() - ref_hidden.double())
    hidden_max_abs = float(dh.abs().max().item())
    hidden_rel = hidden_max_abs / (ref_hidden.double().abs().max().item() + 1e-12)

    # client LM head on both -> logits + top-1 agreement
    lm_head = retrofit_model.get_output_embeddings()
    ref_logits = lm_head(ref_hidden.to(lm_head.weight.dtype)).double()
    srv_logits = lm_head(srv_hidden.to(lm_head.weight.dtype)).double()
    dl = (srv_logits - ref_logits)
    logit_max_abs = float(dl.abs().max().item())
    top1_agree = float((srv_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item())

    return {
        "hidden_max_abs_err": hidden_max_abs,
        "hidden_rel_err": hidden_rel,
        "logit_max_abs_err": logit_max_abs,
        "logit_top1_agreement": top1_agree,
        # correctness = argmax preserved (decisive for greedy). hidden_rel is just
        # numerical precision: ~1e-7 in fp32, ~1e-2 in bf16 (3584-dim orthogonal
        # round-off) -- both keep top-1 == 1.0, i.e. token-identical decoding.
        "equivariant": bool(top1_agree >= 0.999),
        "high_precision": bool(hidden_rel < 1e-4),
        "n_tokens": int(input_ids.numel()),
    }


@torch.no_grad()
def _run_body(model: nn.Module, input_ids: torch.Tensor, rotate_emb) -> torch.Tensor:
    """Run embed -> (optional rotate) -> decoder layers -> final norm; return the
    final normed hidden state [.., S, hidden]. Uses the model's own rotary/mask
    plumbing by calling model.model(...) with precomputed inputs_embeds."""
    base = model.model  # Qwen2Model / LlamaModel
    embeds = base.embed_tokens(input_ids)
    if rotate_emb is not None:
        embeds = rotate_in(embeds, rotate_emb, hp=True)
    out = base(inputs_embeds=embeds, use_cache=False)
    return out.last_hidden_state


# ---------------------------------------------------------------------------
# Cost / structural audit vs ours (A_rightmul folded_remote)
# ---------------------------------------------------------------------------
def structural_audit() -> dict[str, Any]:
    return {
        "architecture_change_required": "per-channel RMSNorm -> scalar RMSNorm (all norm layers)",
        "needs_finetune_to_recover_on_pretrained_model": True,
        "obfuscation_is_exact": True,               # orthogonal conjugation, Lemma 2.1
        "obfuscation_numerically_benign": True,     # kappa=1, norm-preserving (unlike our N^-1 W N fold)
        "server_runs_full_model_in_rotated_basis": True,
        "extra_per_token_tee_crossings": 0,         # client only embed+rotate+lm_head; no TEE round-trip
        "server_observable_invariants": [
            "l2 norms", "pairwise distances", "repeated-token patterns", "attention logits",
        ],
        "cryptographic_or_dp": False,
        "paper_measured_metrics": ["perplexity (OpenWebText / PubMed)"],
        "paper_measured_task_accuracy": False,
        "paper_max_model": "Llama-3.2-1B",
        "contrast_ours_A_rightmul": {
            "architecture_change_required": "none (serves the unmodified instruct model)",
            "needs_finetune": False,
            "measured_task_parity_fp32": {
                "ifeval": [66.17, 68.76], "gsm8k": [90.22, 90.52],
                "humaneval": [78.0, 82.3], "mt_bench": [6.844, 6.763],
            },
            "per_token_tee_crossings": 2,           # single trusted entry + exit
            "note": "ours pays latency (TEE round-trip + fp32 wire); conjformer pays "
                    "an architecture change + retraining and cannot serve an "
                    "off-the-shelf instruct model losslessly.",
        },
    }


def build_conjformer_server(
    model: nn.Module, seed: int = 0, rmsnorm_init: str = "mean",
    rope_compatible: bool = True,
) -> tuple[nn.Module, nn.Module, ConjFormerSecrets, dict[str, Any]]:
    """End-to-end: (retrofit copy, server copy, secrets, audit).

    - retrofit_model: scalar-RMSNorm copy of `model` (the equivariant W~).
    - server_model:   conjugated copy of the retrofit model (W^ sent to server).
    Both are deep copies; the input `model` is left untouched.
    """
    retrofit_model = copy.deepcopy(model)
    n_norm = retrofit_scalar_rmsnorm(retrofit_model, init=rmsnorm_init)
    secrets = sample_secrets(model.config, seed=seed, rope_compatible=rope_compatible)
    server_model = copy.deepcopy(retrofit_model)
    conjugate_qwen2(server_model, secrets)
    audit = structural_audit()
    audit["norm_layers_retrofitted"] = n_norm
    return retrofit_model, server_model, secrets, audit


__all__ = [
    "ScalarRMSNorm",
    "ConjFormerSecrets",
    "retrofit_scalar_rmsnorm",
    "sample_secrets",
    "conjugate_qwen2",
    "rotate_in",
    "rotate_out",
    "verify_equivariance",
    "structural_audit",
    "build_conjformer_server",
]
