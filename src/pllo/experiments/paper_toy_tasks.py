"""Stage 7.5b - paper toy-task workload runner (CPU only).

Drives plain vs. masked execution on three deterministic *task-like*
workloads that do not require external data, tokenizers, or the network:

* ``token_parity_classification`` -- label = ``sum(input_ids) % 2``;
* ``first_last_token_relation`` -- label = ``1 if first_token < last_token else 0``;
* ``next_token_toy_lm`` -- target = ``(input_ids.sum() + last_token) %
  vocab_size``; emulates a tiny LM next-token prediction.

For each task we share a fresh tiny synthetic decoder stack
(embedding -> pooled / sequence Linear-with-LoRA -> classifier head),
run ``num_train_steps`` SGD steps on a deterministic dataset, and record
per-step plain vs. masked loss / accuracy / logits / token-match. The
plain reference is the algebraic identity from Stage 7.0 / 7.2 / 7.4;
the masked path uses ``run_masked_lora_linear`` end-to-end so that the
report exercises Theorem 7 / 8 / 9 on task-like inputs rather than only
on random tensors.

This module does NOT introduce new obfuscation primitives or attackers,
does NOT change any default of the existing inference / LoRA paths, and
does NOT publish raw tensors, masks, adapters, gradients, or input ids.
Reports are summary statistics only.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from pllo.ops.lora import (
    LoRAConfig,
    MaskedLoRAForwardConfig,
    init_lora_adapters,
    plain_lora_linear_forward,
    run_masked_lora_linear,
)


_TASK_NAMES = (
    "token_parity_classification",
    "first_last_token_relation",
    "next_token_toy_lm",
)


@dataclass
class PaperToyTaskConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    num_samples: int = 128
    seq_len: int = 8
    vocab_size: int = 128
    hidden_size: int = 32
    num_classes: int = 2
    num_layers: int = 2
    true_rank: int = 4
    padded_rank: int = 8
    batch_size: int = 8
    num_train_steps: int = 20
    lr: float = 1e-2
    use_pad: bool = True
    mitigation_bundle: str = "fresh_perm_plus_sandwich_plus_pad"
    inter_block_mask_mode: str = "masked_boundary_experimental"
    constant_time_decode_mode: str = "proxy_equalized"
    dtype: str = "float64"
    device: str = "cpu"
    allclose_atol: float = 1e-9
    allclose_rtol: float = 1e-9


_LIMITATIONS = [
    "Task-like inputs are deterministic synthetic sequences; no external dataset, no tokenizer.",
    "Targets are deterministic functions of input ids and do NOT reflect any real-world label distribution.",
    "The toy model is a tiny LoRA-augmented decoder stack; not a full Qwen / TinyLlama / LLaMA fine-tune.",
    "Loss, optimizer, and backward remain trusted-side (Stage 7.0 / 7.1 contract).",
    "Local CPU runtime only; not real TEE wall-time and not GPU throughput.",
    "No formal / cryptographic / semantic security is claimed.",
    "No PEFT / DeepSpeed / vLLM / FlashAttention integration.",
    "Adapter is NEVER merged into the public base weight W (Stage 7.0 contract).",
    "Reports publish summary metrics only; raw tensors / masks / adapters / gradients / input ids are never emitted.",
]


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def _make_inputs(
    cfg: PaperToyTaskConfig, generator: torch.Generator,
) -> torch.Tensor:
    """Deterministic synthetic token sequences ``(num_samples, seq_len)``."""
    return torch.randint(
        low=0, high=cfg.vocab_size,
        size=(cfg.num_samples, cfg.seq_len),
        generator=generator, dtype=torch.long, device=torch.device(cfg.device),
    )


def _make_labels(task: str, input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    if task == "token_parity_classification":
        return (input_ids.sum(dim=1) % 2).long()
    if task == "first_last_token_relation":
        return (input_ids[:, 0] < input_ids[:, -1]).long()
    if task == "next_token_toy_lm":
        return ((input_ids.sum(dim=1) + input_ids[:, -1]) % vocab_size).long()
    raise ValueError(f"unknown toy task {task!r}")


def _make_embedding(
    cfg: PaperToyTaskConfig, generator: torch.Generator,
) -> torch.Tensor:
    dtype = _torch_dtype(cfg.dtype)
    return torch.randn(
        cfg.vocab_size, cfg.hidden_size,
        generator=generator, dtype=dtype, device=torch.device(cfg.device),
    ) * (1.0 / math.sqrt(max(cfg.hidden_size, 1)))


def _make_head(
    cfg: PaperToyTaskConfig, num_classes: int, generator: torch.Generator,
) -> torch.Tensor:
    dtype = _torch_dtype(cfg.dtype)
    return torch.randn(
        cfg.hidden_size, num_classes,
        generator=generator, dtype=dtype, device=torch.device(cfg.device),
    ) * (1.0 / math.sqrt(max(cfg.hidden_size, 1)))


def _make_lora_block(
    cfg: PaperToyTaskConfig, generator: torch.Generator,
) -> dict[str, Any]:
    """Public base weight + private LoRA factors for one block."""
    inner = LoRAConfig(
        d_in=cfg.hidden_size, d_out=cfg.hidden_size, rank=cfg.true_rank,
        alpha=float(cfg.true_rank), use_bias=False,
        dtype=cfg.dtype, device=cfg.device,
    )
    dtype = _torch_dtype(cfg.dtype)
    w = torch.randn(
        cfg.hidden_size, cfg.hidden_size,
        generator=generator, dtype=dtype, device=torch.device(cfg.device),
    ) * (1.0 / math.sqrt(max(cfg.hidden_size, 1)))
    a, b = init_lora_adapters(inner, generator=generator)
    # Stage 7.0 init has B=0, which kills gradient signal across steps when
    # only B is updated. Seed B with a small non-zero so SGD has somewhere to
    # move and downstream loss / accuracy can change across the 20 steps.
    b = b + 0.01 * torch.randn(
        b.shape[0], b.shape[1],
        generator=generator, dtype=dtype, device=torch.device(cfg.device),
    )
    return {"config": inner, "w": w, "a": a, "b": b}


def _plain_block_forward(
    block: dict[str, Any], x: torch.Tensor,
) -> torch.Tensor:
    return plain_lora_linear_forward(
        x, block["w"], block["a"], block["b"], bias=None,
        alpha=block["config"].alpha,
    )


def _masked_block_forward(
    block: dict[str, Any], x: torch.Tensor, cfg: PaperToyTaskConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    fwd = MaskedLoRAForwardConfig(
        use_pad=cfg.use_pad,
        fresh_u_per_call=True,
        fresh_masks_per_call=True,
        dtype=cfg.dtype, device=cfg.device,
    )
    y, _ = run_masked_lora_linear(
        x, block["w"], block["a"], block["b"], None,
        block["config"], fwd, generator=generator,
    )
    return y


def _relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)


def _forward(
    blocks: list[dict[str, Any]], embed: torch.Tensor,
    head: torch.Tensor, input_ids: torch.Tensor,
    task: str, cfg: PaperToyTaskConfig,
    masked: bool, generator: torch.Generator | None,
) -> torch.Tensor:
    """Run the tiny toy decoder stack.

    For sequence-pooled tasks (``token_parity_classification`` /
    ``first_last_token_relation``) we mean-pool over the sequence axis after
    the LoRA stack; for ``next_token_toy_lm`` we project the last position.
    """
    h = embed[input_ids]  # (B, T, D)
    B, T, D = h.shape
    h = h.reshape(B * T, D)
    for block in blocks:
        if masked:
            assert generator is not None
            h = _masked_block_forward(block, h, cfg, generator)
        else:
            h = _plain_block_forward(block, h)
        h = _relu(h)
    h = h.reshape(B, T, D)
    if task == "next_token_toy_lm":
        pooled = h[:, -1, :]
    else:
        pooled = h.mean(dim=1)
    logits = pooled @ head
    return logits


def _cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return -log_probs.gather(1, labels.unsqueeze(1)).mean()


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def _token_match_rate(plain_logits: torch.Tensor, masked_logits: torch.Tensor) -> float:
    plain_preds = plain_logits.argmax(dim=-1)
    masked_preds = masked_logits.argmax(dim=-1)
    return float((plain_preds == masked_preds).float().mean().item())


def _sgd_step(
    blocks: list[dict[str, Any]], head: torch.Tensor,
    grads: dict[str, list[torch.Tensor]], lr: float,
) -> None:
    """In-place SGD update on plaintext (A, B) and the classifier head."""
    for block, ga, gb in zip(blocks, grads["a"], grads["b"]):
        block["a"] = block["a"] - lr * ga
        block["b"] = block["b"] - lr * gb
    # head update is in-place mutation of the head tensor reference; the
    # caller propagates by reassignment via the returned head.


def _backward_lora_stack(
    blocks: list[dict[str, Any]], head: torch.Tensor, embed: torch.Tensor,
    input_ids: torch.Tensor, labels: torch.Tensor, task: str,
    cfg: PaperToyTaskConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Autograd-based backward; returns ``(logits, loss, {'a': [...], 'b': [...]})``.

    We deliberately use autograd over the *plain* path so the masked path is
    NOT part of the gradient graph; the trusted side applies the update
    using these gradients to *both* the plain and masked references (they
    share parameters by construction).
    """
    for block in blocks:
        block["a"] = block["a"].clone().requires_grad_(True)
        block["b"] = block["b"].clone().requires_grad_(True)
    head_t = head.clone().requires_grad_(True)
    logits = _forward(blocks, embed, head_t, input_ids, task, cfg, masked=False, generator=None)
    loss = _cross_entropy(logits, labels)
    grad_list = torch.autograd.grad(
        loss,
        [block["a"] for block in blocks] + [block["b"] for block in blocks] + [head_t],
        create_graph=False, retain_graph=False,
    )
    n = len(blocks)
    grads = {
        "a": [g.detach() for g in grad_list[:n]],
        "b": [g.detach() for g in grad_list[n:2 * n]],
        "head": grad_list[-1].detach(),
    }
    # Reset to non-requires-grad plain tensors for the next step.
    for block in blocks:
        block["a"] = block["a"].detach()
        block["b"] = block["b"].detach()
    return logits.detach(), loss.detach(), grads


def _run_one_task(
    task: str, cfg: PaperToyTaskConfig, generator: torch.Generator,
) -> dict[str, Any]:
    """Train ``cfg.num_train_steps`` SGD steps on one toy task."""
    embed = _make_embedding(cfg, generator)
    num_classes = cfg.vocab_size if task == "next_token_toy_lm" else cfg.num_classes
    head = _make_head(cfg, num_classes, generator)
    blocks = [_make_lora_block(cfg, generator) for _ in range(cfg.num_layers)]

    input_ids = _make_inputs(cfg, generator)
    labels = _make_labels(task, input_ids, cfg.vocab_size)

    per_step: list[dict[str, float]] = []
    train_loss_plain = float("nan")
    train_loss_masked = float("nan")
    final_acc_plain = float("nan")
    final_acc_masked = float("nan")
    final_token_match = float("nan")
    final_logits_err = float("nan")

    num_batches = max(1, cfg.num_samples // cfg.batch_size)
    t0 = time.perf_counter()
    for step in range(cfg.num_train_steps):
        bi = step % num_batches
        s = bi * cfg.batch_size
        e = s + cfg.batch_size
        batch_ids = input_ids[s:e]
        batch_labels = labels[s:e]
        # Plain forward + backward for grads.
        plain_logits, plain_loss, grads = _backward_lora_stack(
            blocks, head, embed, batch_ids, batch_labels, task, cfg,
        )
        # Masked forward sharing the same parameter values.
        masked_logits = _forward(
            blocks, embed, head, batch_ids, task, cfg,
            masked=True, generator=generator,
        )
        masked_loss = _cross_entropy(masked_logits, batch_labels)
        logits_err = float((plain_logits - masked_logits).abs().max().item())
        # Apply SGD update trusted-side using plain analytic gradients.
        for block, ga, gb in zip(blocks, grads["a"], grads["b"]):
            block["a"] = block["a"] - cfg.lr * ga
            block["b"] = block["b"] - cfg.lr * gb
        head = head - cfg.lr * grads["head"]
        per_step.append({
            "step": step,
            "plain_loss": float(plain_loss.item()),
            "masked_loss": float(masked_loss.item()),
            "logits_max_abs_error": logits_err,
            "token_match_rate": _token_match_rate(plain_logits, masked_logits),
        })
        train_loss_plain = float(plain_loss.item())
        train_loss_masked = float(masked_loss.item())
        final_logits_err = logits_err
        final_token_match = _token_match_rate(plain_logits, masked_logits)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    # Evaluate on the full dataset after training.
    plain_logits = _forward(
        blocks, embed, head, input_ids, task, cfg, masked=False, generator=None,
    )
    masked_logits = _forward(
        blocks, embed, head, input_ids, task, cfg, masked=True, generator=generator,
    )
    final_acc_plain = _accuracy(plain_logits, labels)
    final_acc_masked = _accuracy(masked_logits, labels)
    final_logits_err = float((plain_logits - masked_logits).abs().max().item())
    final_token_match = _token_match_rate(plain_logits, masked_logits)
    allclose = bool(torch.allclose(
        plain_logits, masked_logits,
        atol=cfg.allclose_atol, rtol=cfg.allclose_rtol,
    ))

    return {
        "task_name": task,
        "num_samples": int(cfg.num_samples),
        "num_train_steps": int(cfg.num_train_steps),
        "train_loss_plain": train_loss_plain,
        "train_loss_masked": train_loss_masked,
        "loss_diff": float(abs(train_loss_plain - train_loss_masked)),
        "accuracy_plain": final_acc_plain,
        "accuracy_masked": final_acc_masked,
        "accuracy_diff": float(abs(final_acc_plain - final_acc_masked)),
        "logits_max_abs_error": final_logits_err,
        "token_match_rate": final_token_match,
        "allclose": allclose,
        "runtime_ms": runtime_ms,
        "per_step": per_step,
        "notes": (
            "Deterministic synthetic task; plain analytic backward; masked forward"
            " exercises Theorem 7 (LoRA masked forward) on task-like inputs."
        ),
    }


def _write_outputs(
    output_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_toy_tasks.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    columns = [
        "task_name", "num_samples", "num_train_steps",
        "train_loss_plain", "train_loss_masked", "loss_diff",
        "accuracy_plain", "accuracy_masked", "accuracy_diff",
        "logits_max_abs_error", "token_match_rate", "allclose",
        "runtime_ms",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in columns})
    (output_dir / "paper_toy_tasks.csv").write_text(buf.getvalue(), encoding="utf-8")

    md_lines: list[str] = ["# Paper Toy Task Workload (CPU only)\n"]
    md_lines.append(
        "_This is a CPU local-emulation prototype on deterministic synthetic"
        " task-like inputs. No external dataset, no tokenizer, no network,"
        " not a real Qwen / TinyLlama / LLaMA fine-tune._\n"
    )
    md_lines.append("| " + " | ".join(columns) + " |")
    md_lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        md_lines.append(
            "| " + " | ".join(str(r.get(c, "")) for c in columns) + " |"
        )
    md_lines.append("\n## Limitations\n")
    for lim in _LIMITATIONS:
        md_lines.append(f"- {lim}")
    (output_dir / "paper_toy_tasks.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8",
    )


def run_paper_toy_tasks(config: PaperToyTaskConfig) -> dict[str, Any]:
    """Run all three toy tasks under plain vs. masked execution."""
    generator = torch.Generator(device=torch.device(config.device))
    generator.manual_seed(int(config.seed))
    rows: list[dict[str, Any]] = []
    for task in _TASK_NAMES:
        row = _run_one_task(task, config, generator)
        rows.append(row)
    report: dict[str, Any] = {
        "config": asdict(config),
        "rows": rows,
        "paper_toy_tasks_status": "implemented",
        "stage": "7.5b",
        "wall_time_source": "measured_local_emulation",
        "is_real_tee_wall_time": False,
        "is_gpu_throughput": False,
        "security_profile": "proxy-evaluated, not formal",
        "limitations": list(_LIMITATIONS),
    }
    _write_outputs(Path(config.output_dir), report, rows)
    return report


__all__ = [
    "PaperToyTaskConfig",
    "run_paper_toy_tasks",
]
