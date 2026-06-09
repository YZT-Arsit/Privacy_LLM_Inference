"""Stage 7.6 paper-side lifecycle report.

Summarises which objects are plaintext / trusted-only, GPU-visible
masked, or public during each phase of the masked-gradient LoRA
training-to-inference lifecycle:

* LoRA initialization
* Masked forward
* Masked backward
* Masked SGD
* Masked momentum SGD
* Final adapter recovery / audit
* Trained LoRA inference

This is a paper-claims consistency artifact. It does not run any
training; it serialises the visibility classification implied by the
Stage 7.6 construction.

No raw tensors, masks, or adapters are exported. Outputs only contain
strings, shape descriptors, and short labels.

CPU-only. No formal, cryptographic, or semantic security is claimed.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any


# ---------------------------------------------------------------------------
# Static lifecycle classification
# ---------------------------------------------------------------------------


_VIS_PLAINTEXT_TRUSTED = "plaintext_trusted_only"
_VIS_GPU_MASKED = "gpu_visible_masked"
_VIS_PUBLIC = "public"


_REQUIRED_LIFECYCLE_PHASES: tuple[str, ...] = (
    "lora_initialization",
    "masked_forward",
    "masked_backward",
    "masked_sgd",
    "masked_momentum_sgd",
    "final_adapter_recovery_or_audit",
    "trained_lora_inference",
)


_REQUIRED_HONESTY_PHRASES: tuple[str, ...] = (
    "Base model W is public but transformed to preserve hidden-state "
    "masks; the trusted side never exports plaintext A or B.",
    "User input and token ids are trusted-side only and never leave "
    "the user device in plaintext.",
    "GPU sees masked hidden states, masked LoRA adapters, masked LoRA "
    "gradients, and masked momentum buffers.",
    "GPU does not see plaintext A/B, plaintext gradients, masks, or "
    "plaintext optimizer state.",
    "AdamW under dense masks is unsupported; the module raises "
    "DenseMaskedAdamWUnsupported rather than approximating.",
    "Masked-gradient LoRA provides algebraic equivalence for SGD/"
    "Momentum under orthogonal masks and proxy-evaluated leakage "
    "mitigation; it does not provide formal, cryptographic, or "
    "semantic security.",
)


def _row(
    *,
    phase: str,
    obj: str,
    visibility: str,
    exposed_form: str,
    note: str,
) -> dict[str, str]:
    return {
        "phase": phase,
        "object": obj,
        "visibility": visibility,
        "exposed_form": exposed_form,
        "note": note,
    }


def _lifecycle_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    # --- LoRA initialization -----------------------------------------------
    p = "lora_initialization"
    rows.extend([
        _row(
            phase=p, obj="A_real, B_real",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note="Plaintext LoRA factors live only on the trusted side.",
        ),
        _row(
            phase=p, obj="N_x, N_y, M",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note="Orthogonal masks held only by trusted side.",
        ),
        _row(
            phase=p, obj="A_pad, B_pad",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note=(
                "Cancellation-padded factors: A_pad=[A_real, R, -R], "
                "B_pad=vstack(B_real, S, S); dummy contribution is zero "
                "at init."
            ),
        ),
        _row(
            phase=p, obj="A_tilde, B_tilde",
            visibility=_VIS_GPU_MASKED,
            exposed_form="N_x^T A_pad M, M^T B_pad N_y",
            note=(
                "Masked, rank-padded LoRA factors uploaded to the GPU. "
                "Visible padded rank, true rank hidden."
            ),
        ),
        _row(
            phase=p, obj="base_model_W",
            visibility=_VIS_PUBLIC,
            exposed_form=(
                "transformed_to_preserve_hidden_state_masks (e.g., "
                "boundary linear absorbs N_x / N_y on trusted side)"
            ),
            note=(
                "Base weights are publicly known but their use on the "
                "GPU is composed with the trusted-side mask boundary "
                "so that hidden-state masks survive the linear path."
            ),
        ),
    ])

    # --- Masked forward ----------------------------------------------------
    p = "masked_forward"
    rows.extend([
        _row(
            phase=p, obj="user_input / token_ids",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note="Tokenisation, embedding lookup happen trusted-side.",
        ),
        _row(
            phase=p, obj="X (plaintext hidden state)",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note="Plaintext hidden state never crosses the boundary.",
        ),
        _row(
            phase=p, obj="X_tilde = X N_x",
            visibility=_VIS_GPU_MASKED,
            exposed_form="X @ N_x (masked)",
            note="Only masked hidden state is uploaded.",
        ),
        _row(
            phase=p, obj="A_tilde, B_tilde",
            visibility=_VIS_GPU_MASKED,
            exposed_form="N_x^T A M, M^T B N_y",
            note="GPU sees masked, rank-padded LoRA factors only.",
        ),
        _row(
            phase=p, obj="Y_tilde = X_tilde A_tilde B_tilde",
            visibility=_VIS_GPU_MASKED,
            exposed_form="masked LoRA output",
            note=(
                "Algebraically equals X A B N_y so the trusted side "
                "recovers Y by Y_tilde N_y^T."
            ),
        ),
        _row(
            phase=p, obj="Y (plaintext output)",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note="Recovered only on the trusted side.",
        ),
    ])

    # --- Masked backward ---------------------------------------------------
    p = "masked_backward"
    rows.extend([
        _row(
            phase=p, obj="target_tilde = target @ N_y",
            visibility=_VIS_GPU_MASKED,
            exposed_form="target @ N_y (masked)",
            note=(
                "MSE loss is computed against masked target. Orthogonal "
                "N_y preserves the L2 loss exactly. Plain target never "
                "leaves the trusted side."
            ),
        ),
        _row(
            phase=p, obj="grad_Y (plaintext)",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note="Plaintext output gradient never crosses the boundary.",
        ),
        _row(
            phase=p, obj="grad_Y_tilde = 2 (Y_tilde - target_tilde) / n",
            visibility=_VIS_GPU_MASKED,
            exposed_form="masked output gradient",
            note=(
                "Equals 2 (Y - target) N_y / n; orthogonal masks "
                "commute through MSE."
            ),
        ),
        _row(
            phase=p, obj="grad_A_tilde",
            visibility=_VIS_GPU_MASKED,
            exposed_form="N_x^T grad_A M (masked)",
            note=(
                "Algebraic equivalence verified per step at float64 "
                "machine precision."
            ),
        ),
        _row(
            phase=p, obj="grad_B_tilde",
            visibility=_VIS_GPU_MASKED,
            exposed_form="M^T grad_B N_y (masked)",
            note=(
                "Algebraic equivalence verified per step at float64 "
                "machine precision."
            ),
        ),
        _row(
            phase=p, obj="grad_A, grad_B (plaintext)",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note="Plaintext gradients never leave the trusted side.",
        ),
    ])

    # --- Masked SGD --------------------------------------------------------
    p = "masked_sgd"
    rows.extend([
        _row(
            phase=p, obj="A_tilde_next, B_tilde_next",
            visibility=_VIS_GPU_MASKED,
            exposed_form=(
                "A_tilde - lr * grad_A_tilde, B_tilde - lr * "
                "grad_B_tilde"
            ),
            note=(
                "Linear update; right-multiplication by orthogonal "
                "masks distributes, so masked SGD is algebraically "
                "equivalent to plaintext SGD after trusted-side "
                "recovery."
            ),
        ),
        _row(
            phase=p, obj="plaintext optimizer state",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note="SGD has no persistent state besides the parameters.",
        ),
    ])

    # --- Masked momentum SGD ----------------------------------------------
    p = "masked_momentum_sgd"
    rows.extend([
        _row(
            phase=p, obj="V_A_tilde, V_B_tilde (masked momentum buffers)",
            visibility=_VIS_GPU_MASKED,
            exposed_form=(
                "V_tilde <- mu V_tilde + grad_tilde; param_tilde <- "
                "param_tilde - lr V_tilde"
            ),
            note=(
                "Heavy-ball update is linear in the gradients; "
                "orthogonal masks commute, so masked momentum SGD is "
                "algebraically equivalent to plaintext momentum SGD "
                "after recovery."
            ),
        ),
        _row(
            phase=p, obj="plaintext momentum buffers V_A, V_B",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note=(
                "Plaintext momentum buffers never live on the GPU; "
                "they can be recovered from the masked buffers on the "
                "trusted side only when needed."
            ),
        ),
        _row(
            phase="masked_adamw_unsupported", obj="AdamW second moments",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form=(
                "masked_adamw_step_unsupported raises "
                "DenseMaskedAdamWUnsupported"
            ),
            note=(
                "Coordinate-wise second moments are not invariant "
                "under dense orthogonal mixing; AdamW under dense "
                "masks is unsupported."
            ),
        ),
    ])

    # --- Final adapter recovery / audit -----------------------------------
    p = "final_adapter_recovery_or_audit"
    rows.extend([
        _row(
            phase=p, obj="A_pad_recovered, B_pad_recovered",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form=(
                "A_pad = N_x A_tilde M^T, B_pad = M B_tilde N_y^T "
                "(trusted side)"
            ),
            note=(
                "Recovery uses the orthogonal inverses; never executed "
                "on the GPU."
            ),
        ),
        _row(
            phase=p, obj="A_real, B_real (trained)",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="extracted from A_pad / B_pad (trusted side)",
            note=(
                "True rank slices used for downstream inference; never "
                "exposed to the GPU in plaintext."
            ),
        ),
        _row(
            phase=p, obj="published fingerprints",
            visibility=_VIS_PUBLIC,
            exposed_form="shapes and 16-char SHA-256 prefixes only",
            note=(
                "Outputs publish summary scalars and short fingerprints "
                "to enable third-party audit without exposing raw "
                "tensors."
            ),
        ),
    ])

    # --- Trained LoRA inference --------------------------------------------
    p = "trained_lora_inference"
    rows.extend([
        _row(
            phase=p, obj="user_input / token_ids",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note="As in masked forward, tokens are trusted-side only.",
        ),
        _row(
            phase=p, obj="X_infer (plaintext hidden state)",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note="Plaintext hidden state not uploaded.",
        ),
        _row(
            phase=p, obj="X_infer_tilde = X_infer N_x",
            visibility=_VIS_GPU_MASKED,
            exposed_form="masked hidden state",
            note="GPU only sees the masked hidden state.",
        ),
        _row(
            phase=p, obj="A_tilde, B_tilde (trained)",
            visibility=_VIS_GPU_MASKED,
            exposed_form="N_x^T A_real_pad M, M^T B_real_pad N_y",
            note=(
                "Trained masked LoRA factors uploaded for inference; "
                "true rank still hidden behind cancellation padding."
            ),
        ),
        _row(
            phase=p, obj="base_model_W",
            visibility=_VIS_PUBLIC,
            exposed_form=(
                "transformed_to_preserve_hidden_state_masks at the "
                "boundary"
            ),
            note=(
                "Base model is public but its composition with the "
                "trusted-side mask boundary preserves hidden-state "
                "masks at inference time."
            ),
        ),
        _row(
            phase=p, obj="Y_infer (plaintext output)",
            visibility=_VIS_PLAINTEXT_TRUSTED,
            exposed_form="never_exported",
            note=(
                "Final output recovered on the trusted side by "
                "Y_infer = Y_infer_tilde @ N_y^T."
            ),
        ),
    ])

    return rows


# ---------------------------------------------------------------------------
# Top-level report builder
# ---------------------------------------------------------------------------


def build_lifecycle_report() -> dict[str, Any]:
    rows = _lifecycle_rows()
    phases = sorted({r["phase"] for r in rows})

    # Group view for the JSON consumer.
    by_phase: dict[str, list[dict[str, str]]] = {p: [] for p in phases}
    for r in rows:
        by_phase[r["phase"]].append(r)

    # Sanity invariants.
    invariants = {
        "plaintext_A_or_B_visible_to_gpu": False,
        "plaintext_grad_A_or_B_visible_to_gpu": False,
        "masks_visible_to_gpu": False,
        "plaintext_optimizer_state_visible_to_gpu": False,
        "user_input_visible_to_gpu": False,
        "base_model_W_public": True,
        "base_model_W_transformed_to_preserve_hidden_state_masks": True,
        "gpu_sees_masked_hidden_states": True,
        "gpu_sees_masked_lora_adapters": True,
        "gpu_sees_masked_lora_gradients": True,
        "gpu_sees_masked_momentum_buffers": True,
        "adamw_under_dense_masks_supported": False,
    }

    return {
        "status": "ok",
        "stage": "7.6",
        "report": "lora_training_inference_lifecycle",
        "required_phases": list(_REQUIRED_LIFECYCLE_PHASES),
        "phases_present": phases,
        "lifecycle_rows": rows,
        "lifecycle_by_phase": by_phase,
        "invariants": invariants,
        "honesty_phrases": list(_REQUIRED_HONESTY_PHRASES),
        "paper_safe_wording": (
            "masked-gradient LoRA provides algebraic equivalence for "
            "SGD/Momentum under orthogonal masks and proxy-evaluated "
            "leakage mitigation; it does not provide formal, "
            "cryptographic, or semantic security."
        ),
        "formal_security_claim": False,
        "limitations": [
            "Lifecycle classification is descriptive; it is not a "
            "formal information-flow proof.",
            "Base model W is public; we rely on a trusted-side "
            "boundary transformation to preserve hidden-state masks. "
            "A real deployment would have to verify the boundary "
            "transformation matches the trusted side's mask choice.",
            "AdamW under dense masks is unsupported and is gated by "
            "DenseMaskedAdamWUnsupported in the ops module.",
            "Raw tensors, masks, adapters, gradients, and optimiser "
            "states are NEVER exported.",
        ],
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_json(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)


def _write_csv(report: dict[str, Any], path: str) -> None:
    rows = report["lifecycle_rows"]
    fields = ["phase", "object", "visibility", "exposed_form", "note"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Stage 7.6 — LoRA Training-to-Inference Lifecycle Report")
    w()
    w("## 1. Scope")
    w()
    w(
        "This report enumerates which objects are plaintext / trusted-"
        "only, GPU-visible masked, or public during each phase of the "
        "Stage 7.6 masked-gradient LoRA training-to-inference "
        "lifecycle. It is a paper-claims consistency artifact: it "
        "asserts the visibility classification implied by the "
        "construction but is not itself a formal information-flow "
        "proof."
    )
    w()
    w("## 2. Visibility classes")
    w()
    w("- `plaintext_trusted_only`: held only on the trusted (user) side.")
    w("- `gpu_visible_masked`: uploaded to the cloud accelerator in masked form.")
    w("- `public`: known to the cloud accelerator and any external observer.")
    w()
    w("## 3. Invariants")
    w()
    w("| invariant | value |")
    w("|---|---|")
    for k, v in report["invariants"].items():
        w(f"| `{k}` | {v} |")
    w()
    w("## 4. Lifecycle table")
    w()
    w("| phase | object | visibility | exposed_form | note |")
    w("|---|---|---|---|---|")
    for r in report["lifecycle_rows"]:
        note = r["note"].replace("|", "\\|")
        exposed = r["exposed_form"].replace("|", "\\|")
        w(
            f"| `{r['phase']}` | `{r['object']}` | "
            f"`{r['visibility']}` | {exposed} | {note} |"
        )
    w()
    w("## 5. Per-phase summary")
    w()
    for phase in report["required_phases"]:
        w(f"### {phase}")
        w()
        for r in report["lifecycle_by_phase"].get(phase, []):
            w(
                f"- `{r['object']}` -- **{r['visibility']}** -- "
                f"{r['note']}"
            )
        w()
    # Include the masked_adamw_unsupported pseudo-phase if present.
    extras = [
        ph for ph in report["phases_present"]
        if ph not in report["required_phases"]
    ]
    for phase in extras:
        w(f"### {phase}")
        w()
        for r in report["lifecycle_by_phase"].get(phase, []):
            w(
                f"- `{r['object']}` -- **{r['visibility']}** -- "
                f"{r['note']}"
            )
        w()
    w("## 6. Honesty phrases (verbatim)")
    w()
    for phrase in report["honesty_phrases"]:
        w(f"- {phrase}")
    w()
    w("## 7. Limitations")
    w()
    for lim in report["limitations"]:
        w(f"- {lim}")
    w()
    w("## 8. Paper-safe wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()
    w(f"`formal_security_claim`: `{report['formal_security_claim']}`")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, Any], *, outputs_dir: str = "outputs",
    json_filename: str = "lora_training_inference_lifecycle.json",
    csv_filename: str = "lora_training_inference_lifecycle.csv",
    md_filename: str = "lora_training_inference_lifecycle.md",
) -> tuple[str, str, str]:
    os.makedirs(outputs_dir, exist_ok=True)
    json_path = os.path.join(outputs_dir, json_filename)
    csv_path = os.path.join(outputs_dir, csv_filename)
    md_path = os.path.join(outputs_dir, md_filename)
    _write_json(report, json_path)
    _write_csv(report, csv_path)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    return json_path, csv_path, md_path


__all__ = [
    "build_lifecycle_report",
    "render_markdown",
    "write_reports",
]
