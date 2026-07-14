# ObfuscaTune Protocol/Fidelity Preparation

Status: **PREPARATION ONLY — GPU launch prohibited**.

The repository already contains a Qwen inference simulator with orthogonal/random/conditioned
matrices, explicit nonlinear TEE boundaries, GQA handling, decode cache support, and honest exposure
labels. It does not yet contain a validated ObfuscaTune LoRA training path on the frozen E2E protocol.

Before any run, implement the training path in a new versioned module and require: orthogonal
condition number 1; plaintext-reference equivalence; prefill and cached-decode agreement; exact
Qwen2.5-0.5B, E2E pool, target set, rank, alpha, optimizer, schedule, dtype, and decoding match;
explicit Q/K/V, attention-score, and plaintext-KV exposure accounting; no claim that simulator timing
is real TEE timing; and fail-closed rejection of silent plaintext or trusted-shortcut fallback.

Required negative controls include wrong inverse, non-orthogonal ill-conditioned matrix, stale matrix,
wrong cache position, missing nonlinear boundary, mismatched adapter target set, and unreported QKV/KV
exposure. The fidelity checklist remains launch-blocking until every required implementation gate passes.
