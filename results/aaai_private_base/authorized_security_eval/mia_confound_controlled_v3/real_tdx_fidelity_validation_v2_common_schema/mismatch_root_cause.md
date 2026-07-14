# Mismatch root cause

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample subset
Schema: common-semantics schema v2

1. **Protocol-semantic mismatch.** `hidden.final.last_l2` observes post-gamma `hidden_states[-1]` in simulation, but the real package exposes only a gamma-free RMSNorm core or a pre-norm residual because gamma is folded into `lm_head`.
2. **Finite-precision numeric difference.** BF16 kernels, reduction order, and FP32 serialization produce small matched-path differences. The three layer-0 K reductions are near-constant and therefore correlation/rank diagnostics are ill-conditioned despite mean absolute errors below `8e-6`.
3. **Collector implementation difference.** The v1 primary and corrective collector intentionally probed two distinct real final-hidden points; neither is a substitute for the simulated post-gamma tensor. Other 610 columns use matched graph points.
4. **Metadata contamination.** None detected: feature matrices contain no file, run/session, checkpoint, mode, step/epoch, path-presence, member, or split field.
