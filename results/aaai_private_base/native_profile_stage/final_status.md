# FINAL STATUS — EXACT_AND_NATIVE_PROFILES_PARTIAL

Not `COMPLETE` because the live-A10+TDX Profile-N training/lifecycle gates and the transport
live-revalidation are not executed this session. What passed vs what is missing:

## Passed (implemented + locally verified)
- **Security protocol fixes (Phase 1):**
  - 1.1 attestation **gates execution** — TDX refuses all protected ops and the A10 refuses
    optimizer init / any step under an invalid attested session (predicate + gate; **12/12** test).
  - 1.2 secure tensor frame replaces pickle on network payloads with a validated fixed binary
    frame (op/names/count/dtype/rank/shape/limits/target-set/no-dups/no-unknown-metadata; reject
    before allocation) — **13/13** test incl. all 8 required attacks.
  - 1.3 HMAC claim corrected in counters + comments (both files): channel-integrity + replay vs
    third-party only; does NOT prove honest-A10 origin or malicious-GPU integrity.
  - 1.4 dlogits instrumented + honest wording (A10 consumes transient masked dlogits; not
    persisted/logged; no plaintext labels/loss).
- **Profile registration (Phase 2):** Profile E (exact) and Profile N (native transformed-coord
  AdamW) frozen with correct labels (`plaintext_trajectory_equivalence=false` for N).
- **Profile N + local gate (Phase 2/3.1-3.2 local):** E exact vs plaintext (rel-diff 0); N is a
  distinct optimizer (ΔW rel-diff 0.96) reaching identical utility (top-1 1.0) at **0** trusted
  optimizer bytes (vs E 49152); signed-perm equivariance finding recorded.
- **Security-delta audit (Phase 3.3 local):** N adds transformed m,v exposure but does NOT reduce
  plaintext-ΔW recovery below E (both 1.0); no equal-security claim.
- **Adapter handoff (Phase 4):** export + loader + **10/10** fail-closed negative controls.

## Missing cells (enumerated)
1. **Phase 1.2 live-transport wiring**: the secure codec is unit+attack tested, but wiring it into
   the live A10<->TDX transport (replacing dump/load at all op sites) is NOT applied — requires an
   on-hardware transport smoke re-validation before the next real run (documented gate).
2. **Phase 3.2 real-Qwen 1/10/50-step gate** for Profile N on A10+TDX (loss/utility/ΔW cosine/logits
   KL/top-1/stability vs plaintext + Profile E) — hardware, not run.
3. **Phase 3.4 resource comparison** (TDX state bytes, A10<->TDX bytes/step, trusted compute, GPU
   optimizer time, total step time, trusted calls, peak VRAM, TDX CPU/mem) on the live path — not run.
4. **Phase 5 fresh-process lifecycle** (train -> export -> NEW inference process -> adapted
   generation vs trusted reference; restart) on A10+TDX — local pieces tested, hardware run deferred.
5. Profile N's on-hardware implementation inside the real `a10_batch_runner`/`tdx_persistent_service`
   training loop (GPU-resident m/v) — designed + locally gated, not wired into the live runner.

## Constraints honored
No commit; frozen artifacts untouched (baseline hashed before/after); completed security/utility
matrix not rerun. Additive protocol edits only, syntax-checked, with negative tests.
