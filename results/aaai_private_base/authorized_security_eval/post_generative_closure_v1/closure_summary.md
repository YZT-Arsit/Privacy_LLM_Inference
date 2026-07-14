# Authorized security evaluation closure

**Overall verdict: FAIL_CLOSED.** All new evaluated views are labeled **SIMULATED_PROTOCOL_VIEW**. No real TDX execution occurred; the later handoff is **PREPARED_NOT_EXECUTED**. The researcher-owned resource is an NVIDIA A10 (`GPU-209b577b-b17f-22ab-eed7-4914b5b6c7a5`), despite being described as A100 in the supplied note.

| Phase | Status | Principal evidence |
|---|---|---|
| 0 — protocol coverage | Complete | Existing attacks classified by semantic compatibility; filename-only differences did not trigger reruns. |
| 1 — artifact misuse-resistance | **Fail** | All 242 transformed tensors initialize; R1/R2 stop before model forward without session and create no output. However, the copied native entry still has an unmanifested raw `--adapter` load path, so runtime binding is not fully enforced. |
| 2 — `SURROGATE_FIDELITY_PILOT` | **Fail** | At 1%/5 queries, coarse property agreement is V0 0.4667, V2 0.5556, and V0+V2 0.5778. The apparent gain is not claimable because the fixed-length source classifier is perfect. V2−V0 becomes negative at 5%, 10%, and 20%. |
| 3 — fixed-length attention/KV | **Fail** | At one sequence length (51), shape-only AUC is 0.5 but attention, KV, hidden, logits, and combined metadata-free V2 each have source AUC 1.0. This is collection/source confounding, not membership or plaintext recovery. |
| 4 — fusion | Not run | The prerequisite phases did not pass controls. |

Membership inference remains **WITHHELD** because corrected shadow plans exist but completed corrected shadow feature captures do not. The 500-ID/706-feature later real-TDX comparison handoff is frozen without requesting or configuring a TDX resource.

No registry entry, active experiment, or shared active script was modified; no commit was created. Remote-to-local hashes for the two final GPU result directories match exactly (30 files).
