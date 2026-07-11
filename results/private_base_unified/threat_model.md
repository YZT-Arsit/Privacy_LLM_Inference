# Private-base threat model (FROZEN v1.0)

The current main-line threat model. The public GitHub README may still describe a
public-weight scope; it is **not** the source of truth here.

## Adversary
An honest-but-curious (and, for W1/A1, actively analytic) party with **full
control of the untrusted GPU/host**: it sees every tensor and weight resident in
the GPU process and on the host filesystem/path used by the experiment, knows the
architecture, tensor shapes, residual topology, normalization structure, QKV
grouping, the public tokenizer, and output-head behavior. It can hold a **static
transformed package** offline and analyze it without time limit. It does **not**
know the plaintext base weights, the mask secrets, or the private labels.

Trust anchor: a real Intel TDX confidential guest (attested) + a trusted offline
packager. Only these hold plaintext model material and mask secrets.

## Protected assets
base weights · embeddings · user inputs · hidden states · KV cache · LoRA
adapters · LoRA gradients · labels/loss · optimizer state (where applicable) ·
logits.

## Untrusted GPU/host must NEVER receive (plaintext)
checkpoint tensors · plaintext embeddings · plaintext hidden states · plaintext
LoRA parameters/gradients · plaintext labels · plaintext logits.

## What crosses each boundary
| boundary | may hold plaintext | crosses to untrusted |
|---|---|---|
| trusted packager (offline) | full plaintext checkpoint, mask secrets | transformed weights/embeddings/head + hashes only |
| TDX guest (online) | mask secrets, labels, per-step private CE, (O1-C) optimizer | masked logits in / masked dlogits out |
| untrusted GPU | — | transformed weights + masked activations only |

## Known / accepted leakage (recorded, not claimed away)
1. **Attention score matrix** `Q_rope K_rope^T` + softmax `probs` are exposed on
   the GPU by the paired-feature construction (design_spec §C/§G). Not claimed
   confidential. Partial mitigation: layer-0 TEE relocation.
2. **Structural weight leakage** — a static transformed package exposes the folded
   chain `W_tilde_l = N_l^{-1} W_l N_{l+1}`. Whether this permits functional model
   recovery is the W1 go/no-go question; orthogonal masks are known-broken by
   self-Gram alignment, non-orthogonal is an untested candidate.
3. bf16 numerical residuals (characterized in C1 fp32-vs-bf16).

## Explicit non-claims (until the evidence exists)
- No private-base security is claimed from "the package is transformed" alone.
- No non-orthogonal-mask security is claimed before W1 + S1 + A1 complete.
- No attention-score confidentiality, zero-leakage, or formal-security claim.

## Fail-closed rules
- `private_base_required = true`: if a plaintext checkpoint/model is present in the
  GPU process or the experiment's filesystem path, the run aborts.
- `execution_profile = paper_safe`: `current`/`trusted_shortcut` backends,
  `nonlinear_trusted_calls > 0`, `plaintext_hidden_materializations > 0`, or a
  silent fallback are all rejected.
- Mode/mask-family/backend mismatch vs the frozen config is fail-closed, never a
  silent downgrade.
