# Message flow

## Provisioning

1. Model provider -> trusted: plaintext model and model/config binding.
2. Trusted: sample orthogonal transforms, create W* and initial A*/B*.
3. Trusted -> GPU: transformed base package, transformed factors, public shapes
   and contract ID. R and plaintext W/A/B are retained or destroyed according
   to provisioning policy and never sent.

## Input-side forward/backward

1. Trusted -> GPU: X* and operation metadata.
2. GPU: compute `Y = X*(W*+sA*B*)`; return plaintext Y as implied by Eq. 1-3.
3. Trusted executes the next nonlinear operation.
4. During backward, trusted -> GPU: upstream G.
5. GPU: compute dA*, dB*, dX* and update external factors; GPU -> trusted: dX*.
6. Trusted restores dX where a trusted predecessor requires it.

## Output-side forward/backward

1. Trusted -> GPU: plaintext nonlinear result X.
2. GPU -> trusted: `Y*=X(W*+sA*B*)`.
3. Trusted restores Y and continues residual/nonlinear computation.
4. Trusted transforms canonical upstream gradient to `G*=G R_o` and sends it.
5. GPU computes dA*, dB*, dX and updates external factors; dX is returned.

## Checkpoint/restart

GPU persists A*/B*/m*/v*, step, optimizer contract, tensor hashes and exact
base/config/transform-generation bindings. Restart fails closed on any mismatch.
The minimal checkpoint does not support rekeying or canonical adapter export.

## Instrumentation requirements for the later runner

Every message records direction, operation, sequence number, tensor names,
shapes, dtypes, payload bytes, serialization time, transport time and trusted
compute time. Fields are labeled MEASURED, DERIVED_FROM_MEASURED, PROJECTED or
UNAVAILABLE.
