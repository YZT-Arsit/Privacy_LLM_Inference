# System partition contract

| Operation/state | Trusted runtime | Untrusted accelerator | Basis/status |
|---|---:|---:|---|
| Tokenization/decryption/embedding | yes | no | plaintext |
| Orthogonal matrix generation and base transformation | yes | no | secret R/R^T |
| q/k/v/gate/up dense matmul | no | yes | transformed input/weight; plaintext output |
| RoPE, attention scores, softmax | yes | no | plaintext |
| K/V cache storage | no | yes | plaintext/exposed |
| o/down dense matmul | no | yes | plaintext input; transformed output |
| RMSNorm, SiLU, SwiGLU product, residual, loss | yes | no | plaintext |
| LoRA factors | no, unless adaptation requires conversion | yes | paper says outside; exact v2 coordinates versioned |
| Base plaintext weights | yes during provisioning | never | plaintext trusted only |
| Optimizer moments | unresolved until optimizer contract | unresolved | ambiguous in paper |
| Output head and generated tokens | yes | no | plaintext |

Every physical message will carry a v2 runtime ID, run ID, sequence number,
operation name, tensor inventory, dtype/shape, and byte count. A later real TDX
adaptation must fail closed on mismatched model/config/adapter bindings and must
not import or invoke the G2 training runner.
