# Operator execution table

| Operator | Trusted | GPU | Input -> output | Classification |
|---|---:|---:|---|---|
| Token embedding | yes | no | token IDs -> X | PAPER_EXPLICIT |
| RMSNorm | yes | no | plaintext -> plaintext | ADAPTED_WITH_JUSTIFICATION |
| Obfuscate input | yes | no | X -> X* | PAPER_EXPLICIT |
| q/k/v/gate/up base+LoRA matmul | no | yes | X*, W*, A*, B* -> plaintext Y | PAPER_INFERRED for LoRA |
| RoPE | yes | no | plaintext Q/K -> rotated Q/K | ADAPTED_WITH_JUSTIFICATION |
| GQA repeat_kv | yes | no | pre-repeat K/V -> repeated K/V | ADAPTED_WITH_JUSTIFICATION |
| Attention score/softmax | yes | no | Q,K -> probabilities | PAPER_EXPLICIT |
| Context multiply | trusted primary | no | probabilities,V -> context | Conservative Fig. 2 reading |
| o/down base+LoRA matmul | no | yes | plaintext X, W*, A*, B* -> Y* | PAPER_INFERRED for LoRA |
| De-obfuscate output/add bias | yes | no | Y* -> Y | PAPER_EXPLICIT |
| SiLU and gate*up | yes | no | plaintext gate/up -> MLP activation | ADAPTED_WITH_JUSTIFICATION |
| Residual/dropout | yes | no | plaintext -> plaintext | PAPER_EXPLICIT |
| LM head/loss | yes | no | hidden/labels -> loss | PAPER_EXPLICIT |
| LoRA gradient matmuls | no | yes | saved external operands -> dA*,dB*,dX* | REPOSITORY_ADAPTATION |
| Gradient coordinate conversion | yes only at nonlinear boundary | no | dX* <-> dX | PAPER_INFERRED |
| AdamW A*/B* update | no | yes | factor/grad/m/v -> updated state | REPOSITORY_ADAPTATION |
| Checkpoint validation | local owner plus trusted binding check later | yes | package -> accept/reject | REPOSITORY_ADAPTATION |

No operator delegates to the G2 selective-correction, masked-dlogits, or split
AdamW implementation.
