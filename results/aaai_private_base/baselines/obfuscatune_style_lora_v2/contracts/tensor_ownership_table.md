# Tensor ownership table

| Tensor | Shape | Stored coordinates | Owner | Lifetime | Crosses boundary | Persisted | Basis |
|---|---|---|---|---|---|---|---|
| token IDs | B x S | plaintext | trusted | request | no | no | PAPER_EXPLICIT |
| embedding/residual X | N x d | plaintext | trusted | block | no | no | PAPER_EXPLICIT |
| transformed X* | N x d | `X R_i` | GPU | operator | TEE->GPU | no | PAPER_EXPLICIT |
| base W* input | d_i x d_o | `R_i^T W` | GPU | package | provisioning only | yes | PAPER_EXPLICIT |
| base W* output | d_i x d_o | `W R_o` | GPU | package | provisioning only | yes | PAPER_EXPLICIT |
| A* input target | d_i x r | `R_i^T A` | GPU | run | no | yes | PAPER_INFERRED |
| B* input target | r x d_o | canonical B | GPU | run | no | yes | PAPER_INFERRED |
| A* output target | d_i x r | canonical A | GPU | run | no | yes | PAPER_INFERRED |
| B* output target | r x d_o | `B R_o` | GPU | run | no | yes | PAPER_INFERRED |
| optimizer m*/v* | factor-shaped | stored coordinate | GPU | run | no | yes | REPOSITORY_ADAPTATION |
| Q/K/V | B x heads x S x head_dim | plaintext | GPU then trusted op | layer | GPU->TEE | cache K/V | PAPER_EXPLICIT |
| attention scores/probs | B x heads x S x S | plaintext | trusted | operator | as required | no | PAPER_EXPLICIT |
| KV cache | layer-dependent | plaintext | GPU | session | yes | transient | AMBIGUOUS_IN_PAPER |
| MLP gate/up | N x d_ff | plaintext | GPU then trusted | operator | GPU->TEE | no | PAPER_INFERRED |
| transformed o/down output | N x d | `Y R_o` | GPU | operator | GPU->TEE | no | PAPER_EXPLICIT |
| loss/dlogits | task-shaped | plaintext | trusted | step | d-boundary only | no | PAPER_EXPLICIT/adapted |
| R/R^T | d x d | secret orthogonal | trusted | package/session | never | trusted only | PAPER_EXPLICIT |
| plaintext W/A/B | factor-dependent | canonical | trusted provisioning only or absent | provisioning | never | primary checkpoint: no | Contract |

The primary training checkpoint must not contain R, plaintext W, canonical A/B,
tokens, labels, or trusted plaintext activations.
