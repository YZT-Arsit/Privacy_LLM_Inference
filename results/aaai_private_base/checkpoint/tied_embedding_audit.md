# Tied embedding / LM-head audit (real model)

- config tie_word_embeddings: True
- storage shared (same tensor object): True
- numerically identical: True
- lm_head stored in checkpoint: False
- loading: HF ties lm_head to embed_tokens when tie_word_embeddings=True and lm_head.weight is absent from the checkpoint; they share storage

Implication for the private package: embedding and LM head derive from ONE plaintext source matrix E. The package must apply the frozen tied-weight convention (two transformed views of E) and D2 must include the tied-view cross-view attack (see checkpoint/tied_weight_convention.md).
