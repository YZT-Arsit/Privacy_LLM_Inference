# Frozen transformed tied-weight convention

The plaintext model ties embedding and LM head to ONE source matrix E
(shape [vocab, hidden]); see tied_embedding_audit.json for the measured tie.

## Convention (row-vector code convention y = x W^T)
- Embedding view (masked hidden entering the stack):
      E_embed_tilde = E @ N_embed
  where N_embed is the (orthogonal) residual mask at the input boundary. A token
  id's masked embedding row is `E[id] @ N_embed`.
- LM-head view (masked logits boundary):
      W_lm_tilde = N_final^{-1} @ E^T @ M_vocab
  where N_final is the final-residual mask (orthogonal) and M_vocab = D·Pi is the
  O(V) monomial vocab mask. Masked logits = rmsnorm_core(h_tilde) @ W_lm_tilde.

## Explicit statements (to verify at package build)
- tying preserved as physical storage sharing:  **NO** — the package stores TWO
  transformed views (E_embed_tilde, W_lm_tilde), not one shared tensor.
- transformed_storage_shared: false ; transformed_views_count: 2 ; tied_plaintext_source: true
- can either view be derived from the other WITHOUT secret masks? Deriving
  W_lm_tilde from E_embed_tilde requires N_embed^{-1}, N_final^{-1}, M_vocab (all
  secret) -> NO closed-form cross-derivation without secrets.
- cross_view_attack_surface: **true** — both views expose transformed functions of
  the SAME E, giving D2 extra constraints (E_embed_tilde^T and W_lm_tilde share E^T
  up to the secret feature/vocab masks). This is why the D2 tied-view cross-attack
  is MANDATORY (correction 5).

## Manifest fields to emit (package build)
tied_plaintext_source=true, transformed_views_count=2, transformed_storage_shared=false,
embedding_view_formula="E N_embed", lm_head_view_formula="N_final^{-1} E^T M_vocab",
cross_view_attack_surface=true.
