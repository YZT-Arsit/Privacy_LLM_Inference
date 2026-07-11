# Gate 3 — limitations

1. **Base weights plaintext on the GPU.** This profile protects the LoRA adapters +
   optimizer (masked-domain SGD) and the loss/labels (TDX). The frozen base weights run
   in plaintext HF Qwen on the GPU. The final private-base threat model is NOT claimed;
   base folding is the separately-validated inference layer, not applied here.
2. **Permutation logit mask is protocol-structural under a plaintext base.** vocab=151936
   forces an O(vocab) permutation (a dense head is ~92 GB). It hides token<->logit
   identity and keeps labels in TDX, but leaks the logit-value multiset, and because the
   GPU legitimately holds true Z + pi here, it adds no confidentiality beyond the AEAD.
   Its confidentiality role is realised only with a folded LM head (GPU never holds Z/pi).
3. **Training input visible to the GPU** (it tokenises/embeds the text); only the loss/
   label boundary is trusted. Full input protection is the separate staged path.
4. **bf16 residuals not yet proven purely numerical.** ~3.6% ΔW/gradB, ~1.8% next-logits.
   Consistent with bf16 + order-of-operations; loss diff 2.2e-5 and top1=1.0 indicate
   numerical origin, but the fp32/reorder control has not been run.
5. **run_id fixed to the service instance.** Freshness per training run is provided by a
   fresh single-use verifier nonce (=> fresh quote + fresh AEAD session each run), not by
   a distinct run_id; a distinct run_id would require restarting + re-quoting the service.
6. **Throughput.** The single H800<->TDX SSH tunnel runs ~0.5 MB/s through the AutoDL
   gateway, so each 38.6 MB (bf16) logit RPC takes ~80 s. This is a connectivity limit of
   the shared gateway, NOT a system performance claim (0.5B is a feasibility/correctness
   milestone). The 500 MB analysis tensors are therefore kept + compared ON the H800.
7. **Validation loss** not run in the protected loop (would add trusted invocations); the
   training-loss trajectory alignment vs plaintext SGD is the stability signal.
8. **TDX guest memory.** The 14 GB TD OOM-killed an earlier fp32 CE at vocab 151936; the
   service CE was made memory-frugal (F.cross_entropy + freed intermediates + bf16 return).
