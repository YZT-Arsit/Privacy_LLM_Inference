# Gate 3 — network topology (frozen)

## Roles
- **External verifier (Mac):** control plane ONLY. Issues the fresh challenge nonce,
  verifies the signed DCAP appraisal + measurement policy + runtime hash + report_data,
  and authorizes the session. **Never relays model tensors.**
- **H800 GPU worker:** untrusted data-plane peer. Runs real Qwen2.5-0.5B forward/backward
  + masked-domain SGD; sends masked (vocab-permuted) logits and receives masked dlogits.
- **TDX trusted loss service:** private CE + masked dlogits; holds the labels + the vocab
  permutation; runs inside the real Intel TDX guest (39.96.4.252). Binds 127.0.0.1 only.

## Data plane (direct H800 <-> TDX; no Mac relay)
- Proven reachability: H800 (container `autodl-container-35eb4cbcf2-9d84e25e`, internal
  172.17.0.8) -> TDX 39.96.4.252: **ICMP RTT ~23.6 ms, 0% loss; TCP :22 OPEN**.
  TDX (172.30.25.153) -> H800 gateway (connect.westb.seetacloud.com = 116.172.66.188):19948 **OPEN**.
- The trusted service binds `127.0.0.1:18091` (never public). The data path is a **single
  SSH tunnel between H800 and TDX**, established as a **TDX-initiated reverse forward**
  (`ssh -R 18091:127.0.0.1:18091` from TDX to the H800 gateway). Chosen over a forward
  H800->TDX tunnel specifically so that **no TDX private key ever resides on the untrusted
  GPU**; the H800 login secret lives only on the trusted TDX (SSHPASS env, not argv).
- Effect: the H800 worker connects to its own `127.0.0.1:18091`, which the reverse tunnel
  carries directly to the TDX service. Logits/dlogits traverse H800 <-> gateway <-> TDX
  only. The Mac is not on this path.

## Control plane (Mac <-> TDX, control only)
- The Mac verifies the quote from the challenge bundle relayed by the H800 (the H800 makes
  the /train/challenge call directly to TDX; the Mac receives only the public bundle over
  SSH, verifies, and authorizes). No tensors cross the Mac.

## Transport security
- Attestation-bound X25519 ECDH -> HKDF-SHA256 (salt = quote hash) -> ChaCha20-Poly1305
  AEAD, established directly between H800 and TDX after the Mac authorizes. Safe binary
  codec (no pickle). The trusted service is never exposed publicly without this
  authenticated transport.

## Topology diagram
    External verifier (Mac)
        -- fresh nonce + signed-attestation verification + authorization --> (control)
    H800 worker  <==== direct attestation-bound AEAD session (reverse SSH tunnel) ====>  TDX loss service
                 (masked logits / masked dlogits; NO Mac relay)
