"""Gate 3 — H800-side TDX private-loss client (vocab-permutation mask).

Runs ON the untrusted H800. Establishes an attestation-bound AEAD session DIRECTLY
with the TDX loss service (challenge nonce is MAC-issued; the guest quote is verified
externally on the Mac; this client only carries out the ECDH handshake + data plane).
Model logits/dlogits go H800<->TDX directly -- never through the Mac.

Per step (loss boundary): Z_tilde = Z[:, pi]  ->  TDX recovers Z = Z_tilde[:, pi_inv],
computes private CE (fp32) with labels held in TDX, returns G_tilde = G[:, pi]; the GPU
un-permutes G = G_tilde[:, pi_inv]. pi is delivered by TDX at authorized init.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pllo.experiments.real_tdx_training_client import TrustedBoundaryClient  # noqa: E402


class TdxLossSession:
    """Direct attested H800<->TDX session for the permutation private-loss boundary."""

    def __init__(self, base_url: str):
        self.client = TrustedBoundaryClient(base_url=base_url)
        self.pi = None
        self.pi_inv = None
        self.wire_bytes = {"init_out": 0, "logits_out": 0, "dlogits_in": 0}
        self.rpc_latency = []

    # ---- control-plane: challenge (Mac-issued nonce) -> export bundle for Mac verify ----
    def challenge(self, *, run_id, config_digest, model_id, gradient_convention,
                  optimizer_mode, verifier_nonce) -> dict:
        self.client.run_id = run_id
        self.client.config_digest = config_digest
        bundle = self.client._post_json("/train/challenge", {
            "run_id": run_id, "config_digest": config_digest,
            "verifier_nonce": verifier_nonce, "model_id": model_id,
            "gradient_convention": gradient_convention, "optimizer_mode": optimizer_mode})
        self.client.attestation_bundle = bundle
        return bundle

    # ---- after Mac authorizes: attested ECDH handshake (direct) ----
    def handshake(self, bundle: dict, *, run_id, config_digest, gradient_convention,
                  optimizer_mode):
        from pllo.experiments.real_tdx_kex import (
            AeadReceiver, AeadSender, VerifierKeyExchange)
        self.client.run_id = run_id                 # run phase is a fresh process
        self.client.config_digest = config_digest
        vkex = VerifierKeyExchange()
        kx = vkex.complete(guest_public_hex=bundle["guest_ephemeral_public_hex"],
                           run_id=run_id, config_digest=config_digest,
                           optimizer_mode=optimizer_mode,
                           gradient_convention=gradient_convention,
                           quote_hash_hex=bundle["quote_hash"])
        self.client._post_json("/train/handshake", {"verifier_public_hex": vkex.public_hex})
        self.client._aead_tx = AeadSender(key=kx.c2s_key, run_id=run_id, direction=1)
        self.client._aead_rx = AeadReceiver(key=kx.s2c_key, run_id=run_id, direction=2)
        self.client.attested = True

    # ---- authorized init: register private labels in TDX, receive pi (input boundary) ----
    def init(self, *, lora_manifest, vocab_size, labels_by_step):
        t = time.time()
        resp = self.client.init_session(run_id=self.client.run_id,
                                        config_digest=self.client.config_digest,
                                        lora_manifest=lora_manifest, vocab_size=vocab_size,
                                        labels_by_step=labels_by_step)
        self.rpc_latency.append(("init", time.time() - t))
        self.pi = resp["logit_permutation"].to(torch.long)
        self.pi_inv = torch.empty_like(self.pi); self.pi_inv[self.pi] = torch.arange(self.pi.numel())
        assert int(self.pi.numel()) == vocab_size and int(torch.unique(self.pi).numel()) == vocab_size
        return resp

    # ---- loss boundary: masked logits -> private CE -> masked dlogits ----
    def logits_loss(self, step_id: int, Z: torch.Tensor, wire_dtype=torch.bfloat16):
        """Z: (n, vocab) real logits on GPU. Returns (loss, dlogits (n,vocab) on GPU)."""
        pi = self.pi.to(Z.device)
        Zt = Z.detach().index_select(1, pi).to(wire_dtype).cpu()   # Z_tilde = Z[:, pi], bf16 wire
        self.wire_bytes["logits_out"] += Zt.numel() * Zt.element_size()
        t = time.time()
        resp = self.client.logits_loss(step_id, Zt)
        self.rpc_latency.append(("logits_loss", time.time() - t))
        g_masked = resp["masked_logit_gradient"]                   # G_tilde (n, vocab)
        self.wire_bytes["dlogits_in"] += g_masked.numel() * g_masked.element_size()
        pi_inv = self.pi_inv.to(g_masked.device)
        G = g_masked.to(torch.float32).index_select(1, pi_inv)     # un-permute
        return resp["loss"], G.to(Z.device)
