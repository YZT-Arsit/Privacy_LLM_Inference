# Network topology — direct H800 <-> TDX transport

```
   ┌─────────┐  control-plane only (setup + attestation record; NO payload)
   │   Mac   │───────────────────────────────────────────────┐
   └─────────┘                                                 │
      authorize H800 key on TDX; provision session key/config  │
                                                               ▼
   ┌──────────────┐   persistent SSH (port 22, forced cmd)  ┌──────────────┐
   │  H800 (GPU)  │═══════════════════════════════════════▶ │  TDX (enclave)│
   │ package-only │   framed msgs: logits→CE/dlogits;       │ labels, γ,    │
   │ MaskedQwen   │◀═══════════════════════════════════════ │ vocab-perm,   │
   │ LoRA in-mem  │   corrgrads→corrected                   │ session key   │
   └──────────────┘   ONE channel reused across all steps   └──────────────┘
```

- **Data plane:** H800 ↔ TDX only, over one persistent SSH channel (port 22, the only open port).
  The TDX service is an SSH **forced command** (no shell, no port-forward). No new inbound port.
- **No Mac in the data plane:** the Mac never downloads/writes/re-uploads per-step payloads and does
  no step-by-step orchestration. It authorizes the H800→TDX key and records attestation (control).
- **Measured:** RTT 22.9 ms (raw) / 65 ms (SSH-exec); up 12.7 MB/s; down ~0.6 MB/s (throttled →
  dlogits return dominates; mitigate with bf16 wire). End-to-end step latency measured in validation.
- **Session:** one fresh attestation at setup (NOT per step); one verified session reused across steps.
