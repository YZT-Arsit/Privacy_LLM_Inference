# Firewall constraints (H800 <-> TDX)

- **H800 -> TDX**: ONLY TCP/22 (SSH) reachable. 443/8443/9000/12345/50051 filtered. ICMP allowed
  (RTT ~22.9 ms). Outbound to arbitrary internet hosts appears restricted (1.1.1.1:443 blocked) —
  egress is allow-listed, but TDX:22 is reachable.
- **TDX inbound**: arbitrary service ports firewalled (Alibaba security group) → a direct
  app-listen socket on TDX is NOT reachable from H800. Hence transport option **B**: run the TDX
  service as an SSH **forced command**; the single persistent SSH channel (port 22) carries all
  framed application messages. No new inbound port needed.
- **TDX -> H800 egress appears bandwidth-throttled** (~0.6 MB/s bulk down vs 12.7 MB/s up). The
  returned dlogits (~12.5 MB bf16) dominate step latency; mitigations: bf16 wire, and (future)
  payload reduction. Measured end-to-end step latency is the definitive metric (Step 6 validation).
- **Mac**: control-plane only after setup (authorized the H800 key on TDX, records attestation).
  Not in the per-step data path.
