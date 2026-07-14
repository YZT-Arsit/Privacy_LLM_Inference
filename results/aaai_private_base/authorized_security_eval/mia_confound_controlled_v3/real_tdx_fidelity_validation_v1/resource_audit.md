# Resource audit

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample fidelity-validation subset

- Pair1 A10: `39.107.123.173` / `172.30.25.154`, hostname `iZ2zeajy3dc0ssoyi2cf3vZ`, NVIDIA A10, UUID `GPU-0729b83b-60dd-c88d-2b9c-2049e0127eae`, 23028 MiB total / 22588 MiB free before launch, no competing process or tmux/screen session.
- Pair1 TDX: `39.96.43.122` / `172.30.25.153`, hostname `iZ2zebdihkqll10ragqp0cZ`, real KVM Intel TDX guest with `/dev/tdx_guest`, quote handler available, no pre-existing trusted service.
- Pair1 T3 was terminal (`G2_ALL_DONE`) and its PIDs had exited before this validation.
- Pair2/T4 remained independent and running; no Pair2 write, stop, restart, signal, or registry update occurred.
- Repository HEAD recorded during final offline analysis: `d30a9a3b59367819e1b6520d87307abd8c5916d8`; working tree remained dirty with researcher-owned changes.
- WD0 files excluded from deployment: `scripts/a10_batch_runner.py`, `scripts/generative_lora/gate_e2e_protected.py`, `scripts/tdx_adamw_protocol.py`, `scripts/tdx_persistent_service.py`.
- Current status entries observed: 32 (includes this append-only output and unrelated researcher work).
