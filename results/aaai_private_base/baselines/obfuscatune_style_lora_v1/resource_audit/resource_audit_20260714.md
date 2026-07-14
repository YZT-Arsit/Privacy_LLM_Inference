# A10 + TDX resource audit — 2026-07-14

## Available isolated pair

- A10: `root@39.107.123.173`, host `iZ2zeajy3dc0ssoyi2cf3vZ`, NVIDIA A10,
  UUID `GPU-0729b83b-60dd-c88d-2b9c-2049e0127eae`, 23028 MiB total,
  0 MiB used, 0% utilization, no GPU process at 2026-07-14 20:57 CST.
- TDX: `root@39.96.43.122`, host `iZ2zebdihkqll10ragqp0cZ`,
  `/dev/tdx_guest` present, load 0.00, no experiment Python/service process at
  2026-07-14 20:58 CST.
- Historical private data-plane addresses: A10 `172.30.25.154`, TDX
  `172.30.25.153`, same VPC/vSwitch according to the frozen migration manifest.
- No fresh quote was generated because the fidelity gate blocked execution before
  an attested service could be started.

## Occupied pair left untouched

The newer pair A10 `8.147.119.67` / TDX `8.147.112.139` was not free. It was
running another session's `e2e_T4_all7_s1234_protected` process (A10 PID 26413,
about 5227 MiB allocated) and the paired TDX service. Nothing was stopped,
signalled, overwritten, or modified.

