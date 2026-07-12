# Direct-channel transport anti-pattern audit

Files: `h800_direct_runner.py` (client framing), `tdx_persistent_service.py` (server framing),
`tdx_bench_service.py` (benchmark). Static code review of the anti-pattern checklist:

| Anti-pattern | Present? | Evidence |
|---|---|---|
| byte-at-a-time reads | **No** | `_read` loops `f.read(n-len(buf))` — reads all currently-available bytes per syscall, not 1 byte. Bench reader caps at a 4 MB buffer. |
| tiny fixed chunks | **No** | No fixed small chunk; reads are payload-sized (client) / ≤4 MB (bench). |
| line-buffered stdout | **No** | Service uses `sys.stdout.buffer` (binary, block-buffered); client `Popen(bufsize=0)` raw pipe. No text-mode/line buffering. |
| explicit flush per chunk | **No** | Exactly **one `flush()` per frame** (after the whole payload write), not per chunk. |
| base64 / JSON expansion of payload | **No** | Payload is **raw binary** (`torch.save` bytes); only the tiny frame header is JSON. `grep base64` → none. |
| repeated tensor copies | **Minimal** | One serialize (`torch.save`→BytesIO→bytes) and one deserialize (bytes→BytesIO→`torch.load`) per message; no redundant clones on the wire path. |
| SSH compression | **Off** | `ssh` invoked without `-o Compression=yes`; OpenSSH default is `no`. (Correct: bf16 dlogits are weakly compressible and compression would burn CPU.) |
| PTY allocation | **No** | `ssh` invoked piped, no `-t`/`RequestTTY`; forced-command key also sets `no-pty`. |
| proxy / jump-host throttling | **LIKELY (infra)** | H800 is an AutoDL container; egress/ingress to the Alibaba TDX public IP traverses AutoDL NAT. Observed **asymmetry** — H800→TDX upload fast (~12.7 MB/s) vs TDX→H800 download throttled (~0.1–0.6 MB/s) — is the signature of an **ingress rate cap on the container**, not framing code. |
| MTU / TCP-window limits | Unlikely primary | Asymmetry points to a rate cap, not a window ceiling; benchmark quantifies. |

## Preliminary conclusion (pending benchmark numbers)
The transport **code is already clean** — binary framing, no base64, no PTY, no compression,
block-buffered stdout, one flush per frame, large reads, single persistent channel, no per-message
process restart, no Mac relay. The dominant limiter is therefore expected to be the **asymmetric
AutoDL ingress throttle** on the H800 container (infra), which no framing change can raise. The
benchmark (`framed_throughput_results.csv`) tests this: if framed-no-HMAC download ≈ the S5
torch+HMAC download (~0.1 MB/s), the limit is the network, and the correctness-preserving remedies
are **byte reduction** (bf16 already applied; dlogits compression pending a label-leak security
review) and/or a **faster network path**, not transport-code changes.

## Verdict (from real measurements)
- **Per-message non-network overhead = 8.3 ms** (serialize 3.6 + deserialize 0.6 + HMAC 4.0), measured
  locally over a 42×151936 bf16 tensor — vs **123,000 ms** network for one 12.76 MB dlogits message.
  Non-network fraction = **0.007 %**.
- **Download throughput flat with size:** two real S5 points — 3.48 MB @ **0.109 MB/s**, 12.76 MB @
  **0.104 MB/s** — i.e. ~constant ~0.10–0.11 MB/s regardless of payload → a **rate cap**, not a
  window/MTU or per-frame issue. The synthetic 1/8/16/32 MB sweep was **STOPPED_EARLY_SUFFICIENT_EVIDENCE**
  (the two real points already establish flatness; no need to move 114 MB over a 0.1 MB/s link).
- **Directional asymmetry:** upload H800→TDX ~12.7 MB/s vs download TDX→H800 ~0.10 MB/s.

`bottleneck_class = infrastructure_ingress_throttle` (consistent_with an AutoDL container ingress
throttle — not asserted as provider policy). `code_path / framing / serialization / hmac / gpu_compute
/ tdx_compute` bottleneck = **all false**. No transport-code change can raise throughput; the code
already adds only 8 ms. Correctness-preserving levers: **byte reduction** (bf16 applied, 25.5→12.76 MB;
dlogits compression pending a label-leak security review) or a **faster network path** (migration).
