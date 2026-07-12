# Direct H800↔TDX transport — security model

`transport_profile = direct_h800_tdx`. Replaces `mac_ferried_authenticated_prototype` (Gate-0 only).
Data plane is a single persistent SSH channel H800→TDX (port 22, the only reachable port). The Mac is
**control-plane only**: it generates the session key + binding manifest, provisions them + the γ bundle
+ the service/runner code, launches the runner, and collects results. It never touches per-step payloads.

## Two independent protection layers

1. **SSH transport encryption** (channel confidentiality/integrity of bytes on the wire). Necessary but
   NOT sufficient: SSH alone does not bind a message to the *attested TDX session* — a different process
   behind the same SSH endpoint would still decrypt.
2. **Application-layer attestation + HMAC binding** (this design). Every request carries
   `HMAC-SHA256(session_key, sha256(payload) ‖ seq ‖ run_id ‖ op)` with a strictly monotonic `seq`
   (replay reject). The `session_key` is **bound into the attested report_data**: the enclave commits to
   `hmac_key_commitment = sha256(session_key ‖ nonce)` inside the TDX quote, and at startup the service
   recomputes that commitment from the key it actually holds and **refuses to serve on mismatch**
   (`exit(3)`, verified). So a verifier of the quote confirms the exact key used for message
   authentication was the one present at attestation time — not merely "a key in a static file".

These are independent: compromising one (e.g. an SSH MITM, or a swapped process) does not defeat the
other. SSH is not treated as a substitute for the attestation binding.

## Key material & session lifecycle
- `session_key`: 32 random bytes, fresh per run (`secrets.token_bytes`). Provisioned by the Mac to both
  endpoints at setup; committed into the attested manifest (above).
- `nonce`: 16 random bytes; enters `report_data` and the key commitment.
- Attestation: performed **once** at session setup (`do_attestation`, reuses `gate0_d4_attestation`),
  not per step. `report_data = SHA-512(binding_manifest)`; the manifest carries the full D4 binding set
  plus `transport_profile`, `hmac_key_commitment`, `session_key_derivation`. Verified = appraisal
  SUCCESS ∧ reportdata-bound ∧ debug=false.
- Session termination: `close` op or channel EOF → service writes final counters and exits. A new run
  mints a fresh key + nonce + quote (no key reuse across runs).

## Forced-command boundary (TDX authorized_keys, H800 key)
The H800 key on TDX is pinned to a forced command with restrictions:
```
command="…/python …/tdx_persistent_service.py /tmp/direct_session.json",no-pty,no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-user-rc  ssh-ed25519 … h800-direct-tdx
```
Consequences (tested):
- **No arbitrary shell**: `ssh … 'echo LEAK; id; cat /etc/passwd'` runs the pinned service instead;
  `LEAK`/`id`/`/etc/passwd` never execute (verified).
- **Client cannot override the service command**: `SSH_ORIGINAL_COMMAND` is discarded; the runner's
  `--service-cmd` is advisory (kept so a mis-provisioned non-forced key fails loudly).
- **No client-chosen path args**: `sys.argv[1]` is fixed by the forced command to the pinned session
  config; the service never accepts a client-supplied file path.
- **No forwarding / pty**: port/agent/X11 forwarding and pty allocation disabled.

## Service-side fail-closed guarantees (verified by `test_direct_transport_faults.py`)
- Bad HMAC → `reject:auth`; wrong `run_id` in the MAC → `reject:auth`.
- `seq ≤ last_seq` → `reject:replay`.
- Correction payload containing a forbidden key (`o_proj`, `down_proj`, `N_inv`, `Nr`, `gamma`, `pad`,
  `token`, `input_ids`, `label`) → `reject:forbidden_key`; the enclave only ever corrects
  `{q,k,v,gate,up}` A-grads.
- Incomplete correction set (≠ all L×{q,k,v,gate,up}) → `reject:incomplete`.
- Oversize header/payload (>1 MB header, >200 MB payload) → bounded read raises; truncated frame → EOF
  → clean exit. The session survives isolated rejects (a valid request after several rejects still
  succeeds — verified) but never continues on a *failed* step.
- The enclave never returns γ or gram_inv; `untrusted_gamma_returns` counter stays 0. Labels, vocab
  perm (seed 8000), and γ never leave the enclave.

## Client-side fail-closed (runner, req 6)
- Every frame read is bounded (≤200 MB payload, ≤1 MB header) and watchdog-timed (`READ_TIMEOUT`,
  `select` on the pipe) → a hung/crashed service trips a `TransportError` instead of blocking forever.
- A dead SSH channel (`p.poll()` set), any `reject`, an HMAC mismatch, or a wrong ack op each raise
  `TransportError`, which **aborts the run** — there is no fallback to local computation or stale
  results. `stderr` of the SSH subprocess is `DEVNULL` on the client; the service writes no secrets to
  stderr.

## Session reuse (proven per run, req 2)
One SSH channel for the whole run; `channel_requests == 1 + 2·steps` (handshake + ce_dlogits + correct
per step); `seq` strictly monotone across all steps; attestation once; gram_inv built once in-enclave;
model + LoRA + γ bundle loaded once and held in memory (no per-step reload).

_(Measured attestation/counter/latency evidence appended after the direct gate runs.)_
