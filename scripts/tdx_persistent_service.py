"""Persistent attested TDX boundary service (direct transport).

Runs INSIDE the Intel TDX guest as an SSH FORCED COMMAND: the untrusted H800 opens ONE
persistent SSH channel (port 22 -- the only reachable port) and this service reads framed
requests from stdin and writes framed responses to stdout, looping across MANY training
steps. No per-step process restart, no Mac in the data path, no temp-file ferry.

Ops (one verified session reused across steps):
  handshake     -> fresh attestation evidence + service info (attestation ONCE, not per step)
  ce_dlogits    -> masked logits in  -> CE + masked dlogits out (in-enclave, labels private)
  correct       -> packed masked A-grads in -> gram_inv-corrected grads out (in-enclave)
  close         -> exit

Security: HMAC-SHA256 over each message + monotonic seq (replay reject); bounded message
size; fail-closed on bad tag / seq / oversize / malformed. Holds labels, vocab perm (seed
8000), and the gamma bundle (gram_inv rebuilt in-enclave). Never logs plaintext; never
returns gamma / gram_inv. Session key provisioned by the Mac control-plane at setup.
"""
from __future__ import annotations
import hashlib, hmac, io, json, struct, sys, time
from pathlib import Path

import torch

MAX_MSG = 1200 * 1024 * 1024       # 1.2 GB bound (causal-LM full-vocab supervised logits, batch 16)
ATTN = ("q_proj", "k_proj", "v_proj"); MLP = ("gate_proj", "up_proj")
ALLOWED = set(ATTN) | set(MLP)
FORBIDDEN = ("o_proj", "down_proj", "N_inv", "Nr", "gamma", "pad", "token", "input_ids", "label")


def orthogonal_signed_perm(n, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    signs = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(dtype)
    N = torch.zeros(n, n, dtype=dtype); N[torch.arange(n), perm] = signs
    return N


def vocab_perm(V, seed=8000):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(V, generator=g)
    inv = torch.empty_like(perm); inv[perm] = torch.arange(V)
    return perm, inv


def build_gram_inv(gb):
    H, L = gb["hidden"], gb["num_layers"]
    Nr = orthogonal_signed_perm(H, gb["nr_seed"], torch.float64)
    attn = {l: Nr.T @ torch.diag(gb["ga"][l].double() ** 2) @ Nr for l in range(L)}
    mlp = {l: Nr.T @ torch.diag(gb["gm"][l].double() ** 2) @ Nr for l in range(L)}
    return attn, mlp


# ---- framing over binary stdin/stdout ----
def _read(f, n):
    buf = b""
    while len(buf) < n:
        c = f.read(n - len(buf))
        if not c:
            raise EOFError
        buf += c
    return buf


def read_frame(f):
    (hlen,) = struct.unpack(">I", _read(f, 4))
    if hlen > 1 << 20:
        raise ValueError("header_too_large")
    header = json.loads(_read(f, hlen).decode())
    (plen,) = struct.unpack(">Q", _read(f, 8))
    if plen > MAX_MSG:
        raise ValueError("payload_too_large")
    payload = _read(f, plen) if plen else b""
    return header, payload


def write_frame(f, header, payload=b""):
    hb = json.dumps(header).encode()
    f.write(struct.pack(">I", len(hb))); f.write(hb)
    f.write(struct.pack(">Q", len(payload))); f.write(payload)
    f.flush()


def load_tensor(b):
    return torch.load(io.BytesIO(b), map_location="cpu")


def dump_tensor(t):
    bio = io.BytesIO(); torch.save(t, bio); return bio.getvalue()


def mac(key, payload, seq, run_id, op):
    m = hashlib.sha256(payload).digest() + str(seq).encode() + run_id.encode() + op.encode()
    return hmac.new(key, m, hashlib.sha256).hexdigest()


def do_attestation(cfg, out_dir):
    """Fresh attestation at session setup (once). Reuses gate0_d4_attestation."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from gate0_d4_attestation import verify_quote_jwt, generate_quote_alibaba, d4_report_data_hex
        manifest = cfg["binding_manifest"]
        rd = d4_report_data_hex(manifest)
        outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
        quote = generate_quote_alibaba(rd, outp, quote_out_name="session_quote.dat")
        parsed = verify_quote_jwt(quote)
        rdx = (parsed.get("tdx_reportdata") or "").lower().removeprefix("0x")
        verified = bool(parsed.get("overall_appraisal_result") == "SUCCESS"
                        and rdx == rd.lower() and parsed.get("debug") is False)
        ev = {"attestation_verified": verified,
              "overall_appraisal_result": parsed.get("overall_appraisal_result"),
              "reportdata_bound": rdx == rd.lower(), "debug_false": parsed.get("debug") is False,
              "mr_td": parsed.get("mr_td"), "report_data": rd, "session_setup_once": True}
        (outp / "session_attestation.json").write_text(json.dumps(ev, indent=2))
        return ev
    except Exception as e:
        return {"attestation_verified": False, "error": repr(e)[:300]}


def main():
    # Under the SSH forced command, sys.argv is fixed by authorized_keys (H800 cannot choose it);
    # we always read the pinned session config path. SSH_ORIGINAL_COMMAND is ignored.
    cfg = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else \
        json.loads(Path("/tmp/direct_session.json").read_text())
    key = bytes.fromhex(cfg["session_key_hex"]); run_id = cfg["run_id"]
    # Verify the app-layer HMAC key is the one bound into the attested report_data. This ties
    # message authentication to attestation: refuse to serve if the held key does not match the
    # commitment the enclave will (or did) attest. Fail-closed, before any request is served.
    _bm = cfg.get("binding_manifest", {})
    if _bm.get("hmac_key_commitment"):
        _expect = hashlib.sha256(key + bytes.fromhex(_bm["nonce"])).hexdigest()
        if _expect != _bm["hmac_key_commitment"]:
            sys.stderr.write("hmac_key_commitment_mismatch\n"); sys.exit(3)
    labels = torch.tensor(cfg["labels"])
    gb = torch.load(cfg["gamma_bundle"], map_location="cpu")
    gi_attn, gi_mlp = build_gram_inv(gb)
    perm = perm_inv = None
    last_seq = -1
    adamw = {"tw": None}     # lazy TrustedAdamW (L12); holds trusted theta_plain + m + v in-enclave
    # ---- batched training/eval data plane (PHASE 1-6) ----
    from batch_dataplane import (BatchLedger, batch_mac, canonical_desc, unpermute,
                                 cls_ce_dlogits, cls_eval, clm_ce_dlogits, EVAL_SPLITS as EVAL)
    bstate = {"ledger": BatchLedger(), "labels": {}, "sched": {}, "loaded": False,
              "dataset_id": cfg.get("dataset_id", ""), "task": cfg.get("task", ""),
              "template_hash": cfg.get("template_hash", ""),
              "label_schema_hash": cfg.get("label_schema_hash", ""),
              "tokenizer_hash": cfg.get("tokenizer_hash", "")}

    def _load_batch_assets():
        if bstate["loaded"]:
            return
        for split, p in cfg.get("batch_label_tables", {}).items():
            bstate["labels"][split] = torch.load(p, map_location="cpu", weights_only=False)
        for split, p in cfg.get("batch_schedules", {}).items():
            bstate["sched"][split] = {b["global_batch_index"]: b for b in json.loads(Path(p).read_text())["schedule"]}
        bstate["loaded"] = True

    def _expected_ids(desc):
        sc = bstate["sched"].get(desc["split"])
        if sc is None:
            return desc["sample_ids"]              # eval one-pass w/o schedule: accept provided
        b = sc.get(desc["global_batch_index"])
        return b["sample_ids"] if b else None
    counters = {"ce_calls": 0, "correct_calls": 0, "auth_failures": 0, "replay_rejected": 0,
                "malformed_rejected": 0, "forbidden_key_rejected": 0,
                "untrusted_gamma_returns": 0, "correction_missing_targets": -1,
                "adamw_init_calls": 0, "adamw_step_calls": 0, "trusted_target_state_missing": -1,
                "untrusted_moment_returns": 0,
                # --- PHASE 2 authoritative FP32 master-state flow invariants ---
                "authoritative_state_location": "TDX",
                "authoritative_state_dtype": None,          # set to FP32 at init_adamw
                "runtime_copy_dtype": "BF16",               # A10 casts re-folded fp32 -> bf16 runtime
                "runtime_copy_matches_master_transform": None,  # verified each step (re-fold == master@M)
                "gpu_trusted_factor_optimizer_step": "forbidden",
                "untrusted_m_materializations": 0, "untrusted_v_materializations": 0,
                "untrusted_fp32_master_materializations": 0, "silent_fallbacks": 0,
                "adamw_checkpoint_calls": 0, "adamw_restore_calls": 0, "adamw_state_version": 0,
                "checkpoint_restore_rejected": 0}
    fin = sys.stdin.buffer; fout = sys.stdout.buffer
    att = do_attestation(cfg, cfg.get("attest_out", "/tmp/direct_attest")) if cfg.get("attest") else {"attestation_skipped": True}

    while True:
        try:
            header, payload = read_frame(fin)
        except EOFError:
            break
        except ValueError as _fe:
            # oversize / malformed framing is unrecoverable (stream desynced) -> fail closed, exit cleanly
            counters["malformed_rejected"] += 1
            try:
                write_frame(fout, {"op": "reject", "reason": f"frame:{_fe}"})
            except Exception:
                pass
            break
        op = header.get("op")
        if op == "close":
            break
        if op == "handshake":
            write_frame(fout, {"op": "handshake_ack", "attestation": att, "counters": counters,
                               "service": "tdx_persistent", "run_id": run_id})
            continue
        # authenticated ops
        seq = header.get("seq", -1)
        try:
            if header.get("hmac") != mac(key, payload, seq, run_id, op):
                counters["auth_failures"] += 1
                write_frame(fout, {"op": "reject", "reason": "auth", "seq": seq}); continue
            if seq <= last_seq:
                counters["replay_rejected"] += 1
                write_frame(fout, {"op": "reject", "reason": "replay", "seq": seq}); continue
        except Exception:
            counters["malformed_rejected"] += 1
            write_frame(fout, {"op": "reject", "reason": "malformed"}); continue

        if op == "ce_dlogits":
            t0 = time.time()
            masked = load_tensor(payload).float()
            V = masked.shape[1]
            if perm is None:
                perm, perm_inv = vocab_perm(V, cfg.get("vocab_seed", 8000))
            plain = masked[:, perm]
            lg = plain[:-1]; tg = labels[1:len(plain)]
            loss = torch.nn.functional.cross_entropy(lg, tg)
            probs = torch.softmax(lg, -1); dp = probs.clone()
            dp[torch.arange(tg.shape[0]), tg] -= 1.0; dp /= tg.shape[0]
            dpl = torch.zeros_like(plain); dpl[:-1] = dp
            dmask = dpl[:, perm_inv]
            counters["ce_calls"] += 1; last_seq = seq
            # Return dlogits in the RUN dtype. For bf16 runs the client casts to bf16 on receipt
            # anyway (backward runs in bf16), so returning bf16 here is semantics-preserving and
            # HALVES the throttled TDX->H800 download (masked was .float()'d, so masked.dtype was
            # fp32 -> the old code shipped 25.5MB fp32 even for bf16 runs; now 12.75MB bf16).
            out = dump_tensor(dmask.to(torch.bfloat16) if header.get("dtype") == "bf16" else dmask)
            resp_seq = seq + 1
            write_frame(fout, {"op": "ce_dlogits_ack", "seq": resp_seq,
                               "hmac": mac(key, out, resp_seq, run_id, "ce_dlogits_ack"),
                               "ce_loss": float(loss), "compute_sec": time.time() - t0,
                               "bytes_in": len(payload), "bytes_out": len(out)}, out)

        elif op == "correct":
            t0 = time.time()
            grads = load_tensor(payload)
            seen = set(); corrected = {}
            bad = False
            for k, gA in grads.items():
                l_str, proj = k.split(".", 1); l = int(l_str)
                if proj not in ALLOWED or any(s in k for s in FORBIDDEN):
                    counters["forbidden_key_rejected"] += 1; bad = True; break
                gi = gi_attn[l] if proj in ATTN else gi_mlp[l]
                corrected[k] = (gA.double() @ gi).to(gA.dtype)
                seen.add((l, proj))
            if bad:
                write_frame(fout, {"op": "reject", "reason": "forbidden_key", "seq": seq}); continue
            expected = {(l, p) for l in range(gb["num_layers"]) for p in ALLOWED}
            counters["correction_missing_targets"] = len(expected - seen)
            if seen != expected:
                write_frame(fout, {"op": "reject", "reason": "incomplete", "seq": seq}); continue
            counters["correct_calls"] += 1; last_seq = seq
            out = dump_tensor(corrected)
            resp_seq = seq + 1
            write_frame(fout, {"op": "correct_ack", "seq": resp_seq,
                               "hmac": mac(key, out, resp_seq, run_id, "correct_ack"),
                               "corrected": len(corrected), "missing": counters["correction_missing_targets"],
                               "compute_sec": time.time() - t0,
                               "bytes_in": len(payload), "bytes_out": len(out)}, out)
        elif op == "init_adamw":
            # receive initial masked trusted factors; un-fold to plaintext in-enclave, zero m/v.
            t0 = time.time()
            from tdx_adamw_protocol import TrustedAdamW, TRUSTED_A, TRUSTED_B
            data = load_tensor(payload)
            hp = header.get("hparams", {})
            _sdt = torch.float32 if hp.get("state_dtype", "fp32") == "fp32" else torch.float64
            tw = TrustedAdamW(gb, cfg["model_cfg"], lr=hp.get("lr", 1e-3), b1=hp.get("b1", 0.9),
                              b2=hp.get("b2", 0.999), eps=hp.get("eps", 1e-8), wd=hp.get("wd", 0.01),
                              state_dtype=_sdt, optimizer=hp.get("optimizer", "adamw"), mom=hp.get("mom", 0.9))
            tw.set_binding(run_id=run_id, package_root_hash=cfg.get("package_root_hash", "unpinned"),
                           adapter_id=hp.get("adapter_id", run_id),
                           optimizer_profile=hp.get("optimizer_profile", "L12_adamw"),
                           model_config_hash=cfg.get("model_config_hash", ""),
                           service_hash=cfg.get("service_hash", ""))
            counters["authoritative_state_dtype"] = "FP32" if _sdt == torch.float32 else "FP64"
            bad = False
            for k, t in data.get("A", {}).items():
                l_str, proj = k.split(".", 1); l = int(l_str)
                if proj not in TRUSTED_A or any(s in k for s in FORBIDDEN):
                    counters["forbidden_key_rejected"] += 1; bad = True; break
                tw.init_factor(l, proj, A_tilde=t)
            for k, t in data.get("B", {}).items():
                l_str, proj = k.split(".", 1); l = int(l_str)
                if proj not in TRUSTED_B or any(s in k for s in FORBIDDEN):
                    counters["forbidden_key_rejected"] += 1; bad = True; break
                tw.init_factor(l, proj, B_tilde=t)
            if bad:
                write_frame(fout, {"op": "reject", "reason": "forbidden_key", "seq": seq}); continue
            adamw["tw"] = tw
            counters["adamw_init_calls"] += 1; last_seq = seq
            present = tw.state_present()
            counters["trusted_target_state_missing"] = 0 if present else 1
            resp_seq = seq + 1
            write_frame(fout, {"op": "init_adamw_ack", "seq": resp_seq,
                               "hmac": mac(key, b"", resp_seq, run_id, "init_adamw_ack"),
                               "trusted_adamw_state_present": present,
                               "trusted_A_factors": len(tw.stateA), "trusted_B_factors": len(tw.stateB),
                               "compute_sec": time.time() - t0})

        elif op == "adamw_step":
            # masked trusted grads in -> exact plaintext AdamW in-enclave -> re-folded masked factors out.
            t0 = time.time()
            tw = adamw["tw"]
            if tw is None or not tw.state_present():          # fail-closed on missing/stale state
                counters["trusted_target_state_missing"] = 1
                write_frame(fout, {"op": "reject", "reason": "stale_or_missing_adamw_state", "seq": seq}); continue
            data = load_tensor(payload); bad = False
            for k in list(data.get("gA", {})) + list(data.get("gB", {})):
                if any(s in k for s in FORBIDDEN):
                    counters["forbidden_key_rejected"] += 1; bad = True; break
            if bad:
                write_frame(fout, {"op": "reject", "reason": "forbidden_key", "seq": seq}); continue
            try:
                outA, outB, missing = tw.step(data.get("gA", {}), data.get("gB", {}))
            except KeyError:
                write_frame(fout, {"op": "reject", "reason": "stale_or_missing_adamw_state", "seq": seq}); continue
            counters["trusted_target_state_missing"] = missing
            if missing != 0:
                write_frame(fout, {"op": "reject", "reason": "incomplete_trusted_set", "seq": seq}); continue
            # Verify the re-folded runtime copy is EXACTLY master @ fold (regenerated from FP32 master,
            # not an independently-tracked BF16 tensor). Cheap invariant on one representative factor.
            _lp = next(iter(tw.stateA)); _foldM = tw.tinA[_lp][0]
            _match = bool(torch.allclose(outA[f"{_lp[0]}.{_lp[1]}"], tw.stateA[_lp][0] @ _foldM, atol=0, rtol=0))
            counters["runtime_copy_matches_master_transform"] = _match
            counters["adamw_state_version"] = tw.version
            # return ONLY re-folded masked factors (never m/v/gamma/plaintext theta), FP32 master image
            out = dump_tensor({"A": {k: v.to(torch.float32) for k, v in outA.items()},
                               "B": {k: v.to(torch.float32) for k, v in outB.items()}})
            counters["adamw_step_calls"] += 1; last_seq = seq
            resp_seq = seq + 1
            write_frame(fout, {"op": "adamw_step_ack", "seq": resp_seq,
                               "hmac": mac(key, out, resp_seq, run_id, "adamw_step_ack"),
                               "updated_A": len(outA), "updated_B": len(outB), "missing": missing,
                               "adam_t": tw.t, "state_version": tw.version,
                               "runtime_copy_matches_master_transform": _match,
                               "compute_sec": time.time() - t0,
                               "bytes_in": len(payload), "bytes_out": len(out)}, out)

        elif op == "rebase_adamw":
            # rank-basis refresh applied to trusted plaintext state (A_new=R@A, B_new=B@R^T + moments).
            tw = adamw["tw"]
            if tw is None or not tw.state_present():
                write_frame(fout, {"op": "reject", "reason": "no_adamw_state", "seq": seq}); continue
            R = load_tensor(payload)["R"]; mode = header.get("mode", "signed_perm")
            info = tw.rebase(R, mode=mode)
            counters["adamw_rebase_calls"] = counters.get("adamw_rebase_calls", 0) + 1
            counters["adamw_state_version"] = tw.version; last_seq = seq; resp_seq = seq + 1
            write_frame(fout, {"op": "rebase_adamw_ack", "seq": resp_seq,
                               "hmac": mac(key, b"", resp_seq, run_id, "rebase_adamw_ack"),
                               "state_version": tw.version, **info})

        elif op == "checkpoint_adamw":
            # seal (theta_master_fp32, m, v, step, version, binding) with AEAD to a durable file.
            tw = adamw["tw"]
            if tw is None or not tw.state_present():
                counters["checkpoint_restore_rejected"] += 1
                write_frame(fout, {"op": "reject", "reason": "no_adamw_state", "seq": seq}); continue
            blob = tw.checkpoint(session_key=key)   # ChaCha20-Poly1305; checkpoint_seq incremented inside
            ckpt_dir = Path(cfg.get("ckpt_dir", "/tmp/l12_ckpt")); ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"{run_id}.v{tw.version}.s{tw.checkpoint_seq}.enc"
            ckpt_path.write_bytes(blob)                      # durable state is ciphertext only
            counters["adamw_checkpoint_calls"] += 1; last_seq = seq
            resp_seq = seq + 1
            write_frame(fout, {"op": "checkpoint_adamw_ack", "seq": resp_seq,
                               "hmac": mac(key, b"", resp_seq, run_id, "checkpoint_adamw_ack"),
                               "state_version": tw.version, "adam_t": tw.t,
                               "checkpoint_seq": tw.checkpoint_seq, "binding": tw.binding,
                               "aead": "ChaCha20Poly1305",
                               "ckpt_path": str(ckpt_path), "ckpt_bytes": len(blob),
                               "durable_plaintext": False})

        elif op == "restore_adamw":
            # authenticated decrypt + binding + monotonic-version validation, then resume.
            from tdx_adamw_protocol import TrustedAdamW
            hp = header.get("hparams", {})
            ckpt_path = Path(header.get("ckpt_path", ""))
            if not ckpt_path.exists():
                counters["checkpoint_restore_rejected"] += 1
                write_frame(fout, {"op": "reject", "reason": "ckpt_missing", "seq": seq}); continue
            _sdt = torch.float32 if hp.get("state_dtype", "fp32") == "fp32" else torch.float64
            tw = TrustedAdamW(gb, cfg["model_cfg"], lr=hp.get("lr", 1e-3), b1=hp.get("b1", 0.9),
                              b2=hp.get("b2", 0.999), eps=hp.get("eps", 1e-8), wd=hp.get("wd", 0.01),
                              state_dtype=_sdt)
            try:
                info = tw.restore(ckpt_path.read_bytes(), session_key=key,
                                  expected_binding=header["expected_binding"],
                                  min_version=int(header.get("min_version", 0)))
            except Exception as e:
                counters["checkpoint_restore_rejected"] += 1
                write_frame(fout, {"op": "reject", "reason": f"restore_failed:{repr(e)[:80]}", "seq": seq}); continue
            adamw["tw"] = tw
            counters["adamw_restore_calls"] += 1; counters["adamw_state_version"] = tw.version
            counters["trusted_target_state_missing"] = 0 if tw.state_present() else 1
            counters["authoritative_state_dtype"] = "FP32" if _sdt == torch.float32 else "FP64"
            last_seq = seq; resp_seq = seq + 1
            write_frame(fout, {"op": "restore_adamw_ack", "seq": resp_seq,
                               "hmac": mac(key, b"", resp_seq, run_id, "restore_adamw_ack"),
                               "trusted_adamw_state_present": tw.state_present(),
                               "restored_version": info["restored_version"], "restored_t": info["restored_t"]})

        elif op in ("ce_batch", "eval_batch"):
            # ---- authenticated variable-batch CE + dlogits (classification / causal-LM) ----
            t0 = time.time()
            _load_batch_assets()
            desc = header.get("desc", {})
            # re-verify the HMAC WITH the batch descriptor bound in (transport-layer mac() above did not
            # include desc; fail-closed if the descriptor was tampered).
            if header.get("bmac") != batch_mac(key, payload, seq, run_id, op, desc):
                counters["auth_failures"] += 1
                write_frame(fout, {"op": "reject", "reason": "batch_auth", "seq": seq}); continue
            # binding checks
            binderr = None
            if desc.get("run_id") != run_id: binderr = "wrong_run_id"
            elif desc.get("dataset_id") != bstate["dataset_id"]: binderr = "wrong_dataset"
            elif desc.get("package_hash") != cfg.get("package_root_hash"): binderr = "wrong_package"
            elif desc.get("template_hash") != bstate["template_hash"]: binderr = "wrong_template"
            elif desc.get("tokenizer_hash") != bstate["tokenizer_hash"]: binderr = "wrong_tokenizer"
            elif desc.get("label_schema_hash") != bstate["label_schema_hash"]: binderr = "wrong_label_schema"
            if binderr:
                counters.setdefault("batch_binding_rejected", 0); counters["batch_binding_rejected"] += 1
                write_frame(fout, {"op": "reject", "reason": binderr, "seq": seq}); continue
            split = desc["split"]
            labels = bstate["labels"].get(split)
            if labels is None:
                write_frame(fout, {"op": "reject", "reason": "no_label_table", "seq": seq}); continue
            exp_ids = _expected_ids(desc)
            if exp_ids is None:
                write_frame(fout, {"op": "reject", "reason": "unknown_global_batch_index", "seq": seq}); continue
            ledger = bstate["ledger"]
            # causal-LM transport is chunked (row-slices of the concatenated supervised logits) so the
            # CPU-only TDX guest can softmax+serialize each frame under the read window; the ledger
            # advances ONCE per optimizer batch (on chunk 0), continuation chunks validate ids only.
            chunk = int(desc.get("chunk", 0)); nchunks = int(desc.get("nchunks", 1))
            if op == "eval_batch" and split in EVAL:
                if list(desc["sample_ids"]) != list(exp_ids):
                    counters.setdefault("batch_ledger_rejected", 0); counters["batch_ledger_rejected"] += 1
                    write_frame(fout, {"op": "reject", "reason": "wrong_sample_ids", "seq": seq}); continue
            elif chunk == 0:
                ok, reason = ledger.check_and_advance(desc, exp_ids)
                if not ok:
                    counters.setdefault("batch_ledger_rejected", 0); counters["batch_ledger_rejected"] += 1
                    write_frame(fout, {"op": "reject", "reason": reason, "seq": seq}); continue
            else:                                        # continuation chunk of an advanced training batch
                if list(desc["sample_ids"]) != list(exp_ids):
                    counters.setdefault("batch_ledger_rejected", 0); counters["batch_ledger_rejected"] += 1
                    write_frame(fout, {"op": "reject", "reason": "wrong_sample_ids", "seq": seq}); continue
            data = load_tensor(payload)                 # {"logits":..., optional "sup_positions":...}
            masked = data["logits"].float()             # cls: [n,V]; clm: [rows_in_chunk,V]
            V = masked.shape[1]
            if perm is None:
                perm, perm_inv = vocab_perm(V, cfg.get("vocab_seed", 8000))
            plain = masked[:, perm]
            sample_ids = desc["sample_ids"]; task = desc["task"]
            if task == "cls":
                # one supervised logit per example
                if masked.shape[0] != len(sample_ids):
                    write_frame(fout, {"op": "reject", "reason": "wrong_sequence_shape", "seq": seq}); continue
                tgt = torch.tensor([labels[int(s)]["target_tok"] for s in sample_ids])
                lab = torch.tensor([labels[int(s)]["label"] for s in sample_ids])
                neg = labels[int(sample_ids[0])]["neg_tok"]; pos = labels[int(sample_ids[0])]["pos_tok"]
                if op == "eval_batch":
                    pred, correct, nll = cls_eval(plain, tgt, neg, pos, lab)
                    counters.setdefault("eval_batch_calls", 0); counters["eval_batch_calls"] += 1; last_seq = seq
                    resp_seq = seq + 1
                    out = dump_tensor({"pred": pred.cpu(), "correct": correct.cpu(), "nll": nll.cpu(),
                                       "label": lab.cpu()})
                    write_frame(fout, {"op": "eval_batch_ack", "seq": resp_seq,
                                       "hmac": mac(key, out, resp_seq, run_id, "eval_batch_ack"),
                                       "n": len(sample_ids), "correct": int(correct.sum()),
                                       "mean_nll": float(nll.mean()), "compute_sec": time.time() - t0}, out)
                    continue
                ce, dpl, nll = cls_ce_dlogits(plain, tgt)
                dmask = torch.zeros_like(plain); dmask[:] = dpl
                dmask = dmask[:, perm_inv]
            elif task == "clm":
                # verify the ignore_index / supervised-position pattern vs the label table, build the FULL
                # concatenated target vector, then serve only this chunk's row-slice [row_start:row_start+rows].
                sup_counts = desc["sup_counts"]
                tgt_list, bad = [], False
                for s, sc in zip(sample_ids, sup_counts):
                    ent = labels[int(s)]
                    if len(ent["sup_positions"]) != sc:
                        bad = True; break
                    tgt_list += [ent["target"][p] for p in ent["sup_positions"]]
                if bad:
                    counters.setdefault("batch_ledger_rejected", 0); counters["batch_ledger_rejected"] += 1
                    write_frame(fout, {"op": "reject", "reason": "wrong_ignore_index_pattern", "seq": seq}); continue
                total_sup = len(tgt_list)
                denom = int(desc.get("sup_denominator", total_sup))    # mean-reduction denominator (full batch)
                row_start = int(desc.get("row_start", 0)); rows = masked.shape[0]
                if row_start + rows > total_sup:
                    write_frame(fout, {"op": "reject", "reason": "wrong_sequence_shape", "seq": seq}); continue
                tgt = torch.tensor(tgt_list[row_start:row_start + rows])
                logp = torch.log_softmax(plain, -1)
                nll = -logp[torch.arange(tgt.shape[0]), tgt]
                if op == "eval_batch":
                    counters.setdefault("eval_batch_calls", 0); counters["eval_batch_calls"] += 1; last_seq = seq
                    resp_seq = seq + 1
                    out = dump_tensor({"nll": nll.cpu()})
                    write_frame(fout, {"op": "eval_batch_ack", "seq": resp_seq,
                                       "hmac": mac(key, out, resp_seq, run_id, "eval_batch_ack"),
                                       "n_sup": int(tgt.shape[0]), "sum_nll": float(nll.sum()),
                                       "mean_nll": float(nll.mean()), "compute_sec": time.time() - t0}, out)
                    continue
                # training: dlogits scaled by the FULL-batch denominator so summing chunks == mean gradient
                probs = torch.softmax(plain, -1)
                probs[torch.arange(tgt.shape[0]), tgt] -= 1.0
                dpl = probs / denom
                dmask = dpl[:, perm_inv]
                ce = float(nll.sum())                    # partial CE sum; client aggregates / denom
            else:
                write_frame(fout, {"op": "reject", "reason": "unknown_task", "seq": seq}); continue
            counters.setdefault("ce_batch_calls", 0); counters["ce_batch_calls"] += 1; last_seq = seq
            out = dump_tensor(dmask.to(torch.bfloat16) if header.get("dtype") == "bf16" else dmask)
            resp_seq = seq + 1
            write_frame(fout, {"op": "ce_batch_ack", "seq": resp_seq,
                               "hmac": mac(key, out, resp_seq, run_id, "ce_batch_ack"),
                               "ce_loss": float(ce), "n_out": int(dmask.shape[0]),
                               "global_batch_index": desc["global_batch_index"],
                               "ledger_last": ledger.last.get((desc["dataset_id"], split), -1),
                               "compute_sec": time.time() - t0,
                               "bytes_in": len(payload), "bytes_out": len(out)}, out)

        elif op == "checkpoint_ledger":
            # seal the batch ledger (ChaCha20-Poly1305) bound to the run for restart continuity.
            from tdx_adamw_protocol import aead_seal_std, canonical_aad
            _bind = adamw["tw"].binding if adamw["tw"] else {
                "run_id": run_id, "package_root_hash": cfg.get("package_root_hash", ""),
                "adapter_id": header.get("adapter_id", run_id), "optimizer_profile": "ledger",
                "model_config_hash": cfg.get("model_config_hash", ""), "service_hash": cfg.get("service_hash", "")}
            aad = canonical_aad(_bind, header.get("ledger_version", 0), header.get("ledger_seq", 0))
            blob = aead_seal_std(key, bstate["ledger"].snapshot(), aad)
            lp = Path(cfg.get("ckpt_dir", "/tmp/l12_ckpt")); lp.mkdir(parents=True, exist_ok=True)
            fpath = lp / f"{run_id}.ledger.enc"; fpath.write_bytes(blob)
            counters.setdefault("ledger_checkpoint_calls", 0); counters["ledger_checkpoint_calls"] += 1
            last_seq = seq; resp_seq = seq + 1
            write_frame(fout, {"op": "checkpoint_ledger_ack", "seq": resp_seq,
                               "hmac": mac(key, b"", resp_seq, run_id, "checkpoint_ledger_ack"),
                               "ledger_path": str(fpath), "ledger_bytes": len(blob),
                               "ledger_state": {f"{a}|{b}": v for (a, b), v in bstate["ledger"].last.items()}})

        elif op == "restore_ledger":
            from tdx_adamw_protocol import aead_open_std, canonical_aad
            _load_batch_assets()
            _bind = header["expected_binding"]
            aad = canonical_aad(_bind, header.get("ledger_version", 0), header.get("ledger_seq", 0))
            fpath = Path(header.get("ledger_path", ""))
            if not fpath.exists():
                write_frame(fout, {"op": "reject", "reason": "ledger_missing", "seq": seq}); continue
            try:
                pt = aead_open_std(key, fpath.read_bytes(), aad)
                bstate["ledger"].load(pt)
            except Exception as e:
                counters.setdefault("ledger_restore_rejected", 0); counters["ledger_restore_rejected"] += 1
                write_frame(fout, {"op": "reject", "reason": f"ledger_restore_failed:{repr(e)[:60]}", "seq": seq}); continue
            counters.setdefault("ledger_restore_calls", 0); counters["ledger_restore_calls"] += 1
            last_seq = seq; resp_seq = seq + 1
            write_frame(fout, {"op": "restore_ledger_ack", "seq": resp_seq,
                               "hmac": mac(key, b"", resp_seq, run_id, "restore_ledger_ack"),
                               "ledger_state": {f"{a}|{b}": v for (a, b), v in bstate["ledger"].last.items()}})

        else:
            counters["malformed_rejected"] += 1
            write_frame(fout, {"op": "reject", "reason": "unknown_op"})
    # final counters to a file for the Mac control-plane to record
    Path(cfg.get("counters_out", "/tmp/direct_session_counters.json")).write_text(
        json.dumps({"counters": counters, "attestation": att}, indent=2))


if __name__ == "__main__":
    main()
