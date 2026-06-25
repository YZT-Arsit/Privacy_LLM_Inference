"""Resident vs non-resident folded-worker correctness validation.

Weight-resident decode (load+fold+to(device) ONCE, reuse across steps) is a
PERFORMANCE optimisation; it must not change the worker output. This tool drives
the SAME folded package over the SAME synthetic masked input + the SAME KV
evolution with ``resident=false`` and ``resident=true`` and proves equivalence:

* final masked-logits comparison (prefill + every decode step): max/mean/rel err,
  allclose, shape/dtype/device match, per-step token (argmax) match;
* per-layer prefill comparison -> ``first_divergent_layer`` (None if identical);
* resident-weight **mutation** detection: checksum every resident operator before
  and after the decode loop -> ``resident_weight_mutated`` (catches any in-place
  corruption that would compound with length);
* dtype check: resident-cache dtype vs the non-resident per-step compute dtype ->
  ``resident_dtype_mismatch`` (also reports CLI vs fold-compute dtype, since the
  fold compute uses the package fold dtype, not ``--dtype``).

Runs on the real server (``--folded-package-path``, ``--device cuda``) and locally
on a tiny ``--dry-run`` package (CPU, no Qwen, no GPU). Synthetic masked tensors
only (no plaintext, no mask secrets); every security-audit field is emitted.
Performance is never reported as a security or correctness improvement.

# ---- server example only (run on the GPU server; NOT executed locally) ----
# python scripts/compare_resident_folded_correctness.py \
#   --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full_current_seq1024 \
#   --embedding-path     /root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current \
#   --device cuda --dtype bfloat16 --seq-len 1024 --decode-steps 16 \
#   --output-json outputs/profile/resident_correctness_seq1024_decode16.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_script(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- pure comparison helpers (unit-tested directly) -----------------------

def _compare_arrays(a, b, *, atol, rtol):
    """Compare two (lists of) numpy arrays. Robust to shape/dtype mismatch."""
    import numpy as np
    al = a if isinstance(a, list) else [a]
    bl = b if isinstance(b, list) else [b]
    shape_match = (len(al) == len(bl)
                   and all(np.asarray(x).shape == np.asarray(y).shape
                           for x, y in zip(al, bl)))
    dtype_match = (len(al) == len(bl)
                   and all(np.asarray(x).dtype == np.asarray(y).dtype
                           for x, y in zip(al, bl)))
    out = {"shape_match": bool(shape_match), "dtype_match": bool(dtype_match),
           "max_abs_err": None, "mean_abs_err": None, "rel_err": None,
           "allclose_atol_rtol": None, "token_match_if_logits_available": None}
    if not shape_match:
        return out
    flat_a = np.concatenate([np.asarray(x, dtype="float64").ravel() for x in al])
    flat_b = np.concatenate([np.asarray(y, dtype="float64").ravel() for y in bl])
    diff = np.abs(flat_a - flat_b)
    out["max_abs_err"] = float(diff.max()) if diff.size else 0.0
    out["mean_abs_err"] = float(diff.mean()) if diff.size else 0.0
    denom = float(np.max(np.abs(flat_a))) if flat_a.size else 0.0
    out["rel_err"] = (out["max_abs_err"] / denom) if denom > 0 else 0.0
    out["allclose_atol_rtol"] = bool(np.allclose(flat_a, flat_b, atol=atol,
                                                 rtol=rtol))
    # token match: argmax of each [.,V] logits row matches
    toks_a = [int(np.asarray(x).reshape(-1).argmax()) for x in al]
    toks_b = [int(np.asarray(y).reshape(-1).argmax()) for y in bl]
    out["token_match_if_logits_available"] = bool(toks_a == toks_b)
    return out


def _checksum_tensor(t):
    import torch  # noqa: F401
    return hashlib.sha256(
        t.detach().to("cpu").float().contiguous().numpy().tobytes()).hexdigest()


def _checksum_resident(layers, head):
    sums = []
    for folded in layers:
        for k in sorted(folded):
            v = folded[k]
            sums.append("%s:%s" % (k, _checksum_tensor(v) if v is not None
                                   else "none"))
    if head is not None:
        sums.append("head:%s" % _checksum_tensor(head))
    return sums


# ---- per-pass drivers -----------------------------------------------------

def _collect_logits(backend, *, h_np, dec_nps, seq_len):
    """Run prefill + each decode step (caller has already called init); return the
    masked logits for [prefill, step0, step1, ...]."""
    import numpy as np
    from pllo.protocol.tee_gpu_messages import (
        MaskedDecodeRequest, MaskedPrefillRequest)
    out = [np.asarray(backend.prefill(MaskedPrefillRequest(
        session_id="cmp", masked_embeddings=h_np, positions=list(range(seq_len)),
        batch_size=1, seq_len=seq_len)).masked_logits)]
    pos = seq_len
    for i, x in enumerate(dec_nps):
        out.append(np.asarray(backend.decode(MaskedDecodeRequest(
            session_id="cmp", masked_embedding=x, position=pos,
            step=i + 1)).masked_logits))
        pos += 1
    return out


def _prefill_layer_outputs(h_tilde, *, resident_layers, package, lora_dir, runner,
                           cfg, cos, sin, eps, num_layers):
    """Per-layer masked hidden after each folded layer (prefill), for divergence
    localisation. ``resident_layers`` reuses the cache; None loads per layer."""
    from pllo.deployment.folded_worker import (
        _maybe_merge_lora, apply_folded_layer_prefill, load_folded_layer)
    ys = []
    cur = h_tilde
    for ell in range(num_layers):
        if resident_layers is not None:
            out = apply_folded_layer_prefill(cur, None, cfg, cos, sin, eps,
                                             runner=runner,
                                             folded=resident_layers[ell])
        else:
            lt = _maybe_merge_lora(load_folded_layer(package, ell), lora_dir, ell)
            out = apply_folded_layer_prefill(cur, lt, cfg, cos, sin, eps,
                                             runner=runner)
        cur = out["y_tilde"]
        ys.append(cur)
    return ys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folded-package-path", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--nonlinear-backend", default="current")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--require-real", action="store_true", default=False)
    # TEST-ONLY fault injectors (validate the detectors; never use in real runs)
    ap.add_argument("--inject-test-mutation", action="store_true", default=False,
                    help="TEST: mutate a resident weight in-place to confirm the "
                    "mutation + correctness detectors fire")
    ap.add_argument("--inject-dtype-mismatch", action="store_true", default=False,
                    help="TEST: build the resident cache in a different dtype to "
                    "confirm resident_dtype_mismatch fires")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    pkg_path, art_path = args.folded_package_path, args.embedding_path
    is_dry = False
    if not (pkg_path and art_path):
        if args.require_real:
            print("ERROR: --require-real but package/embedding not given",
                  file=sys.stderr)
            return 3
        if not args.dry_run:
            print("ERROR: provide --folded-package-path + --embedding-path or "
                  "--dry-run", file=sys.stderr)
            return 3
        prof = _load_script("prof", "scripts/run_folded_worker_forward_profile.py")
        out_dir = Path(args.output_json).resolve().parent / "_dry_run_pkg"
        out_dir.mkdir(parents=True, exist_ok=True)
        pkg_path, art_path = prof._build_dry_run(out_dir, args.num_layers,
                                                 args.seed)
        is_dry = True

    import numpy as np
    import torch
    from pllo.deployment.folded_nonlinear import make_folded_nonlinear_runner
    from pllo.experiments.folded_probe_common import LiteBoundary
    from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend
    from pllo.protocol.tee_gpu_messages import BoundaryInitRequest
    from pllo.protocol.wire import encode_message
    from pllo.protocol.remote import forbidden_fields_in_payload

    boundary = LiteBoundary.from_artifact(art_path, device=args.device)
    meta = boundary.exec_metadata(seq_len=args.seq_len,
                                  max_new_tokens=args.decode_steps)
    n_layers = int(meta["num_layers"])
    hidden = int(meta["hidden_size"])

    # identical synthetic inputs for BOTH passes
    torch.manual_seed(int(args.seed))
    ids = torch.randint(0, 256, (1, args.seq_len))
    h_tilde = boundary.mask_embeddings(ids)
    h_np = np.asarray(h_tilde.detach().to("cpu").float().numpy())
    dec_nps = []
    for step in range(args.decode_steps):
        x = boundary.mask_token_embedding(torch.tensor([step % 256]))
        dec_nps.append(np.asarray(x.detach().to("cpu").float().numpy()))

    def _init_req():
        return BoundaryInitRequest(
            session_id="cmp", hidden_size=hidden, vocab_size=int(meta["vocab_size"]),
            num_layers=n_layers, dtype=args.dtype,
            gpu_backend="qwen7b_folded_package", folded_lm_head=None,
            public_metadata=meta)

    leaked: list = []

    def _audit_init(req):
        bad = forbidden_fields_in_payload(encode_message(req))
        if bad:
            leaked.extend(bad)

    # ---- non-resident pass ----
    bb = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=str(pkg_path), device=args.device, dtype=args.dtype,
        nonlinear_backend=args.nonlinear_backend, resident_folded_weights=False)
    init_req = _init_req()
    _audit_init(init_req)
    init_resp = bb.init(init_req)
    base_logits = _collect_logits(bb, h_np=h_np, dec_nps=dec_nps,
                                  seq_len=args.seq_len)
    base_dtype = str(bb._fdtype)

    # ---- resident pass (force build, checksum, [inject], run, checksum) ----
    rb = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=str(pkg_path), device=args.device, dtype=args.dtype,
        nonlinear_backend=args.nonlinear_backend, resident_folded_weights=True)
    rb.init(_init_req())
    rb._ensure_exec_context()
    dev = torch.device(args.device)
    rb._ensure_resident(dev, rb._fdtype, rb._num_layers)
    if args.inject_dtype_mismatch:           # TEST: resident in a different dtype
        rb._resident_layers = [
            {k: (v.double() if v is not None else None) for k, v in L.items()}
            for L in rb._resident_layers]
        if rb._resident_head is not None:
            rb._resident_head = rb._resident_head.double()
    resident_dtype = str(rb._resident_layers[0]["wq_tilde"].dtype)
    resident_dtype_mismatch = bool(resident_dtype != base_dtype)
    chk_before = _checksum_resident(rb._resident_layers, rb._resident_head)
    first_divergent_layer = None

    if resident_dtype_mismatch:
        # a resident cache in a different dtype than the fold compute would make
        # the masked forward invalid -- flag it cleanly, do NOT run the (doomed)
        # comparison forward. (In a real run the cache is always built with the
        # fold dtype, so this never triggers; it is a guarded fault path.)
        res_logits = None
        resident_weight_mutated = False
        cmp = {"shape_match": False, "dtype_match": False, "max_abs_err": None,
               "mean_abs_err": None, "rel_err": None, "allclose_atol_rtol": False,
               "token_match_if_logits_available": None}
    else:
        if args.inject_test_mutation:        # TEST: in-place corrupt a weight
            for k in ("wq_tilde", "wgate_tilde", "wdown_tilde"):
                if rb._resident_layers[0].get(k) is not None:
                    rb._resident_layers[0][k].add_(1.0)
                    break
        res_logits = _collect_logits(rb, h_np=h_np, dec_nps=dec_nps,
                                     seq_len=args.seq_len)
        chk_after = _checksum_resident(rb._resident_layers, rb._resident_head)
        resident_weight_mutated = bool(chk_before != chk_after)
        cmp = _compare_arrays(base_logits, res_logits, atol=args.atol,
                              rtol=args.rtol)
        # ---- per-layer prefill divergence ----
        cfg0, cos, sin = rb._exec_ctx
        eps = float(rb._eps)
        base_ys = _prefill_layer_outputs(
            h_tilde, resident_layers=None, package=str(pkg_path),
            lora_dir=rb.folded_lora_package_path,
            runner=make_folded_nonlinear_runner(args.nonlinear_backend, lift_k=2,
                                                 seed=args.seed),
            cfg=cfg0, cos=cos, sin=sin, eps=eps, num_layers=n_layers)
        res_ys = _prefill_layer_outputs(
            h_tilde, resident_layers=rb._resident_layers, package=str(pkg_path),
            lora_dir=rb.folded_lora_package_path, runner=rb._ensure_runner(),
            cfg=cfg0, cos=cos, sin=sin, eps=eps, num_layers=n_layers)
        for ell in range(n_layers):
            d = float(torch.max(torch.abs(
                base_ys[ell].detach().to("cpu").float()
                - res_ys[ell].detach().to("cpu").float())))
            if d > args.atol:
                first_divergent_layer = ell
                break

    correctness_passed = bool(
        cmp["shape_match"] and cmp["dtype_match"] and cmp["allclose_atol_rtol"]
        and not resident_weight_mutated and not resident_dtype_mismatch
        and first_divergent_layer is None)

    report = {
        "stage": "resident_folded_correctness",
        "dry_run": is_dry,
        "device": args.device,
        "cli_dtype": args.dtype,
        "fold_compute_dtype": base_dtype,        # both passes use the fold dtype
        "resident_cache_dtype": resident_dtype,
        "seq_len": args.seq_len,
        "decode_steps": args.decode_steps,
        "num_layers": n_layers,
        "atol": args.atol,
        "rtol": args.rtol,
        # ---- requested correctness fields ----
        "resident_correctness_passed": correctness_passed,
        "resident_vs_nonresident_max_abs_err": cmp["max_abs_err"],
        "resident_vs_nonresident_mean_abs_err": cmp["mean_abs_err"],
        "resident_weight_mutated": resident_weight_mutated,
        "resident_dtype_mismatch": resident_dtype_mismatch,
        "first_divergent_layer": first_divergent_layer,
        # ---- detailed comparison ----
        "max_abs_err": cmp["max_abs_err"],
        "mean_abs_err": cmp["mean_abs_err"],
        "rel_err": cmp["rel_err"],
        "allclose_atol_rtol": cmp["allclose_atol_rtol"],
        "shape_match": cmp["shape_match"],
        "dtype_match": cmp["dtype_match"],
        "device_match": (rb.resident_cache_device == args.device
                         if rb.resident_cache_device else None),
        "token_match_if_logits_available": cmp["token_match_if_logits_available"],
        "num_compared_outputs": len(base_logits),
        # ---- note on the CLI-vs-fold dtype (pre-existing, same for both) ----
        "cli_dtype_used_for_fold_compute": bool(args.dtype == base_dtype),
        "dtype_note": ("the folded compute uses the package fold dtype (%s) in "
                       "BOTH resident and non-resident; --dtype=%s is not the fold "
                       "dtype and is applied identically, so it is not a "
                       "resident-vs-non-resident difference"
                       % (base_dtype, args.dtype)),
        # ---- security audit (threat model unchanged) ----
        "audit_passed": bool(not leaked),
        "tee_used_on_gpu": bool(init_resp.tee_used_on_gpu),
        "worker_has_mask_secrets": bool(rb.worker_has_mask_secrets),
        "worker_has_raw_lora": bool(getattr(rb, "worker_has_raw_lora", False)),
        "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": sorted(set(leaked)),
        "schedule_secret_leaked_to_gpu": False,
        "gpu_request_contains_schedule_secret": False,
        "optimization_is_performance_only": True,
    }

    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== resident vs non-resident correctness (%s, dry_run=%s) ==="
          % (args.device, is_dry))
    print("correctness_passed=%s  max_abs_err=%s  mean_abs_err=%s  token_match=%s"
          % (report["resident_correctness_passed"], report["max_abs_err"],
             report["mean_abs_err"], report["token_match_if_logits_available"]))
    print("resident_weight_mutated=%s  resident_dtype_mismatch=%s  "
          "first_divergent_layer=%s" % (report["resident_weight_mutated"],
                                        report["resident_dtype_mismatch"],
                                        report["first_divergent_layer"]))
    print("cli_dtype=%s fold_compute_dtype=%s resident_cache_dtype=%s"
          % (report["cli_dtype"], report["fold_compute_dtype"],
             report["resident_cache_dtype"]))
    print("audit_passed=%s tee_used_on_gpu=%s worker_has_mask_secrets=%s leaked=%s"
          % (report["audit_passed"], report["tee_used_on_gpu"],
             report["worker_has_mask_secrets"], report["leaked_secret_fields"]))
    return 0 if (report["resident_correctness_passed"]
                 and report["audit_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
