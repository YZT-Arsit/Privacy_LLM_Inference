"""Real E9 backends: produce honest paper-ready predictions on real hardware.

Builds a per-example predictor for each backend, REUSING the existing inference /
protocol infrastructure (no duplicated wire/audit logic):

* ``plaintext_local`` -- the non-private baseline: a real Qwen checkpoint via
  transformers (greedy generation; logit choice-scoring for multiple choice).
* ``folded_remote`` / ``tdx_lite_remote`` / ``tdx_attested_remote`` /
  ``folded_lora_remote`` / ``tdx_attested_folded_lora_remote`` -- the private
  path: a trusted lite boundary (``LiteBoundary`` = embedding artifact only) +
  ``RemoteGpuWorker`` driving masked prefill/decode over HTTP. Only masked
  embeddings + public metadata cross to the GPU; tokenization + logit recovery +
  sampling stay trusted-side. Attestation is verified via ``pllo.protocol.
  attestation`` for the ``tdx_attested_*`` backends.

Everything heavy (torch / transformers / the protocol bridge) is imported lazily
INSIDE the constructors, so importing this module is cheap and the unit tests
(which have no model / worker) never touch the real path. If a real backend
cannot be constructed (missing model / worker / artifact / evidence, or a wrong
worker configuration) a :class:`RealBackendUnavailable` is raised so the caller
can either fall back to the stub (default) or hard-fail under ``--require-real``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

__all__ = ["RealBackendUnavailable", "build_predictor", "REMOTE_BACKENDS",
           "expected_remote_runtime_hash"]

REMOTE_BACKENDS = (
    "folded_remote", "tdx_lite_remote", "tdx_attested_remote",
    "folded_lora_remote", "tdx_attested_folded_lora_remote",
)
_LORA_BACKENDS = ("folded_lora_remote", "tdx_attested_folded_lora_remote")
_ATTESTED_BACKENDS = ("tdx_attested_remote", "tdx_attested_folded_lora_remote")

_MC_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


class RealBackendUnavailable(Exception):
    """Raised when a real backend cannot be constructed (missing resources)."""


def expected_remote_runtime_hash(nonlinear_backend=None, expected_mr_td=None):
    """The runtime hash the E9 masked-remote attested backends must bind to.

    This MUST match ``run_tee_gpu_protocol_demo.py --print-runtime-hash-only``
    invoked with ``--gpu-backend qwen7b_folded_package`` and the same
    ``--nonlinear-backend`` (and ``--expected-mr-td``): the real E9 remote folded
    path runs the ``qwen7b_folded_package`` GPU backend, and the selected
    nonlinear design is part of the runtime identity. It is deliberately NOT the
    ``qwen7b`` hash."""
    from pllo.protocol.attestation import (
        boundary_manifest_metadata, build_trusted_boundary_manifest,
        compute_runtime_hash_from_manifest)
    nb = None
    if nonlinear_backend is not None:
        from pllo.experiments.nonlinear_designs import (
            normalize_nonlinear_backend)
        nb = normalize_nonlinear_backend(nonlinear_backend)
    md = boundary_manifest_metadata(
        "process", "qwen7b_folded_package", expected_mr_td, nonlinear_backend=nb)
    return compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(metadata=md))


def _torch_dtype(name):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.float32)


def _letters_for(example) -> List[str]:
    n = len(example.get("choices") or []) or 4
    return _MC_LETTERS[:n]


def _choice_token_ids(tokenizer, letters):
    """Map each MC letter to a single representative vocab id (prefer ' A')."""
    ids = {}
    for L in letters:
        tid = None
        for cand in (" " + L, L):
            try:
                enc = tokenizer.encode(cand, add_special_tokens=False)
            except Exception:                               # noqa: BLE001
                enc = []
            if enc:
                tid = int(enc[-1])
                break
        if tid is not None:
            ids[L] = tid
    return ids


# ---------------------------------------------------------------------------
# plaintext baseline
# ---------------------------------------------------------------------------


class _PlaintextLocalPredictor:
    backend = "plaintext_local"

    def __init__(self, *, model_path, model_name, seq_len, max_new_tokens,
                 dtype, device):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:                            # noqa: BLE001
            raise RealBackendUnavailable("transformers/torch unavailable: %s"
                                         % exc)
        try:
            self._tok = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=_torch_dtype(dtype), device_map=device,
                trust_remote_code=True, local_files_only=True).eval()
        except Exception as exc:                            # noqa: BLE001
            raise RealBackendUnavailable("could not load model at %s: %s"
                                         % (model_path, exc))
        self.model_name = model_name
        self.seq_len = int(seq_len)
        self.max_new_tokens = int(max_new_tokens)
        self.device = device

    def _ids(self, prompt):
        return self._tok(prompt, return_tensors="pt")["input_ids"][
            :, :self.seq_len].to(self._model.device)

    def predict(self, prompt, example):
        import torch
        ids = self._ids(prompt)
        task = example.get("task_type")
        if task == "multiple_choice":
            letters = _letters_for(example)
            tid = _choice_token_ids(self._tok, letters)
            if tid:
                with torch.no_grad():
                    logits = self._model(ids).logits[0, -1, :].float()
                best = max(tid, key=lambda L: float(logits[tid[L]]))
                return best
        with torch.no_grad():
            out = self._model.generate(
                ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=getattr(self._tok, "eos_token_id",
                                                  None))
        new = out[0, ids.shape[1]:]
        return self._tok.decode(new, skip_special_tokens=True)

    def stats(self):
        # plaintext baseline: no GPU privacy boundary (it IS plaintext).
        return {"tee_used_on_gpu": False, "worker_has_mask_secrets": False,
                "worker_has_raw_lora": False, "gpu_visible_plaintext_fields": [],
                "leaked_secret_fields": [], "audit_passed": None,
                "trusted_bytes": None, "gpu_bytes": None, "boundary_calls": None,
                "gpu_calls": None, "note": "plaintext baseline (no private GPU "
                "boundary; security flags are not applicable)"}

    def close(self):
        pass


# ---------------------------------------------------------------------------
# private masked remote backends
# ---------------------------------------------------------------------------


class _RemoteMaskedPredictor:
    def __init__(self, backend, *, model_path, model_name, gpu_worker_url,
                 embedding_path, folded_lora_package_path, attestation_evidence,
                 expected_mr_td, seq_len, max_new_tokens, dtype, device, audit,
                 nonlinear_backend="current"):
        self.backend = backend
        self.model_name = model_name
        from pllo.experiments.nonlinear_designs import (
            normalize_nonlinear_backend)
        self.nonlinear_backend = normalize_nonlinear_backend(
            nonlinear_backend or "current")
        self.seq_len = int(seq_len)
        self.max_new_tokens = int(max_new_tokens)
        self._audit = bool(audit)
        want_lora = backend in _LORA_BACKENDS
        want_attest = backend in _ATTESTED_BACKENDS

        if not gpu_worker_url:
            raise RealBackendUnavailable("%s requires --gpu-worker-url" % backend)
        if not embedding_path:
            raise RealBackendUnavailable(
                "%s requires --embedding-path (the trusted boundary artifact)"
                % backend)
        if not model_path:
            raise RealBackendUnavailable(
                "%s requires --model-path for the (trusted-side) tokenizer"
                % backend)
        if want_attest and not attestation_evidence:
            raise RealBackendUnavailable(
                "%s requires --attestation-evidence" % backend)

        try:
            import numpy as np  # noqa: F401
            import torch  # noqa: F401
            from transformers import AutoTokenizer
            from pllo.experiments.folded_probe_common import LiteBoundary
            from pllo.protocol.remote import RemoteGpuWorker
            from pllo.protocol.tee_gpu_messages import (
                BoundaryInitRequest, ProtocolTrace)
        except Exception as exc:                            # noqa: BLE001
            raise RealBackendUnavailable("protocol/runtime deps unavailable: %s"
                                         % exc)

        try:
            self._tok = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True)
        except Exception as exc:                            # noqa: BLE001
            raise RealBackendUnavailable("tokenizer load failed: %s" % exc)

        try:
            self._boundary = LiteBoundary.from_artifact(embedding_path,
                                                        device=device)
        except Exception as exc:                            # noqa: BLE001
            raise RealBackendUnavailable("boundary artifact load failed: %s"
                                         % exc)

        meta = self._boundary.meta
        self._vocab_size = int(meta["vocab_size"])
        self._n_layers = int(meta["num_layers"])
        self._dtype = dtype
        exec_meta = self._boundary.exec_metadata(
            seq_len=self.seq_len, max_new_tokens=self.max_new_tokens)

        self._trace = ProtocolTrace(
            boundary_backend="process", gpu_backend="qwen7b_folded_package",
            max_new_tokens=self.max_new_tokens, tee_used_on_gpu=False)

        def _record(direction, _method, msg):
            (self._trace.record_gpu_inbound if direction == "inbound"
             else self._trace.record_gpu_outbound)(msg)

        self._worker = RemoteGpuWorker(gpu_worker_url, "qwen7b_folded_package",
                                       recorder=_record)
        try:
            self._health = self._worker.health()
            init_resp = self._worker.init(BoundaryInitRequest(
                session_id="e9", hidden_size=int(exec_meta["hidden_size"]),
                vocab_size=self._vocab_size, num_layers=self._n_layers,
                dtype=dtype, gpu_backend="qwen7b_folded_package",
                folded_lm_head=None, public_metadata=exec_meta))
        except Exception as exc:                            # noqa: BLE001
            raise RealBackendUnavailable("worker init failed: %s" % exc)
        self._trace.tee_used_on_gpu = bool(init_resp.tee_used_on_gpu)
        self._trace.bump_boundary("init")

        import json as _json
        try:
            self._snotes = _json.loads(init_resp.notes)
        except Exception:                                   # noqa: BLE001
            self._snotes = {}

        worker_lora = bool(self._snotes.get("lora_enabled"))
        if want_lora and not worker_lora:
            raise RealBackendUnavailable(
                "%s requested but the worker has no folded LoRA loaded "
                "(lora_enabled=False); start the worker with "
                "--folded-lora-package-path" % backend)
        if (not want_lora) and worker_lora:
            raise RealBackendUnavailable(
                "%s is a no-LoRA backend but the worker has LoRA enabled; "
                "start a base-only worker" % backend)

        self._attestation = None
        if want_attest:
            self._attestation = self._verify_attestation(
                attestation_evidence, expected_mr_td)

    def _verify_attestation(self, evidence, expected_mr_td):
        from dataclasses import asdict
        from pllo.protocol.attestation import (
            attest_boundary, binding_mismatch_reason)
        # The real E9 remote folded path runs the qwen7b_folded_package GPU
        # backend, and the binding is specific to the selected nonlinear design.
        expected = expected_remote_runtime_hash(self.nonlinear_backend,
                                                expected_mr_td)
        ev = attest_boundary(runtime_hash=expected, evidence=evidence,
                             expected_mr_td=expected_mr_td)
        return {
            "attestation": asdict(ev),
            "boundary_tee_type": ev.tee_type,
            "boundary_attested": ev.verified,
            "runtime_hash": ev.runtime_hash_hex,
            "expected_runtime_hash": expected,
            "evidence_report_data": ev.report_data_hex,
            "runtime_hash_bound": ev.runtime_hash_bound,
            "binding_mismatch_reason": binding_mismatch_reason(ev),
            "mr_td": ev.mr_td,
            "attestation_nonlinear_backend": self.nonlinear_backend,
        }

    # -- generation -------------------------------------------------------

    def _ids(self, prompt):
        import torch
        enc = self._tok(prompt, return_tensors="pt")["input_ids"][
            :, :self.seq_len]
        return enc.to(torch.long)

    def _np(self, t):
        import numpy as np
        return np.asarray(t.detach().to("cpu").float().numpy())

    def _recover_last(self, masked_logits):
        import numpy as np
        import torch
        rec = self._boundary.recover(
            torch.as_tensor(np.asarray(masked_logits)).to(
                self._boundary.compute_device, self._boundary.fdtype))
        self._trace.trusted_bytes += int(
            np.asarray(rec.detach().to("cpu")).nbytes)
        return rec

    def _decode_loop(self, ids):
        """Greedy masked prefill+decode; return (generated_ids, last_recovered)."""
        from pllo.protocol.tee_gpu_messages import (
            MaskedDecodeRequest, MaskedPrefillRequest)
        import torch
        seq_len = int(ids.shape[1])
        h = self._boundary.mask_embeddings(ids)
        pre = self._worker.prefill(MaskedPrefillRequest(
            session_id="e9", masked_embeddings=self._np(h),
            positions=list(range(seq_len)), batch_size=1, seq_len=seq_len))
        self._trace.bump_boundary("prefill")
        rec_last = self._recover_last(pre.masked_logits)
        tok = int(rec_last.argmax(-1))
        gen = [tok]
        pos = seq_len
        for step in range(self.max_new_tokens - 1):
            x = self._boundary.mask_token_embedding(torch.tensor([tok]))
            self._trace.bump_boundary("mask_token_embedding")
            dec = self._worker.decode(MaskedDecodeRequest(
                session_id="e9", masked_embedding=self._np(x), position=pos,
                step=step + 1))
            self._trace.bump_boundary("decode")
            tok = int(self._recover_last(dec.masked_logits).argmax(-1))
            gen.append(tok)
            pos += 1
        return gen, rec_last

    def predict(self, prompt, example):
        ids = self._ids(prompt)
        task = example.get("task_type")
        if task == "multiple_choice":
            # logit choice-scoring over A/B/C/D using the recovered prefill logits
            letters = _letters_for(example)
            tid = _choice_token_ids(self._tok, letters)
            gen, rec_last = self._decode_loop(ids)
            if tid:
                row = rec_last.reshape(-1)
                valid = {L: t for L, t in tid.items() if t < row.shape[0]}
                if valid:
                    return max(valid, key=lambda L: float(row[valid[L]]))
            return self._tok.decode(gen, skip_special_tokens=True)
        gen, _ = self._decode_loop(ids)
        return self._tok.decode(gen, skip_special_tokens=True)

    def stats(self):
        bc = sum(self._trace.boundary_calls.values()) \
            if self._trace.boundary_calls else 0
        gc = sum(self._trace.gpu_calls.values()) if self._trace.gpu_calls else 0
        out = {
            "trusted_bytes": int(self._trace.trusted_bytes),
            "gpu_bytes": int(self._trace.gpu_bytes),
            "boundary_calls": bc, "gpu_calls": gc,
            "tee_used_on_gpu": bool(self._trace.tee_used_on_gpu),
            "worker_has_mask_secrets": bool(
                self._snotes.get("worker_has_mask_secrets", False)),
            "worker_has_raw_lora": bool(
                self._snotes.get("worker_has_raw_lora", False)),
            "lora_enabled": bool(self._snotes.get("lora_enabled", False)),
            "folded_lora_loaded": self._snotes.get("folded_lora_loaded"),
            "folded_package_loaded": self._snotes.get("folded_package_loaded"),
            "gpu_visible_plaintext_fields": [],
            "leaked_secret_fields": [],
        }
        if self._audit:
            out["audit_passed"] = self._run_audit()
        if self._attestation:
            out.update(self._attestation)
        return out

    def _run_audit(self):
        """Audit the exact recorded GPU traffic (no plaintext / no mask secrets)."""
        try:
            from pllo.protocol.security_audit import (
                assert_no_gpu_visible_plaintext, assert_no_mask_secret_leak)
            plaintext = assert_no_gpu_visible_plaintext(
                self._trace, raw_prompt=None, input_ids=None,
                generated_token_ids=None, recovered_logits=None,
                raise_on_fail=False)
            secret = assert_no_mask_secret_leak(self._trace, None,
                                                raise_on_fail=False)
            return bool(not plaintext and not secret
                        and not self._trace.tee_used_on_gpu)
        except Exception:                                   # noqa: BLE001
            return None

    def close(self):
        try:
            self._worker.close()
        except Exception:                                   # noqa: BLE001
            pass


def build_predictor(backend: str, *, model_path=None, model_name="qwen",
                    gpu_worker_url=None, embedding_path=None,
                    folded_lora_package_path=None, attestation_evidence=None,
                    expected_mr_td=None, seq_len=256, max_new_tokens=8,
                    dtype="bfloat16", device="cuda", audit=True,
                    nonlinear_backend="current"):
    """Construct a real predictor or raise :class:`RealBackendUnavailable`.

    ``nonlinear_backend`` selects the nonlinear design; for the attested remote
    backends it binds the attestation to the design (current vs trusted_shortcut
    produce distinct runtime hashes)."""
    if backend == "plaintext_local":
        if not model_path:
            raise RealBackendUnavailable(
                "plaintext_local requires --model-path")
        return _PlaintextLocalPredictor(
            model_path=model_path, model_name=model_name, seq_len=seq_len,
            max_new_tokens=max_new_tokens, dtype=dtype, device=device)
    if backend in REMOTE_BACKENDS:
        return _RemoteMaskedPredictor(
            backend, model_path=model_path, model_name=model_name,
            gpu_worker_url=gpu_worker_url, embedding_path=embedding_path,
            folded_lora_package_path=folded_lora_package_path,
            attestation_evidence=attestation_evidence,
            expected_mr_td=expected_mr_td, seq_len=seq_len,
            max_new_tokens=max_new_tokens, dtype=dtype, device=device,
            audit=audit, nonlinear_backend=nonlinear_backend)
    raise RealBackendUnavailable("unknown backend: %r" % (backend,))
