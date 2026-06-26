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


def format_prompt_for_generation(prompt, tokenizer, use_chat_template):
    """Single source of truth for prompt formatting (trusted-side), shared by the
    plaintext and folded-remote predictors so they decode the IDENTICAL string.

    ``use_chat_template`` True -> ``tokenizer.apply_chat_template([{user}],
    tokenize=False, add_generation_prompt=True)`` (Qwen injects its default system
    prompt + the assistant generation prefix); False or no tokenizer -> the raw
    prompt. Returns a STRING (tokenization happens downstream, trusted-side)."""
    if not use_chat_template or tokenizer is None:
        return prompt
    fn = getattr(tokenizer, "apply_chat_template", None)
    if fn is None:
        return prompt
    return fn([{"role": "user", "content": prompt}], tokenize=False,
              add_generation_prompt=True)


def prompt_format_info(tokenizer, raw, use_chat_template, seq_len):
    """Trusted-side, non-secret prompt-formatting metadata for the report: the
    format, a sha256 of the FORMATTED string (not the string itself), and the raw
    / chat / actually-used token counts. Never returns the prompt or token ids."""
    import hashlib
    formatted = format_prompt_for_generation(raw, tokenizer, use_chat_template)

    def _count(s):
        try:
            return len(tokenizer(s)["input_ids"])
        except Exception:                                    # noqa: BLE001
            return None
    raw_n = _count(raw)
    chat_n = _count(format_prompt_for_generation(raw, tokenizer, True))
    used = _count(formatted)
    used = min(used, int(seq_len)) if used is not None else None
    return {
        "prompt_format": "chat" if use_chat_template else "raw",
        "formatted_prompt_sha256": hashlib.sha256(
            formatted.encode("utf-8")).hexdigest(),
        "raw_prompt_token_count": raw_n,
        "chat_prompt_token_count": chat_n,
        "prompt_token_count": used,
    }


def apply_repetition_penalty(rec, seen_ids, penalty):
    """Pure HF-equivalent repetition_penalty over recovered logits (trusted-side).

    For each already-seen token id: ``logit < 0 -> *penalty`` else ``/penalty``
    (HF ``RepetitionPenaltyLogitsProcessor``). ``rec`` is a torch tensor; returns
    a possibly-adjusted tensor of the same shape. No-op for penalty in (None, 1.0)
    or empty history. Shared by the remote predictor and the correctness tool so
    both apply the IDENTICAL processor."""
    if penalty is None or penalty == 1.0 or not seen_ids:
        return rec
    import torch
    row = rec.reshape(-1)
    idx = torch.as_tensor(sorted({int(t) for t in seen_ids}),
                          dtype=torch.long, device=row.device)
    idx = idx[idx < row.shape[0]]
    if idx.numel() == 0:
        return rec
    sel = row.index_select(0, idx)
    penal = torch.where(sel < 0, sel * penalty, sel / penalty)
    row = row.clone()
    row.index_copy_(0, idx, penal)
    return row.reshape(rec.shape)


def _normalize_eos_ids(eos) -> set:
    """Normalise an eos_token_id (int, list[int], or None) to a set of ints, to
    match ``model.generate``'s acceptance of a single id or a list of ids
    (Qwen2.5 uses ``[151645, 151643]``)."""
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(e) for e in eos if e is not None}
    return {int(eos)}


# ---------------------------------------------------------------------------
# plaintext baseline
# ---------------------------------------------------------------------------


class _PlaintextLocalPredictor:
    backend = "plaintext_local"

    def __init__(self, *, model_path, model_name, seq_len, max_new_tokens,
                 dtype, device, use_chat_template=False):
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
        self._use_chat_template = bool(use_chat_template)

    def format_prompt(self, prompt):
        """Apply the shared chat-template formatting (trusted-side)."""
        return format_prompt_for_generation(prompt, self._tok,
                                            self._use_chat_template)

    def prompt_info(self, prompt):
        return prompt_format_info(self._tok, prompt, self._use_chat_template,
                                  self.seq_len)

    def _encode(self, prompt):
        """Format (chat template if enabled) -> tokenize -> (input_ids,
        attention_mask), truncated to seq_len and on the model device. The
        attention mask is always passed so padding / left-truncation never
        corrupts the logits."""
        enc = self._tok(self.format_prompt(prompt), return_tensors="pt")
        ids = enc["input_ids"][:, :self.seq_len].to(self._model.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn[:, :self.seq_len].to(self._model.device)
        return ids, attn

    def _ids(self, prompt):
        return self._encode(prompt)[0]

    def predict(self, prompt, example):
        import torch
        ids, attn = self._encode(prompt)
        task = example.get("task_type")
        if task == "multiple_choice":
            letters = _letters_for(example)
            tid = _choice_token_ids(self._tok, letters)
            if tid:
                with torch.no_grad():
                    logits = self._model(
                        ids, attention_mask=attn).logits[0, -1, :].float()
                best = max(tid, key=lambda L: float(logits[tid[L]]))
                return best
        with torch.no_grad():
            out = self._model.generate(
                ids, attention_mask=attn, max_new_tokens=self.max_new_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=getattr(self._tok, "eos_token_id", None))
        new = out[0, ids.shape[1]:]
        return self._tok.decode(new, skip_special_tokens=True)

    def generate(self, prompt):
        """Greedy open-ended generation -> {text, token_ids} (deterministic).

        Used by the generation benchmarks. Passes the attention mask so the
        decode is correct, and returns the newly generated token ids so
        token-level preservation can be measured."""
        import torch
        ids, attn = self._encode(prompt)
        with torch.no_grad():
            out = self._model.generate(
                ids, attention_mask=attn, max_new_tokens=self.max_new_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=getattr(self._tok, "eos_token_id", None))
        new = out[0, ids.shape[1]:]
        info = self.prompt_info(prompt)
        return {"text": self._tok.decode(new, skip_special_tokens=True),
                "token_ids": [int(t) for t in new.tolist()], **info}

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
                 nonlinear_backend="current", align_generation_config=False,
                 repetition_penalty=None, stop_on_eos=True,
                 length_hide_generation=False, use_chat_template=False):
        self.backend = backend
        self.model_name = model_name
        self._use_chat_template = bool(use_chat_template)
        from pllo.experiments.nonlinear_designs import (
            normalize_nonlinear_backend)
        self.nonlinear_backend = normalize_nonlinear_backend(
            nonlinear_backend or "current")
        self.seq_len = int(seq_len)
        self.max_new_tokens = int(max_new_tokens)
        self._audit = bool(audit)
        # TRUSTED-SIDE generation processor alignment (default OFF -> old raw
        # greedy behaviour). When on, the trusted side applies the same
        # repetition_penalty the plaintext baseline (model.generate) applies, AFTER
        # logits recovery and BEFORE argmax/sampling -- entirely trusted-side. The
        # token history, the recovered logits, and the sampling decision NEVER
        # cross to the GPU.
        self._align_gen = bool(align_generation_config)
        self._rep_penalty = (float(repetition_penalty)
                             if repetition_penalty is not None else None)
        self._gen_processor_applied = False
        # TRUSTED-SIDE EOS stop (default ON -> aligns model.generate, which stops
        # at generation_config.eos_token_id). The eos decision + token history are
        # trusted-only; nothing extra crosses to the GPU. The generation-config /
        # tokenizer values are resolved below, once the tokenizer is loaded.
        self._stop_on_eos = bool(stop_on_eos)
        self._examples_finish: list = []        # per-example finish records
        self._model_path = model_path
        self._eos_ids: set = set()
        self._pad_token_id = None
        # STRICT length-hiding mode (default OFF -> high-performance default path).
        # When on, after the trusted side detects EOS it keeps issuing DUMMY masked
        # decode rounds to a fixed budget so the GPU only sees a constant number of
        # decode rounds and cannot infer the true output length. The dummy token id
        # is trusted-only and never reported / sent to the GPU.
        self._length_hide = bool(length_hide_generation)
        self._dummy_token_id = 0                 # set properly once pad is resolved
        self._lh_true_latency_s = 0.0
        self._lh_dummy_latency_s = 0.0
        self._lh_gpu_rounds = 0
        self._lh_returned_tokens = 0
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

        # Resolve generation-config alignment values (repetition_penalty + eos/pad)
        # now that the tokenizer is loaded, aligned with model.generate. Reads
        # generation_config.json only -- never loads model weights.
        gc = None
        try:
            from transformers import GenerationConfig
            gc = GenerationConfig.from_pretrained(model_path,
                                                  local_files_only=True)
        except Exception:                                    # noqa: BLE001
            gc = None
        if self._align_gen and self._rep_penalty is None:
            rp = getattr(gc, "repetition_penalty", None) if gc else None
            self._rep_penalty = float(rp) if rp else None
        eos = getattr(gc, "eos_token_id", None) if gc else None
        if eos is None:
            eos = getattr(self._tok, "eos_token_id", None)
        self._eos_ids = _normalize_eos_ids(eos)
        pad = getattr(gc, "pad_token_id", None) if gc else None
        if pad is None:
            pad = getattr(self._tok, "pad_token_id", None)
        self._pad_token_id = int(pad) if pad is not None else None
        # fixed trusted-side dummy token for length-hiding (pad, else 0). Trusted
        # only -- never reported, never sent to the GPU (only its masked embedding).
        self._dummy_token_id = (self._pad_token_id
                                if self._pad_token_id is not None else 0)

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
        # real runtime model config (so schedule/report sizing uses the genuine
        # hidden_size/dtype, never a mock placeholder)
        self._hidden_size = int(exec_meta["hidden_size"])

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

        # Optional trusted-side precomputed obfuscation schedule (default off).
        # When attached, each decode step consumes its step-specific slot. This
        # is a trusted-only object: nothing from it ever crosses to the GPU.
        self._schedule = None
        # Optional per-token, per-stage decode profiler (default off).
        self._profiler = None

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

    def format_prompt(self, prompt):
        """Apply the shared chat-template formatting (trusted-side); identical to
        the plaintext predictor so both decode the SAME formatted string."""
        return format_prompt_for_generation(prompt, self._tok,
                                            self._use_chat_template)

    def prompt_info(self, prompt):
        return prompt_format_info(self._tok, prompt, self._use_chat_template,
                                  self.seq_len)

    def _ids(self, prompt):
        import torch
        enc = self._tok(self.format_prompt(prompt), return_tensors="pt")[
            "input_ids"][:, :self.seq_len]
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

    def model_runtime_config(self) -> Dict[str, Any]:
        """The genuine runtime model config (no mock placeholders) so the schedule
        + report size against the real ``hidden_size``/``dtype`` -- e.g. Qwen2.5-7B
        is hidden_size=3584, dtype=bfloat16, not the local mock defaults."""
        return {"hidden_size": self._hidden_size, "dtype": self._dtype,
                "num_layers": self._n_layers, "vocab_size": self._vocab_size}

    def attach_obfuscation_schedule(self, schedule) -> None:
        """Attach a trusted-side precomputed obfuscation schedule (optional). Its
        per-step slots are consumed during decode; its secrets never leave the
        trusted runtime (no schedule field is ever added to a GPU request)."""
        self._schedule = schedule

    def enable_decode_profiling(self, enabled: bool = True,
                                request_worker_timing: bool = False) -> None:
        """Turn on per-token, per-stage decode profiling (default off). Times the
        9 decode stages + per-step boundary/gpu call & byte deltas.

        ``request_worker_timing`` additionally asks the untrusted worker to return
        its PUBLIC forward-timing metadata so each step's gpu_worker_roundtrip can
        be split into network vs worker compute. No secret is ever sent/received;
        the worker timing is audited on receipt."""
        from pllo.benchmarks.decode_profiler import DecodeProfiler
        self._profiler = DecodeProfiler(counters=self._trace_counters,
                                        enabled=bool(enabled))
        if request_worker_timing:
            try:
                self._worker.request_worker_timing = True
            except Exception:                                # noqa: BLE001
                pass

    def decode_profiler(self):
        return self._profiler

    def _record_worker_timing(self, resp) -> None:
        """Attach the worker's PUBLIC timing metadata (if any) to the current
        profiler step, after a defense-in-depth secret audit."""
        if self._profiler is None:
            return
        wt = getattr(resp, "worker_timing", None)
        if not isinstance(wt, dict):
            return
        try:
            from pllo.protocol.worker_timing import audit_worker_timing_no_secrets
            audit_worker_timing_no_secrets(wt)
            self._profiler.set_worker_timing(wt)
        except Exception:                                    # noqa: BLE001
            # never let timing metadata corrupt decode; just skip it
            pass

    def _trace_counters(self):
        t = self._trace
        bc = sum(t.boundary_calls.values()) if t.boundary_calls else 0
        gc = sum(t.gpu_calls.values()) if t.gpu_calls else 0
        return {"boundary_calls": bc, "gpu_calls": gc,
                "trusted_bytes": int(t.trusted_bytes),
                "gpu_bytes": int(t.gpu_bytes)}

    def _consume_slot(self, step_id: int) -> None:
        if self._schedule is None:
            return
        try:
            self._schedule.consume(step_id)
        except Exception:                                    # noqa: BLE001
            # schedule too short / already consumed: do not corrupt decode; the
            # report's slots_consumed reflects what actually happened.
            pass

    def _pstage(self, name):
        from contextlib import nullcontext
        return (self._profiler.stage(name) if self._profiler is not None
                else nullcontext())

    def _pbegin(self, step_id, phase):
        if self._profiler is not None:
            self._profiler.begin_step(step_id, phase)

    def _pend(self, token_id=None):
        if self._profiler is not None:
            self._profiler.end_step(token_id)

    def _apply_generation_processors(self, rec, seen_ids):
        """TRUSTED-SIDE generation-config logit processors (currently only
        repetition_penalty), applied to the recovered logits BEFORE argmax. Matches
        HF ``RepetitionPenaltyLogitsProcessor``: for each already-seen token id,
        ``logit < 0 -> *penalty`` else ``logit / penalty``. ``seen_ids`` (the token
        history) is trusted-only and is never sent to the GPU. Returns the
        (possibly) adjusted logits; a no-op when alignment is off."""
        if (not self._align_gen or self._rep_penalty is None
                or self._rep_penalty == 1.0 or not seen_ids):
            return rec
        out = apply_repetition_penalty(rec, seen_ids, self._rep_penalty)
        self._gen_processor_applied = True
        return out

    def _decode_loop(self, ids):
        """Greedy masked prefill+decode; return (generated_ids, last_recovered).

        Each stage is timed when profiling is enabled (no overhead otherwise).
        No schedule secret / mask / pad / inverse is ever placed in a request."""
        from pllo.protocol.tee_gpu_messages import (
            MaskedDecodeRequest, MaskedPrefillRequest)
        import time
        import torch
        seq_len = int(ids.shape[1])
        # trusted-only token history for generation-config processors (never sent
        # to the GPU); seeded with the prompt tokens like HF's running input_ids.
        seen_ids = [int(t) for t in ids.reshape(-1).tolist()]

        # ---- prefill step ----
        self._pbegin(0, "prefill")
        with self._pstage("trusted_input_embedding"):
            h = self._boundary.mask_embeddings(ids)
        with self._pstage("http_request_serialization"):
            pre_req = MaskedPrefillRequest(
                session_id="e9", masked_embeddings=self._np(h),
                positions=list(range(seq_len)), batch_size=1, seq_len=seq_len)
        with self._pstage("gpu_worker_roundtrip"):
            pre = self._worker.prefill(pre_req)
            self._trace.bump_boundary("prefill")
        self._record_worker_timing(pre)
        with self._pstage("schedule_slot_lookup"):
            self._consume_slot(0)                   # step 0 obfuscation slot
        with self._pstage("trusted_nonlinear_restore_logits"):
            rec_last = self._recover_last(pre.masked_logits)
        with self._pstage("sampling"):
            rec_proc = self._apply_generation_processors(rec_last, seen_ids)
            tok = int(rec_proc.argmax(-1))
        self._pend(tok)
        gen = [tok]
        seen_ids.append(tok)
        pos = seq_len
        # TRUSTED-SIDE EOS detection. The eos decision + token history stay
        # trusted-side; nothing extra is ever placed in a GPU request. ``real_done``
        # = the true sequence has ended; in length-hiding mode we keep issuing
        # DUMMY decode rounds afterwards so the GPU sees a fixed round count.
        eos_hit = (tok in self._eos_ids)
        real_done = bool(eos_hit and (self._stop_on_eos or self._length_hide))
        stopped_by_eos = real_done
        real_decode_rounds = 0
        dummy_decode_rounds = 0
        t_real = 0.0
        t_dummy = 0.0

        # ---- decode steps ----
        for step in range(self.max_new_tokens - 1):
            if real_done and not self._length_hide:
                break                               # DEFAULT mode: stop at EOS
            if real_done and self._length_hide:
                # DUMMY round: keep the GPU at a fixed decode-round budget. The
                # dummy token id stays trusted-side; the GPU sees only its masked
                # embedding. Recovered logits / token are DISCARDED (never appended
                # to the real output, never recovered, never reported).
                t0 = time.perf_counter()
                xd = self._boundary.mask_token_embedding(
                    torch.tensor([self._dummy_token_id]))
                self._trace.bump_boundary("mask_token_embedding")
                ddec_req = MaskedDecodeRequest(
                    session_id="e9", masked_embedding=self._np(xd), position=pos,
                    step=step + 1)
                self._worker.decode(ddec_req)       # discard response
                self._trace.bump_boundary("decode")
                pos += 1
                dummy_decode_rounds += 1
                t_dummy += time.perf_counter() - t0
                continue

            # ---- real decode round ----
            t0 = time.perf_counter()
            self._pbegin(step + 1, "decode")
            with self._pstage("prompt_token_prep"):
                tok_tensor = torch.tensor([tok])
            with self._pstage("trusted_input_embedding"):
                x = self._boundary.mask_token_embedding(tok_tensor)
                self._trace.bump_boundary("mask_token_embedding")
            with self._pstage("schedule_slot_lookup"):
                self._consume_slot(step + 1)        # fresh per-step slot
            with self._pstage("http_request_serialization"):
                dec_req = MaskedDecodeRequest(
                    session_id="e9", masked_embedding=self._np(x), position=pos,
                    step=step + 1)
            with self._pstage("gpu_worker_roundtrip"):
                dec = self._worker.decode(dec_req)
                self._trace.bump_boundary("decode")
            self._record_worker_timing(dec)
            with self._pstage("trusted_nonlinear_restore_logits"):
                rec = self._recover_last(dec.masked_logits)
            with self._pstage("sampling"):
                rec_proc = self._apply_generation_processors(rec, seen_ids)
                tok = int(rec_proc.argmax(-1))
            with self._pstage("kv_cache_update"):
                pos += 1
            gen.append(tok)
            seen_ids.append(tok)
            self._pend(tok)
            real_decode_rounds += 1
            t_real += time.perf_counter() - t0
            if tok in self._eos_ids and (self._stop_on_eos or self._length_hide):
                real_done = True
                stopped_by_eos = True
                if not self._length_hide:
                    break

        finish_reason = "eos" if stopped_by_eos else "length"
        gpu_decode_rounds = 1 + real_decode_rounds + dummy_decode_rounds  # +prefill
        self._examples_finish.append({
            "finish_reason": finish_reason,
            "true_finish_reason": finish_reason,
            "stopped_by_eos": bool(stopped_by_eos),
            "generated_tokens": len(gen),
            "true_generated_tokens": len(gen),
            "output_tokens_returned": len(gen),
            "gpu_decode_rounds": gpu_decode_rounds,
            "dummy_decode_rounds": dummy_decode_rounds,
        })
        self._lh_true_latency_s += t_real
        self._lh_dummy_latency_s += t_dummy
        self._lh_gpu_rounds += gpu_decode_rounds
        self._lh_returned_tokens += len(gen)
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

    def generate(self, prompt):
        """Greedy masked prefill+decode over the remote folded worker ->
        {text, token_ids}. Only masked embeddings + public metadata cross to the
        GPU; tokenization + recovery + sampling stay trusted-side."""
        ids = self._ids(prompt)
        gen, _ = self._decode_loop(ids)
        info = self.prompt_info(prompt)
        return {"text": self._tok.decode(gen, skip_special_tokens=True),
                "token_ids": [int(t) for t in gen], **info}

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
            # trusted-side generation-config alignment (repetition_penalty)
            "generation_processors_applied": bool(self._gen_processor_applied),
            "repetition_penalty": (self._rep_penalty if self._align_gen
                                   else None),
            "generation_config_aligned_with_plaintext": bool(
                self._align_gen and self._rep_penalty not in (None, 1.0)),
            "generation_processor_location": "trusted_side",
            "plaintext_logits_or_sampling_on_gpu": False,
        }
        # trusted-side EOS stop, aligned with model.generate (per-example records)
        fin = self._examples_finish
        reasons = [e["finish_reason"] for e in fin]
        counts = [e["generated_tokens"] for e in fin]
        true_counts = [e["true_generated_tokens"] for e in fin]
        gpu_rounds = [e["gpu_decode_rounds"] for e in fin]
        dummy_rounds = [e["dummy_decode_rounds"] for e in fin]
        returned = [e["output_tokens_returned"] for e in fin]
        total_dummy = sum(dummy_rounds)
        total_returned = sum(returned)
        out.update({
            "eos_token_id": (sorted(self._eos_ids) if self._eos_ids else None),
            "pad_token_id": self._pad_token_id,
            "stop_on_eos": bool(self._stop_on_eos),
            "stopped_by_eos": bool(any(e["stopped_by_eos"] for e in fin)),
            "finish_reason": (None if not reasons
                              else ("eos" if all(r == "eos" for r in reasons)
                                    else ("length"
                                          if all(r == "length" for r in reasons)
                                          else "mixed"))),
            "true_finish_reason": (None if not reasons
                                   else ("eos" if all(r == "eos" for r in reasons)
                                         else ("length"
                                               if all(r == "length"
                                                      for r in reasons)
                                               else "mixed"))),
            "finish_reason_per_example": reasons,
            "stopped_by_eos_per_example": [e["stopped_by_eos"] for e in fin],
            "generated_tokens_per_example": counts,
            "true_generated_tokens_per_example": true_counts,
            "output_tokens_returned_per_example": returned,
            "gpu_decode_rounds_per_example": gpu_rounds,
            "dummy_decode_rounds_per_example": dummy_rounds,
            "max_new_tokens_requested": int(self.max_new_tokens),
            "max_new_tokens_consumed": (max(counts) if counts else None),
            # ---- strict length-hiding mode (public, no secret) ----
            "length_hiding_enabled": bool(self._length_hide),
            "dummy_decode_after_eos": bool(self._length_hide),
            "length_hiding_overhead_tokens": int(total_dummy),
            "length_hiding_overhead_ratio": (
                round(total_dummy / total_returned, 6) if total_returned else None),
            "length_hiding_security_note": (
                "strict mode: after trusted-side EOS the boundary issues dummy "
                "masked decode rounds to a fixed max_new_tokens budget, so the GPU "
                "sees a constant decode-round count and cannot infer the true "
                "output length; dummy tokens/logits stay trusted-side"
                if self._length_hide else
                "default mode: trusted-side EOS stop; the GPU observes the number "
                "of decode rounds (coarse output length), as in any online decode"),
            # ---- performance split (default vs strict length-hiding) ----
            "true_output_latency_s": round(self._lh_true_latency_s, 6),
            "dummy_decode_latency_s": round(self._lh_dummy_latency_s, 6),
            "latency_per_returned_token_s": (
                round(self._lh_true_latency_s / self._lh_returned_tokens, 6)
                if self._lh_returned_tokens else None),
            "latency_per_gpu_decode_round_s": (
                round((self._lh_true_latency_s + self._lh_dummy_latency_s)
                      / self._lh_gpu_rounds, 6) if self._lh_gpu_rounds else None),
            "plaintext_logits_or_sampling_on_gpu": False,
            "dummy_token_id_on_gpu": False,        # dummy id never leaves trusted
        })
        # measured nonlinear-design execution evidence from the worker (post-run
        # health): proves design B genuinely lifted the activation on the GPU.
        out["nonlinear_backend"] = self.nonlinear_backend
        out["nonlinear_op_backend"] = self._snotes.get("nonlinear_op_backend")
        # resident-cache status reported at worker init notes (public, no secret)
        out["resident_folded_weights"] = bool(
            self._snotes.get("resident_folded_weights", False))
        try:
            health = self._worker.health() or {}
            ev = health.get("nonlinear_execution_evidence") or {}
            if ev:
                out.update(ev)
            # public weight-resident cache status from worker /health
            rs = health.get("resident_status") or {}
            for k in ("resident_folded_weights", "resident_weight_init_latency_s",
                      "resident_weight_memory_gb", "resident_cache_num_layers",
                      "resident_cache_device", "resident_cache_dtype",
                      "resident_cache_oom", "resident_cache_fallback_used",
                      "resident_cache_active",
                      # per-decode weight-movement counters (the resident win)
                      "weight_reloaded_each_step",
                      "weight_shard_loads_per_decode_step",
                      "folded_layer_dict_builds_per_decode_step",
                      "cpu_to_gpu_weight_copies_per_decode_step"):
                if k in rs:
                    out[k] = rs[k]
        except Exception:                                    # noqa: BLE001
            pass
        if self._audit:
            out["audit_passed"] = self._run_audit()
        if self._attestation:
            out.update(self._attestation)
        # per-token, per-stage profile (only when profiling was enabled)
        if self._profiler is not None:
            agg = self._profiler.aggregate()
            out["decode_profile"] = agg
            out["decode_bottleneck_stage"] = agg.get("bottleneck_stage")
            out["decode_stage_total_s"] = agg.get("stage_total_s")
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
                    nonlinear_backend="current", align_generation_config=False,
                    repetition_penalty=None, stop_on_eos=True,
                    length_hide_generation=False, use_chat_template=False):
    """Construct a real predictor or raise :class:`RealBackendUnavailable`.

    ``nonlinear_backend`` selects the nonlinear design; for the attested remote
    backends it binds the attestation to the design (current vs trusted_shortcut
    produce distinct runtime hashes). ``use_chat_template`` applies the SAME
    trusted-side chat-template formatting to both plaintext and folded backends."""
    if backend == "plaintext_local":
        if not model_path:
            raise RealBackendUnavailable(
                "plaintext_local requires --model-path")
        return _PlaintextLocalPredictor(
            model_path=model_path, model_name=model_name, seq_len=seq_len,
            max_new_tokens=max_new_tokens, dtype=dtype, device=device,
            use_chat_template=use_chat_template)
    if backend in REMOTE_BACKENDS:
        return _RemoteMaskedPredictor(
            backend, model_path=model_path, model_name=model_name,
            gpu_worker_url=gpu_worker_url, embedding_path=embedding_path,
            folded_lora_package_path=folded_lora_package_path,
            attestation_evidence=attestation_evidence,
            expected_mr_td=expected_mr_td, seq_len=seq_len,
            max_new_tokens=max_new_tokens, dtype=dtype, device=device,
            audit=audit, nonlinear_backend=nonlinear_backend,
            align_generation_config=align_generation_config,
            repetition_penalty=repetition_penalty, stop_on_eos=stop_on_eos,
            length_hide_generation=length_hide_generation,
            use_chat_template=use_chat_template)
    raise RealBackendUnavailable("unknown backend: %r" % (backend,))
