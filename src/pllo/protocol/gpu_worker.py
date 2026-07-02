"""Untrusted GPU worker backends + a process-isolated worker client.

The GPU worker is the **untrusted** side of the protocol. It only ever sees the
:class:`~pllo.protocol.tee_gpu_messages.MaskedPrefillRequest` /
:class:`~pllo.protocol.tee_gpu_messages.MaskedDecodeRequest` messages (masked
embeddings + public metadata + the folded LM head) and returns masked logits.

Two backends:

* ``MockGpuBackend`` -- numpy identity "decoder": next-token logits depend only
  on the current masked hidden state via the folded head. Lets the whole
  trusted/untrusted round trip run with no torch and exact recovery. It stores
  only a masked KV cache (the masked hidden states it was handed).
* ``Qwen7BGpuBackend`` -- lazy-torch adapter for the real masked Qwen pipeline.
  Constructible with numpy only (no torch import at construction); a real
  forward requires CUDA + a checkpoint and is meant to run on the GPU server.
  ``tee_used`` is ``False`` either way -- the model never enters the TEE.

``LocalGpuWorker`` runs a chosen backend in a separate ``spawn`` process and
talks to it over a ``Pipe`` -- this is the ``local_two_process`` deployment and
lets a recorder capture exactly what crosses to the untrusted side.

numpy + standard library only (torch is imported lazily by the qwen backend).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from pllo.protocol.tee_gpu_messages import (
    BoundaryInitRequest,
    BoundaryInitResponse,
    MaskedDecodeRequest,
    MaskedDecodeResponse,
    MaskedPrefillRequest,
    MaskedPrefillResponse,
)

__all__ = [
    "GpuBackend",
    "MockGpuBackend",
    "Qwen7BGpuBackend",
    "Qwen7BFoldedPackageGpuBackend",
    "make_gpu_backend",
    "LocalGpuWorker",
]


# ---------------------------------------------------------------------------
# Backend interface + implementations
# ---------------------------------------------------------------------------


class GpuBackend(ABC):
    """Untrusted masked backend. Sees masked tensors + public metadata only."""

    name: str
    tee_used: bool = False                  # the GPU side is NEVER a TEE
    # Set by the server (under its lock) when the client opts into worker timing
    # via the X-PLLO-WorkerTiming header. Default False -> zero timing overhead and
    # responses identical to the historical path.
    collect_worker_timing: bool = False

    @abstractmethod
    def init(self, req: BoundaryInitRequest) -> BoundaryInitResponse: ...

    @abstractmethod
    def prefill(self, req: MaskedPrefillRequest) -> MaskedPrefillResponse: ...

    @abstractmethod
    def decode(self, req: MaskedDecodeRequest) -> MaskedDecodeResponse: ...

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "tee_used": self.tee_used}


class MockGpuBackend(GpuBackend):
    """Identity-decoder mock: ``masked_logits = masked_hidden @ folded_head``.

    With an identity decoder the next-token logits depend only on the last
    masked hidden state, so ``X_tilde @ W_tilde = (X @ N) @ (N^{-1} W M) =
    L @ M`` -- the trusted side recovers the exact plaintext logits. Stores only
    the masked hidden states it receives (the "masked KV cache")."""

    name = "mock"
    tee_used = False

    def __init__(self) -> None:
        self._folded_head: np.ndarray | None = None
        self._session: str | None = None
        self._masked_kv: list[np.ndarray] = []   # masked hidden states only

    def init(self, req: BoundaryInitRequest) -> BoundaryInitResponse:
        if req.folded_lm_head is None:
            raise ValueError("mock backend requires a folded LM head")
        self._folded_head = np.asarray(req.folded_lm_head)
        self._session = req.session_id
        self._masked_kv = []
        return BoundaryInitResponse(
            session_id=req.session_id, ok=True, gpu_backend=self.name,
            tee_used_on_gpu=False,
            notes="mock identity decoder; masked KV cache only; no TEE")

    def _logits(self, masked_hidden_last: np.ndarray) -> np.ndarray:
        assert self._folded_head is not None
        return masked_hidden_last @ self._folded_head        # [B, V]

    def _coarse_timing(self, phase, forward_s):
        from pllo.protocol.worker_timing import coarse_forward_metadata
        dt = str(self._folded_head.dtype) if self._folded_head is not None \
            else None
        # mock runs on CPU (synchronous) -> wall_clock is accurate
        return coarse_forward_metadata(
            phase=phase, backend_name=self.name, device="cpu", dtype=dt,
            forward_s=forward_s, num_layers=None, timing_method="wall_clock",
            is_cuda_synchronized=False)

    def prefill(self, req: MaskedPrefillRequest) -> MaskedPrefillResponse:
        timing = bool(getattr(self, "collect_worker_timing", False))
        t0 = time.perf_counter() if timing else None
        x = np.asarray(req.masked_embeddings)                # [B, T, H] masked
        for t in range(x.shape[1]):                          # store masked KV
            self._masked_kv.append(x[:, t, :])
        masked_logits = self._logits(x[:, -1, :])
        wt = self._coarse_timing("prefill", time.perf_counter() - t0) \
            if timing else None
        return MaskedPrefillResponse(
            session_id=req.session_id, masked_logits=masked_logits,
            kv_cache_len=len(self._masked_kv), worker_timing=wt)

    def decode(self, req: MaskedDecodeRequest) -> MaskedDecodeResponse:
        timing = bool(getattr(self, "collect_worker_timing", False))
        t0 = time.perf_counter() if timing else None
        x = np.asarray(req.masked_embedding)                 # [B, 1, H] masked
        self._masked_kv.append(x[:, -1, :])
        masked_logits = self._logits(x[:, -1, :])
        wt = self._coarse_timing("decode", time.perf_counter() - t0) \
            if timing else None
        return MaskedDecodeResponse(
            session_id=req.session_id, masked_logits=masked_logits,
            kv_cache_len=len(self._masked_kv), worker_timing=wt)


class Qwen7BGpuBackend(GpuBackend):
    """Masked Qwen worker over the protocol, backed by ``MaskedQwenSession``.

    Runs the validated Stage 8.4 masked decoder on the untrusted GPU. It consumes
    only the masked hidden ``h_tilde`` (``MaskedPrefill/Decode`` payloads) + public
    metadata and returns **masked logits** ``logits_tilde``; it never sees the raw
    prompt, input ids, plaintext hidden, recovered logits, or sampled tokens. The
    decoder / attention / MLP / KV cache / LM head all run here, **never in a TEE**
    (``tee_used = False``).

    torch is imported lazily. The real Qwen2.5-7B path needs CUDA + a local
    checkpoint (``model_path``); for tests a tiny model can be injected directly
    via ``model`` / ``model_config``. Masked KV cache is kept on the worker; the
    boundary keeps the masks + does embedding/recovery/sampling."""

    name = "qwen7b"
    tee_used = False

    def __init__(self, model_path: str | None = None,
                 device: str = "cuda", dtype: str = "bfloat16",
                 seq_len: int = 128, num_layers: int = 28,
                 model: Any = None, model_config: Any = None,
                 folded_weight_device: str | None = None,
                 mlp_down_chunk_size: int = 512, seed: int = 2035) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.seq_len = int(seq_len)
        self.num_layers = int(num_layers)
        self.folded_weight_device = folded_weight_device or device
        self.mlp_down_chunk_size = int(mlp_down_chunk_size)
        self.seed = int(seed)
        self._session_id: str | None = None
        self._public: dict[str, Any] = {}
        self._model = model
        self._model_config = model_config
        self._sess: Any = None
        self._kv: Any = None
        self.peak_gpu_memory_mb: float | None = None

    # -- lazy model load + session build (untrusted compute) ------------------
    def _ensure_session(self):
        if self._sess is not None:
            return self._sess
        from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession
        from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig
        if self._model is None:
            if not self.model_path:
                raise RuntimeError(
                    "qwen7b worker needs a local checkpoint (model_path) or an "
                    "injected model; none provided")
            from transformers import AutoModelForCausalLM  # lazy
            import torch
            dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                  "float32": torch.float32}.get(self.dtype, torch.bfloat16)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=dt, device_map=self.device,
                trust_remote_code=True, local_files_only=True).eval()
            self._model_config = self._model.config
        cfg = MemoryOptimizedConfig(
            num_layers=self.num_layers, batch_size=1, seq_len=self.seq_len,
            max_new_tokens=max(1, int(self._public.get("max_new_tokens", 64))),
            device=self.device, dtype=self.dtype, folding_dtype="float32",
            folded_weight_device=self.folded_weight_device,
            mlp_down_chunk_size=self.mlp_down_chunk_size, seed=self.seed)
        self._sess = MaskedQwenSession(self._model, self._model_config, cfg)
        return self._sess

    def init(self, req: BoundaryInitRequest) -> BoundaryInitResponse:
        self._session_id = req.session_id
        self._public = dict(req.public_metadata)
        self._kv = None
        return BoundaryInitResponse(
            session_id=req.session_id, ok=True, gpu_backend=self.name,
            tee_used_on_gpu=False,
            notes="qwen7b masked decoder runs on the untrusted GPU; not a TEE")

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "tee_used": self.tee_used,
                "model_path": self.model_path, "device": self.device,
                "dtype": self.dtype, "seq_len": self.seq_len,
                "num_layers": self.num_layers,
                "peak_gpu_memory_mb": self.peak_gpu_memory_mb}

    def _to_torch(self, arr):
        import torch
        import numpy as np
        t = arr if isinstance(arr, torch.Tensor) else torch.as_tensor(
            np.asarray(arr))
        return t

    def _masked_logits_out(self, logits_tilde):
        import numpy as np
        return np.asarray(logits_tilde.detach().to("cpu").float().numpy())

    def _track_peak(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.peak_gpu_memory_mb = float(
                    torch.cuda.max_memory_allocated() / (1024 ** 2))
        except Exception:
            pass

    def _cuda_sync(self) -> None:
        """Synchronise CUDA so a wall-clock forward time is honest (not just the
        async kernel-launch time). No-op off CUDA. Only called when timing."""
        try:
            import torch
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
        except Exception:                                    # noqa: BLE001
            pass

    def _coarse_timing(self, phase, forward_s):
        from pllo.protocol.worker_timing import coarse_forward_metadata
        # forward_s is measured with a synchronize bracket (see prefill/decode),
        # so the TOTAL is accurate even though there is no per-substage split.
        on_cuda = str(self.device).startswith("cuda")
        return coarse_forward_metadata(
            phase=phase, backend_name=self.name, device=self.device,
            dtype=self.dtype, forward_s=forward_s, num_layers=self.num_layers,
            timing_method=("cuda_synchronize" if on_cuda else "wall_clock"),
            is_cuda_synchronized=on_cuda)

    def prefill(self, req: MaskedPrefillRequest) -> MaskedPrefillResponse:
        timing = bool(getattr(self, "collect_worker_timing", False))
        sess = self._ensure_session()
        h_tilde = self._to_torch(req.masked_embeddings).to(
            sess.compute_device, sess.fdtype)
        if timing:
            self._cuda_sync()
        t0 = time.perf_counter() if timing else None
        out = sess.worker_prefill(h_tilde)             # masked logits [B,T,V]
        if timing:
            self._cuda_sync()
        fwd = (time.perf_counter() - t0) if timing else None
        self._kv = out["kv"]
        self._track_peak()
        last = out["logits_tilde"][:, -1, :]           # masked next-token logits
        return MaskedPrefillResponse(
            session_id=req.session_id,
            masked_logits=self._masked_logits_out(last),
            kv_cache_len=int(h_tilde.shape[1]),
            worker_timing=self._coarse_timing("prefill", fwd) if timing else None)

    def decode(self, req: MaskedDecodeRequest) -> MaskedDecodeResponse:
        timing = bool(getattr(self, "collect_worker_timing", False))
        sess = self._ensure_session()
        x_tilde = self._to_torch(req.masked_embedding).to(
            sess.compute_device, sess.fdtype)
        if timing:
            self._cuda_sync()
        t0 = time.perf_counter() if timing else None
        out = sess.worker_decode(x_tilde, self._kv, int(req.position))
        if timing:
            self._cuda_sync()
        fwd = (time.perf_counter() - t0) if timing else None
        self._kv = out["kv"]
        self._track_peak()
        last = out["logits_tilde"][:, -1, :]
        return MaskedDecodeResponse(
            session_id=req.session_id,
            masked_logits=self._masked_logits_out(last),
            kv_cache_len=int(req.position) + 1,
            worker_timing=self._coarse_timing("decode", fwd) if timing else None)


class Qwen7BFoldedPackageGpuBackend(GpuBackend):
    """Untrusted worker that loads PRE-FOLDED weights from a local package.

    This is the strict cross-machine deployment backend: the trusted setup phase
    (``scripts/build_qwen7b_folded_package.py``) wrote folded operators
    ``W_tilde = N_in^{-1} W N_out`` to disk; this worker loads them locally and
    would compute over masked runtime tensors **without ever holding the masks**
    (``worker_has_mask_secrets = False``). The base model ``W`` is public, so the
    folded weights are not chiefly hiding ``W`` -- they let the GPU operate on
    masked activations without learning ``N_in``/``N_out``. ``tee_used = False``.

    ``init`` loads + verifies the package manifest (a fast load/attestation
    probe). The full masked prefill/decode over package shards is a documented
    TODO -- the tiny end-to-end equivalence is proven in
    ``tests/test_folded_package_tiny.py``; wiring 28-layer shard-streamed decode
    is the remaining work."""

    name = "qwen7b_folded_package"
    tee_used = False

    def __init__(self, folded_package_path: str | None = None,
                 device: str = "cuda", dtype: str = "bfloat16",
                 verify_on_init: bool = True,
                 folded_lora_package_path: str | None = None,
                 nonlinear_backend: str = "current",
                 nonlinear_lift_k: int = 2, nonlinear_seed: int = 2035,
                 resident_folded_weights: bool = False,
                 native_logits_wire: bool = False,
                 fold_dtype_override: str | None = None,
                 **_ignored: Any) -> None:
        self.folded_package_path = folded_package_path
        self.folded_lora_package_path = folded_lora_package_path
        self.device = device
        self.dtype = dtype
        self.verify_on_init = bool(verify_on_init)
        # Optional precision override for the folded compute dtype. The exec
        # metadata's ``fold_dtype`` (from the boundary artifact meta) normally
        # drives ``self._fdtype`` -- the dtype the incoming masked embeddings,
        # RoPE caches, resident folded weights and head matmuls run in. When the
        # package shards are stored at higher precision than the meta label (e.g.
        # F32 shards under a ``bfloat16`` meta), forcing this to ``float32`` runs
        # the whole folded forward in fp32 with NO design or security change
        # (same masking, same single TEE crossing) -- it only raises numerical
        # precision, closing bf16 near-tie argmax flips. Default None -> honour
        # the meta fold_dtype (unchanged behaviour).
        self._fold_dtype_override = (str(fold_dtype_override)
                                     if fold_dtype_override else None)
        # Send masked logits in their native bf16 compute dtype (half the wire
        # bytes) instead of upcasting to fp32; bit-identical after the boundary's
        # bf16 -> fp32 upcast for recovery. Default OFF (historical fp32 wire).
        self.native_logits_wire = bool(native_logits_wire)
        # Weight-resident decode (default OFF -> historical per-step load path).
        # When on, all folded layers + head are loaded/folded/moved-to-device ONCE
        # (lazily, on first forward) and reused across decode steps: no per-token
        # shard reload, no build_folded_layer_dict, no CPU->GPU copy. The cache
        # holds only PUBLIC folded operators the worker already reads -- no mask
        # secrets, no raw LoRA. Falls back to the per-step path on OOM.
        self.resident_folded_weights = bool(resident_folded_weights)
        self._resident_layers: Any = None
        self._resident_head: Any = None
        self.resident_weight_init_latency_s: float | None = None
        self.resident_weight_memory_gb: float | None = None
        self.resident_cache_num_layers: int | None = None
        self.resident_cache_device: str | None = None
        self.resident_cache_dtype: str | None = None
        self.resident_cache_oom: bool = False
        self.resident_cache_fallback_used: bool = False
        # per-decode weight-movement counters (so the END-TO-END report can show
        # the resident win without the microbench's monkeypatch). These mirror the
        # exact branch taken: resident -> 0 per step; per-step path -> one shard
        # load + one folded_layer_dict build (= one H2D copy) per executed layer.
        self._decode_steps_run: int = 0
        self._decode_shard_loads: int = 0
        self._decode_dict_builds: int = 0
        # Selected nonlinear DESIGN -- the worker GENUINELY executes it: design B
        # (trusted_shortcut/amulet_migrated) lifts the MLP activation onto this
        # untrusted accelerator and migrates softmax/RMSNorm with a trusted
        # reduction shortcut. The runner + counters are built lazily (torch).
        from pllo.experiments.nonlinear_designs import (  # stdlib, no torch
            normalize_nonlinear_backend, op_backend_for_design)
        self.nonlinear_backend = normalize_nonlinear_backend(
            nonlinear_backend or "current")
        self.nonlinear_op_backend = op_backend_for_design(self.nonlinear_backend)
        self.nonlinear_lift_k = int(nonlinear_lift_k)
        self.nonlinear_seed = int(nonlinear_seed)
        self._runner: Any = None
        self._session_id: str | None = None
        # folded-LoRA state (optional; the no-LoRA path leaves these defaults)
        self.lora_enabled = bool(folded_lora_package_path)
        self.folded_lora_loaded = False
        self.folded_lora_valid: bool | None = None
        self.lora_rank: int | None = None
        self.lora_alpha: float | None = None
        self.lora_target_modules: Any = None
        self.lora_adapter_hash: str | None = None
        self.worker_has_raw_lora = False
        self.folded_package_loaded = False
        # A_rightmul compatible-mask binding read back from the package manifest
        # (set by a paper-facing A_rightmul build). The worker refuses to execute
        # A_rightmul unless the manifest certifies a verified compatible family.
        self.compatible_mask_family: str | None = None
        self.compatible_mask_audit: dict[str, Any] = {}
        self.compatible_masks_verified: bool | None = None
        self.folded_package_size_gb: float | None = None
        self.manifest_hash: str | None = None
        self.num_shards: int | None = None
        self.package_valid: bool | None = None
        self.worker_has_mask_secrets = False
        self._kv: Any = None
        self._exec_layers: int | None = None
        # public exec context (rebuilt from init public_metadata; NO mask secrets)
        self._public: dict[str, Any] = {}
        self._exec_ctx: Any = None        # (cfg0, cos, sin)
        self._eps: float | None = None
        self._num_layers: int | None = None
        self._fdtype: Any = None
        self.peak_gpu_memory_mb: float | None = None

    def init(self, req: BoundaryInitRequest) -> BoundaryInitResponse:
        from pllo.deployment import (
            compute_manifest_hash,
            load_manifest,
            package_size_gb,
            verify_package,
        )
        self._session_id = req.session_id
        # Public metadata only (model dims + RoPE config); the worker rebuilds its
        # per-layer config + RoPE caches from it and NEVER receives mask secrets.
        self._public = dict(req.public_metadata or {})
        self._exec_ctx = None
        self._kv = None
        self._decode_steps_run = 0
        self._decode_shard_loads = 0
        self._decode_dict_builds = 0
        if not self.folded_package_path:
            raise RuntimeError("qwen7b_folded_package worker needs "
                               "folded_package_path")
        manifest = load_manifest(self.folded_package_path)
        self.manifest_hash = compute_manifest_hash(manifest)
        self.num_shards = manifest.num_shards
        self.folded_package_size_gb = round(
            package_size_gb(self.folded_package_path), 6)
        if self.verify_on_init:
            rep = verify_package(self.folded_package_path)
            self.package_valid = bool(rep["package_valid"])
            if not self.package_valid:
                raise RuntimeError(f"folded package failed verification: {rep}")
        # the package, by construction + verification, carries no mask secrets
        self.worker_has_mask_secrets = bool(manifest.contains_mask_secrets)
        self.folded_package_loaded = True

        # A_rightmul: read the compatible-mask binding from the manifest (the
        # build verified the REAL masks are signed-permutation / orthogonal QK /
        # shared SwiGLU permutation). The worker NEVER sees the masks; it trusts
        # the manifest-hash-bound audit. Refuse A_rightmul without it.
        self.compatible_mask_family = manifest.compatible_mask_family
        self.compatible_mask_audit = dict(manifest.compatible_mask_audit or {})
        self.compatible_masks_verified = (
            bool(self.compatible_mask_audit.get("compatible_masks_verified"))
            if self.compatible_mask_audit else None)
        if self.nonlinear_backend == "A_rightmul" and \
                self.compatible_masks_verified is not True:
            raise RuntimeError(
                "A_rightmul worker refused: the folded package manifest does not "
                "certify a verified compatible mask family "
                "(compatible_masks_verified != True). Rebuild with "
                "`build_qwen7b_folded_package.py --paper-facing --nonlinear-backend "
                "A_rightmul` so the masks are verified + bound into the manifest.")

        # optional private folded-LoRA package (no raw A/B, no masks)
        if self.folded_lora_package_path:
            from pllo.deployment.lora_folded_package import (
                load_lora_meta,
                verify_lora_folded_package,
            )
            lrep = verify_lora_folded_package(
                self.folded_lora_package_path,
                base_manifest_hash=self.manifest_hash if self.verify_on_init
                else None)
            self.folded_lora_valid = bool(lrep["lora_package_valid"])
            if self.verify_on_init and not self.folded_lora_valid:
                raise RuntimeError("folded-LoRA package failed verification: "
                                   "%s" % lrep)
            lmeta = load_lora_meta(self.folded_lora_package_path)
            self.lora_rank = lmeta.get("rank")
            self.lora_alpha = lmeta.get("alpha")
            self.lora_target_modules = lmeta.get("target_modules")
            self.lora_adapter_hash = lmeta.get("adapter_hash")
            self.worker_has_raw_lora = bool(lmeta.get("contains_raw_lora", False))
            self.lora_enabled = True
            self.folded_lora_loaded = True

        notes = json.dumps({
            "folded_package_loaded": True,
            "nonlinear_backend": self.nonlinear_backend,
            "nonlinear_op_backend": self.nonlinear_op_backend,
            "folded_package_valid": bool(self.package_valid)
            if self.package_valid is not None else None,
            "folded_package_path": self.folded_package_path,
            "folded_package_size_gb": self.folded_package_size_gb,
            "manifest_hash": self.manifest_hash, "num_shards": self.num_shards,
            "worker_has_mask_secrets": self.worker_has_mask_secrets,
            "lora_enabled": self.lora_enabled,
            "folded_lora_loaded": self.folded_lora_loaded,
            "folded_lora_valid": self.folded_lora_valid,
            "lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha,
            "lora_target_modules": self.lora_target_modules,
            "lora_adapter_hash": self.lora_adapter_hash,
            "worker_has_raw_lora": self.worker_has_raw_lora,
            "resident_folded_weights": self.resident_folded_weights,
            "tee_used_on_gpu": False})
        return BoundaryInitResponse(
            session_id=req.session_id, ok=True, gpu_backend=self.name,
            tee_used_on_gpu=False, notes=notes)

    def linear_boundary_pad_status(self) -> dict[str, Any]:
        """Linear-boundary additive-pad audit, read back from the LOADED package's
        shard tensor names (a module counts only if BOTH its xpad+cpad tensors are
        present). Reflects what the worker will actually execute; no secrets."""
        from pllo.deployment.linear_boundary_pad import (
            default_linear_boundary_pad_report_fields,
            layer_pad_coverage, linear_boundary_pad_report_fields)
        if not self.folded_package_loaded or not self.folded_package_path:
            return default_linear_boundary_pad_report_fields()
        from pllo.deployment import load_manifest
        layer_names: list[str] = []
        head_names: list[str] = []
        for sh in load_manifest(self.folded_package_path).shard_index:
            tn = sh.get("tensors", [])
            if str(sh.get("path", "")).startswith("head"):
                head_names += tn
            else:
                layer_names += tn
        cov = layer_pad_coverage(layer_names, head_names)
        return linear_boundary_pad_report_fields(
            enabled=any(cov.values()), coverage=cov)

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "tee_used": self.tee_used,
                "nonlinear_backend": self.nonlinear_backend,
                "nonlinear_op_backend": self.nonlinear_op_backend,
                "nonlinear_execution_evidence": self.nonlinear_execution_evidence(),
                "compatible_mask_family": self.compatible_mask_family,
                "compatible_masks_verified": self.compatible_masks_verified,
                # P1: surface the FOUR formal compatible-mask sub-conditions so the
                # paper-facing gate can prove each (signed-perm residual /
                # score-preserving Q,K / shared SwiGLU channel perm / dense
                # rejected) -- not only the rolled-up compatible_masks_verified.
                # The audit is loaded verbatim from the package manifest (built by
                # session.verify_compatible_masks()); the worker holds no masks.
                "compatible_mask_audit": dict(self.compatible_mask_audit or {}),
                "folded_package_path": self.folded_package_path,
                "folded_package_loaded": self.folded_package_loaded,
                "folded_package_size_gb": self.folded_package_size_gb,
                "manifest_hash": self.manifest_hash, "num_shards": self.num_shards,
                "package_valid": self.package_valid,
                "worker_has_mask_secrets": self.worker_has_mask_secrets,
                "lora_enabled": self.lora_enabled,
                "folded_lora_loaded": self.folded_lora_loaded,
                "folded_lora_valid": self.folded_lora_valid,
                "folded_lora_package_path": self.folded_lora_package_path,
                "lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha,
                "lora_target_modules": self.lora_target_modules,
                "worker_has_raw_lora": self.worker_has_raw_lora,
                "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
                **self.resident_status(),
                **self.linear_boundary_pad_status(),
                "tee_used_on_gpu": False}

    def _ensure_exec_context(self):
        """Rebuild the PUBLIC per-layer config + RoPE caches from the init
        metadata so the folded shards can be executed remotely. None of these are
        mask secrets: the per-layer block config (head counts, head_dim, RoPE
        theta, rms_norm_eps, biases, mask_family) and the deterministic RoPE
        cos/sin caches are public artifacts -- the masks stay on the boundary."""
        if self._exec_ctx is not None:
            return self._exec_ctx
        meta = self._public or {}
        required = ("hidden_size", "intermediate_size", "num_heads",
                    "num_key_value_heads", "head_dim", "rope_theta",
                    "rms_norm_eps", "num_layers", "rope_max_pos")
        missing = [k for k in required if meta.get(k) is None]
        if missing:
            raise RuntimeError(
                "qwen7b_folded_package remote exec needs public model/RoPE "
                "metadata in the init request public_metadata; missing %s "
                "(required: %s). No mask secrets are ever required."
                % (missing, list(required)))
        import torch
        from pllo.hf_wrappers.llama_qwen_single_block import HFSingleBlockConfig
        from pllo.ops.rope import build_rope_cache
        # ``fold_dtype_override`` (opt-in) wins over the meta label so a package
        # whose shards are stored above the meta precision can run the folded
        # forward at full precision (purely numerical; no design/security change).
        _fold_dtype_name = self._fold_dtype_override or str(
            meta.get("fold_dtype", "float32"))
        fdtype = {"float32": torch.float32, "float64": torch.float64,
                  "bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            _fold_dtype_name, torch.float32)
        cfg0 = HFSingleBlockConfig(
            model_type=str(meta.get("model_type", "qwen2")),
            hidden_size=int(meta["hidden_size"]),
            intermediate_size=int(meta["intermediate_size"]),
            num_heads=int(meta["num_heads"]),
            num_key_value_heads=int(meta["num_key_value_heads"]),
            head_dim=int(meta["head_dim"]),
            rope_theta=float(meta["rope_theta"]),
            rms_norm_eps=float(meta["rms_norm_eps"]),
            attention_bias=bool(meta.get("attention_bias", False)),
            mlp_bias=bool(meta.get("mlp_bias", False)),
            mask_family=str(meta.get("mask_family", "pairwise_complex_scaling")),
            dtype=fdtype, device=str(self.device))
        cos, sin = build_rope_cache(int(meta["rope_max_pos"]), cfg0.head_dim,
                                    cfg0.rope_theta, fdtype, self.device)
        self._eps = float(meta["rms_norm_eps"])
        self._num_layers = int(meta["num_layers"])
        self._fdtype = fdtype
        self._exec_ctx = (cfg0, cos, sin)
        return self._exec_ctx

    def _track_peak(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                self.peak_gpu_memory_mb = float(
                    torch.cuda.max_memory_allocated() / (1024 ** 2))
        except Exception:                                    # noqa: BLE001
            pass

    def _cuda_sync(self) -> None:
        """Synchronise CUDA so the per-region wall times reflect real device
        compute (not async kernel launch). No-op off CUDA; only used when timing."""
        try:
            import torch
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
        except Exception:                                    # noqa: BLE001
            pass

    def _make_timer(self):
        """A ``WorkerTimer`` iff the client opted into worker timing, else None
        (so the folded forward runs the historical path with no timing overhead).

        Prefers CUDA events on a CUDA device (accurate device time, one final
        synchronize), falls back to wall-clock on CPU (synchronous -> accurate)."""
        if not bool(getattr(self, "collect_worker_timing", False)):
            return None
        from pllo.protocol.worker_timing import WorkerTimer
        method = "wall_clock"
        if str(self.device).startswith("cuda"):
            try:
                import torch
                method = ("cuda_event" if torch.cuda.is_available()
                          else "cuda_synchronize")
            except Exception:                                # noqa: BLE001
                method = "cuda_synchronize"
        # optional explicit override (profiler tooling); never weakens timing.
        override = getattr(self, "_timing_method_override", None)
        if override:
            method = override
        return WorkerTimer(enabled=True, method=method, sync=self._cuda_sync)

    def _ensure_runner(self):
        """Lazily build the nonlinear runner that EXECUTES the selected design
        over the folded layers (design B lifts the activation onto this GPU)."""
        if self._runner is None:
            from pllo.deployment.folded_nonlinear import (
                make_folded_nonlinear_runner)
            audit = dict(self.compatible_mask_audit or {})
            if self.compatible_mask_family:
                audit.setdefault("compatible_mask_family",
                                 self.compatible_mask_family)
            self._runner = make_folded_nonlinear_runner(
                self.nonlinear_backend, lift_k=self.nonlinear_lift_k,
                seed=self.nonlinear_seed,
                compatible_masks_verified=self.compatible_masks_verified,
                compatible_mask_audit=audit)
        return self._runner

    def _ensure_resident(self, device: Any, fdtype: Any, num_layers: int) -> bool:
        """Build the GPU-resident folded-layer cache ONCE (lazy, on first forward).

        Returns True if the resident cache is available for use. On OOM it clears
        any partial cache, records the failure, and returns False so the caller
        falls back to the per-step load path (correctness preserved). Caches only
        PUBLIC folded operators -> no mask secrets, no raw LoRA."""
        if not self.resident_folded_weights:
            return False
        if self._resident_layers is not None:
            return True
        if self.resident_cache_fallback_used:        # already tried + failed
            return False
        from pllo.deployment.folded_worker import (
            build_resident_folded_layers, folded_layers_nbytes,
            load_resident_head_dict)
        t0 = time.perf_counter()
        try:
            layers = build_resident_folded_layers(
                self.folded_package_path, int(num_layers), device=device,
                dtype=fdtype, lora_package_dir=self.folded_lora_package_path)
            # full head dict so the optional Linear-boundary head pad survives
            head = load_resident_head_dict(self.folded_package_path,
                                           device=device, dtype=fdtype)
        except RuntimeError as exc:                   # CUDA OOM or similar
            self._resident_layers = None
            self._resident_head = None
            self.resident_cache_oom = ("out of memory" in str(exc).lower()
                                       or "oom" in str(exc).lower())
            self.resident_cache_fallback_used = True
            return False
        self._resident_layers = layers
        self._resident_head = head
        self.resident_weight_init_latency_s = round(time.perf_counter() - t0, 9)
        self.resident_weight_memory_gb = round(
            folded_layers_nbytes(layers, head) / (1024 ** 3), 6)
        self.resident_cache_num_layers = len(layers)
        self.resident_cache_device = str(device)
        self.resident_cache_dtype = str(fdtype)
        return True

    def resident_status(self) -> dict[str, Any]:
        """Public, non-secret resident-cache status + per-decode weight-movement
        counters for reports (measured from the actual decode path)."""
        steps = self._decode_steps_run
        loads_ps = (round(self._decode_shard_loads / steps, 3)
                    if steps else None)
        builds_ps = (round(self._decode_dict_builds / steps, 3)
                     if steps else None)
        return {
            "resident_folded_weights": bool(self.resident_folded_weights),
            "resident_weight_init_latency_s": self.resident_weight_init_latency_s,
            "resident_weight_memory_gb": self.resident_weight_memory_gb,
            "resident_cache_num_layers": self.resident_cache_num_layers,
            "resident_cache_device": self.resident_cache_device,
            "resident_cache_dtype": self.resident_cache_dtype,
            "resident_cache_oom": bool(self.resident_cache_oom),
            "resident_cache_fallback_used": bool(self.resident_cache_fallback_used),
            "resident_cache_active": self._resident_layers is not None,
            # per-decode weight-movement counters (the resident win, end-to-end)
            "weight_reloaded_each_step": (bool(loads_ps and loads_ps >= 1)
                                          if loads_ps is not None else None),
            "weight_shard_loads_per_decode_step": loads_ps,
            "folded_layer_dict_builds_per_decode_step": builds_ps,
            "cpu_to_gpu_weight_copies_per_decode_step": builds_ps,
        }

    def _tag_resident(self, wt: dict[str, Any] | None) -> None:
        """Stamp the (public, non-secret) resident-cache flags onto a worker-timing
        dict so a trusted client / ifeval report can see resident mode without any
        secret crossing. These are bools/ints/strings only."""
        if not isinstance(wt, dict):
            return
        wt["resident_folded_weights"] = bool(self.resident_folded_weights)
        wt["resident_cache_active"] = self._resident_layers is not None
        wt["resident_cache_fallback_used"] = bool(self.resident_cache_fallback_used)
        wt["resident_cache_oom"] = bool(self.resident_cache_oom)

    def nonlinear_execution_evidence(self) -> dict[str, Any]:
        """Measured runtime evidence of the nonlinear design actually executed
        over the folded layers (empty until a prefill/decode/head has run). A
        paper-facing report stamps this AFTER ``nonlinear_design_report_fields``
        so a wired ``trusted_shortcut`` run carries genuine lift counters."""
        if self._runner is None:
            return {}
        return self._runner.execution_evidence()

    def _masked_last_logits(self, h_tilde: Any, timer: Any = None):
        """Apply the folded head, return last-position MASKED logits.

        Default: fp32 numpy (historical wire). When ``native_logits_wire`` is set
        AND the logits are bf16, return the bf16 tensor directly so the wire moves
        HALF the bytes; the trusted boundary upcasts bf16 -> fp32 for recovery, so
        the recovered logits are BIT-IDENTICAL to the fp32-upcast payload."""
        import numpy as np
        import torch
        logits_tilde = self.run_head(h_tilde, float(self._eps),
                                     timer=timer)              # [B, T, V]
        last = logits_tilde[:, -1, :].detach().to("cpu")
        if getattr(self, "native_logits_wire", False) \
                and last.dtype == torch.bfloat16:
            return last.contiguous()                          # wire -> "bfloat16"
        return np.asarray(last.float().numpy())

    def run_single_layer_prefill(self, x_tilde: Any, layer_index: int,
                                 config: Any, cos: Any, sin: Any,
                                 eps: float) -> dict[str, Any]:
        """Load ONE folded layer from the package and run its masked prefill.

        The worker holds NO masks: the down projection is pre-folded in the
        package and attention/MLP use the folded ``*_tilde`` operators directly.
        This is the incremental step toward full shard-streamed decode; it matches
        the in-process folded path (see tests/test_folded_package_qwen_1layer.py).
        ``config``/``cos``/``sin`` are public artifacts supplied by the boundary."""
        from pllo.deployment.folded_worker import (
            apply_folded_layer_prefill,
            load_folded_layer,
        )
        if not self.folded_package_loaded:
            raise RuntimeError("call init() to load the folded package first")
        layer_tensors = load_folded_layer(self.folded_package_path, layer_index)
        return apply_folded_layer_prefill(x_tilde, layer_tensors, config, cos,
                                          sin, eps, runner=self._ensure_runner())

    def run_prefill(self, h_tilde: Any, num_exec_layers: int, config: Any,
                    cos: Any, sin: Any, eps: float, timer: Any = None
                    ) -> dict[str, Any]:
        """Execute ``num_exec_layers`` folded layers from the package over masked
        ``h_tilde`` (no masks on the worker). Stores the masked KV; returns the
        masked hidden + KV. ``config``/``cos``/``sin`` are public artifacts from
        the boundary. ``timer`` (optional) records the per-layer breakdown.
        Tested in tests/test_folded_package_prefill_exec.py."""
        from pllo.deployment.folded_worker import apply_folded_prefill
        if not self.folded_package_loaded:
            raise RuntimeError("call init() to load the folded package first")
        resident = (self._resident_layers
                    if self._ensure_resident(h_tilde.device, h_tilde.dtype,
                                             int(num_exec_layers)) else None)
        out = apply_folded_prefill(h_tilde, self.folded_package_path,
                                   int(num_exec_layers), config, cos, sin, eps,
                                   lora_package_dir=self.folded_lora_package_path,
                                   runner=self._ensure_runner(), timer=timer,
                                   resident_layers=resident)
        self._kv = out["kv"]
        self._exec_layers = int(num_exec_layers)
        return out

    def run_head(self, h_tilde: Any, eps: float, timer: Any = None) -> Any:
        """Masked logits from the package's folded head (no masks)."""
        from pllo.deployment.folded_worker import apply_folded_head
        if not self.folded_package_loaded:
            raise RuntimeError("call init() to load the folded package first")
        head = self._resident_head if self._ensure_resident(
            h_tilde.device, h_tilde.dtype,
            int(self._num_layers or self._exec_layers or 0)) else None
        return apply_folded_head(h_tilde, self.folded_package_path, eps,
                                 runner=self._ensure_runner(), timer=timer,
                                 folded_head=head)

    def run_decode(self, x_next_tilde: Any, position: int, config: Any,
                   cos: Any, sin: Any, eps: float,
                   num_exec_layers: int | None = None, timer: Any = None
                   ) -> dict[str, Any]:
        """One-token masked decode over the package's folded layers, threading the
        masked KV from the preceding prefill/decode. ``timer`` (optional) records
        the per-layer breakdown."""
        from pllo.deployment.folded_worker import apply_folded_decode
        if getattr(self, "_kv", None) is None:
            raise RuntimeError("run_prefill() must populate the KV cache first")
        k = int(num_exec_layers if num_exec_layers is not None
                else getattr(self, "_exec_layers", len(self._kv)))
        resident = (self._resident_layers
                    if self._ensure_resident(x_next_tilde.device,
                                             x_next_tilde.dtype, k) else None)
        # count exactly what this step's path does: resident reuses the cached
        # layers (0 loads/builds/copies); the per-step path loads + builds (one
        # H2D copy) per executed layer.
        self._decode_steps_run += 1
        if resident is None:
            self._decode_shard_loads += k
            self._decode_dict_builds += k
        out = apply_folded_decode(x_next_tilde, self.folded_package_path,
                                  self._kv, int(position), k, config, cos, sin,
                                  eps,
                                  lora_package_dir=self.folded_lora_package_path,
                                  runner=self._ensure_runner(), timer=timer,
                                  resident_layers=resident)
        self._kv = out["kv"]
        return out

    def prefill(self, req: MaskedPrefillRequest) -> MaskedPrefillResponse:
        """Remote package-backed prefill: stream all folded layers over the masked
        embeddings, apply the folded head, return MASKED last-position logits.

        The worker rebuilds the public per-layer config + RoPE caches from the
        init metadata (``_ensure_exec_context``); it holds NO mask secrets and is
        NOT a TEE. The boundary keeps the masks + recovers/samples."""
        import torch
        import numpy as np
        if not self.folded_package_loaded:
            raise RuntimeError("call init() to load the folded package first")
        cfg0, cos, sin = self._ensure_exec_context()
        h_tilde = torch.as_tensor(np.asarray(req.masked_embeddings)).to(
            self.device, self._fdtype)
        timer = self._make_timer()
        if timer is not None:
            self._cuda_sync()
        t0 = time.perf_counter() if timer is not None else None
        out = self.run_prefill(h_tilde, int(self._num_layers), cfg0, cos, sin,
                               float(self._eps), timer=timer)  # stores masked KV
        masked = self._masked_last_logits(out["y_tilde"], timer=timer)
        if timer is not None:
            self._cuda_sync()
        fwd = (time.perf_counter() - t0) if timer is not None else None
        self._track_peak()
        wt = None
        if timer is not None:
            wt = timer.forward_metadata(
                phase="prefill", backend_name=self.name, device=self.device,
                dtype=self.dtype, forward_s=fwd, num_layers=self._num_layers)
            self._tag_resident(wt)
        return MaskedPrefillResponse(
            session_id=req.session_id, masked_logits=masked,
            kv_cache_len=int(h_tilde.shape[1]), worker_timing=wt)

    def decode(self, req: MaskedDecodeRequest) -> MaskedDecodeResponse:
        """Remote package-backed one-token decode over the folded layers, threading
        the masked KV cache held on the worker. Returns MASKED next-token logits."""
        import torch
        import numpy as np
        cfg0, cos, sin = self._ensure_exec_context()
        x_tilde = torch.as_tensor(np.asarray(req.masked_embedding)).to(
            self.device, self._fdtype)
        timer = self._make_timer()
        if timer is not None:
            self._cuda_sync()
        t0 = time.perf_counter() if timer is not None else None
        out = self.run_decode(x_tilde, int(req.position), cfg0, cos, sin,
                              float(self._eps),
                              num_exec_layers=int(self._num_layers), timer=timer)
        masked = self._masked_last_logits(out["y_tilde"], timer=timer)
        if timer is not None:
            self._cuda_sync()
        fwd = (time.perf_counter() - t0) if timer is not None else None
        self._track_peak()
        wt = None
        if timer is not None:
            wt = timer.forward_metadata(
                phase="decode", backend_name=self.name, device=self.device,
                dtype=self.dtype, forward_s=fwd, num_layers=self._num_layers)
            self._tag_resident(wt)
        return MaskedDecodeResponse(
            session_id=req.session_id, masked_logits=masked,
            kv_cache_len=int(req.position) + 1, worker_timing=wt)


def make_gpu_backend(name: str, **kwargs: Any) -> GpuBackend:
    if name == "mock":
        return MockGpuBackend()
    if name == "qwen7b":
        return Qwen7BGpuBackend(**kwargs)
    if name == "qwen7b_folded_package":
        return Qwen7BFoldedPackageGpuBackend(**kwargs)
    raise ValueError(
        f"unknown gpu backend {name!r}; expected 'mock', 'qwen7b', or "
        f"'qwen7b_folded_package'")


# ---------------------------------------------------------------------------
# Process-isolated worker (the untrusted domain)
# ---------------------------------------------------------------------------


def _gpu_worker_loop(conn: Any, backend_name: str,
                     backend_kwargs: dict[str, Any]) -> None:
    """Untrusted worker process: build a backend, service masked-tensor RPCs."""
    backend = make_gpu_backend(backend_name, **backend_kwargs)
    while True:
        try:
            method, payload = conn.recv()
        except EOFError:
            break
        try:
            if method == "shutdown":
                conn.send(("ok", None))
                break
            if method == "init":
                conn.send(("ok", backend.init(payload)))
            elif method == "prefill":
                conn.send(("ok", backend.prefill(payload)))
            elif method == "decode":
                conn.send(("ok", backend.decode(payload)))
            elif method == "describe":
                conn.send(("ok", backend.describe()))
            else:
                conn.send(("error", f"unknown method {method!r}"))
        except Exception as exc:  # noqa: BLE001 - propagate to parent
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
    conn.close()


class LocalGpuWorker:
    """Drives a GPU backend in a separate (untrusted) ``spawn`` process.

    Optionally takes a ``recorder`` callback ``(direction, method, message)``
    invoked for every object crossing the boundary, so the orchestrator can log
    the exact GPU-inbound / GPU-outbound traffic for the security audit."""

    def __init__(self, backend_name: str = "mock",
                 backend_kwargs: dict[str, Any] | None = None,
                 recorder: Any = None) -> None:
        self.backend_name = backend_name
        self.backend_kwargs = dict(backend_kwargs or {})
        self._recorder = recorder
        self._ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_gpu_worker_loop,
            args=(child_conn, backend_name, self.backend_kwargs), daemon=True)
        self._proc.start()
        child_conn.close()
        self._closed = False

    def _rpc(self, method: str, payload: Any) -> Any:
        if self._closed:
            raise RuntimeError("gpu worker is closed")
        if self._recorder is not None and payload is not None:
            self._recorder("inbound", method, payload)
        self._parent_conn.send((method, payload))
        status, result = self._parent_conn.recv()
        if status == "error":
            raise RuntimeError(f"GPU worker error: {result}")
        if self._recorder is not None and result is not None:
            self._recorder("outbound", method, result)
        return result

    def init(self, req: BoundaryInitRequest) -> BoundaryInitResponse:
        return self._rpc("init", req)

    def prefill(self, req: MaskedPrefillRequest) -> MaskedPrefillResponse:
        return self._rpc("prefill", req)

    def decode(self, req: MaskedDecodeRequest) -> MaskedDecodeResponse:
        return self._rpc("decode", req)

    def describe(self) -> dict[str, Any]:
        return self._rpc("describe", {})

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._parent_conn.send(("shutdown", None))
            self._parent_conn.recv()
        except (EOFError, OSError, BrokenPipeError):
            pass
        finally:
            try:
                self._parent_conn.close()
            except OSError:
                pass
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5)

    def __enter__(self) -> "LocalGpuWorker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
