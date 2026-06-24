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

    def prefill(self, req: MaskedPrefillRequest) -> MaskedPrefillResponse:
        x = np.asarray(req.masked_embeddings)                # [B, T, H] masked
        for t in range(x.shape[1]):                          # store masked KV
            self._masked_kv.append(x[:, t, :])
        masked_logits = self._logits(x[:, -1, :])
        return MaskedPrefillResponse(
            session_id=req.session_id, masked_logits=masked_logits,
            kv_cache_len=len(self._masked_kv))

    def decode(self, req: MaskedDecodeRequest) -> MaskedDecodeResponse:
        x = np.asarray(req.masked_embedding)                 # [B, 1, H] masked
        self._masked_kv.append(x[:, -1, :])
        masked_logits = self._logits(x[:, -1, :])
        return MaskedDecodeResponse(
            session_id=req.session_id, masked_logits=masked_logits,
            kv_cache_len=len(self._masked_kv))


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

    def prefill(self, req: MaskedPrefillRequest) -> MaskedPrefillResponse:
        sess = self._ensure_session()
        h_tilde = self._to_torch(req.masked_embeddings).to(
            sess.compute_device, sess.fdtype)
        out = sess.worker_prefill(h_tilde)             # masked logits [B,T,V]
        self._kv = out["kv"]
        self._track_peak()
        last = out["logits_tilde"][:, -1, :]           # masked next-token logits
        return MaskedPrefillResponse(
            session_id=req.session_id,
            masked_logits=self._masked_logits_out(last),
            kv_cache_len=int(h_tilde.shape[1]))

    def decode(self, req: MaskedDecodeRequest) -> MaskedDecodeResponse:
        sess = self._ensure_session()
        x_tilde = self._to_torch(req.masked_embedding).to(
            sess.compute_device, sess.fdtype)
        out = sess.worker_decode(x_tilde, self._kv, int(req.position))
        self._kv = out["kv"]
        self._track_peak()
        last = out["logits_tilde"][:, -1, :]
        return MaskedDecodeResponse(
            session_id=req.session_id,
            masked_logits=self._masked_logits_out(last),
            kv_cache_len=int(req.position) + 1)


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
                 verify_on_init: bool = True, **_ignored: Any) -> None:
        self.folded_package_path = folded_package_path
        self.device = device
        self.dtype = dtype
        self.verify_on_init = bool(verify_on_init)
        self._session_id: str | None = None
        self.folded_package_loaded = False
        self.folded_package_size_gb: float | None = None
        self.manifest_hash: str | None = None
        self.num_shards: int | None = None
        self.package_valid: bool | None = None
        self.worker_has_mask_secrets = False

    def init(self, req: BoundaryInitRequest) -> BoundaryInitResponse:
        from pllo.deployment import (
            compute_manifest_hash,
            load_manifest,
            package_size_gb,
            verify_package,
        )
        self._session_id = req.session_id
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
        notes = json.dumps({
            "folded_package_loaded": True,
            "folded_package_path": self.folded_package_path,
            "folded_package_size_gb": self.folded_package_size_gb,
            "manifest_hash": self.manifest_hash, "num_shards": self.num_shards,
            "worker_has_mask_secrets": self.worker_has_mask_secrets,
            "tee_used_on_gpu": False})
        return BoundaryInitResponse(
            session_id=req.session_id, ok=True, gpu_backend=self.name,
            tee_used_on_gpu=False, notes=notes)

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "tee_used": self.tee_used,
                "folded_package_path": self.folded_package_path,
                "folded_package_loaded": self.folded_package_loaded,
                "folded_package_size_gb": self.folded_package_size_gb,
                "manifest_hash": self.manifest_hash, "num_shards": self.num_shards,
                "package_valid": self.package_valid,
                "worker_has_mask_secrets": self.worker_has_mask_secrets,
                "tee_used_on_gpu": False}

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
                                          sin, eps)

    def prefill(self, req: MaskedPrefillRequest) -> MaskedPrefillResponse:
        raise NotImplementedError(
            "TODO: full masked prefill from folded-package shards. The folded "
            "operators load + verify on init, and single-layer masked prefill "
            "from the package is implemented + tested "
            "(run_single_layer_prefill; tests/test_folded_package_qwen_1layer.py). "
            "Full 28-layer shard-streamed prefill/decode is not yet wired.")

    def decode(self, req: MaskedDecodeRequest) -> MaskedDecodeResponse:
        raise NotImplementedError(
            "TODO: full masked decode from folded-package shards (see prefill).")


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
