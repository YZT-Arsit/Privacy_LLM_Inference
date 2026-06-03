# Runtime Boundary -- Trusted Controller / Accelerator Backend

Stage 7.5c introduces an explicit *deployable* runtime boundary between
the **trusted controller** (TEE-like) and the **accelerator backend**
(GPU-like). The current artifact ships exactly one accelerator backend,
`LocalCPUBackend`. The protocol logic does **not** depend on which
backend is wired up; a future TEE or GPU deployment only has to provide
a replacement backend object.

**This is not a real TEE deployment.** It is not a real GPU deployment
either. The interfaces are *backend-ready*; the hardware wire-up is
future work.

## 1. Object boundary

```
+-----------------------------------+
|        TrustedController          |   trusted side
|   (sample_mask, sample_pad,       |
|    transform_linear, ...          |
|    recover_output, optimizer_step)|
|                                   |
+----------------+------------------+
                 |  masked tensors only
                 v
+-----------------------------------+
|       AcceleratorBackend          |   untrusted side
|   (linear, matmul, attention,     |
|    softmax, activation,           |
|    rmsnorm_core, layernorm_core,  |
|    append_kv_cache,               |
|    lora_forward, lora_backward,   |
|    measure_runtime,               |
|    collect_transcript_summary)    |
+-----------------------------------+
```

Anything on the trusted side may hold raw secrets (user input, LoRA
adapter, optimizer state, masks, pads, loss closure, sampler). Anything
on the accelerator side only ever sees masked tensors and a sanitised
`RuntimeTranscript`.

## 2. Files

| Path | Role |
|---|---|
| `src/pllo/runtime/interfaces.py` | `AcceleratorBackend` protocol and the `UnsupportedBackendOp` error type. |
| `src/pllo/runtime/local_cpu_backend.py` | The concrete `LocalCPUBackend` -- the only backend shipped in Stage 7.5c. |
| `src/pllo/runtime/trusted_controller.py` | `TrustedController` + `TrustedControllerConfig` -- mask / pad sampling, linear and LoRA boundary transforms, optimiser step, transcript sanitisation. |
| `src/pllo/runtime/transcript.py` | `RuntimeTranscript` -- JSON-safe summary that never carries raw tensors / masks / adapters / gradients. |
| `src/pllo/runtime/backend_registry.py` | `register_backend("tee", ...)` / `register_backend("gpu", ...)` indirection. |
| `src/pllo/runtime/__init__.py` | Public re-exports. |

## 3. Operations still trusted in this stage

The trusted controller still performs:

- mask sampling (`sample_mask`)
- pad sampling (`sample_pad`)
- linear boundary transform (`transform_linear`)
- output recovery (`recover_output`, `recover_logits`)
- LoRA adapter transform / gradient recovery
- optimiser step
- transcript sanitisation (`sanitize_summary`)
- loss closure
- sampler (greedy / top-k / top-p)

The accelerator backend performs:

- dense linear matmul (`linear`, `matmul`)
- attention dot product (`attention_scores`)
- softmax (`softmax`)
- pointwise activations (`activation`)
- RMSNorm / LayerNorm cores (`rmsnorm_core`, `layernorm_core`)
- KV-cache concatenation on the token axis (`append_kv_cache`)
- LoRA forward / backward (`lora_forward`, `lora_backward`)
- runtime measurement (`measure_runtime`)
- transcript reporting (`collect_transcript_summary`)

## 4. What a future TEE backend has to implement

To plug in a real TEE deployment, write a class that satisfies the
`AcceleratorBackend` protocol and register it:

```python
from pllo.runtime import register_backend, AcceleratorBackend

class TEEBackend:
    name = "tee"
    def linear(self, x_tilde, w_tilde, bias_tilde): ...
    def matmul(self, a, b): ...
    def attention_scores(self, q_tilde, k_tilde): ...
    def softmax(self, x, dim=-1): ...
    def activation(self, kind, x_tilde): ...
    def rmsnorm_core(self, x_tilde): ...
    def layernorm_core(self, x_tilde): ...
    def append_kv_cache(self, ck, cv, nk, nv): ...
    def lora_forward(self, x_tilde, w_tilde, a_tilde, b_tilde,
                     bias_tilde, pad_compensation, alpha): ...
    def lora_backward(self, x_tilde, a_tilde, b_tilde, grad_y_tilde,
                      alpha, w_tilde=None, recover_grad_x=False): ...
    def measure_runtime(self, fn, *, num_warmup=1, num_repeats=3): ...
    def collect_transcript_summary(self): ...

register_backend("tee", TEEBackend)
```

Then the trusted controller is constructed with the new backend:

```python
from pllo.runtime import TrustedController, get_backend
ctrl = TrustedController(backend=get_backend("tee"))
```

No change to the protocol logic is required. Likewise for a GPU
backend.

## 5. What this stage does NOT implement

- No real TEE attestation, no sealed storage, no remote attestation.
- No real-hardware wall-time measurement; `LocalCPUBackend.measure_runtime`
  uses `time.perf_counter` only and never calls `time.sleep`.
- No GPU kernel launches; every backend call runs on the local CPU under
  ordinary `torch` matmuls.
- No PEFT / DeepSpeed / vLLM / FlashAttention integration on either side
  of the boundary.
- No raw mask, adapter, gradient, or input ever crosses the boundary;
  the controller redacts any field that should not be published.

## 6. Test coverage

`tests/test_runtime_boundary.py` enforces:

- A fresh `TrustedController` can sample masks and pads.
- `LocalCPUBackend` satisfies the `AcceleratorBackend` runtime check.
- The `RuntimeTranscript` published by `collect_transcript_summary`
  carries `contains_raw_secret = False`.
- `register_backend("local_cpu", ...)` is present in the registry.
- This document exists.
