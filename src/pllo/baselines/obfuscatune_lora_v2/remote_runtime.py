"""Autograd-capable A10 client for trusted operators hosted by TDX."""

from __future__ import annotations

import torch

from .rpc_protocol import RPCClient
from .runtime import RuntimeCounters, TrustedRuntime


class _RemoteOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, client: RPCClient, operation: str, attributes: dict, *tensors):
        training = any(t.is_floating_point() and t.requires_grad for t in tensors)
        result = client.call("forward", operation=operation, attributes=attributes,
                             tensors=[t.detach().cpu() for t in tensors], training=training)
        ctx.client = client; ctx.call_id = result.get("call_id"); ctx.n_inputs = len(tensors)
        ctx.devices = [t.device for t in tensors]; ctx.dtypes = [t.dtype for t in tensors]
        outputs = tuple(value.to(tensors[0].device) for value in result["outputs"])
        return outputs[0] if len(outputs) == 1 else outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.call_id is None:
            return (None, None, None, *([None] * ctx.n_inputs))
        result = ctx.client.call("backward", call_id=ctx.call_id,
                                 gradients=[None if g is None else g.detach().cpu() for g in grad_outputs])
        values = []
        for gradient, device, dtype in zip(result["input_gradients"], ctx.devices, ctx.dtypes):
            values.append(None if gradient is None else gradient.to(device=device, dtype=dtype))
        return (None, None, None, *values)


def remote_op(client: RPCClient, operation: str, attributes: dict, *tensors):
    return _RemoteOp.apply(client, operation, attributes, *tensors)


class RemoteTrustedRuntime(TrustedRuntime):
    """Same operator interface as the local oracle, backed by real RPC."""

    def __init__(self, client: RPCClient, counters: RuntimeCounters | None = None):
        super().__init__(counters, real_transport=True)
        self.client = client

    def _remote(self, counter: str, op: str, attributes: dict, *tensors):
        before_s, before_r, before_t = self.client.bytes_sent, self.client.bytes_received, self.client.transport_ns
        value = remote_op(self.client, op, attributes, *tensors)
        setattr(self.counters, counter, getattr(self.counters, counter) + 1)
        self.counters.physical_messages += 2
        self.counters.bytes_a10_to_tdx += self.client.bytes_sent - before_s
        self.counters.bytes_tdx_to_a10 += self.client.bytes_received - before_r
        self.counters.transport_ns += self.client.transport_ns - before_t
        return value

    def rmsnorm(self, x, weight, eps):
        return self._remote("trusted_norm_calls", "rmsnorm_inline", {"eps": float(eps)}, x, weight)

    def rmsnorm_key(self, x, *, key: str, eps: float, layer: int | None = None, which: str | None = None):
        attrs = {"key": key, "eps": float(eps), "layer": layer, "which": which}
        return self._remote("trusted_norm_calls", "rmsnorm_key", attrs, x)

    def rope(self, q, k, cos, sin):
        return self._remote("trusted_rope_calls", "rope", {}, q, k, cos, sin)

    def softmax(self, scores, dim=-1):
        return self._remote("trusted_softmax_calls", "softmax", {"dim": int(dim)}, scores)

    def swiglu(self, gate, up):
        return self._remote("trusted_nonlinear_calls", "swiglu", {}, gate, up)

    def residual(self, residual, update):
        return self._remote("trusted_residual_calls", "residual", {}, residual, update)

    def embedding(self, input_ids, weight):
        # The real worker ignores the inline weight; it owns the embedding.
        return self._remote("trusted_embedding_calls", "embedding", {}, input_ids)

    def output(self, hidden, weight):
        return self._remote("trusted_output_calls", "output", {}, hidden)

    def cross_entropy(self, logits, labels, ignore_index=-100):
        return self._remote("trusted_loss_calls", "cross_entropy", {"ignore_index": int(ignore_index)}, logits, labels)

    def output_loss(self, hidden, labels, ignore_index=-100, chunk_tokens=16):
        self.counters.trusted_output_calls += 1
        return self._remote("trusted_loss_calls", "output_loss",
                            {"ignore_index": int(ignore_index), "chunk_tokens": int(chunk_tokens)}, hidden, labels)

    def transform_input(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return self._remote("trusted_residual_calls", "transform_input", {"key": key}, x)

    def restore_output(self, key: str, y: torch.Tensor) -> torch.Tensor:
        return self._remote("trusted_residual_calls", "restore_output", {"key": key}, y)

    def add_bias(self, key: str, y: torch.Tensor) -> torch.Tensor:
        return self._remote("trusted_residual_calls", "add_bias", {"key": key}, y)
