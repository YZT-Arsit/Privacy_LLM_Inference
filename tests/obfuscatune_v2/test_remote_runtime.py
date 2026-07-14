from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import torch

from pllo.baselines.obfuscatune.qwen_config import make_tiny_qwen_config
from pllo.baselines.obfuscatune_lora_v2.remote_model import RemoteAdaptedQwenCausalLM
from pllo.baselines.obfuscatune_lora_v2.remote_runtime import RemoteTrustedRuntime
from pllo.baselines.obfuscatune_lora_v2.rpc_protocol import RPCClient


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); return sock.getsockname()[1]


def test_remote_qwen_forward_backward_and_cache_cleanup(tmp_path):
    from transformers import Qwen2ForCausalLM
    root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "model"; package = tmp_path / "package"; status = tmp_path / "status.json"
    torch.manual_seed(9); Qwen2ForCausalLM(make_tiny_qwen_config(num_hidden_layers=1)).save_pretrained(model_path)
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    subprocess.run([sys.executable, str(root / "scripts/obfuscatune_v2/prepare_transformed_package.py"),
                    "--model", str(model_path), "--output", str(package), "--seed", "1234",
                    "--rank", "8", "--alpha", "16"], check=True, env=env, capture_output=True)
    port = _port(); secret = "37" * 32
    worker = subprocess.Popen([sys.executable, str(root / "scripts/obfuscatune_v2/tdx_worker.py"),
                               "--model", str(model_path), "--seed", "1234", "--host", "127.0.0.1",
                               "--port", str(port), "--secret-hex", secret, "--status", str(status)],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            if status.exists(): break
            if worker.poll() is not None: raise RuntimeError(worker.stderr.read().decode())
            time.sleep(.05)
        client = RPCClient("127.0.0.1", port, secret)
        runtime = RemoteTrustedRuntime(client)
        model = RemoteAdaptedQwenCausalLM(package, runtime, device="cpu")
        ids = torch.tensor([[1, 2, 3, 4]])
        logits, caches, loss = model(ids, labels=ids)
        assert logits.shape == (1, 4, 64) and len(caches) == 1 and torch.isfinite(loss)
        loss.backward(); parameters = model.transformed_parameters()
        assert len(parameters) == 14 and all(value.grad is not None and torch.isfinite(value.grad).all() for value in parameters.values())
        assert client.call("health")["cached_calls"] == 0
        assert runtime.counters.physical_messages > 0 and runtime.counters.bytes_a10_to_tdx > 0
        client.close()
    finally:
        try: worker.wait(timeout=5)
        except subprocess.TimeoutExpired: worker.terminate(); worker.wait(timeout=5)
