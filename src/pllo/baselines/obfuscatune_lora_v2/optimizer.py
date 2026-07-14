"""Explicit AdamW over externally stored transformed-coordinate factors.

This optimizer deliberately does not claim rotation equivalence to canonical
AdamW.  It is the internally consistent optimizer frozen by contract
OBFUSCATUNE_LORA_V2_TRANSFORMED_EXTERNAL_ADAMW_V1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdamWHyperparameters:
    lr: float = 2e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01

    def validate(self) -> None:
        if self.lr < 0 or self.eps <= 0 or self.weight_decay < 0:
            raise ValueError("invalid AdamW scalar hyperparameter")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must be in [0,1)")


class TransformedAdamW:
    def __init__(self, parameters: dict[str, torch.Tensor], hparams: AdamWHyperparameters | None = None):
        if not parameters:
            raise ValueError("parameters must not be empty")
        self.parameters = parameters
        self.hparams = hparams or AdamWHyperparameters()
        self.hparams.validate()
        self.step_index = 0
        self.m = {name: torch.zeros_like(value) for name, value in parameters.items()}
        self.v = {name: torch.zeros_like(value) for name, value in parameters.items()}

    @torch.no_grad()
    def step(self, gradients: dict[str, torch.Tensor]) -> None:
        if set(gradients) != set(self.parameters):
            raise ValueError("gradient names must exactly match parameter names")
        self.step_index += 1
        h = self.hparams
        correction1 = 1.0 - h.beta1**self.step_index
        correction2 = 1.0 - h.beta2**self.step_index
        for name, parameter in self.parameters.items():
            gradient = gradients[name]
            if gradient.shape != parameter.shape or gradient.dtype != parameter.dtype:
                raise ValueError(f"gradient metadata mismatch for {name}")
            if not torch.isfinite(gradient).all():
                raise ValueError(f"non-finite gradient for {name}")
            self.m[name].mul_(h.beta1).add_(gradient, alpha=1.0 - h.beta1)
            self.v[name].mul_(h.beta2).addcmul_(gradient, gradient, value=1.0 - h.beta2)
            m_hat = self.m[name] / correction1
            v_hat = self.v[name] / correction2
            parameter.mul_(1.0 - h.lr * h.weight_decay)
            parameter.addcdiv_(m_hat, v_hat.sqrt().add_(h.eps), value=-h.lr)

    def state_dict(self) -> dict:
        return {
            "contract_id": "OBFUSCATUNE_LORA_V2_TRANSFORMED_EXTERNAL_ADAMW_V1",
            "step": self.step_index,
            "hparams": vars(self.hparams).copy(),
            "parameters": {k: v.detach().clone() for k, v in self.parameters.items()},
            "m": {k: v.detach().clone() for k, v in self.m.items()},
            "v": {k: v.detach().clone() for k, v in self.v.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("contract_id") != "OBFUSCATUNE_LORA_V2_TRANSFORMED_EXTERNAL_ADAMW_V1":
            raise ValueError("optimizer contract mismatch")
        if state.get("hparams") != vars(self.hparams):
            raise ValueError("optimizer hyperparameter mismatch")
        names = set(self.parameters)
        if set(state.get("parameters", {})) != names or set(state.get("m", {})) != names or set(state.get("v", {})) != names:
            raise ValueError("optimizer tensor inventory mismatch")
        for name in names:
            for group in ("parameters", "m", "v"):
                if state[group][name].shape != self.parameters[name].shape:
                    raise ValueError(f"optimizer tensor shape mismatch: {group}.{name}")
            self.parameters[name].copy_(state["parameters"][name])
            self.m[name].copy_(state["m"][name])
            self.v[name].copy_(state["v"][name])
        self.step_index = int(state["step"])
