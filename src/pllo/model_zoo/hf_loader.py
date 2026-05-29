"""HuggingFace model loader."""

from __future__ import annotations

import torch

from pllo.model_zoo.base import ExternalModelConfig, torch_dtype_from_string


class HuggingFaceModelLoader:
    """Load HuggingFace causal language models and tokenizers."""

    def _model_path(self, config: ExternalModelConfig) -> str:
        return config.local_dir or config.model_id

    def load_tokenizer(self, config: ExternalModelConfig):
        """Load tokenizer via AutoTokenizer.from_pretrained."""
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install HuggingFace dependencies with pip install -e '.[hf]'") from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_path(config),
                trust_remote_code=config.trust_remote_code,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer for {config.model_id!r}. "
                "Check network access, local cache, or local_dir."
            ) from exc
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self, config: ExternalModelConfig):
        """Load model via AutoModelForCausalLM.from_pretrained."""
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError("Install HuggingFace dependencies with pip install -e '.[hf]'") from exc

        dtype = torch_dtype_from_string(config.dtype, config.device)
        try:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self._model_path(config),
                    dtype=dtype,
                    trust_remote_code=config.trust_remote_code,
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    self._model_path(config),
                    torch_dtype=dtype,
                    trust_remote_code=config.trust_remote_code,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model {config.model_id!r}. "
                "Check network access, local cache, dtype/device support, or local_dir."
            ) from exc

        model.to(torch.device(config.device))
        model.eval()
        return model

    def load(self, config: ExternalModelConfig):
        """Load tokenizer and model."""
        return self.load_tokenizer(config), self.load_model(config)
