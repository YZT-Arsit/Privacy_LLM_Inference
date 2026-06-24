"""Folded-weight provisioning for strict cross-machine deployment.

The trusted setup phase (TDX or another attested environment) reads the public
frozen base model ``W`` + an internally-generated mask schedule and writes a
**folded weight package** -- folded operators ``W_tilde = N_in^{-1} W N_out`` --
to disk. The untrusted GPU worker (H800) loads the package locally and computes
over masked runtime tensors **without ever holding the mask secrets**
(``N_in``/``N_out``). Folding is a one-time setup/provisioning cost; online
decoding reuses the provisioned folded weights and never refolds or resends the
base model per token.

Security: the GPU may hold folded weights (the base ``W`` is public anyway); it
must NEVER hold mask matrices, vocab inverse masks, raw prompts/input_ids/labels,
raw LoRA A/B, LoRA gradients, optimizer state, or recovered logits. Those are
rejected from every package artifact by name.
"""

from pllo.deployment.folded_package import (
    FORBIDDEN_PACKAGE_SUBSTRINGS,
    FoldedPackageWriter,
    compute_file_sha256,
    fold_linear,
    forbidden_tensor_names,
    list_package_shards,
    load_shard,
    package_size_bytes,
    package_size_gb,
    save_shard,
    verify_package,
)
from pllo.deployment.folded_package_manifest import (
    PACKAGE_FORMAT_VERSION,
    FoldedPackageManifest,
    build_manifest,
    compute_manifest_hash,
    load_manifest,
    validate_manifest,
    write_manifest,
)

__all__ = [
    "FORBIDDEN_PACKAGE_SUBSTRINGS",
    "FoldedPackageWriter",
    "compute_file_sha256",
    "fold_linear",
    "forbidden_tensor_names",
    "list_package_shards",
    "load_shard",
    "package_size_bytes",
    "package_size_gb",
    "save_shard",
    "verify_package",
    "PACKAGE_FORMAT_VERSION",
    "FoldedPackageManifest",
    "build_manifest",
    "compute_manifest_hash",
    "load_manifest",
    "validate_manifest",
    "write_manifest",
]
