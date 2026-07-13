# From-scratch assumption status

The assumption is **not supported** for the currently selected victim. The actual backbone is the public pretrained `Qwen/Qwen2.5-0.5B` checkpoint, used as a secret/private-checkpoint proxy after trusted transformation.

This distinction does not invalidate experiments scoped to transformed confidentiality when the attacker lacks plaintext weights. It does invalidate the stronger premise that no public training lineage or public checkpoint exists.

No result from the existing Qwen package may be relabeled as a from-scratch black-box-security result.
