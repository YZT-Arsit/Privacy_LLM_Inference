# Training-to-generation adapter handoff (PHASE 4)

Implementation: `scripts/native_profile/adapter_handoff.py`; controls: `test_adapter_handoff.py`
(**10/10 pass**, incl. the 7 required negative controls).

`adapter_package/` = `adapter_manifest.json` + `adapter_tensors.safetensors` +
`artifact_hashes.sha256` + `provenance.json`. The manifest binds: adapter_id, base
transformed-package root hash, model config hash, tokenizer hash, LoRA targets, rank, alpha, dtype,
optimization profile, transform-algebra version, feature/rank-basis/vocabulary-mask profiles,
optimizer-state version, final training step, checkpoint sequence, source run_id, creation
timestamp, and per-tensor hashes.

Export policy: **only masked runtime A/B** are written for BOTH profiles — never plaintext
master/m/v, never a reconstructed plaintext adapter. Profile E exports re-folded masked runtime
A/B; Profile N exports the final transformed-coordinate A/B already resident in the runtime.

Generation loader (`load_adapter_for_generation`): verifies the manifest + tensor-blob hashes,
requires exact base-package-hash match, checks model/target/rank/transform-version compatibility,
enforces a minimum optimizer-state version, per-tensor content hashes, and refuses any
plaintext/optimizer-state adapter — loading masked tensors directly, **without rebuilding the base
package**. Negative controls (all fail-closed): wrong package, wrong rank, wrong target map,
modified tensor, modified manifest, wrong transform version, stale adapter version.
