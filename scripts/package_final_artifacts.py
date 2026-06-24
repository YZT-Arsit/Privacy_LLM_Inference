"""Package the full no-LoRA + private-LoRA evaluation artifacts into one tarball.

Collects the (optional) outputs of the whole pipeline -- no-LoRA H800 local/remote
decode, no-LoRA TDX-attested decode, E3/E4/E5 reports, E6 folded-LoRA outputs, the
E7 training prototype, the E8 LoRA report, runtime-hash / attestation-evidence
files, artifact-hash files, and the package manifests (base folded package,
embedding artifact, folded LoRA package) -- into a single ``.tar.gz`` with a
``MANIFEST.json`` (per-file sha256 + size + category) and a ``MANIFEST.md``.

Missing optional inputs are handled gracefully: they are recorded under
``missing`` (per category) and never abort the run. stdlib only.

Example::

    python scripts/package_final_artifacts.py \\
        --outputs-dir outputs \\
        --tee-artifacts-dir /root/privacy_llm_tee_artifacts \\
        --base-folded-package-path /root/.../qwen7b_folded_full \\
        --embedding-artifact-path /root/.../qwen7b_boundary_artifact_cuda \\
        --lora-folded-package-path /root/.../qwen7b_lora_folded_synth_r4 \\
        --extra-file /root/artifact_cuda_tdx.sha256 \\
        --output-tar /root/privacy_llm_final_artifacts.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

# category -> glob patterns (relative to --outputs-dir).
OUTPUT_CATEGORIES = {
    "no_lora_local_remote": [
        "qwen7b_folded*decode*.json", "qwen7b_folded*decode*.md",
        "qwen7b_folded_package*probe*.json", "tee_gpu_protocol*.json",
        "tee_gpu_protocol*.md", "e1_nolora_qwen.*", "e2_token_scaling_qwen.*",
    ],
    "no_lora_tdx_attested": [
        "tdx_attested*.json", "tdx_attested*.md", "*attested*.json",
        "*attested*.md",
    ],
    "e3_e4_e5_reports": [
        "e3_*.json", "e3_*.md", "e4_*.json", "e4_*.md", "e5_*.json", "e5_*.md",
        "examples_e3_e5/*",
    ],
    "e6_lora_inference": [
        "qwen7b_lora_folded*.json", "qwen7b_lora_folded*.md",
        "e6_lora*.json", "e6_lora*.md", "validate_lora_effect*.json",
        "tdx_lora_*.json", "run_tdx_lora_lite_decode.sh",
    ],
    "e7_training_prototype": [
        "lora_private_training*.json", "lora_private_training*.md",
    ],
    "e8_lora_report": [
        "e8_lora_final_report.json", "e8_lora_final_report.md",
    ],
    "examples_e6_e8": ["examples_e6_e8/*"],
    "runtime_hash_evidence": [
        "*runtime_hash*", "*runtime_manifest*", "*evidence*.json",
        "evidence_manifest.*",
    ],
    "artifact_hashes": ["*.sha256"],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(outputs_dir, *, tee_artifacts_dir=None, extra_files=(),
            base_package_path=None, embedding_artifact_path=None,
            lora_folded_package_path=None) -> dict:
    """Resolve which files exist; return {included:[...], missing:{cat:[...]}}."""
    outputs = Path(outputs_dir)
    included: list = []           # (arcname, abspath, category)
    seen: set = set()
    missing: dict = {}

    def _add(arcname, path, category):
        rp = Path(path).resolve()
        if rp in seen or not Path(path).is_file():
            return False
        seen.add(rp)
        included.append((arcname, str(path), category))
        return True

    for cat, patterns in OUTPUT_CATEGORIES.items():
        hit = False
        for pat in patterns:
            for p in sorted(outputs.glob(pat)):
                if p.is_file() and _add("outputs/" + p.relative_to(outputs)
                                        .as_posix(), p, cat):
                    hit = True
        if not hit:
            missing.setdefault(cat, []).append("outputs/{%s}" % ",".join(patterns))

    # package manifests (optional)
    man_specs = [
        ("base folded package", base_package_path, ["manifest.json"]),
        ("embedding artifact", embedding_artifact_path,
         ["manifest.json", "meta.json", "artifact_meta.json"]),
        ("folded LoRA package", lora_folded_package_path,
         ["manifest.json", "lora_meta.json"]),
    ]
    for label, root, names in man_specs:
        if not root:
            missing.setdefault("package_manifests", []).append(
                "%s (not provided)" % label)
            continue
        any_hit = False
        for name in names:
            src = Path(root) / name
            arc = "manifests/%s/%s" % (Path(root).name, name)
            if _add(arc, src, "package_manifests"):
                any_hit = True
        if not any_hit:
            missing.setdefault("package_manifests", []).append(
                "%s: none of %s under %s" % (label, names, root))

    # TEE artifacts dir (optional; include all files)
    if tee_artifacts_dir and Path(tee_artifacts_dir).is_dir():
        root = Path(tee_artifacts_dir)
        for p in sorted(root.rglob("*")):
            if p.is_file():
                _add("tee_artifacts/" + p.relative_to(root).as_posix(), p,
                     "tee_artifacts")
    elif tee_artifacts_dir:
        missing.setdefault("tee_artifacts", []).append(
            "%s (dir not found)" % tee_artifacts_dir)

    # explicit extra files (optional)
    for ef in extra_files:
        if Path(ef).is_file():
            _add("extra/" + Path(ef).name, ef, "extra_files")
        else:
            missing.setdefault("extra_files", []).append("%s (not found)" % ef)

    return {"included": included, "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--tee-artifacts-dir", default=None)
    ap.add_argument("--base-folded-package-path", default=None)
    ap.add_argument("--embedding-artifact-path", default=None)
    ap.add_argument("--lora-folded-package-path", default=None)
    ap.add_argument("--extra-file", action="append", default=[])
    ap.add_argument("--output-tar", required=True)
    args = ap.parse_args()

    info = collect(
        args.outputs_dir, tee_artifacts_dir=args.tee_artifacts_dir,
        extra_files=args.extra_file,
        base_package_path=args.base_folded_package_path,
        embedding_artifact_path=args.embedding_artifact_path,
        lora_folded_package_path=args.lora_folded_package_path)

    files_manifest = []
    for arcname, path, category in info["included"]:
        p = Path(path)
        files_manifest.append({
            "arcname": arcname, "category": category,
            "sha256": _sha256(p), "size": p.stat().st_size,
            "source": str(p),
        })
    manifest = {
        "stage": "final_artifact_package",
        "output_tar": str(args.output_tar),
        "num_files": len(files_manifest),
        "categories_present": sorted({f["category"] for f in files_manifest}),
        "files": files_manifest,
        "missing": info["missing"],
    }

    out_tar = Path(args.output_tar)
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    manifest_md = _render_md(manifest)
    with tarfile.open(out_tar, "w:gz") as tar:
        for arcname, path, _cat in info["included"]:
            tar.add(path, arcname=arcname)
        _add_text(tar, "MANIFEST.json",
                  json.dumps(manifest, indent=2, default=str))
        _add_text(tar, "MANIFEST.md", manifest_md)

    print("=== final artifact package ===")
    print("output_tar=%s" % out_tar)
    print("files_included=%d categories=%s"
          % (len(files_manifest), manifest["categories_present"]))
    if info["missing"]:
        print("missing (optional, skipped gracefully):")
        for cat, items in sorted(info["missing"].items()):
            print("  %s: %d" % (cat, len(items)))
    print("\nFINAL ARTIFACTS PACKAGED")
    return 0


def _add_text(tar, arcname, text):
    import io
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _render_md(m: dict) -> str:
    L = ["# Final artifacts package", "",
         "- output_tar: `%s`" % m["output_tar"],
         "- files: %d  categories: %s"
         % (m["num_files"], ", ".join(m["categories_present"])), "",
         "## Included files", "",
         "| arcname | category | size | sha256 |",
         "| --- | --- | --- | --- |"]
    for f in m["files"]:
        L.append("| %s | %s | %d | %s |"
                 % (f["arcname"], f["category"], f["size"], f["sha256"][:16]))
    L += ["", "## Missing (optional, not packaged)", ""]
    if not m["missing"]:
        L.append("- none")
    else:
        for cat, items in sorted(m["missing"].items()):
            for it in items:
                L.append("- **%s**: %s" % (cat, it))
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
