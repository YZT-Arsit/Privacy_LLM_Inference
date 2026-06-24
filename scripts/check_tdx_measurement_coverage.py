"""Check the TDX runtime-hash measurement actually covers the boundary code.

Remote attestation only means something if the runtime hash measures every
trusted-side source file the boundary process loads inside the TD. This tool:

1. starts from the boundary entry point(s) (the ``boundary_client`` demo);
2. walks the **first-party** (``pllo`` / repo ``scripts``) import closure via AST,
   pruning modules that are provably NOT part of the boundary runtime -- worker-
   only code (gpu_worker / folded_worker / folded-package writer / private
   ``lora_folded_package``), trusted-setup-only code, and the full ``hf_wrappers``
   model stack (used only in the non-lite full-reference path, never in the TD);
3. compares that closure against the files measured into the runtime hash
   (``attestation.DEFAULT_TRUSTED_BOUNDARY_PATHS``).

It exits non-zero if the boundary imports a trusted-side module that is NOT
measured (a real attestation gap) -- so this guards every future boundary edit,
including the private-LoRA path. The LoRA folding/merging runs on the untrusted
worker, so the LoRA path must add NO new measured boundary file; if it does, this
check tells you to either measure it or justify pruning it.

stdlib only (ast / pathlib).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.protocol.attestation import (  # noqa: E402
    DEFAULT_TRUSTED_BOUNDARY_PATHS,
    _expand_paths,
)

# Boundary runtime entry point(s): what actually runs inside the TD guest.
BOUNDARY_ENTRY = ["scripts/run_tee_gpu_protocol_demo.py"]

# First-party modules that are NOT part of the boundary runtime and are therefore
# intentionally excluded from the measurement (each with a reason). The demo file
# also contains the gpu_worker_server branch, so worker modules are reachable by
# import but never execute inside the TD.
#
# Exact module names (prune only that module, e.g. the ``pllo.deployment`` package
# __init__ whose ``load_manifest`` is used only on the non-lite reference path --
# NOT its ``embedding_artifact`` submodule, which the lite boundary does load).
ALLOW_UNMEASURED_EXACT = {
    "pllo.protocol.gpu_worker": "untrusted GPU worker backend (runs on GPU host)",
    "pllo.protocol.orchestrator": "local/worker orchestration (not boundary)",
    "pllo.deployment.folded_worker": "untrusted GPU folded kernels (GPU host)",
    "pllo.deployment.lora_folded_package": "folded-LoRA build/merge (setup+worker)",
    "pllo.deployment.folded_package": "folded-weight package writer (trusted setup)",
    "pllo.deployment.folded_package_manifest": "package manifest (trusted setup)",
    "pllo.deployment": "pkg init re-export; load_manifest only in non-lite path",
}
# Whole subtrees that never run inside the TD.
ALLOW_UNMEASURED_PREFIXES = {
    "pllo.hf_wrappers": "full model stack (non-lite full-reference only, not TD)",
    "pllo.training": "private LoRA training prototype (trusted setup, not TD)",
    "pllo.models": "model definitions (non-lite only)",
}


def _module_to_path(mod: str):
    rel = mod.replace(".", "/")
    for cand in (REPO_ROOT / "src" / (rel + ".py"),
                 REPO_ROOT / "src" / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _first_party_imports(path: Path):
    """All ``pllo.*`` modules imported by ``path`` (module + function level)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("pllo"):
                    mods.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.startswith("pllo"):
                mods.add(node.module)
    return mods


def _pruned(mod: str):
    if mod in ALLOW_UNMEASURED_EXACT:
        return ALLOW_UNMEASURED_EXACT[mod]
    for pref, reason in ALLOW_UNMEASURED_PREFIXES.items():
        if mod == pref or mod.startswith(pref + "."):
            return reason
    return None


def compute_coverage():
    measured = {p.resolve()
                for p in _expand_paths(DEFAULT_TRUSTED_BOUNDARY_PATHS, REPO_ROOT)
                if p.is_file()}

    seen_files: set = set()
    pruned: dict = {}
    queue = []
    for rel in BOUNDARY_ENTRY:
        queue.append(REPO_ROOT / rel)

    closure: set = set()
    while queue:
        f = queue.pop()
        rf = f.resolve()
        if rf in seen_files or not f.is_file():
            continue
        seen_files.add(rf)
        closure.add(rf)
        for mod in sorted(_first_party_imports(f)):
            reason = _pruned(mod)
            if reason is not None:
                pruned.setdefault(mod, reason)
                continue
            mp = _module_to_path(mod)
            if mp is None:                       # not a module (e.g. attr import)
                continue
            if mp.resolve() not in seen_files:
                queue.append(mp)

    # entry scripts themselves must be measured too
    unmeasured = sorted(str(f.relative_to(REPO_ROOT))
                        for f in closure if f not in measured)
    return {
        "measured": sorted(str(p.relative_to(REPO_ROOT)) for p in measured),
        "closure": sorted(str(f.relative_to(REPO_ROOT)) for f in closure),
        "unmeasured_boundary_imports": unmeasured,
        "pruned": pruned,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cov = compute_coverage()
    print("=== TDX measurement coverage ===")
    print("measured files: %d   boundary closure: %d"
          % (len(cov["measured"]), len(cov["closure"])))
    if args.verbose:
        print("\n-- measured --")
        for m in cov["measured"]:
            print("  " + m)
        print("\n-- boundary import closure --")
        for c in cov["closure"]:
            print("  " + c)
        print("\n-- pruned (intentionally unmeasured) --")
        for m, reason in sorted(cov["pruned"].items()):
            print("  %s  (%s)" % (m, reason))

    gap = cov["unmeasured_boundary_imports"]
    if gap:
        print("\nMEASUREMENT GAP: boundary imports unmeasured trusted files:")
        for g in gap:
            print("  - " + g)
        print("\nFix: add each to attestation.DEFAULT_TRUSTED_BOUNDARY_PATHS, or "
              "(if it truly never runs in the TD) add a prefix + reason to "
              "ALLOW_UNMEASURED_PREFIXES in this checker.")
        print("\nTDX MEASUREMENT COVERAGE: FAILED")
        return 1
    print("\nall boundary-imported trusted files are measured")
    print("TDX MEASUREMENT COVERAGE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
