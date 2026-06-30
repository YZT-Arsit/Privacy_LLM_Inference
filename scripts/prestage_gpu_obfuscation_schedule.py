"""Pre-generate + stage a NON-SECRET obfuscation schedule for the GPU worker.

Builds a per-session staged schedule (one slot per Linear-boundary pad site) and
writes a manifest containing ONLY masked-basis / public artifacts
(``xpad_tilde`` / ``cpad_tilde`` refs, commitments, shapes). The schedule is
audited for secrets BEFORE writing; under ``--paper-facing-aaai`` any of these is
fatal: raw mask / raw inverse / raw pad / plaintext input / token ids / recovery
secret present, or nonlinear backend != A_rightmul.

This reduces online TEE<->GPU interaction WITHOUT moving any secret to the GPU.

Example::

    python scripts/prestage_gpu_obfuscation_schedule.py \\
      --folded-package-path <PKG> --embedding-artifact-path <EMB> \\
      --nonlinear-backend A_rightmul --seq-len 1024 --max-new-tokens 512 \\
      --num-layers 28 --output-dir outputs/staged/qwen_A_rightmul \\
      --paper-facing-aaai --output-json outputs/staged/qwen_prestage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.nonlinear_designs import (  # noqa: E402
    normalize_nonlinear_backend)
from pllo.runtime.gpu_staged_schedule import (  # noqa: E402
    StagedScheduleSecretLeak, audit_gpu_staged_schedule_no_secrets,
    build_staged_schedule, staged_schedule_report_fields,
    write_gpu_staged_schedule)


def _num_layers(args):
    if args.num_layers:
        return int(args.num_layers)
    # best-effort from the folded package manifest
    try:
        from pllo.deployment import load_manifest
        return int(load_manifest(args.folded_package_path).num_layers)
    except Exception:                                            # noqa: BLE001
        return 28


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folded-package-path", required=True)
    ap.add_argument("--embedding-artifact-path", default=None)
    ap.add_argument("--nonlinear-backend", default="A_rightmul")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--schedule-id", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--paper-facing-aaai", action="store_true", default=False)
    args = ap.parse_args()

    nb = normalize_nonlinear_backend(args.nonlinear_backend or "A_rightmul")
    if args.paper_facing_aaai and nb != "A_rightmul":
        print("ERROR: --paper-facing-aaai requires --nonlinear-backend A_rightmul",
              file=sys.stderr)
        return 3

    n = _num_layers(args)
    sched_id = args.schedule_id or ("staged_%s_n%d_s%d" % (nb, n, args.seq_len))
    manifest = build_staged_schedule(
        schedule_id=sched_id, nonlinear_backend=nb, seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens, num_layers=n,
        session_id=args.session_id)

    try:
        audit = audit_gpu_staged_schedule_no_secrets(manifest)
    except StagedScheduleSecretLeak as exc:
        print("ERROR: staged schedule failed no-secret audit: %s" % exc,
              file=sys.stderr)
        return 3

    path = write_gpu_staged_schedule(args.output_dir, manifest)
    rep = {"stage": "prestage_gpu_obfuscation_schedule",
           "staged_schedule_dir": str(args.output_dir),
           "manifest_path": str(path), "nonlinear_backend": nb,
           "num_layers": n, "seq_len": args.seq_len,
           "max_new_tokens": args.max_new_tokens,
           "paper_facing_aaai": bool(args.paper_facing_aaai), **audit}
    rep.update(staged_schedule_report_fields(manifest, audit))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(rep, indent=2),
                                          encoding="utf-8")
    print("=== prestage gpu staged schedule ===")
    print("dir=%s slots=%s nonlinear=%s audit_passed=%s"
          % (args.output_dir, manifest["num_slots"], nb,
             rep["staged_schedule_no_secret_audit_passed"]))
    print("raw_masks=%s raw_pad=%s plaintext_input=%s token_ids=%s"
          % (rep["gpu_staged_schedule_contains_raw_masks"],
             rep["gpu_staged_schedule_contains_raw_pad"],
             rep["gpu_staged_schedule_contains_plaintext_input"],
             rep["gpu_staged_schedule_contains_token_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
