"""Runner for Stage 7.7c paged KV abstraction."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.paged_kv_abstraction import (  # noqa: E402
    PagedKVConfig,
    run_paged_kv_abstraction,
    write_reports,
)


def main() -> None:
    cfg = PagedKVConfig()
    rep = run_paged_kv_abstraction(cfg=cfg)
    j, m = write_reports(rep, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {j}")
    print(f"Wrote: {m}")
    print(f"status={rep['status']} paged_kv_supported={rep['paged_kv_supported']}")
    for a in rep["per_session_audit"]:
        print(
            f"  session {a['session_id']}: "
            f"per_block_max={a['per_block_invariant_max_abs_error']:.2e} "
            f"full_max={a['full_cache_invariant_max_abs_error']:.2e}"
        )


if __name__ == "__main__":
    main()
