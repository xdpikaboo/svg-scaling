"""Phase 11: evaluate the best checkpoint.

Reads runs/best_run_id.txt to locate checkpoint + samples, then runs
src.eval_svg.main with sensible defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_svg import main as eval_main  # noqa: E402


def main() -> int:
    run_id_file = ROOT / "runs" / "best_run_id.txt"
    if not run_id_file.exists():
        print(f"ERROR: {run_id_file} not found.", file=sys.stderr)
        return 1
    run_id = run_id_file.read_text(encoding="utf-8").strip()
    ckpt = ROOT / "checkpoints" / run_id / "model.pt"
    samples = ROOT / "runs" / "samples" / run_id
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1
    return eval_main([
        "--checkpoint", str(ckpt),
        "--samples-dir", str(samples),
        "--test-bin", str(ROOT / "data" / "tokens" / "test.bin"),
        "--output", str(ROOT / "runs" / f"eval_{run_id}.json"),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
