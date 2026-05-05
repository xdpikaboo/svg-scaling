"""Phase 10: generate samples from the best model checkpoint.

Reads runs/best_run_id.txt to locate checkpoints/{run_id}/model.pt, then
calls src.generate.main with the standard knobs (temps {0.5, 0.8, 1.0},
top-p=0.9, 10 unconditional + 5 prefix).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generate import main as generate_main  # noqa: E402


def main() -> int:
    run_id_file = ROOT / "runs" / "best_run_id.txt"
    if not run_id_file.exists():
        print(f"ERROR: {run_id_file} not found. Run scripts/08_train_best.py first.",
              file=sys.stderr)
        return 1
    run_id = run_id_file.read_text(encoding="utf-8").strip()
    ckpt = ROOT / "checkpoints" / run_id / "model.pt"
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    print(f"Generating from: {ckpt}")
    return generate_main([
        "--checkpoint", str(ckpt),
        "--tokenizer", str(ROOT / "data" / "tokenizer" / "tokenizer.json"),
        "--prefixes", str(ROOT / "configs" / "prefixes.txt"),
        "--output-dir", str(ROOT / "runs" / "samples" / run_id),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
