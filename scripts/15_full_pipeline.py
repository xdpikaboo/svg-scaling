"""Full end-to-end pipeline from a clean slate.

Runs every stage of the project in order:

    1. Data prep (50k-fonts subsample, ~24M tokens)
    2. BPE tokenizer (4096 vocab)
    3. Tokenize corpus to .bin files
    4. SP LR sweep (7 LRs on tiny)
    5. SP scaling (5 sizes)
    6. muP coordinate check
    7. muP LR sweep + scaling (5 sizes)
    8. Best model (muP-XL x 3 epochs) + generate 45 samples
    9. Eval + render for report
   10. Bonus tiny-long (mup_tiny x 15 epochs) for Appendix C
   11. Run analysis.ipynb + report.ipynb, export HTML + PDF

Estimated wall-time on RTX 3060 laptop: ~2.5 hours.

If any stage fails, the script exits with a clear message. To resume,
re-run the failed stage manually with the relevant scripts/0X_*.py command.

Usage:
    .\\.venv\\Scripts\\Activate.ps1            # Windows; or source .venv/bin/activate on Linux
    python scripts/15_full_pipeline.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PY = sys.executable

_t_start = time.monotonic()


def hr(s: str) -> None:
    print(f"\n{'=' * 80}\n{s}\n{'=' * 80}", flush=True)


def run(cmd: list[str], label: str) -> None:
    hr(label)
    t0 = time.monotonic()
    rc = subprocess.run(cmd).returncode
    elapsed = (time.monotonic() - t0) / 60
    total = (time.monotonic() - _t_start) / 60
    if rc != 0:
        sys.exit(f"\nFAILED: {label} (rc={rc}, {elapsed:.1f} min)\n"
                 f"Total elapsed before failure: {total:.1f} min.\n"
                 f"You can resume by running individual commands from this point.")
    print(f"\n[OK] {label}  ({elapsed:.1f} min, total {total:.1f} min)")


def main() -> int:
    print(f"Working dir: {ROOT}")
    print("Plan: full pipeline from clean state (~2.5 hr on RTX 3060).\n")

    # Sanity check: warn if a partial state exists.
    for d in ("runs", "checkpoints"):
        p = ROOT / d
        if p.exists() and any(p.iterdir()):
            print(f"WARNING: {d}/ is not empty. Existing runs may be merged with new ones.")

    # ---- Data + tokenizer + .bin files ----
    run([PY, "-m", "src.data", "--no-render", "--fonts-subsample", "50000"],
        "Stage 1, download + clean (50k fonts, ~15 min)")
    # Subset BPE for speed; vocabulary converges fast on char n-grams.
    run([PY, "-m", "src.tokenizer", "--max-records", "100000"],
        "Stage 2, train BPE (4096 vocab, ~30 sec)")
    run([PY, "-m", "src.tokenize_corpus"],
        "Stage 3, tokenize corpus to .bin (~1 min)")

    # ---- SP scaling ----
    run([PY, "-m", "src.lr_sweep", "--config", "configs/sp_tiny.yaml"],
        "Stage 4, SP LR sweep (7 LRs on tiny, ~7 min)")
    run([PY, "scripts/05_train_all_sp.py"],
        "Stage 5, SP scaling (5 sizes, ~30 min)")

    # ---- muP ----
    run([PY, "-m", "src.coord_check"],
        "Stage 6, muP coordinate check (~15 sec)")
    run([PY, "-m", "src.lr_sweep",
         "--config", "configs/mup_tiny.yaml",
         "--output-dir", "runs/sweep_mup_tiny",
         "--winner-file", "configs/mup_winning_lr.txt"],
        "Stage 7a, muP LR sweep (~7 min)")
    run([PY, "scripts/07_train_all_mup.py"],
        "Stage 7b, muP scaling (5 sizes, ~30 min)")

    # ---- Best model + eval ----
    run([PY, "scripts/08_train_best.py"],
        "Stage 8a, best model (muP-XL x 3 epochs, ~25 min)")
    run([PY, "scripts/09_generate.py"],
        "Stage 8b, generate 45 samples (~5 min)")
    run([PY, "scripts/10_eval.py"],
        "Stage 9a, eval (perplexity + validity rates, ~30 sec)")
    run([PY, "scripts/11_render_for_report.py"],
        "Stage 9b, render training examples + sample grid (~30 sec)")

    # ---- Bonus tiny-long for Appendix C ----
    run([PY, "scripts/14_train_tiny_long.py"],
        "Stage 10, tiny-long bonus (mup_tiny x 15 epochs, ~30 min)")

    # ---- Notebooks ----
    run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "analysis.ipynb"],
        "Notebook, analysis.ipynb (~1 min)")
    run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "report/report.ipynb", "--output", "report.ipynb"],
        "Notebook, report.ipynb (populate outputs, ~1 min)")
    run([PY, "-m", "jupyter", "nbconvert", "--to", "html",
         "--no-input", "report/report.ipynb"],
        "Export, report.html")
    run([PY, "scripts/12_html_to_pdf.py"],
        "Export, report.pdf")

    total = (time.monotonic() - _t_start) / 60
    print(f"\n\nFULL PIPELINE COMPLETE  ({total:.1f} min total)")
    print("Outputs:")
    print(f"  runs/                     all metrics JSONs")
    print(f"  checkpoints/              all model checkpoints")
    print(f"  report/figures/           updated PDFs and PNGs")
    print(f"  report/report.html        clean HTML report")
    print(f"  report/report.pdf         PDF version")
    print()
    print("Open report/report.html or report/report.pdf to view the deliverable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
