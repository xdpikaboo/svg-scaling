"""Overnight: train mup_large for ~10 epochs (~2 hr), then mup_xl for ~15
epochs (~5.5 hr), past the Chinchilla token-per-parameter optimum. Generate
samples from each, render, run eval. Total wall time ~7.5 hours.

After this finishes the report's Appendix D auto-populates from the new
run-id files (runs/large_long_run_id.txt, runs/xl_long_run_id.txt). To
update the exported PDF/HTML afterward:

    jupyter nbconvert --to notebook --execute report/report.ipynb --output report.ipynb
    jupyter nbconvert --to html --no-input report/report.ipynb
    python scripts/12_html_to_pdf.py

Idempotent: if a stage's run-id file already exists and its checkpoint is on
disk, that stage is skipped. To force a retrain, delete the corresponding
runs/{name}_run_id.txt file.

Usage:
    python scripts/16_train_overnight.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PY = sys.executable

LARGE_EPOCHS = 10  # mup_large (~17.7M params) at ~38:1 tokens/param
XL_EPOCHS    = 15  # mup_xl    (~36.2M params) at ~28:1 tokens/param

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
        sys.exit(
            f"\nFAILED: {label} (rc={rc}, {elapsed:.1f} min)\n"
            f"Total elapsed: {total:.1f} min.\n"
            f"To resume, re-run this script. Completed stages will be skipped."
        )
    print(f"\n[OK] {label}  ({elapsed:.1f} min, total {total:.1f} min)")


def write_long_config(name: str, extends: str, max_steps: int, warmup: int,
                      n_epochs: int, train_tokens: int, n_params_estimate: int) -> None:
    cfg = ROOT / "configs" / f"{name}.yaml"
    ratio = (max_steps * 524288) / max(1, n_params_estimate)
    cfg.write_text(
        f"extends: {extends}\n"
        f"name: {name}\n"
        f"parameterization: mup\n"
        f"\n"
        f"# Long training run for the report's Appendix D.\n"
        f"# {n_epochs} epochs over the {train_tokens // 1_000_000}M-token corpus,\n"
        f"# ~{ratio:.0f}:1 tokens/param (past Chinchilla 20:1).\n"
        f"\n"
        f"schedule:\n"
        f"  warmup_steps: {warmup}\n"
        f"  max_steps: {max_steps}\n"
        f"\n"
        f"optimizer:\n"
        f"  lr: 1.0e-1   # winning muP LR (transfers across widths)\n",
        encoding="utf-8",
    )
    print(f"  wrote {cfg.name}: max_steps={max_steps}, warmup={warmup}")


def _train_one(name: str, run_id_file: str) -> str:
    """Train one config and return its run_id. Skip if already trained."""
    rid_path = ROOT / "runs" / run_id_file
    if rid_path.exists():
        run_id = rid_path.read_text(encoding="utf-8").strip()
        ckpt = ROOT / "checkpoints" / run_id / "model.pt"
        if ckpt.exists():
            print(f"\n[skip] {name} already trained (run_id={run_id})")
            return run_id

    existing = set((ROOT / "runs").glob("*.json"))
    run([PY, "-m", "src.train", "--config", f"configs/{name}.yaml"],
        f"Train {name}")
    new = sorted(set((ROOT / "runs").glob("*.json")) - existing,
                 key=lambda p: p.stat().st_mtime)
    if not new:
        sys.exit(f"FAILED: no new run JSON for {name}")
    metrics = json.loads(new[-1].read_text(encoding="utf-8"))
    run_id = metrics["run_id"]
    print(f"  {name} run_id: {run_id}")
    print(f"  final val_loss: {metrics.get('val_loss_final')}")
    rid_path.write_text(run_id + "\n", encoding="utf-8")
    return run_id


def _gen_and_eval(run_id: str) -> None:
    """Generate samples + run eval for the given checkpoint. Skip if done."""
    ckpt = ROOT / "checkpoints" / run_id / "model.pt"
    if not ckpt.exists():
        sys.exit(f"FAILED: checkpoint not found for {run_id}")
    samples_dir = ROOT / "runs/samples" / run_id
    eval_path = ROOT / f"runs/eval_{run_id}.json"
    if eval_path.exists() and (samples_dir / "manifest.json").exists():
        print(f"\n[skip] generate + eval already done for {run_id}")
        return

    if not (samples_dir / "manifest.json").exists():
        run([PY, "-m", "src.generate",
             "--checkpoint", str(ckpt),
             "--tokenizer", str(ROOT / "data/tokenizer/tokenizer.json"),
             "--prefixes", str(ROOT / "configs/prefixes.txt"),
             "--output-dir", str(samples_dir),
             "--seed", "0"],
            f"Generate 45 samples for {run_id}")

    if not eval_path.exists():
        run([PY, "-m", "src.eval_svg",
             "--checkpoint", str(ckpt),
             "--samples-dir", str(samples_dir),
             "--test-bin", str(ROOT / "data/tokens/test.bin"),
             "--output", str(eval_path)],
            f"Eval (perplexity + validity rates) for {run_id}")


def main() -> int:
    print(f"Working dir: {ROOT}")
    print(f"Plan:")
    print(f"  1. mup_large x {LARGE_EPOCHS} epochs (~2 hr)")
    print(f"  2. mup_xl    x {XL_EPOCHS} epochs (~5.5 hr)")
    print(f"  Each followed by sample generation + render + eval.")
    print(f"  Total estimated wall time: ~7.5 hours.\n")

    # 1. Compute step counts from the actual current corpus.
    seq_path = ROOT / "data/splits/seqlen_hist.json"
    if not seq_path.exists():
        sys.exit(
            f"ERROR: {seq_path} missing. Run scripts/15_full_pipeline.py first "
            f"(or otherwise restore the data pipeline)."
        )
    seq = json.loads(seq_path.read_text(encoding="utf-8"))
    n_tokens = seq["per_split"]["train"]["stats"]["n_tokens"]
    steps_per_epoch = (n_tokens + 524287) // 524288
    print(f"  train tokens: {n_tokens:,}")
    print(f"  steps per epoch (524288-token batch): {steps_per_epoch}\n")

    # 2. Write the two configs with current-corpus step counts.
    large_steps = steps_per_epoch * LARGE_EPOCHS
    xl_steps    = steps_per_epoch * XL_EPOCHS
    write_long_config(
        "mup_large_long", "mup_large.yaml",
        max_steps=large_steps,
        warmup=max(10, large_steps // 10),
        n_epochs=LARGE_EPOCHS,
        train_tokens=n_tokens,
        n_params_estimate=17_701_248,
    )
    write_long_config(
        "mup_xl_long", "mup_xl.yaml",
        max_steps=xl_steps,
        warmup=max(10, xl_steps // 10),
        n_epochs=XL_EPOCHS,
        train_tokens=n_tokens,
        n_params_estimate=36_186_624,
    )

    # 3. Train + eval mup_large_long first (faster, lets you check progress sooner).
    large_id = _train_one("mup_large_long", "large_long_run_id.txt")
    _gen_and_eval(large_id)

    # 4. Train + eval mup_xl_long.
    xl_id = _train_one("mup_xl_long", "xl_long_run_id.txt")
    _gen_and_eval(xl_id)

    total = (time.monotonic() - _t_start) / 60
    print(f"\n\nOVERNIGHT RUN COMPLETE  ({total:.1f} min total)")
    print()
    print(f"Run IDs:")
    print(f"  large_long: {large_id}")
    print(f"  xl_long:    {xl_id}")
    print()
    print(f"Re-execute the report to populate Appendix D:")
    print(f"  jupyter nbconvert --to notebook --execute report/report.ipynb --output report.ipynb")
    print(f"  jupyter nbconvert --to html --no-input report/report.ipynb")
    print(f"  python scripts/12_html_to_pdf.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
