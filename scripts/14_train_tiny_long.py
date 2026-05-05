"""Option 1, Bonus run for Appendix C: train mup_tiny for many epochs to
demonstrate that grammar emerges when tokens/parameter is Chinchilla-optimal.

Reuses the data/clean and data/splits already on disk from the in-progress
Option 3 rerun (so the 30+ min Phase 1 work isn't wasted), but skips the
slow full-corpus BPE in favour of a 100k-record subset (vocabulary converges
fast on char n-gram statistics).

Stages:
    0. Restore archived runs/checkpoints if Option 3 had moved them away.
    1. Train BPE on a 100k-record subset (~30 sec).
    2. Tokenize the corpus (~3 min).
    3. Compute step count for ~15 epochs over the new train set.
    4. Write configs/tiny_long.yaml.
    5. Train mup_tiny for ~15 epochs (~25-35 min on RTX 3060).
    6. Generate 45 samples from the tiny-long checkpoint.
    7. Run eval (test perplexity + XML/structural/render rates).
    8. Render samples for the report.

After this finishes, re-execute report/report.ipynb to pick up Appendix C.

Usage:
    python scripts/14_train_tiny_long.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PY = sys.executable
N_EPOCHS = 15  # ~15 epochs gives ~800:1 tokens/param at 80M corpus / 1.5M params

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
        sys.exit(f"\nFAILED: {label} (rc={rc}, {elapsed:.1f} min)")
    print(f"\n[OK] {label}  ({elapsed:.1f} min, total {total:.1f} min)")


def restore_archives() -> None:
    """If a previous run archived runs/ and checkpoints/, restore them so the
    report can still find the SP/muP scaling data alongside the new tiny-long
    run."""
    hr("Stage 0, restore archived runs/checkpoints if needed")
    for d in ["runs", "checkpoints"]:
        archive = ROOT / f"{d}_24M_archive"
        live = ROOT / d
        if archive.exists() and not live.exists():
            print(f"  restoring {archive.name}/ -> {d}/")
            archive.rename(live)
        elif archive.exists() and live.exists():
            # Both exist, merge archive contents into live dir.
            print(f"  merging {archive.name}/* -> {d}/")
            for src in list(archive.iterdir()):
                target = live / src.name
                if not target.exists():
                    shutil.move(str(src), str(target))
            try:
                if not any(archive.iterdir()):
                    archive.rmdir()
            except OSError:
                pass
        else:
            print(f"  no archive at {archive.name}; nothing to restore")
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "checkpoints").mkdir(exist_ok=True)


def main() -> int:
    print(f"Working dir: {ROOT}")
    print(f"Plan: tiny-long bonus run, {N_EPOCHS} epochs over the current corpus.\n")

    restore_archives()

    # 1. Tokenizer (skip if already present from a previous run).
    if not (ROOT / "data/tokenizer/tokenizer.json").exists():
        run([PY, "-m", "src.tokenizer", "--max-records", "100000"],
            "Stage 1, train BPE (100k-record subset)")
    else:
        print("\n[skip] data/tokenizer/tokenizer.json exists; reusing")

    # 2. Tokenize corpus (skip if .bin files exist).
    if not (ROOT / "data/tokens/train.bin").exists():
        run([PY, "-m", "src.tokenize_corpus"], "Stage 2, tokenize corpus")
    else:
        print("[skip] data/tokens/train.bin exists; reusing")

    # 3. Compute step count.
    seq = json.loads((ROOT / "data/splits/seqlen_hist.json").read_text(encoding="utf-8"))
    n_tokens = seq["per_split"]["train"]["stats"]["n_tokens"]
    steps_per_epoch = (n_tokens + 524287) // 524288
    max_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = max(10, max_steps // 10)
    tokens_seen = max_steps * 524288
    tiny_n_params = 1_573_504  # mup_tiny n_params from the report
    ratio = tokens_seen / tiny_n_params
    print(f"\n  train tokens = {n_tokens:,}")
    print(f"  1 epoch = {steps_per_epoch} steps")
    print(f"  {N_EPOCHS} epochs = {max_steps} steps")
    print(f"  tokens seen = {tokens_seen:,} (~{ratio:.0f}:1 tokens/param) "
          f", Chinchilla optimum is 20:1")

    # 4. Write the tiny_long config.
    tiny_long_yaml = ROOT / "configs/tiny_long.yaml"
    tiny_long_yaml.write_text(
        f"extends: mup_tiny.yaml\n"
        f"name: tiny_long\n"
        f"parameterization: mup\n"
        f"\n"
        f"# Bonus run for report Appendix C: train mup_tiny for {N_EPOCHS} epochs over\n"
        f"# the {n_tokens // 1_000_000}M-token corpus (~{ratio:.0f}:1 tokens/param, wildly past\n"
        f"# Chinchilla's 20:1). Tests whether grammar emerges with enough training.\n"
        f"\n"
        f"schedule:\n"
        f"  warmup_steps: {warmup_steps}\n"
        f"  max_steps: {max_steps}\n"
        f"\n"
        f"optimizer:\n"
        f"  lr: 1.0e-1   # µP winning LR transfers across widths and across training durations\n",
        encoding="utf-8",
    )
    print(f"  wrote {tiny_long_yaml.name}")

    # 5. Train tiny_long.
    # NB: the run_id uses the suffix of cfg.name after the first '_', so
    # `tiny_long` becomes `mup_long_lr1e-1_<ts>`, glob broadly.
    existing_runs = set((ROOT / "runs").glob("*.json"))
    run([PY, "-m", "src.train", "--config", "configs/tiny_long.yaml"],
        f"Stage 5, train tiny-long ({N_EPOCHS} epochs)")

    # Locate the new run JSON.
    new_runs = sorted(
        set((ROOT / "runs").glob("*.json")) - existing_runs,
        key=lambda p: p.stat().st_mtime,
    )
    if not new_runs:
        sys.exit("FAILED: no tiny_long run JSON written")
    metrics = json.loads(new_runs[-1].read_text(encoding="utf-8"))
    run_id = metrics["run_id"]
    print(f"  tiny_long run_id: {run_id}")
    print(f"  final val_loss: {metrics.get('val_loss_final')}")

    # Persist the run id so the report notebook can find it.
    (ROOT / "runs/tiny_long_run_id.txt").write_text(run_id + "\n", encoding="utf-8")

    ckpt = ROOT / "checkpoints" / run_id / "model.pt"
    samples_dir = ROOT / "runs/samples" / run_id

    # 6. Generate.
    run([PY, "-m", "src.generate",
         "--checkpoint", str(ckpt),
         "--tokenizer", str(ROOT / "data/tokenizer/tokenizer.json"),
         "--prefixes", str(ROOT / "configs/prefixes.txt"),
         "--output-dir", str(samples_dir),
         "--seed", "0"],
        "Stage 6, generate 45 samples from tiny-long")

    # 7. Eval.
    run([PY, "-m", "src.eval_svg",
         "--checkpoint", str(ckpt),
         "--samples-dir", str(samples_dir),
         "--test-bin", str(ROOT / "data/tokens/test.bin"),
         "--output", str(ROOT / f"runs/eval_{run_id}.json")],
        "Stage 7, eval (perplexity + validity rates)")

    # 8. Already rendered by generate.py; nothing extra.
    total = (time.monotonic() - _t_start) / 60
    print(f"\nDONE  ({total:.1f} min total)")
    print(f"  tiny_long checkpoint: {ckpt}")
    print(f"  samples:              {samples_dir}")
    print(f"  eval JSON:            runs/eval_{run_id}.json")
    print(f"\nNext: re-execute report/report.ipynb to populate Appendix C.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
