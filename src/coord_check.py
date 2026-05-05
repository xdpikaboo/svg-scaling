"""µP coordinate check (the project spec), hand-rolled.

Trains 2-layer µP models at widths {64, 128, 256, 512} for a few steps with
MuAdamW, records the L1 norm of each module's output, and plots one line
per (module, step) across widths. If µP is correctly implemented, the lines
stay roughly horizontal across widths.

We bypass ``mup.coord_check.get_coord_data`` because it crashes natively
(Windows access violation, not a Python exception) on this PyTorch + mup
combo. The full mup machinery isn't needed for the check, we just need
forward hooks recording L1 norms.

Output:
    report/figures/coord_check.csv      (raw data)
    report/figures/coord_check.png      (plot)
    report/figures/coord_check.pdf      (plot, vector)

CLI:
    python -m src.coord_check
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mup import MuAdamW

from src.config import ModelConfig
from src.mup_model import build_mup_gpt


WIDTHS = (64, 128, 256, 512)
N_STEPS = 5
N_SEEDS = 3
LR = 1e-3
MICRO_BATCH = 8
SEQ_LEN = 64
VOCAB = 4096


def _print(msg: str) -> None:
    print(msg, flush=True)


def _make_hook(records: list, width: int, name: str, step: int):
    """Forward hook that records the L1 norm of the module's output tensor.
    Tuple/None outputs are silently skipped (handles MuGPT's optional loss
    return)."""
    def hook(module: nn.Module, _args, output) -> None:
        if isinstance(output, torch.Tensor):
            records.append({
                "width": width,
                "module": name,
                "t": step,
                "l1": output.detach().abs().mean().item(),
            })
    return hook


def _build_model(width: int) -> nn.Module:
    cfg = ModelConfig(
        vocab_size=VOCAB,
        block_size=SEQ_LEN * 2,
        n_layer=2,
        n_head=4 if width >= 64 else 2,
        d_model=width,
        d_ff=width * 4,
        dropout=0.0,
        bias=False,
    )
    return build_mup_gpt(cfg)


def _is_leaf_module(name: str, module: nn.Module) -> bool:
    """Keep leaf modules where activation L1 is meaningful: Linear,
    Embedding, LayerNorm. Skip containers (ModuleList, top-level GPT)."""
    if name == "":
        return False
    if isinstance(module, (nn.ModuleList,)):
        return False
    # Keep modules that have parameters of their own (i.e. leaf-ish).
    has_own_params = any(True for _ in module.parameters(recurse=False))
    return has_own_params


def run_coord_check() -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print(f"  device={device}")
    records: list[dict] = []
    # Fixed input across widths/seeds/steps so any drift is from µP, not data.
    torch.manual_seed(0)
    x_fixed = torch.randint(0, VOCAB, (MICRO_BATCH, SEQ_LEN), device=device)
    y_fixed = torch.randint(0, VOCAB, (MICRO_BATCH, SEQ_LEN), device=device)

    for width in WIDTHS:
        for seed in range(N_SEEDS):
            _print(f"  width={width:>4d}  seed={seed} ...")
            torch.manual_seed(seed)
            model = _build_model(width).to(device)
            model.train()
            optimizer = MuAdamW(model.parameters(), lr=LR)

            for step in range(N_STEPS):
                # Attach hooks for this step.
                handles = []
                for name, module in model.named_modules():
                    if not _is_leaf_module(name, module):
                        continue
                    handles.append(
                        module.register_forward_hook(_make_hook(records, width, name, step))
                    )

                # Forward + backward + step.
                optimizer.zero_grad(set_to_none=True)
                logits, loss = model(x_fixed, y_fixed)
                loss.backward()
                optimizer.step()

                # Remove hooks.
                for h in handles:
                    h.remove()

            # Free per-width state. Aggressive cleanup helps the Windows
            # PyTorch+CUDA combo that crashed mup's loop.
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return pd.DataFrame(records)


def plot(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "coord_check.csv"
    df.to_csv(csv_path, index=False)
    _print(f"CSV saved: {csv_path}")

    # Average over seeds.
    agg = df.groupby(["module", "width", "t"], as_index=False)["l1"].mean()

    modules = sorted(agg["module"].unique())
    n = len(modules)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.4),
                              sharey=False, squeeze=False)

    for i, mod in enumerate(modules):
        ax = axes[i // cols][i % cols]
        sub = agg[agg["module"] == mod]
        for t in sorted(sub["t"].unique()):
            line = sub[sub["t"] == t].sort_values("width")
            ax.plot(line["width"], line["l1"], marker="o", label=f"t={t}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(mod, fontsize=7)
        ax.set_xlabel("width")
        ax.set_ylabel("L1")
        ax.tick_params(labelsize=7)

    # Hide unused subplots.
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=7)
    fig.suptitle("µP coord check, L1 of module outputs vs width", fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.95, 0.97])

    png_path = out_dir / "coord_check.png"
    pdf_path = out_dir / "coord_check.pdf"
    fig.savefig(png_path, dpi=150)
    _print(f"PNG saved: {png_path}")
    fig.savefig(pdf_path)
    _print(f"PDF saved: {pdf_path}")
    plt.close(fig)


def main() -> int:
    out_dir = Path("report/figures")
    _print(f"Coord check (hand-rolled): widths={WIDTHS} steps={N_STEPS} seeds={N_SEEDS} lr={LR}")

    df = run_coord_check()
    _print(f"Collected {len(df)} L1-norm records across "
           f"{df['module'].nunique()} modules")

    plot(df, out_dir)

    _print("")
    _print("Verdict: lines should be roughly horizontal across widths "
           "(per-module). Strong upward drift = µP broken.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
