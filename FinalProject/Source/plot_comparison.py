#!/usr/bin/env python3
"""
plot_comparison.py
==================
Reads one rendered view and its diff from each method produced by
compare_renderers.py and assembles them into a single figure:

  Row 1 – rendered view  (view_<VIEW>.png)
  Row 2 – absolute diff vs. Original  (diff_<VIEW>.png)
           Original column shows a blank/zero panel (no self-diff)

Usage
-----
    python plot_comparison.py [--view 0] [--out comparison_plot.png]
                              [--base comparison_output]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SCRIPT_DIR = Path(__file__).resolve().parent

# Methods in display order – first entry is always the reference (no diff row).
METHODS = [
    ("original",        "Original\n(NumPy)"),
    ("cython",          "Cython"),
    ("multiprocessing", "Multiprocessing"),
    ("gpu",             "GPU / PyTorch"),
    ("dask",            "Dask"),
]


def load_image(path: Path) -> np.ndarray | None:
    """Return an RGB float32 array, or None if the file doesn't exist."""
    if path.exists():
        img = mpimg.imread(str(path))
        # Convert RGBA → RGB if needed
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        return img.astype(np.float32)
    return None


def blank_like(ref: np.ndarray) -> np.ndarray:
    """Return a black image the same size as ref."""
    return np.zeros_like(ref)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot render vs. diff comparison grid.")
    parser.add_argument("--view", type=int, default=0,
                        help="Which view index to display (default: 0)")
    parser.add_argument("--out", type=str, default="comparison_plot.png",
                        help="Output filename (default: comparison_plot.png)")
    parser.add_argument("--base", type=str, default="comparison_output",
                        help="Base output directory from compare_renderers.py "
                             "(default: comparison_output)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI (default: 150)")
    args = parser.parse_args()

    base_dir = (SCRIPT_DIR / args.base).resolve()
    view_idx = args.view
    view_file = f"view_{view_idx:02d}.png"
    diff_file = f"diff_{view_idx:02d}.png"

    # ── discover which methods actually have output ───────────────────────────
    available = []
    for key, label in METHODS:
        img = load_image(base_dir / key / view_file)
        if img is not None:
            available.append((key, label, img))

    if not available:
        print(f"No rendered views found in {base_dir}. "
              "Run compare_renderers.py first.")
        return

    ncols = len(available)
    nrows = 2
    ref_img = available[0][2]   # Original is always column 0

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 7),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )

    # Ensure axes is always 2-D even with a single column
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    for col, (key, label, render_img) in enumerate(available):
        # ── Row 1: rendered view ──────────────────────────────────────────────
        ax_top = axes[0, col]
        ax_top.imshow(render_img, interpolation="nearest")
        ax_top.set_title(label, fontsize=9, pad=4)
        ax_top.axis("off")

        # ── Row 2: diff vs. Original ──────────────────────────────────────────
        ax_bot = axes[1, col]
        if key == "original":
            # No self-diff – show a plain black panel
            diff_img = blank_like(ref_img)
            ax_bot.imshow(diff_img, vmin=0, vmax=1, interpolation="nearest")
        else:
            diff_img = load_image(base_dir / key / diff_file)
            if diff_img is None:
                diff_img = np.abs(ref_img - render_img)
            # Show diff with a consistent scale across all columns
            diff_max = np.max(np.abs(ref_img - render_img))
            im = ax_bot.imshow(
                diff_img,
                vmin=0,
                vmax=max(diff_max, 1e-6),
                cmap="inferno",
                interpolation="nearest",
            )
            # Compact MAE annotation
            mae = float(np.mean(np.abs(ref_img.astype(float) - render_img.astype(float))))
            ax_bot.set_title(f"MAE={mae:.5f}", fontsize=7, pad=2)

        ax_bot.axis("off")

    # ── Row labels on left-most column ───────────────────────────────────────
    axes[0, 0].set_ylabel("Render", fontsize=10, labelpad=6)
    axes[1, 0].set_ylabel("Diff vs. Original", fontsize=10, labelpad=6)
    for row in range(nrows):
        axes[row, 0].axis("on")
        axes[row, 0].tick_params(left=False, bottom=False,
                                 labelleft=False, labelbottom=False)
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

    fig.suptitle(f"Volume Render Comparison — view {view_idx}", fontsize=12, y=1.01)

    out_path = SCRIPT_DIR / args.out
    fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
