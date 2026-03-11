#!/usr/bin/env python3
"""
benchmark_renderers.py
======================
Iterates over every HDF5 / TIFF volume in a dataset directory, times each
rendering method across all 10 views, and produces a line plot of

    execution time (s)  vs  dataset file size (GB)

one line per method.  The X axis uses the actual on-disk file size so that
different volume files show distinct positions even when the DataLoader
applies padding.

Usage
-----
    python benchmark_renderers.py [--dataset ../dataset] [--out benchmark.png]
"""

import argparse
import math
import multiprocessing
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── reuse all rendering logic from compare_renderers ────────────────────────
from compare_renderers import (
    NANGLES,
    N,
    SCRIPT_DIR,
    load_datacube,
    make_points,
    make_camera_grid,
    render_original,
    render_gpu,
    _render_cython,
    _build_cython_renderer,
    render_all_dask,
    _mp_worker,
)

DATASET_DIR = (SCRIPT_DIR / "../dataset").resolve()

VOLUME_EXTENSIONS = {".tif", ".tiff", ".h5", ".hdf5"}

# Colour / marker per method
METHOD_STYLE = {
    "original":        {"color": "#4c72b0", "marker": "o", "label": "Original (NumPy)"},
    "cython":          {"color": "#dd8452", "marker": "s", "label": "Cython"},
    "multiprocessing": {"color": "#55a868", "marker": "^", "label": "Multiprocessing"},
    "gpu":             {"color": "#c44e52", "marker": "D", "label": "GPU / PyTorch"},
    "dask":            {"color": "#8172b3", "marker": "P", "label": "Dask"},
}


def file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def time_method(renderer, camera_grids, datacube, points) -> float:
    """Return total wall-clock time (s) to render all views with one method."""
    t0 = time.perf_counter()

    if renderer == "mp":
        mp_args = [
            (datacube, points, math.pi / 2 * i / NANGLES)
            for i in range(NANGLES)
        ]
        with multiprocessing.Pool() as pool:
            pool.map(_mp_worker, mp_args)

    elif renderer == "dask":
        render_all_dask(datacube, points)

    else:
        for cg in camera_grids:
            renderer(cg)

    return time.perf_counter() - t0


def build_methods(cython_fn) -> list[tuple[str, object]]:
    """Return (key, renderer) pairs for every available method."""
    methods = [("original", render_original)]

    if cython_fn:
        methods.append(("cython", lambda cg, fn=cython_fn: _render_cython(cg, fn)))
    else:
        methods.append(("cython", render_original))  # Python fallback

    methods.append(("multiprocessing", "mp"))

    try:
        import torch  # noqa: F401
        methods.append(("gpu", render_gpu))
    except ImportError:
        print("  PyTorch not available – GPU method skipped.")

    try:
        import dask  # noqa: F401
        methods.append(("dask", "dask"))
    except ImportError:
        print("  Dask not available – Dask method skipped.")

    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark rendering methods across datasets.")
    parser.add_argument(
        "--dataset", type=Path, default=DATASET_DIR,
        help=f"Directory containing volume files (default: {DATASET_DIR})",
    )
    parser.add_argument(
        "--out", type=str, default="benchmark.png",
        help="Output plot filename (default: benchmark.png)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output DPI (default: 150)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset directory not found: {dataset_dir}")
        return

    # ── collect volume files, sorted by file size ────────────────────────────
    volume_files = sorted(
        (p for p in dataset_dir.iterdir() if p.suffix.lower() in VOLUME_EXTENSIONS),
        key=lambda p: p.stat().st_size,
    )
    if not volume_files:
        print(f"No volume files found in {dataset_dir}")
        return

    print(f"Found {len(volume_files)} volume file(s):")
    for vf in volume_files:
        print(f"  {vf.name:30s}  {vf.stat().st_size / 1024**2:.1f} MB")

    # ── build Cython renderer once ───────────────────────────────────────────
    print("\nBuilding Cython renderer …")
    cython_fn = _build_cython_renderer()
    print("  Ready." if cython_fn else "  Failed – using Python fallback.")

    methods = build_methods(cython_fn)

    # results[method_key] = list of (size_gb, time_s)
    results: dict[str, list[tuple[float, float]]] = {key: [] for key, _ in methods}

    # ── benchmark ────────────────────────────────────────────────────────────
    for vf in volume_files:
        size_gb = file_size_gb(vf)
        print(f"\n{'─'*60}")
        print(f"  Dataset : {vf.name}  ({size_gb * 1024:.2f} MB)")

        print("  Loading …", end=" ", flush=True)
        datacube = load_datacube(vf)
        points = make_points(datacube)
        print(f"shape={datacube.shape}")

        print(f"  Pre-computing {NANGLES} camera grids …", end=" ", flush=True)
        camera_grids = [
            make_camera_grid(datacube, points, math.pi / 2 * i / NANGLES)
            for i in range(NANGLES)
        ]
        print("done")

        for method_key, renderer in methods:
            elapsed = time_method(renderer, camera_grids, datacube, points)
            results[method_key].append((size_gb, elapsed))
            label = METHOD_STYLE[method_key]["label"]
            print(f"  {label:<28s}  {elapsed:.3f}s  ({elapsed / NANGLES:.4f}s/view)")

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    for method_key, pts in results.items():
        if not pts:
            continue
        style = METHOD_STYLE[method_key]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs, ys,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=1.8,
            markersize=7,
        )

    # Vertical guide lines + filename labels
    for vf in volume_files:
        size_gb = file_size_gb(vf)
        ax.axvline(size_gb, color="grey", linewidth=0.5, linestyle=":")
        ax.text(
            size_gb, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
            f" {vf.stem}", rotation=90, fontsize=7, color="grey",
            va="top", ha="right",
        )

    ax.set_xlabel("Dataset file size (GB)", fontsize=11)
    ax.set_ylabel("Total execution time — all 10 views (s)", fontsize=11)
    ax.set_title("Volume Rendering Benchmark: Execution Time vs. Dataset Size", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_path = SCRIPT_DIR / args.out
    fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    main()
