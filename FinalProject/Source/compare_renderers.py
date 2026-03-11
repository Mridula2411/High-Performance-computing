#!/usr/bin/env python3
"""
compare_renderers.py
====================
Renders 10 views of ``datacube.hdf5`` with every available method, saves
each view as a PNG, and saves the absolute pixel-wise difference images
(and a 5× brightened copy for easy visual inspection) against the
Original method.

Output layout (relative to this script)
-----------------------------------------
comparison_output/
  original/          view_00.png … view_09.png
  cython/            view_00.png … view_09.png   diff_00.png … diff_09.png
  multiprocessing/   view_00.png … view_09.png   diff_*
  gpu/               view_00.png … view_09.png   diff_*
  dask/              view_00.png … view_09.png   diff_*

Methods
-------
1. Original        – sequential NumPy / SciPy loops  (reference)
2. Cython          – compiled C inner-loops via pyximport; Python fallback
3. Multiprocessing – multiprocessing.Pool, one angle per worker
4. GPU / PyTorch   – rendering loop on MPS / CUDA / CPU (same camera grids)
5. Dask            – dask.delayed angle-level parallelism

All methods share:
  • the same interpolated camera grids (scipy.interpolate.interpn)
  • the same transfer function (alpha peaks: 0.6 / 0.1 / 0.01)
  • log(value + 1e-9) to guard against log(0)
"""

import os
import sys
import math
import time
import multiprocessing
from pathlib import Path

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

from dataloader import DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NANGLES = 10
N = 180
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATACUBE = SCRIPT_DIR / "datacube.hdf5"
OUT_DIR = SCRIPT_DIR / "comparison_output"
RENDERS_DIR = SCRIPT_DIR / "renders"

# ---------------------------------------------------------------------------
# Shared transfer function  –  identical to volumerender_original.py
# ---------------------------------------------------------------------------
def transferFunction(x):
    r = (
        1.0 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * np.exp(-((x + 3.0) ** 2) / 0.5)
    )
    g = (
        1.0 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 1.0 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * np.exp(-((x + 3.0) ** 2) / 0.5)
    )
    b = (
        0.1 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 1.0 * np.exp(-((x + 3.0) ** 2) / 0.5)
    )
    a = (
        0.6 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.01 * np.exp(-((x + 3.0) ** 2) / 0.5)
    )
    return r, g, b, a


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_datacube(path: Path) -> np.ndarray:
    """Load a volume file (HDF5 or TIFF) via DataLoader."""
    loader = DataLoader(virtual_stack=False)
    p = str(path)
    if p.endswith(".h5") or p.endswith(".hdf5"):
        return loader.load_h5(path)
    elif p.endswith(".tif") or p.endswith(".tiff"):
        return loader.load_tiff(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def make_points(datacube: np.ndarray):
    Nx, Ny, Nz = datacube.shape
    return (
        np.linspace(-Nx / 2, Nx / 2, Nx),
        np.linspace(-Ny / 2, Ny / 2, Ny),
        np.linspace(-Nz / 2, Nz / 2, Nz),
    )


def make_camera_grid(datacube: np.ndarray, points, angle: float, n: int = N) -> np.ndarray:
    """Build the interpolated (n, n, n) camera grid for one viewing angle."""
    c = np.linspace(-n / 2, n / 2, n)
    qx, qy, qz = np.meshgrid(c, c, c)
    qxR = qx
    qyR = qy * np.cos(angle) - qz * np.sin(angle)
    qzR = qy * np.sin(angle) + qz * np.cos(angle)
    qi = np.column_stack((qxR.ravel(), qyR.ravel(), qzR.ravel()))
    return interpn(points, datacube, qi, method="linear").reshape((n, n, n))


# ---------------------------------------------------------------------------
# METHOD 1 – Original (pure NumPy, sequential)
# ---------------------------------------------------------------------------
def render_original(camera_grid: np.ndarray) -> np.ndarray:
    image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))
    for dataslice in camera_grid:
        r, g, b, a = transferFunction(np.log(dataslice + 1e-9))
        image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
        image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
        image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]
    return np.clip(image, 0.0, 1.0)


# ---------------------------------------------------------------------------
# METHOD 2 – Cython
# ---------------------------------------------------------------------------
_CYTHON_SRC = r"""
import numpy as np
cimport numpy as np
from libc.math cimport exp, log

def render_volume_cython(double[:,:,:] camera_grid):
    cdef int Nx = camera_grid.shape[0]
    cdef int Ny = camera_grid.shape[1]
    cdef int Nz = camera_grid.shape[2]
    cdef int i, j, k
    cdef double val, r, g, b, a
    cdef np.ndarray image_np = np.zeros((Ny, Nz, 3))
    cdef double[:,:,:] image = image_np

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                val = log(camera_grid[i, j, k] + 1e-9)

                r = (      exp(-((val-9.0)*(val-9.0))/1.0)
                     + 0.1*exp(-((val-3.0)*(val-3.0))/0.1)
                     + 0.1*exp(-((val+3.0)*(val+3.0))/0.5))

                g = (      exp(-((val-9.0)*(val-9.0))/1.0)
                     +     exp(-((val-3.0)*(val-3.0))/0.1)
                     + 0.1*exp(-((val+3.0)*(val+3.0))/0.5))

                b = (0.1 *exp(-((val-9.0)*(val-9.0))/1.0)
                     + 0.1*exp(-((val-3.0)*(val-3.0))/0.1)
                     +     exp(-((val+3.0)*(val+3.0))/0.5))

                a = (0.6 *exp(-((val-9.0)*(val-9.0))/1.0)
                     + 0.1*exp(-((val-3.0)*(val-3.0))/0.1)
                     + 0.01*exp(-((val+3.0)*(val+3.0))/0.5))

                image[j, k, 0] = a*r + (1.0-a)*image[j, k, 0]
                image[j, k, 1] = a*g + (1.0-a)*image[j, k, 1]
                image[j, k, 2] = a*b + (1.0-a)*image[j, k, 2]

    return image_np
"""


def _build_cython_renderer():
    """
    Write the Cython source to a local cache, compile it via pyximport,
    and return the callable. Returns None if compilation fails.
    """
    try:
        import pyximport

        pyx_dir = SCRIPT_DIR / ".cython_cache"
        pyx_dir.mkdir(parents=True, exist_ok=True)
        pyx_path = pyx_dir / "_render_cython_compare.pyx"
        pyx_path.write_text(_CYTHON_SRC)

        pyximport.install(
            setup_args={"include_dirs": [np.get_include()]},
            reload_support=True,
        )

        if str(pyx_dir) not in sys.path:
            sys.path.insert(0, str(pyx_dir))

        # Drop stale cached module so pyximport picks up the fresh .pyx
        sys.modules.pop("_render_cython_compare", None)

        import _render_cython_compare as _mod  # noqa: PLC0415

        return _mod.render_volume_cython
    except Exception as exc:
        print(f"  [Cython] Build failed: {exc}  →  falling back to Python.")
        return None


def _render_cython(camera_grid: np.ndarray, cython_fn) -> np.ndarray:
    out = cython_fn(np.ascontiguousarray(camera_grid, dtype=np.float64))
    return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


# ---------------------------------------------------------------------------
# METHOD 3 – Multiprocessing
# (worker must be a module-level function for macOS 'spawn' start method)
# ---------------------------------------------------------------------------
def _mp_worker(args):
    datacube, points, angle = args
    cg = make_camera_grid(datacube, points, angle)
    return render_original(cg)


# ---------------------------------------------------------------------------
# METHOD 4 – GPU / PyTorch
# Uses the same (pre-computed, interpn-based) camera grids as all other
# methods so that diffs measure only numerical / precision differences.
# ---------------------------------------------------------------------------
def render_gpu(camera_grid: np.ndarray) -> np.ndarray:
    import torch  # noqa: PLC0415

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    cg = torch.from_numpy(np.ascontiguousarray(camera_grid, dtype=np.float32)).to(device)
    log_cg = torch.log(cg + 1e-9)

    r_vol = (
        1.0 * torch.exp(-((log_cg - 9.0) ** 2) / 1.0)
        + 0.1 * torch.exp(-((log_cg - 3.0) ** 2) / 0.1)
        + 0.1 * torch.exp(-((log_cg + 3.0) ** 2) / 0.5)
    )
    g_vol = (
        1.0 * torch.exp(-((log_cg - 9.0) ** 2) / 1.0)
        + 1.0 * torch.exp(-((log_cg - 3.0) ** 2) / 0.1)
        + 0.1 * torch.exp(-((log_cg + 3.0) ** 2) / 0.5)
    )
    b_vol = (
        0.1 * torch.exp(-((log_cg - 9.0) ** 2) / 1.0)
        + 0.1 * torch.exp(-((log_cg - 3.0) ** 2) / 0.1)
        + 1.0 * torch.exp(-((log_cg + 3.0) ** 2) / 0.5)
    )
    a_vol = (
        0.6 * torch.exp(-((log_cg - 9.0) ** 2) / 1.0)
        + 0.1 * torch.exp(-((log_cg - 3.0) ** 2) / 0.1)
        + 0.01 * torch.exp(-((log_cg + 3.0) ** 2) / 0.5)
    )

    img_h, img_w = cg.shape[1], cg.shape[2]
    image_t = torch.zeros((img_h, img_w, 3), device=device)
    for i in range(cg.shape[0]):
        a_s = a_vol[i].unsqueeze(-1)
        rgb = torch.stack((r_vol[i], g_vol[i], b_vol[i]), dim=-1)
        image_t = a_s * rgb + (1 - a_s) * image_t

    return torch.clamp(image_t, 0.0, 1.0).cpu().numpy()


# ---------------------------------------------------------------------------
# METHOD 5 – Dask (delayed)
# ---------------------------------------------------------------------------
def _dask_render_one(idx: int, datacube: np.ndarray, points, nangles: int, n: int) -> np.ndarray:
    """Module-level so it serialises cleanly under any Dask scheduler."""
    angle = math.pi / 2 * idx / nangles
    cg = make_camera_grid(datacube, points, angle, n=n)
    return render_original(cg)


def render_all_dask(datacube: np.ndarray, points) -> list:
    import dask  # noqa: PLC0415

    tasks = [
        dask.delayed(_dask_render_one)(i, datacube, points, NANGLES, N)
        for i in range(NANGLES)
    ]
    return list(dask.compute(*tasks, scheduler="threads"))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def save_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), np.clip(image.astype(float), 0.0, 1.0))


def save_diff(ref: np.ndarray, cmp: np.ndarray, path: Path) -> None:
    """
    Save:
      <path>         – raw absolute difference |ref - cmp|  (values in [0,1])
      <path>_5x.png  – same, brightened ×5 for easier visual inspection
    """
    diff = np.abs(ref.astype(float) - cmp.astype(float))
    save_png(diff, path)
    save_png(np.clip(diff * 5.0, 0.0, 1.0), path.with_name(path.stem + "_5x" + path.suffix))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Compare volume rendering methods.")
    parser.add_argument(
        "--filepath",
        type=Path,
        default=DEFAULT_DATACUBE,
        help="Path to an HDF5 (.hdf5/.h5) or TIFF (.tif/.tiff) volume file "
             f"(default: {DEFAULT_DATACUBE})",
    )
    args = parser.parse_args()
    datacube_path = args.filepath.resolve()

    print("=" * 62)
    print("  Volume Renderer Comparison")
    print("=" * 62)

    # ── load data ────────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading {datacube_path.name} …")
    datacube = load_datacube(datacube_path)
    points = make_points(datacube)
    print(f"      shape={datacube.shape}  dtype={datacube.dtype}")

    # ── pre-compute camera grids (shared by all non-MP/Dask methods) ─────────
    print(f"\n[2/3] Pre-computing {NANGLES} camera grids (N={N}) …")
    camera_grids = []
    for i in range(NANGLES):
        angle = math.pi / 2 * i / NANGLES
        camera_grids.append(make_camera_grid(datacube, points, angle))
        print(f"      grid {i+1:2d}/{NANGLES}", end="\r", flush=True)
    print(f"      {NANGLES} grids ready.          ")

    # ── build Cython renderer ────────────────────────────────────────────────
    print("\n[3/3] Building Cython renderer …")
    cython_fn = _build_cython_renderer()
    if cython_fn:
        print("      Cython renderer ready.")

    # ── probe optional libraries ─────────────────────────────────────────────
    try:
        import torch  # noqa: F401
        _torch_available = True
        _gpu_device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"\nPyTorch available  –  device: {_gpu_device}")
    except ImportError:
        _torch_available = False
        print("\nPyTorch not available – GPU method will be skipped.")

    try:
        import dask  # noqa: F401
        _dask_available = True
        print("Dask available.")
    except ImportError:
        _dask_available = False
        print("Dask not available – Dask method will be skipped.")

    # ── assemble method list ─────────────────────────────────────────────────
    # Each entry: (key, label, renderer)
    # renderer is one of:
    #   callable(camera_grid) -> np.ndarray
    #   "mp"   – multiprocessing, handled specially
    #   "dask" – dask.delayed, handled specially
    methods: list[tuple[str, str, object]] = []

    methods.append(("original", "Original (NumPy)", render_original))

    if cython_fn:
        methods.append(("cython", "Cython (compiled C)",
                         lambda cg, fn=cython_fn: _render_cython(cg, fn)))
    else:
        methods.append(("cython", "Cython (Python fallback)", render_original))

    methods.append(("multiprocessing", "Multiprocessing", "mp"))

    if _torch_available:
        methods.append(("gpu", f"GPU / PyTorch ({_gpu_device})", render_gpu))
    else:
        print("  Skipping GPU method (PyTorch not installed).")

    if _dask_available:
        methods.append(("dask", "Dask (delayed)", "dask"))
    else:
        print("  Skipping Dask method (Dask not installed).")

    # ── render ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 62)
    all_images: dict[str, list[np.ndarray]] = {}

    for method_key, label, renderer in methods:
        print(f"\n  ▶  {label}")
        t0 = time.perf_counter()
        images: list[np.ndarray] = []

        if renderer == "mp":
            args = [
                (datacube, points, math.pi / 2 * i / NANGLES)
                for i in range(NANGLES)
            ]
            with multiprocessing.Pool() as pool:
                images = pool.map(_mp_worker, args)

        elif renderer == "dask":
            images = render_all_dask(datacube, points)

        else:
            for cg in camera_grids:
                images.append(renderer(cg))

        elapsed = time.perf_counter() - t0
        print(f"     time: {elapsed:.2f}s  ({elapsed / NANGLES:.3f}s / view)")

        for i, img in enumerate(images):
            save_png(img, OUT_DIR / method_key / f"view_{i:02d}.png")
            save_png(img, RENDERS_DIR / method_key / f"volumerender{i}.png")

        all_images[method_key] = images

    # ── diff images (vs original) ────────────────────────────────────────────
    print("\n" + "─" * 62)
    print("\n  Saving diff images (vs Original) …\n")
    ref_images = all_images["original"]

    for method_key, label, _ in methods:
        if method_key == "original" or method_key not in all_images:
            continue
        method_imgs = all_images[method_key]
        maes = []
        for i, (ref, cmp) in enumerate(zip(ref_images, method_imgs)):
            save_diff(ref, cmp, OUT_DIR / method_key / f"diff_{i:02d}.png")
            maes.append(float(np.mean(np.abs(ref - cmp))))
        mean_mae = np.mean(maes)
        max_mae = max(maes)
        print(f"  {method_key:<20s}  mean |diff|={mean_mae:.6f}   "
              f"worst view={max_mae:.6f}")

    print(f"\n  Comparison output : {OUT_DIR.relative_to(SCRIPT_DIR.parent)}")
    print(f"  Renders output    : {RENDERS_DIR.relative_to(SCRIPT_DIR.parent)}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
