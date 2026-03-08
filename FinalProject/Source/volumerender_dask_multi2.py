import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for Dask workers
import matplotlib.pyplot as plt
import dask
from dask.distributed import Client
import tifffile


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
        1.0 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.2 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.05 * np.exp(-((x + 3.0) ** 2) / 0.5)
    )
    return r, g, b, a


def render_angle_delayed(i, tiff_path, Nangles, N=96):
    from scipy.interpolate import interpn

    datacube = tifffile.imread(tiff_path).astype(np.float32)

    # Robust normalization to avoid flat/yellow output
    p1, p99 = np.percentile(datacube, (1.0, 99.5))
    datacube = np.clip((datacube - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    Nx, Ny, Nz = datacube.shape
    x = np.linspace(-Nx / 2, Nx / 2, Nx, dtype=np.float32)
    y = np.linspace(-Ny / 2, Ny / 2, Ny, dtype=np.float32)
    z = np.linspace(-Nz / 2, Nz / 2, Nz, dtype=np.float32)
    points = (x, y, z)

    angle = np.pi / 2 * i / Nangles

    # Smaller sampling cube for lower memory footprint
    half = np.float32(0.45 * min(Nx, Ny, Nz))
    c = np.linspace(-half, half, N, dtype=np.float32)
    qx, qy, qz = np.meshgrid(c, c, c, indexing="ij")

    qxR = qx
    qyR = qy * np.cos(angle) - qz * np.sin(angle)
    qzR = qy * np.sin(angle) + qz * np.cos(angle)

    qi = np.column_stack((qxR.ravel(), qyR.ravel(), qzR.ravel())).astype(np.float32)
    camera_grid = interpn(
        points,
        datacube,
        qi,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    ).reshape((N, N, N))

    image = np.zeros((N, N, 3), dtype=np.float32)
    for dataslice in camera_grid:
        # Map [0,1] -> [-3,9] to match transfer function peaks
        tf_x = dataslice * 12.0 - 3.0
        r, g, b, a = transferFunction(tf_x)
        image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
        image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
        image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

    image = np.clip(image, 0.0, 1.0)
    out_name = f"{Path(tiff_path).stem}_volumerender{i}.png"
    plt.imsave(out_name, image)
    return out_name


if __name__ == "__main__":
    # Mac-safe + lower memory pressure
    client = Client(processes=False, n_workers=1, threads_per_worker=2, memory_limit="3GB")
    print("Dask dashboard:", client.dashboard_link)

    script_dir = Path(__file__).resolve().parent
    dataset_dir = (script_dir / "../dataset").resolve()
    print(f"Looking for TIFFs in: {dataset_dir}")

    if not dataset_dir.is_dir():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        input("Press Enter to close Dask dashboard and exit...")
        client.close()
        raise SystemExit(1)

    tiff_files = sorted(
        p for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )

    if not tiff_files:
        print("No .tif/.tiff files found. Nothing to render.")
        input("Press Enter to close Dask dashboard and exit...")
        client.close()
        raise SystemExit(0)

    Nangles = 6
    for tiff_path in tiff_files:
        try:
            # Compute one angle at a time to keep RAM stable
            results = []
            for i in range(Nangles):
                task = dask.delayed(render_angle_delayed)(i, str(tiff_path), Nangles, 96)
                result = dask.compute(task)[0]
                results.append(result)
            print(f"Rendered images for {tiff_path.name}: {tuple(results)}")
        except Exception as e:
            print(f"Failed for {tiff_path.name}: {e}")
            continue

    input("Rendering complete. Press Enter to close Dask dashboard and exit...")
    client.close()