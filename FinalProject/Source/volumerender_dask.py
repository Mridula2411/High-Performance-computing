import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn
import dask
from dask.distributed import Client

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

def render_angle(i, datacube, points, Nangles):
    import h5py
    import dask.array as da
    with h5py.File(hdf5_path, "r") as f:
        datacube = da.from_array(f["density"], chunks="auto").compute()
        Nx, Ny, Nz = datacube.shape
        x = np.linspace(-Nx / 2, Nx / 2, Nx)
        y = np.linspace(-Ny / 2, Ny / 2, Ny)
        z = np.linspace(-Nz / 2, Nz / 2, Nz)
        points = (x, y, z)
        angle = np.pi / 2 * i / Nangles
        N = 180
        c = np.linspace(-N / 2, N / 2, N)
        qx, qy, qz = np.meshgrid(c, c, c)
        qxR = qx
        qyR = qy * np.cos(angle) - qz * np.sin(angle)
        qzR = qy * np.sin(angle) + qz * np.cos(angle)
        qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
        camera_grid = interpn(points, datacube, qi, method="linear").reshape((N, N, N))
        image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))
        for dataslice in camera_grid:
            r, g, b, a = transferFunction(np.log(dataslice))
            image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
            image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
            image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]
        image = np.clip(image, 0.0, 1.0)
        plt.figure(figsize=(4, 4), dpi=80)
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(f"volumerender{i}.png", dpi=240, bbox_inches="tight", pad_inches=0)
        plt.close()
    return f"volumerender{i}.png"

if __name__ == "__main__":
    import os
    client = Client()
    print("Dask dashboard:", client.dashboard_link)

    hdf5_path = "datacube.hdf5"
    if not os.path.exists(hdf5_path):
        print(f"ERROR: '{hdf5_path}' not found in {os.getcwd()}.")
        print("Please place datacube.hdf5 in this directory or update the path in the script.")
        exit(1)

    Nangles = 10
    render_tasks = [dask.delayed(render_angle)(i, hdf5_path, None, Nangles) for i in range(Nangles)]
    results = dask.compute(*render_tasks)

    print("Rendered images:", results)