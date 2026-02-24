import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn
import imageio.v2 as imageio
import os
from dataloader import DataLoader
import argparse


"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton University, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""


def transferFunction(x):
    r = (
        1.0 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    g = (
        1.0 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 1.0 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    b = (
        0.1 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 1.0 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    a = (
        0.6 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.01 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )

    return r, g, b, a

def render_view(datacube, points, angle, N=180, output_prefix="volumerender"):
    """Render a datacube from a single viewing angle and save an image.

    Args:
        datacube: 3D numpy array with the volume data.
        points: tuple of coordinate arrays used for interpolation (x, y, z).
        Nangles: number of viewing angles to render.
        N: resolution of the camera grid along each axis.
        output_prefix: prefix for saved image files (files will be {prefix}{i}.png).
        create_gif: whether to assemble the rendered images into a GIF named {prefix}.gif.
    """
    print("Rendering Scene " + str(0 + 1) + ".\n")

    # Camera Grid / Query Points -- rotate camera view
    c = np.linspace(-N / 2, N / 2, N)
    qx, qy, qz = np.meshgrid(c, c, c)
    qxR = qx
    qyR = qy * np.cos(angle) - qz * np.sin(angle)
    qzR = qy * np.sin(angle) + qz * np.cos(angle)
    qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

    # Interpolate onto Camera Grid
    camera_grid = interpn(points, datacube, qi, method="linear").reshape((N, N, N))

    # Do Volume Rendering
    image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))

    for dataslice in camera_grid:
        r, g, b, a = transferFunction(np.log(dataslice))
        image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
        image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
        image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

    image = np.clip(image, 0.0, 1.0)

    return image



def main(filepath: str | os.PathLike, save_dir: str | os.PathLike = "./rendering_output", create_gif: bool = True) -> int:
    """Volume Rendering"""

    # Convert filepath to string if it's a PathLike object
    filepath_str = str(filepath)

    # Load datacube
    dataloader = DataLoader(virtual_stack=False)
    if filepath_str.endswith(".h5") or filepath_str.endswith(".hdf5"):
        datacube = dataloader.load_h5(filepath)
    elif filepath_str.endswith(".tif") or filepath_str.endswith(".tiff"):
        datacube = dataloader.load_tiff(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide an HDF5 or TIFF file.")

    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = np.linspace(-Nx / 2, Nx / 2, Nx)
    y = np.linspace(-Ny / 2, Ny / 2, Ny)
    z = np.linspace(-Nz / 2, Nz / 2, Nz)
    points = (x, y, z)

    # Do Volume Rendering at Different Viewing Angles
    Nangles = 10
    for i in range(Nangles):
        # Camera Grid / Query Points -- rotate camera view
        angle = np.pi / 2 * i / Nangles
        image = render_view(datacube, points, angle)


        # Plot Volume Rendering
        plt.figure(figsize=(4, 4), dpi=80)
        plt.imshow(image)
        plt.axis("off")

        # Save figure
        plt.savefig(
            f"{save_dir}_{i}.png", dpi=240, bbox_inches="tight", pad_inches=0
        )

    if create_gif:
        images = []
        for i in range(Nangles):
            images.append(imageio.imread(f"{save_dir}" + str(i) + ".png"))
        imageio.mimsave(f"{save_dir}.gif", images, duration=0.5)


    # Plot Simple Projection -- for Comparison
    plt.figure(figsize=(4, 4), dpi=80)

    plt.imshow(np.log(np.mean(datacube, 0)), cmap="viridis")
    plt.clim(-5, 5)
    plt.axis("off")

    # Save figure
    plt.savefig("projection.png", dpi=240, bbox_inches="tight", pad_inches=0)
    # plt.show()

    # Create GIF from the rendered images
    images = []
    for i in range(Nangles):
        images.append(imageio.imread(f"volumerender{i}.png"))
    imageio.mimsave("volumerender.gif", images, duration=0.5)


    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volume Rendering of a 3D Datacube")
    parser.add_argument(
        "--filepath",
        type=str,
        help="Path to the HDF5 or TIFF file containing the datacube (e.g., 'datacube.h5')",
    )
    args = parser.parse_args()
    main(args.filepath)
