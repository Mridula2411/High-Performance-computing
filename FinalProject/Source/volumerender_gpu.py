import torch
import torch.nn.functional as F
import numpy as np
import os
import math
import argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from dataloader import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def transferFunction(x):
    """
    Vectorised Transfer Function for GPU based rendering.
    Operations are performed on the entire input tensor at once, leveraging GPU parallelism.
    """

    r = (
        1.0 * torch.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * torch.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * torch.exp(-((x - -3.0) ** 2) / 0.5)
    )
    g = (
        1.0 * torch.exp(-((x - 9.0) ** 2) / 1.0)
        + 1.0 * torch.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * torch.exp(-((x - -3.0) ** 2) / 0.5)
    )
    b = (
        0.1 * torch.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * torch.exp(-((x - 3.0) ** 2) / 0.1)
        + 1.0 * torch.exp(-((x - -3.0) ** 2) / 0.5)
    )
    a = (
        0.6 * torch.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * torch.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.01 * torch.exp(-((x - -3.0) ** 2) / 0.5)
    )

    return r, g, b, a

def render_view(datacube_t, angle, N=180):
    """
    Render the datacube from a single viewing angle using pytorch for GPU acceleration.
    """

    # Camera Grid
    c = torch.linspace(-1, 1, N, device=device)
    qx, qy, qz = torch.meshgrid(c, c, c, indexing='xy')
    angle_t = torch.tensor(angle, device=device)

    # Rotate the camera view
    qxR = qx
    qyR = qy * torch.cos(angle_t) - qz * torch.sin(angle_t)
    qzR = qy * torch.sin(angle_t) + qz * torch.cos(angle_t)

    # Stack into a grid for interpolation
    grid = torch.stack((qxR, qyR, qzR), dim=-1).unsqueeze(0)  # Shape: (1, N, N, N, 3)

    # Interpolate the datacube onto the camera grid
    datacube_t = datacube_t.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
    camera_grid = F.grid_sample(datacube_t, grid, mode='bilinear', align_corners=True)  # Shape: (1, 1, N, N, N)
    camera_grid = camera_grid.squeeze()  # Shape: (N, N, N)

    # Volume Rendering
    # Prevent log(0) by adding a small epsilon
    log_data = torch.log(camera_grid + 1e-6)


    r, g, b, a = transferFunction(log_data)

    # Initialize the image tensor
    image = torch.zeros((N, N, 3), device=device)

    for i in range(N):
        a_slice = a[i].unsqueeze(-1)  # Shape: (N, N, 1)
        rgb_slice = torch.stack((r[i], g[i], b[i]), dim=-1)  # Shape: (N, N, 3)
        image = a_slice * rgb_slice + (1 - a_slice) * image  # Alpha blending

    image = torch.clamp(image, 0, 1)  # Ensure pixel values are in [0, 1]

    return image.cpu().numpy()  # Move back to CPU for further processing or saving

def main(filepath: str | os.PathLike, save_dir: str | os.PathLike = "./rendering_output", create_gif: bool = True) -> int:

    filepath_str = str(filepath)


    # Load datacube
    dataloader = DataLoader(virtual_stack=False)
    if filepath_str.endswith(".h5") or filepath_str.endswith(".hdf5"):
        datacube = dataloader.load_h5(filepath)
    elif filepath_str.endswith(".tif") or filepath_str.endswith(".tiff"):
        datacube = dataloader.load_tiff(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide an HDF5 or TIFF file.")
    
    datacube_t = torch.from_numpy(datacube).float().to(device)

    Nangles = 10


    for i in range(Nangles):
        angle = math.pi / 2 * i / Nangles
        image = render_view(datacube_t, angle)

        # Plot Volume Rendering
        plt.figure(figsize=(4, 4), dpi=80)
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(f"{save_dir}_{i}.png", dpi=240, bbox_inches="tight", pad_inches=0)
        plt.close()

    if create_gif:
        images = [imageio.imread(f"{save_dir}_{i}.png") for i in range(Nangles)]
        imageio.mimsave(f"{save_dir}.gif", images, duration=0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volume Rendering of a 3D Datacube using PyTorch")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the HDF5 or TIFF file")
    args = parser.parse_args()
    main(args.filepath)
        