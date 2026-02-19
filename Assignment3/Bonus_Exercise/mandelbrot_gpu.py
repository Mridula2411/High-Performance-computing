import torch

def mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter=100, use_gpu=True):

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    x = torch.linspace(x_min, x_max, width, device=device)
    y = torch.linspace(y_min, y_max, height, device=device)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    c = torch.complex(x_grid, y_grid)
    z = torch.zeros_like(c)


    image = torch.zeros(c.shape, dtype=torch.float32, device=device)

    mask = torch.ones(c.shape, dtype=torch.bool, device=device)


    for n in range(max_iter):

        z[mask] = z[mask] * z[mask] + c[mask]
        
        diverged = torch.abs(z) > 2

        new_divergences = diverged & mask
        image[new_divergences] = n


        mask &= ~diverged

    image[mask] = max_iter

    return image.cpu().numpy()