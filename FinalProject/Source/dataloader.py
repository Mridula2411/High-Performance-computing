import tifffile
import h5py as h5
import numpy as np
import os


class DataLoader:
    def __init__(self, **kwargs):
        self.virtual_stack = kwargs.get("virtual_stack", False)


    def load_h5(self, path):
        """Load data from an HDF5 file."""
        with h5.File(path, "r") as f:
            data = np.array(f["density"])
        return data
    
    def load_tiff(self, path: str | os.PathLike) -> np.ndarray:
        """
        Load a TIFF file from the specified path.

        Args:
            path (str): The path to the TIFF file.

        Returns:
            numpy.ndarray or numpy.memmap: The loaded volume.
                If 'self.virtual_stack' is True, returns a numpy.memmap object.
        Adapted from qim3d (https://github.com/qim-center/qim3d/blob/main/qim3d/io/_loading.py)
        """
        # Get the number of TIFF series (some BigTIFF have multiple series)
        with tifffile.TiffFile(path) as tif:
            series = len(tif.series)

        if self.virtual_stack:
            vol = tifffile.memmap(path)
        else:
            vol = tifffile.imread(path, key=range(series) if series > 1 else None)

        vol = vol.astype(np.float32)

        # Ensure the loaded volume is at least as large as the renderer's camera grid.
        # The renderer uses N=180 by default; padding here prevents interpn xi-out-of-bounds.
        min_size = 256
        if any(s < min_size for s in vol.shape):
            new_shape = tuple(max(s, min_size) for s in vol.shape)
            padded = np.zeros(new_shape, dtype=np.float32)
            # center the original volume inside the padded volume
            insert_slices = []
            for s_old, s_new in zip(vol.shape, new_shape):
                start = (s_new - s_old) // 2
                insert_slices.append(slice(start, start + s_old))
            padded[tuple(insert_slices)] = vol
            vol = padded

        # avoid exact zeros to prevent log(0) in downstream code
        vol = vol + 1e-12
        return vol