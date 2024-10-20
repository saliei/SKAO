#!/usr/bin/env python3

import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from numba import jit

# Speed of light
c = 299792458
# Size of image in pixels
image_size = 2048 
# Size of image on sky, directional cosines
theta = 0.0125
# Dataset path
dataset_path = "./example_simulation.zarr"
# Output image name
image_name = "test_flatten.png"

dataset = xr.open_zarr(dataset_path)

# Flatten UVW and VISIBILITY arrays
uvws_flat = dataset.UVW.values.reshape(-1, 3)  # Shape: (512*351, 3)
visibility_flat = dataset.VISIBILITY.values.reshape(-1, 256)  # Shape: (512*351, 256)

# Tile frequency array to match the length of flattened UVW and VISIBILITY
frequency_tile = np.tile(dataset.frequency.values, (uvws_flat.shape[0], 1))  # Shape: (512*351, 256)

# Create grid
grid = np.zeros((image_size, image_size), dtype=complex)

def gridding(uvws, visibilities, frequencies, grid):
    # Iterate over individual baselines (i.e. pair of antennas)
    for uvw, vis in zip(uvws, visibilities):
        # Iterate over frequencies
        for freq, viss in zip(frequencies, vis):
            # Calculate position in wavelengths
            iu = np.round(theta * uvw[0] * freq / c).astype(int)
            iv = np.round(theta * uvw[1] * freq / c).astype(int)
            # Accumulate visibility at approximate location in the grid
            grid[iu + image_size//2, iv + image_size//2] += viss
    return grid

grid = gridding(uvws_flat, visibility_flat, frequency_tile, grid)
image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real

plt.figure(figsize=(16, 16))
plt.imsave(image_name, image)  # You might want to specify a colormap for the image

