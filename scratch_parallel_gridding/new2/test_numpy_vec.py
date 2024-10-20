#!/usr/bin/env python3

import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt

# Speed of light
c = 299792458
# Size of image in pixels
image_size = 2048 
# Size of image on sky, directional cosines
theta = 0.0125
# Dataset path
dataset_path = "./example_simulation.zarr"
# Output image name
image_name = "image.png"

dataset = xr.open_zarr(dataset_path)

# Flatten UVW and VISIBILITY arrays
uvws_flat = dataset.UVW.values.reshape(-1, 3)  # Shape: (512*351, 3)
visibility_flat = dataset.VISIBILITY.values.reshape(-1, 256)  # Shape: (512*351, 256)

# Tile frequency array to match the length of flattened UVW and VISIBILITY
frequency_tile = np.tile(dataset.frequency.values, (uvws_flat.shape[0], 1))  # Shape: (512*351, 256)

# Create grid
grid = np.zeros((image_size, image_size), dtype=complex)

def calculate_indices(uvw, freq):
    iu = np.round(theta * uvw[0] * freq / c).astype(np.int32)
    iv = np.round(theta * uvw[1] * freq / c).astype(np.int32)
    return iu, iv

def gridding_chunk(uvws, visibilities, frequencies, grid, chunk_size=1000):
    num_chunks = len(uvws) // chunk_size + 1
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(uvws))
        uvws_chunk = uvws[start:end]
        visibilities_chunk = visibilities[start:end]
        frequencies_chunk = frequencies[start:end]
        # Calculate indices for the grid
        iu, iv = calculate_indices(uvws_chunk.T, frequencies_chunk[..., None])
        # Shift indices to be centered at image_size//2
        iu_shifted = iu + image_size//2
        iv_shifted = iv + image_size//2
        # Ensure indices are within grid bounds
        valid_indices = (iu_shifted >= 0) & (iu_shifted < image_size) & (iv_shifted >= 0) & (iv_shifted < image_size)
        # Accumulate visibility at approximate location in the grid
        np.add.at(grid, (iu_shifted[valid_indices], iv_shifted[valid_indices]), visibilities_chunk[valid_indices])
    return grid

grid = gridding_chunk(uvws_flat, visibility_flat, frequency_tile, grid)
image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real

plt.figure(figsize=(16, 16))
plt.imsave(image_name, image, cmap='gray')  # You might want to specify a colormap for the image

