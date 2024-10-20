#!/usr/bin/env python3

import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from numba import njit, jit

# Constants
c = 299792458
image_size = 2048
theta = 0.0125
dataset_path = "./example_simulation.zarr"
image_name = "image.png"

# Numba JIT-compiled function to calculate grid indices for visibility positions
# @jit(parallel=True)
def calculate_indices(uvw, freq):
    iu = np.zeros((uvw.shape[0], uvw.shape[1]), dtype=np.int32)
    iv = np.zeros((uvw.shape[0], uvw.shape[1]), dtype=np.int32)
    for i in range(uvw.shape[0]):
        for j in range(uvw.shape[1]):
            iu[i, j] = np.round(theta * uvw[i, j, 0] * freq / c).astype(np.int32)
            iv[i, j] = np.round(theta * uvw[i, j, 1] * freq / c).astype(np.int32)
    return iu, iv

# Numba JIT-compiled function to accumulate visibility at approximate locations in the grid
@jit(parallel=True)
def accumulate_visibility(grid, flat_iu, flat_iv, flat_vis):
    for i in range(len(flat_iu)):
        grid[flat_iu[i], flat_iv[i]] += flat_vis[i]

# Main function
def main():
    # Load dataset
    dataset = xr.open_zarr(dataset_path)
    uvws = dataset.UVW.compute().data
    visss = dataset.VISIBILITY.compute().data
    frequencies = dataset.frequency.data

    # Initialize grid
    grid = np.zeros((image_size, image_size), dtype=complex)

    # Calculate indices for grid
    iu, iv = calculate_indices(uvws, frequencies)

    # Flatten indices and visibility data for efficient accumulation
    flat_iu = (iu + image_size//2).ravel()
    flat_iv = (iv + image_size//2).ravel()
    flat_vis = visss.ravel()

    # Accumulate visibility at approximate locations in the grid
    accumulate_visibility(grid, flat_iu, flat_iv, flat_vis)

    # Perform Fourier transform
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real

    # Save the resulting image
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

if __name__ == "__main__":
    main()

