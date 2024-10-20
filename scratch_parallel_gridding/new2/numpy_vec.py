#!/usr/bin/env python3

import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt

# Constants
c = 299792458
image_size = 2048
theta = 0.0125
dataset_path = "./example_simulation.zarr"
image_name = "numpy_vec.png"

# Load dataset
dataset = xr.open_zarr(dataset_path)
uvws = dataset.UVW.compute().data
visss = dataset.VISIBILITY.compute().data
frequencies = dataset.frequency.data

# Initialize grid
grid = np.zeros((image_size, image_size), dtype=complex)

# Function to calculate grid indices for visibility positions
def calculate_indices(uvw, freq):
    iu = np.round(theta * uvw[..., 0] * freq[:, np.newaxis, np.newaxis] / c).astype(int)
    iv = np.round(theta * uvw[..., 1] * freq[:, np.newaxis, np.newaxis] / c).astype(int)
    return iu, iv

# Main function
def main():
    # Calculate indices for grid
    iu, iv = calculate_indices(uvws, frequencies)

    # Flatten indices and visibility data for efficient accumulation
    flat_iu = (iu + image_size//2).ravel()
    flat_iv = (iv + image_size//2).ravel()
    flat_vis = visss.ravel()

    # Accumulate visibility at approximate locations in the grid
    np.add.at(grid, (flat_iu, flat_iv), flat_vis)

    # Perform Fourier transform
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real

    # Save the resulting image
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

if __name__ == "__main__":
    main()

