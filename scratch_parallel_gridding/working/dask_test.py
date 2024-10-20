#!/usr/bin/env python3

import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client, progress, LocalCluster

# Speed of light
c = 299792458
# Size of image in pixels
image_size = 2048 
# Size of image on sky, directional cosines
theta = 0.0125
# Dataset path
dataset_path = "./example_simulation.zarr"
# Output image name
image_name = "dask_test.png"


def grid_visibilities_dataset(uvws, visss, frequencies):
    grid = da.zeros((image_size, image_size), dtype=complex)
    # Calculate positions in wavelengths
    iu = (theta * uvws[:, :, 0] * frequencies[:, None] / c).round().astype(int)
    iv = (theta * uvws[:, :, 1] * frequencies[:, None] / c).round().astype(int)
    # Calculate global positions
    iu_global = iu + image_size // 2
    iv_global = iv + image_size // 2
    # Mask for valid positions
    mask = (iu_global >= 0) & (iu_global < image_size) & (iv_global >= 0) & (iv_global < image_size)
    # Transpose visss to match the shape of the mask
    # visss_transposed = visss.transpose(0, 2, 1)
    # Apply mask and accumulate visibility
    grid = da.map_blocks(lambda vis: np.where(mask, np.real(vis), 0), visss, dtype=grid.dtype)

    return grid


if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)

    # Open the Zarr dataset using Dask
    dataset = xr.open_zarr(dataset_path)
    dataset = dataset.isel(time=slice(0, 1))

    uvws = dataset.UVW.values
    visss = dataset.VISIBILITY.values
    frequencies = dataset.frequency.values

    grid = grid_visibilities_dataset(uvws, visss, frequencies)

    # Compute the result
    grid = grid.sum(axis=0).compute()

    # Perform inverse Fourier transform and get the real part of the result
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real

    # Plot and save the image
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

    client.close()
    cluster.close()

