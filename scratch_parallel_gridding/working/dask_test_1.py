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
    # Calculate positions in wavelengths
    iu = (theta * uvws[:, :, 0] * frequencies[:, None] / c).round().astype(int)
    iv = (theta * uvws[:, :, 1] * frequencies[:, None] / c).round().astype(int)
    # Calculate global positions
    iu_global = iu + image_size // 2
    iv_global = iv + image_size // 2
    # Mask for valid positions
    mask = (iu_global >= 0) & (iu_global < image_size) & (iv_global >= 0) & (iv_global < image_size)
    # Apply mask and accumulate visibility
    grid = da.zeros((image_size, image_size), dtype=complex, chunks=(256, 256))
    # for i in range(uvws.shape[0]):
        # grid = da.where(mask[i], visss[i].real, 0, out=grid)
    def apply_mask(vis):
        return np.where(mask, vis.real, 0)
    grid = da.map_blocks(apply_mask, visss, dtype=np.float64)
    return grid

if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)

    # Open the Zarr dataset using Dask
    dataset = xr.open_zarr(dataset_path)
    dataset = dataset.isel(time=slice(0, 1))

    uvws = da.from_array(dataset.UVW, chunks=(256, 175, 3))
    visss = da.from_array(dataset.VISIBILITY, chunks=(256, 175, 256))
    frequencies = da.from_array(dataset.frequency, chunks=(256,))

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

