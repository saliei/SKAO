#!/usr/bin/env python3

import time
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client, LocalCluster

# Speed of light
c = 299792458
# Size of image in pixels
image_size = 2048 
# Size of image on sky, directional cosines
theta = 0.0125
# Dataset path
dataset_path = "./example_simulation.zarr"
# Output image name
image_name = "test_dask.png"

def open_dataset(filepath):
    dataset = xr.open_zarr(filepath)
    return dataset

def calculate_indices(uvw, f):
    iu = theta * uvw[0] * f / c
    iv = theta * uvw[1] * f / c
    iu_round = np.round(iu)
    iv_round = np.round(iv)
    iu_idx = iu_round + image_size // 2
    iv_idx = iv_round + image_size // 2
    return iu_idx, iv_idx

def gridding_single_timestep(uvwt, vist, freq, grid):
    for (uvw, vis) in zip(uvwt, vist):
        for (f, vi) in zip(freq, vis):
            iu_idx, iv_idx = calculate_indices(uvw, f)
            grid[iu_idx, iv_idx] += vi
    return grid

def gridding_compact(dataset):
    grid = da.zeros((image_size, image_size), dtype=complex)
    for t in range(len(dataset.time)):
        uvwt = dataset.UVW[t].compute()
        vist = dataset.VISIBILITY[t].compute()
        freq = dataset.frequency.data
        grid = gridding_single_timestep(uvwt, vist, freq, grid)
    return grid

def fourier_transform(grid):
    image = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(grid))).real
    return image

def save_image(image):
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

def main():
    cluster = LocalCluster()  # Start a local Dask cluster
    client = Client(cluster)  # Connect to the cluster
    dataset = open_dataset(dataset_path)
    dataset = dataset.isel(time=slice(0, 50))

    grid = gridding_compact(dataset)
    # grid = grid.compute()

    image = fourier_transform(grid)
    image = image.compute()

    save_image(image)

    client.close()  # Close the client
    cluster.close()  # Close the cluster

if __name__ == "__main__":
    main()

