#!/usr/bin/env python3

import time
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from numba import njit

# speed of light
c = 299792458
# size of image in pixels
image_size = 2048
# size of image on sky, directional cosines
theta = 0.0125
# dataset path
dataset_path = "../example_simulation.zarr"
image_name = "v5.png"

print("**v5**")
start_dataset_time = time.perf_counter()
dataset = open_dataset(dataset_path)
end_dataset_time = time.perf_counter()
print(f"openning dataset: {end_dataset_time - start_dataset_time}s")
num_timesteps = len(dataset.time)
num_baselines = len(dataset.baseline_id)

def open_dataset(dataset_path):
    dataset = xr.open_zarr(dataset_path)
    return dataset

def fourier_transform(grid):
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    return image

def plot_image(image):
    plt.figure(figsize=(8,8))
    plt.imsave(image_name ,image)


@njit
def gridding_single_timestep_v5(grid, uvwb, visb, freq):
    uvw0 = uvwb[:,0]
    uvw1 = uvwb[:,1]
    uvw0 = np.expand_dims(uvw0, axis=0) # (1, 351)
    uvw1 = np.expand_dims(uvw1, axis=0) # (1, 351)
    freq = np.expand_dims(freq, axis=1) # (256, 1)
    visb = np.swapaxes(visb, 0, -1)     # (256, 351)

    iu = np.round(theta * uvw0 * freq / c).astype(np.int32)
    iv = np.round(theta * uvw1 * freq / c).astype(np.int32)
    iu_idx = iu + image_size // 2
    iv_idx = iv + image_size // 2

    for i in range(num_timesteps):
        for j in range(num_baselines):
            grid[iu_idx[i, j], iv_idx[i, j]] += visb[i, j]

    return grid

@profile
def gridding_v5(uvwt, vist, freq):
    grid = np.zeros((image_size, image_size), dtype=np.complex128)
    for t in range(num_timesteps):
        grid_timestep = gridding_single_timestep_v5(grid, uvwt[t].compute().data, vist[t].compute().data, freq)
        grid += grid_timestep
    return grid


uvwt = dataset.UVW
vist = dataset.VISIBILITY
freq = dataset.frequency.data

start_gridding_time = time.perf_counter()
grid = gridding_v5(uvwt, vist, freq)
end_gridding_time = time.perf_counter()
print(f"gridding: {end_gridding_time - start_dataset_time}s")

start_fft_time = time.perf_counter()
image = fourier_transform(grid)
end_fft_time = time.perf_counter()
print(f"fourier transform: {end_fft_time - start_fft_time}s")

plot_image(image)
