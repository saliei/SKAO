#!/usr/bin/env python3

import time
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt

# speed of light
c = 299792458
# size of image in pixels
image_size = 2048
# size of image on sky, directional cosines
theta = 0.0125
# dataset path
dataset_path = "../example_simulation.zarr"

image_name = "v0.png"

print("**v0**")
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

def gridding_v0(uvwt, vist, frq):
    grid = np.zeros((image_size, image_size), dtype=np.complex128)

    for (uvwb, visb) in zip(uvwt, vist):
        for(uvw, vis) in zip(uvwb.compute().data, visb.compute().data):
            for(fq, vi) in zip(frq, vis):
                iu_idx, iv_idx = calculate_indices(uvw[0], uvw[1], fq)
                grid[iu_idx, iv_idx] += vi

    return grid

uvwt = dataset.UVW
vist = dataset.VISIBILITY
freq = dataset.frequency.data

start_gridding_time = time.perf_counter()
grid = gridding_v0(uvwt, vist, freq)
end_gridding_time = time.perf_counter()
print(f"gridding: {end_gridding_time - start_dataset_time}s")

start_fft_time = time.perf_counter()
image = fourier_transform(grid)
end_fft_time = time.perf_counter()
print(f"fourier transform: {end_fft_time - start_fft_time}s")

plot_image(image)

