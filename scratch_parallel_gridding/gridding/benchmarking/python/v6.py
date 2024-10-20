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
image_name = "v6.png"

print("**v6**")
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


def gridding_v6(uvwt, vist, freq):
    grid = np.zeros((image_size, image_size), dtype=np.complex128)
    uvw0 = uvwt[:,:,0]
    uvw1 = uvwt[:,:,1]

    uvw0 = np.expand_dims(uvw0, axis=1) # (512, 351, 1)
    uvw1 = np.expand_dims(uvw1, axis=1) # (512, 351, 1)
    freq = np.expand_dims(freq, axis=1) # (1, 256)

    iu = np.round(theta * uvw0 * freq / c).astype(int)
    iv = np.round(theta * uvw1 * freq / c).astype(int)
    iu_idx = iu + image_size // 2
    iv_idx = iv + image_size // 2

    vist = np.swapaxes(vist, 1, -1) # (512, 256, 351)
    np.add.at(grid, (iu_idx, iv_idx), vist)
    
    return grid


uvwt_np = dataset.UVW.compute().to_numpy()
vist_np = dataset.VISIBILITY.compute().to_numpy()
freq_np = dataset.frequency.compute().to_numpy()

start_gridding_time = time.perf_counter()
grid = gridding_v6(uvwt_np, vist_np, freq_np)
end_gridding_time = time.perf_counter()
print(f"gridding: {end_gridding_time - start_dataset_time}s")

start_fft_time = time.perf_counter()
image = fourier_transform(grid)
end_fft_time = time.perf_counter()
print(f"fourier transform: {end_fft_time - start_fft_time}s")

plot_image(image)
