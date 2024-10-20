#!/usr/bin/env python3

import time
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from numba import vectorize

# speed of light
c = 299792458
# size of image in pixels
image_size = 2048 
# size of image on sky, directional cosines
theta = 0.0125
# dataset path
dataset_path = "./example_simulation.zarr"
# output image name
image_name = "ufunc.png"

def open_dataset(filepath):
    dataset = xr.open_zarr(filepath)
    return dataset

@vectorize
def calculate_indices(uvw0, uvw1, f):
    iu = round(theta * uvw0 * f / c)
    iv = round(theta * uvw1 * f / c)
    iu_idx = iu + image_size//2
    iv_idx = iv + image_size//2
    return iu_idx, iv_idx

def gridding_single_timestep(uvwt, vist, freq, grid):
    uvw = uvwt.compute()
    vis = vist.compute().to_numpy()
    freq = freq.compute().to_numpy()
    iu_idx, iv_idx = calculate_indices(uvw[:, 0], uvw[:, 1], freq)
    for i in range(len(uvw)):
        grid[iu_idx[i], iv_idx[i]] += vis[i]
    return grid

def gridding_compact(dataset):
    grid = np.zeros((image_size, image_size), dtype=complex)
    uvwt = dataset.UVW.compute()
    vist = dataset.VISIBILITY.compute()
    freq = dataset.frequency.compute()
    grid = gridding_single_timestep(uvwt, vist, freq, grid)
    return grid

def fourier_transform(grid):
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    return image

def save_image(image):
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

def main():
    dataset = open_dataset(dataset_path)
    dataset = dataset.isel(time=slice(0, 50))

    grid = gridding_compact(dataset)

    image = fourier_transform(grid)
    save_image(image)

if __name__ == "__main__":
    main()

