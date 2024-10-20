#!/usr/bin/env python3

import time
import functools
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
import dask.array as da

# speed of light
c = 299792458
# size of image in pixels
image_size = 2048 
# size of image on sky, directional cosines
theta = 0.0125
# dataset path
dataset_path = "./example_simulation.zarr"
# output image name
image_name = "simple_flat.png"

def open_dataset(filepath):
    dataset = xr.open_zarr(filepath)
    return dataset

def calculate_indices(uvw, f):
    iu = xr.DataArray.round(theta * uvw[0] * f / c).astype(int)
    iv = xr.DataArray.round(theta * uvw[1] * f / c).astype(int)
    iu_idx = iu + image_size//2
    iv_idx = iv + image_size//2
    return iu_idx, iv_idx

def gridding_single_timestep(uvwt, vist, freq, grid):
    for (uvw, vis) in zip(uvwt, vist):
        for(f, vi) in zip(freq, vis):
            iu_idx, iv_idx = calculate_indices(uvw, f)
            grid[iu_idx, iv_idx] += vi
    return grid

def gridding_compact(dataset, grid):
    for t in range(len(dataset.time)):
        uvwt = dataset.UVW[t]
        vist = dataset.VISIBILITY[t]
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
    dataset = open_dataset(dataset_path)
    dataset = dataset.isel(time=slice(0, 50))
    grid = np.zeros((image_size, image_size), dtype=complex)

    grid = gridding_compact(dataset, grid)

    image = fourier_transform(grid)
    save_image(image)

if __name__ == "__main__":
    main()

