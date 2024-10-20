#!/usr/bin/env python3

import time
import functools
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from numba import jit

# speed of light
c = 299792458
# size of image in pixels
image_size = 2048 
# size of image on sky, directional cosines
theta = 0.0125
# dataset path
dataset_path = "./example_simulation.zarr"
# output image name
image_name = "mpi_time_domain.png"

def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"in function: {func.__name__} ...")
        start = time.perf_counter()
        return_value = func(*args, **kwargs)
        end = time.perf_counter()
        spent_time = end - start
        print(f"{func.__name__}: {spent_time:10.8f}")
        print("")
        return return_value
    return wrapper


@profile
def open_dataset(filepath):
    dataset = xr.open_zarr(filepath)
    return dataset


@profile
def fourier_transform(grid):
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    return image

@profile
def save_image(image):
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

def gridding_single_timestep(uvwt, vist, freq, grid):
    for (uvw, vis) in zip(uvwt, vist):
        for(f, vi) in zip(freq, vis):
            iu = round(theta * uvw[0] * f / c)
            iv = round(theta * uvw[1] * f / c)
            grid[iu + image_size//2, iv + image_size//2] += vi
    return grid

@profile
def gridding_compact_no_numpy(dataset):
    grid = np.zeros((image_size, image_size), dtype=complex)
    for t in range(len(dataset.time)):
        uvwt = dataset.UVW[t].compute()
        vist = dataset.VISIBILITY[t].compute()
        freq = dataset.frequency.compute()

        grid = gridding_single_timestep(uvwt, vist, freq, grid)
    return grid

@profile
def gridding_compact_to_numpy(dataset):
    grid = np.zeros((image_size, image_size), dtype=complex)
    for t in range(len(dataset.time)):
        uvwt = dataset.UVW[t].compute().to_numpy()
        vist = dataset.VISIBILITY[t].compute().to_numpy()
        freq = dataset.frequency.compute().to_numpy()

        grid = gridding_single_timestep(uvwt, vist, freq, grid)
    return grid

@profile
def gridding_loop_no_zip(dataset):
    grid = np.zeros((image_size, image_size), dtype=complex)

    uvws = dataset.UVW
    visss = dataset.VISIBILITY
    freqs = dataset.frequency

    num_baselines = len(dataset.time)
    num_timesteps = len(dataset.baseline_id)

    for t in range(num_timesteps):
        for b in range(num_baselines):
            uvw = uvws[t, b]
            viss = visss[t, b]
            for idx, f in enumerate(freqs):
                vis = viss[idx]
                iu = (theta * uvw[0] * f / c).round().astype(int)
                iv = (theta * uvw[1] * f / c).round().astype(int)
                iu_global = iu + image_size//2
                iv_global = iv + image_size//2
                grid[iu_global, iv_global] += vis
    return grid

@profile
def main():
    dataset = open_dataset(dataset_path)
    dataset = dataset.isel(time=slice(0, 50))

    grid = gridding_loop_no_zip(dataset)
    image = fourier_transform(grid)
    save_image(image)

if __name__ == "__main__":
    main()

