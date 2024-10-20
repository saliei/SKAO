#!/usr/bin/env python3

import time
import functools
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from numba import jit, njit, vectorize

# speed of light
c = 299792458
# size of image in pixels
image_size = 2048 
# size of image on sky, directional cosines
theta = 0.0125
# dataset path
dataset_path = "./example_simulation.zarr"
# output image name
image_name = "jitting.png"

dataset = xr.open_zarr(dataset_path)
# dataset = dataset.isel(time=slice(0, 100))

UVW = dataset.UVW.compute().to_numpy()
VIS = dataset.VISIBILITY.compute().to_numpy()
FRQ = dataset.frequency.compute().to_numpy()

grid = np.zeros((image_size, image_size), dtype=complex)

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

@jit(fastmath=True)
def calculate_indices(uvw_0, uvw_1, f):
    iu = round(theta * uvw_0 * f / c)
    iv = round(theta * uvw_1 * f / c)
    iu_idx = iu + image_size//2
    iv_idx = iv + image_size//2
    return iu_idx, iv_idx

@jit
def gridding_freq_idx(uvw, viss, freq, grid):
    for (f, v) in zip(freq, viss):
        iu_idx, iv_idx = calculate_indices(uvw[0], uvw[1], f)
        grid[iu_idx, iv_idx] += v
    return grid

@jit(nopython=True, fastmath=True, parallel=True)
def gridding_freq(uvw, viss, freq, grid):
    for (f, v) in zip(freq, viss):
        iu = round(theta * uvw[0] * f / c)
        iv = round(theta * uvw[1] * f / c)
        grid[iu + image_size//2, iv + image_size//2] += v
    return grid

@jit
def gridding_all(UVW, VIS, FRQ, grid):
    for (uvws, visss) in zip(UVW, VIS):
        for (uvw, viss) in zip(uvws, visss):
            # grid = gridding_freq(uvw, viss, FRQ, grid)
            grid = gridding_freq_idx(uvw, viss, FRQ, grid)
    return grid

@jit
def gridding_compact(UVW, VIS, FRQ, grid):
    for (uvws, visss) in zip(UVW, VIS):
        for (uvw, viss) in zip(uvws, visss):
            for (freq, vis) in zip(FRQ, viss):
                iu = round(theta * uvw[0] * freq / c)
                iv = round(theta * uvw[1] * freq / c)
                grid[iu + image_size//2, iv + image_size//2] += vis
    return grid


def do_fft(grid):
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    return image

grid = gridding_all(UVW, VIS, FRQ, grid)
# grid = gridding_compact(UVW, VIS, FRQ, grid)

image = do_fft(grid)

plt.figure(figsize=(16, 16))
plt.imsave(image_name, image)
