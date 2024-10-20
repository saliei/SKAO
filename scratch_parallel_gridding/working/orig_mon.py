#!/usr/bin/env python3

import time
import functools
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from numba import jit, njit, vectorize
import psutil
import threading

# speed of light
c = 299792458
# size of image in pixels
image_size = 2048 
# size of image on sky, directional cosines
theta = 0.0125
# dataset path
dataset_path = "./example_simulation.zarr"
# output image name
image_name = "simple.png"

dataset = xr.open_zarr(dataset_path)
dataset = dataset.isel(time=slice(0, 12))

def get_uvw_vis_freq_numpy(dataset):
    UVW = dataset.UVW.compute().to_numpy()
    VIS = dataset.VISIBILITY.compute().to_numpy()
    FRQ = dataset.frequency.data
    return UVW, VIS, FRQ

def get_uvw_vis_freq_no_numpy(dataset):
    UVW = dataset.UVW
    VIS = dataset.VISIBILITY
    FRQ = dataset.frequency.data
    return UVW, VIS, FRQ

UVW, VIS, FRQ = get_uvw_vis_freq_numpy(dataset)
# UVW, VIS, FRQ = get_uvw_vis_freq_no_numpy(dataset)

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

# Monitoring function
def monitor_usage(interval=1):
    pid = psutil.Process().pid
    p = psutil.Process(pid)
    with open("usage.log", "w") as f:
        while True:
            cpu = p.cpu_percent(interval=interval)
            memory = p.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
            f.write(f"CPU: {cpu}%, Memory: {memory}MB\n")
            f.flush()

# Start monitoring in a separate thread
monitor_thread = threading.Thread(target=monitor_usage, daemon=True)
monitor_thread.start()

# @profile
def calculate_indices_noround(uvw_0, uvw_1, f):
    # uvw_0 = uvw_0.values
    # uvw_1 = uvw_1.values
    iu = round(theta * uvw_0 * f / c)
    iv = round(theta * uvw_1 * f / c)
    iu_idx = iu + image_size//2
    iv_idx = iv + image_size//2
    return iu_idx, iv_idx

# @profile
def calculate_indices(uvw_0, uvw_1, f):
    iu = round(theta * uvw_0 * f / c)
    iv = round(theta * uvw_1 * f / c)
    iu_idx = iu + image_size//2
    iv_idx = iv + image_size//2
    return iu_idx, iv_idx

def gridding_freq_idx(uvw, viss, freq, grid):
    for (f, v) in zip(freq, viss):
        iu_idx, iv_idx = calculate_indices_noround(uvw[0], uvw[1], f)
        grid[iu_idx, iv_idx] += v
    return grid

def gridding_freq(uvw, viss, freq, grid):
    for (f, v) in zip(freq, viss):
        iu = round(theta * uvw[0] * f / c)
        iv = round(theta * uvw[1] * f / c)
        grid[iu + image_size//2, iv + image_size//2] += v
    return grid

def gridding_all(UVW, VIS, FRQ, grid):
    for (uvws, visss) in zip(UVW, VIS):
        for (uvw, viss) in zip(uvws, visss):
            grid = gridding_freq_idx(uvw, viss, FRQ, grid)
    return grid

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

