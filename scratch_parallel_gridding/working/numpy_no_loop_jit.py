#!/usr/bin/env python3

import time
import functools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from numba import jit, njit, prange, vectorize

# Speed of light
c = 299792458
# Size of image in pixels
image_size = 2048 
# Size of image on sky, directional cosines
theta = 0.0125
# Dataset path
dataset_path = "./example_simulation.zarr"
# Output image name
image_name = "numpy_no_loop.png"

def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"In function: {func.__name__} ...")
        start = time.perf_counter()
        return_value = func(*args, **kwargs)
        end = time.perf_counter()
        spent_time = end - start
        print(f"{func.__name__}: {spent_time:10.8f} seconds")
        print("")
        return return_value
    return wrapper

@njit(parallel=True)
def add_at_block_numba(grid_block, iu_block, iv_block, vis_block):
    for i in prange(iu_block.shape[0]):
        for j in range(iu_block.shape[1]):
            for k in range(iu_block.shape[2]):
                grid_block[iu_block[i, j, k], iv_block[i, j, k]] += vis_block[i, j, k]
    return grid_block

@profile
def grid_visibilities_all_timesteps(uvwt, vist, frequencies):
    grid = np.zeros((image_size, image_size), dtype=complex)

    uvw_x = uvwt[:,:,0]
    uvw_y = uvwt[:,:,1]

    uvw_x = np.expand_dims(uvw_x, axis=1)
    uvw_y = np.expand_dims(uvw_y, axis=1)
    freq  = np.expand_dims(frequencies, axis=1)

    iu = np.round(theta * uvw_x * freq / c).astype(int)
    iv = np.round(theta * uvw_y * freq / c).astype(int)

    iu_global = iu + image_size // 2
    iv_global = iv + image_size // 2

    vist = np.swapaxes(vist, 1, -1)

    grid = add_at_block_numba(grid, iu_global, iv_global, vist)

    return grid

@profile
def grid_visibilities_all_timesteps_threads(uvwt, vist, frequencies, n_workers=4):
    grid = np.zeros((image_size, image_size), dtype=complex)

    uvw_x = uvwt[:,:,0]
    uvw_y = uvwt[:,:,1]

    uvw_x = np.expand_dims(uvw_x, axis=1)
    uvw_y = np.expand_dims(uvw_y, axis=1)
    freq  = np.expand_dims(frequencies, axis=1)

    iu = np.round(theta * uvw_x * freq / c).astype(int)
    iv = np.round(theta * uvw_y * freq / c).astype(int)

    iu_global = iu + image_size // 2
    iv_global = iv + image_size // 2

    vist = np.swapaxes(vist, 1, -1)

    iu_chunks = chunk_data(iu_global, n_workers)
    iv_chunks = chunk_data(iv_global, n_workers)
    vis_chunks = chunk_data(vist, n_workers)

    grids = []

    for iu_chunk, iv_chunk, vis_chunk in zip(iu_chunks, iv_chunks, vis_chunks):
        grids.append(add_at_block_numba(np.zeros_like(grid), iu_chunk, iv_chunk, vis_chunk))

    for grid_chunk in grids:
        grid += grid_chunk

    return grid

def chunk_data(data, n_chunks):
    chunk_size = data.shape[0] // n_chunks
    return [data[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]

if __name__ == "__main__":
    dataset = xr.open_zarr(dataset_path)
    uvws = dataset.UVW.compute().values
    visss = dataset.VISIBILITY.compute().values
    frequencies = dataset.frequency.compute().values

    # grid = grid_visibilities_all_timesteps_threads(uvws, visss, frequencies, 4)
    grid = grid_visibilities_all_timesteps(uvws, visss, frequencies)

    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real

    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

