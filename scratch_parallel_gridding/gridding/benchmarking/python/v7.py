#!/usr/bin/env python3

import time
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from numba import njit
from concurrent.futures import ThreadPoolExecutor, as_completed

# speed of light
c = 299792458
# size of image in pixels
image_size = 2048
# size of image on sky, directional cosines
theta = 0.0125
# dataset path
dataset_path = "../example_simulation.zarr"
image_name = "v6.png"
n_workers = 4

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


def sum_vis(grid, iu, iv, vis):
    np.add.at(grid, (iu, iv), vis)
    return grid

def gridding_v7(uvwt, vist, freq, n_workers=4):
    grid = np.zeros((image_size, image_size), dtype=np.complex128)
    uvw0 = uvwt[:,:,0]
    uvw1 = uvwt[:,:,1]

    iu = (theta * uvw0 * freq / c).round().astype(int)  # (512, 351, 256)
    iv = (theta * uvw1 * freq / c).round().astype(int)  # (512, 351, 256)
    iu_idx = iu + image_size // 2
    iv_idx = iv + image_size // 2

    # split data into chunks to be processed by a thread
    def chunk_data(data, n_chunks):
        chunk_size = data.shape[0] // n_chunks
        return [data[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]

    iu_chunks  = chunk_data(iu_idx, n_workers)
    iv_chunks  = chunk_data(iv_idx, n_workers)
    vis_chunks = chunk_data(vist, n_workers)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for iu_chunk, iv_chunk, vis_chunk in zip(iu_chunks, iv_chunks, vis_chunks):
            futures.append(executor.submit(sum_vis, grid.copy(), \
                                           iu_chunk.compute().data, \
                                           iv_chunk.compute().data, \
                                           vis_chunk.compute().data))

        for future in as_completed(futures):
            grid += future.result()

    return grid


uvwt = dataset.UVW
vist = dataset.VISIBILITY
freq = dataset.frequency.data

start_gridding_time = time.perf_counter()
grid = gridding_v7(uvwt, vist, freq, n_workers=n_workers)
end_gridding_time = time.perf_counter()
print(f"gridding: {end_gridding_time - start_dataset_time}s")

start_fft_time = time.perf_counter()
image = fourier_transform(grid)
end_fft_time = time.perf_counter()
print(f"fourier transform: {end_fft_time - start_fft_time}s")

plot_image(image)
