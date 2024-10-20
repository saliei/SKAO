#!/usr/bin/env python3

import time
import functools
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from mpi4py import MPI
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

def gridding_single_timestep(uvwt, vist, freq, grid):
    for (uvw, vis) in zip(uvwt, vist):
        for(f, vi) in zip(freq, vis):
            iu = round(theta * uvw[0] * f / c)
            iv = round(theta * uvw[1] * f / c)
            grid[iu + image_size//2, iv + image_size//2] += vi
    return grid

@profile
def fourier_transform(grid):
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    return image

@profile
def save_image(image):
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

@profile
def gridding_compact(dataset):
    grid = np.zeros((image_size, image_size), dtype=complex)
    for t in range(len(dataset.time)):
        uvwt = dataset.UVW[t].compute().to_numpy()
        vist = dataset.VISIBILITY[t].compute().to_numpy()
        freq = dataset.frequency.compute().to_numpy()

        grid = gridding_single_timestep(uvwt, vist, freq, grid)
    return grid

def gridding_mpi_timestep(dataset, size, rank):

    num_timestamps = len(dataset.time)
    num_baselines = len(dataset.baseline_id)

    chunk_size = num_timestamps // size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank < size - 1 else num_timestamps

    grid_local = np.zeros((image_size, image_size), dtype=complex)

    for t in range(start_idx, end_idx):
        uvws = dataset.UVW[t]
        visss = dataset.VISIBILITY[t]

        for (uvw, viss) in zip(uvws.compute().data, visss.compute().data):
            for (freq, vis) in zip(dataset.frequency.data, viss):
                iu = round(theta * uvw[0] * freq / c)
                iv = round(theta * uvw[1] * freq / c)
                iu_global = iu + image_size // 2
                iv_global = iv + image_size // 2
                if 0 <= iu_global < image_size and 0 <= iv_global < image_size:
                    grid_local[iu_global, iv_global] += vis
    return grid_local

def gridding_mpi_timestep_no_numpy(dataset, size, rank):

    num_timesteps = len(dataset.time)
    num_baselines = len(dataset.baseline_id)

    chunk_size = num_timesteps // size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank < size - 1 else num_timesteps

    grid_local = np.zeros((num_timesteps ,image_size, image_size), dtype=complex)

    for t in range(start_idx, end_idx):
        uvws = dataset.UVW[t]
        visss = dataset.VISIBILITY[t]
        freqs = dataset.frequency
        for (uvw, viss) in zip(uvws, visss):
            for (freq, vis) in zip(freqs, viss):
                iu = (theta * uvw[0] * freq / c).round().astype(int)
                iv = (theta * uvw[1] * freq / c).round().astype(int)
                iu_global = iu + image_size // 2
                iv_global = iv + image_size // 2
                vi_reshaped = np.zeros((num_timesteps, image_size, image_size), dtype=complex)
                vi_reshaped[:, iu_global, iv_global] = vis
                # if 0 <= iu_global < image_size and 0 <= iv_global < image_size:
                    # grid_local[iu_global, iv_global] += vis
                grid_local += vi_reshaped

    return grid_local

def gridding_mpi_baseline(dataset, size, rank):

    num_timestamps = len(dataset.time)
    num_baselines = len(dataset.baseline_id)

    chunk_size = num_baselines // size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank < size - 1 else num_baselines

    grid_local = np.zeros((image_size, image_size), dtype=complex)

    freq = dataset.frequency.compute().to_numpy() # same for all

    for t in range(num_timestamps):
        uvwt_local = dataset.UVW[t].compute().to_numpy()[start_idx:end_idx]
        vist_local = dataset.VISIBILITY[t].compute().to_numpy()[start_idx:end_idx]

        for (uvw, vis) in zip(uvwt_local, vist_local):
            for(f, vi) in zip(freq, vis):
                iu = round(theta * uvw[0] * f / c)
                iv = round(theta * uvw[1] * f / c)
                grid_local[iu + image_size//2, iv + image_size//2] += vi

    return grid_local


@profile
def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    dataset = open_dataset(dataset_path)
    dataset = dataset.isel(time=slice(0, 12))

    # grid_local = gridding_mpi_timestep(dataset, size, rank)
    grid_local = gridding_mpi_timestep_no_numpy(dataset, size, rank)

    grid_global = comm.gather(grid_local, root=0)
    if rank == 0:
        grid_combined = np.sum(grid_global, axis=0)
        image = fourier_transform(grid_combined)
        save_image(image)

if __name__ == "__main__":
    main()

