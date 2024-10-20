#!/usr/bin/env python3

import time
import functools
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from mpi4py import MPI

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
def gridding_compact(dataset):
    grid = np.zeros((image_size, image_size), dtype=complex)
    for t in range(len(dataset.time)):
        uvwt = dataset.UVW[t].compute().to_numpy()
        vist = dataset.VISIBILITY[t].compute().to_numpy()
        freq = dataset.frequency.compute().to_numpy()

        grid = gridding_single_timestep(uvwt, vist, freq, grid)
    return grid

def gridding_mpi(dataset, comm):
    grid = np.zeros((image_size, image_size), dtype=complex)
    size = comm.Get_size()
    rank = comm.Get_rank()

    num_baselines = len(dataset.baseline_id)
    num_timestamps = len(dataset.time)

    mpi_chunk_size = num_baselines // size
    start = rank * mpi_chunk_size
    if rank < size - 1:
        end = start + mpi_chunk_size
    else:
        end = num_baselines - 1

    for t in range(num_timestamps):
        uvwt_local = dataset.UVW[t].compute().to_numpy()[start:end]
        vist_local = dataset.VISIBILITY[t].compute().to_numpy()[start:end]
        freq = dataset.frequency.compute().to_numpy() # same for all
        grid_local = gridding_single_timestep(uvwt_local, vist_local, freq, )


    for t in range(len(dataset.time)):
        uvwt = dataset.UVW[t].compute().to_numpy()
        vist = dataset.VISIBILITY[t].compute().to_numpy()
        freq = dataset.frequency.compute().to_numpy()

        grid = gridding_single_timestep(dataset, uvwt, vist, freq, grid)
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
def main():
    dataset = open_dataset(dataset_path)
    dataset = dataset.isel(time=slice(0, 50))

    comm = MPI.COMM_WORLD

    # grid = gridding_compact(dataset)
    grid = gridding_mpi(dataset, comm)

    image = fourier_transform(grid)
    save_image(image)

if __name__ == "__main__":
    main()

