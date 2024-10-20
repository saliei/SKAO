#!/usr/bin/env python3

import time
import functools
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
import dask as ds
import concurrent.futures
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
image_name = "image.png"

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
    # dataset = dataset.isel(time=slice(0, 10))
    return dataset

@profile
def gridding_parallel_mpi(dataset):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    grid = np.zeros((image_size, image_size), dtype=complex)
    num_time_steps = len(dataset.time)
    steps_per_rank = num_time_steps // size
    remainder = num_time_steps % size

    # Determine the start and end indices for this rank's time steps
    start_idx = rank * steps_per_rank + min(rank, remainder)
    end_idx = start_idx + steps_per_rank + (1 if rank < remainder else 0)

    # Iterate over the assigned time steps
    for i in range(start_idx, end_idx):
        uvws = dataset.UVW.compute().data
        visss = dataset.VISIBILITY.compute().data

        for (uvw, viss) in zip(uvws, visss):
            for (freq, vis) in zip(dataset.frequency.data, viss):
                iu = round(theta * uvw[0] * freq / c)
                iv = round(theta * uvw[1] * freq / c)
                grid[iu + image_size//2, iv + image_size//2] += vis

    # Gather results from all ranks
    all_grids = comm.gather(grid, root=0)

    # Combine grids on root rank
    if rank == 0:
        final_grid = np.zeros((image_size, image_size), dtype=complex)
        for g in all_grids:
            final_grid += g
    else:
        final_grid = None

    return final_grid



@profile
def gridding_parallel(dataset):
    grid = np.zeros((image_size, image_size), dtype=complex)
    
    def process_time_step(uvws, visss):
        nonlocal grid
        for (uvw, viss) in zip(uvws.compute().data, visss.compute().data):
            for (freq, vis) in zip(dataset.frequency.data, viss):
                iu = round(theta * uvw[0] * freq / c)
                iv = round(theta * uvw[1] * freq / c)
                grid[iu + image_size//2, iv + image_size//2] += vis

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for uvws, visss in zip(dataset.UVW, dataset.VISIBILITY):
            futures.append(executor.submit(process_time_step, uvws, visss))
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    return grid


@profile
def gridding(dataset):
    grid = np.zeros((image_size, image_size), dtype=complex)
    
    # iterate over time (baseline UW coordinates rotate with earth)
    for (uvws, visss) in zip(dataset.UVW, dataset.VISIBILITY):
        # iterate over individual baselines (i.e. pair of antennas)
        for (uvw, viss) in zip(uvws.compute().data, visss.compute().data):
            # iterate over frequencies
            for (freq, vis) in zip(dataset.frequency.data, viss):
                # calculate position in wavelengths
                iu = round(theta * uvw[0] * freq / c)
                iv = round(theta * uvw[1] * freq / c)
                # accumulate visibility at approximate location in the grid
                grid[iu + image_size//2, iv + image_size//2] += vis
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
    # grid = gridding(dataset)
    #grid = gridding_parallel(dataset)
    grid = gridding_parallel_mpi(dataset)
    if MPI.COMM_WORLD.Get_rank() == 0:
        image = fourier_transform(grid)
        save_image(image)
    image = fourier_transform(grid)
    save_image(image)

if __name__ == "__main__":
    main()

