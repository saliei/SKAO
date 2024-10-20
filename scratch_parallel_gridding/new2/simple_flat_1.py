#!/usr/bin/env python3

import time
import functools
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask import config as cfg

cfg.set({'distributed.scheduler.worker-ttl': None})
cfg.set({'logging.distributed': None})
cfg.set({"distributed.worker.use-file-locking": False})


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

# @dask.delayed
def calculate_indices(uvw, f):
    iu = xr.DataArray.round(theta * uvw[0] * f / c).astype(int)
    iv = xr.DataArray.round(theta * uvw[1] * f / c).astype(int)
    iu_idx = iu + image_size//2
    iv_idx = iv + image_size//2
    return iu_idx, iv_idx

# @dask.delayed
def gridding_single_timestep(uvwt, vist, freq):
    grid_timestep = da.zeros((image_size, image_size), dtype=complex)
    for (uvw, vis) in zip(uvwt, vist):
        for(f, vi) in zip(freq, vis):
            iu_idx, iv_idx = calculate_indices(uvw, f)
            grid_timestep[iu_idx, iv_idx] = vi
    return grid_timestep

# @dask.delayed
def gridding_compact(dataset, grid_zero):
    all_grids = []
    for t in range(len(dataset.time)):
        uvwt = dataset.UVW[t]
        vist = dataset.VISIBILITY[t]
        freq = dataset.frequency.data

        grid_timestep = gridding_single_timestep(uvwt, vist, freq)
        all_grids.append(grid_timestep)
    return all_grids

def sum_grid_timesptes(all_grids):
    global_grid = da.sum(all_grids, axis=0)
    return global_grid

def fourier_transform(grid):
    image = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(grid))).real
    return image

def save_image(image):
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

def main():
    cluster = LocalCluster(n_workers=4)
    client = Client(cluster)

    dataset = open_dataset(dataset_path)
    dataset = dataset.isel(time=slice(0, 10))
    grid_zero = da.zeros((image_size, image_size), dtype=complex)

    all_grids = gridding_compact(dataset, grid_zero)
    grid = sum_grid_timesptes(all_grids)
    # grid = grid.compute()

    image = fourier_transform(grid)
    image = image.compute()
    # save_image(image)

    client.close()
    cluster.close()

if __name__ == "__main__":
    main()

