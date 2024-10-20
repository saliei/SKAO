#!/usr/bin/env python3

import time
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client, LocalCluster
from numba import vectorize
from dask import config as cfg

cfg.set({'distributed.scheduler.worker-ttl': None})
cfg.set({'logging.distributed': 'error'})
cfg.set({"distributed.worker.use-file-locking": False})

# Speed of light
c = 299792458
# Size of image in pixels
image_size = 2048 
# Size of image on sky, directional cosines
theta = 0.0125
# Dataset path
dataset_path = "./example_simulation.zarr"
# Output image name
image_name = "test_dask_1.png"

def open_dataset(filepath):
    dataset = xr.open_zarr(filepath)
    return dataset

def calculate_indices(uvw, f):
    iu = round(theta * uvw[0] * f / c)
    iv = round(theta * uvw[1] * f / c)
    iu_idx = iu + image_size // 2
    iv_idx = iv + image_size // 2
    return iu_idx, iv_idx

def gridding_single_timestep(uvwt, vist, freq):
    grid = np.zeros((image_size, image_size), dtype=complex)
    for (uvw, vis) in zip(uvwt, vist):
        for (f, vi) in zip(freq, vis):
            iu_idx, iv_idx = calculate_indices(uvw, f)
            grid[iu_idx, iv_idx] += vi
    return grid

def gridding_compact(uvwts, vists, freqs):
    grids = [gridding_single_timestep(uvwt, vist, freq) for uvwt, vist, freq in zip(uvwts, vists, freqs)]
    return np.sum(grids, axis=0)

def fourier_transform(grid):
    image = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(grid))).real
    return image

def save_image(image):
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

def main():
    cluster = LocalCluster()  # Start a local Dask cluster with 4 workers
    client = Client(cluster)  # Connect to the cluster
    dataset = open_dataset(dataset_path)
    dataset = dataset.isel(time=slice(0, 50))

    uvwts = [da.from_array(dataset.UVW[i].compute(), chunks=(351, 3)) for i in range(len(dataset.time))]
    vists = [da.from_array(dataset.VISIBILITY[i].compute(), chunks=(351, 256)) for i in range(len(dataset.time))]
    freqs = [da.from_array(dataset.frequency.compute(), chunks=(256,)) for _ in range(len(dataset.time))]

    grids = client.map(gridding_single_timestep, uvwts, vists, freqs)
    grid = client.submit(np.sum, grids).result()

    image = fourier_transform(grid)
    save_image(image)

    client.close()  # Close the client
    cluster.close()  # Close the cluster

if __name__ == "__main__":
    main()

