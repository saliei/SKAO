#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import time
from numba import jit

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
time_start = time.perf_counter()
# Speed of light
c = 299792458
# Size of image in pixels
image_size = 2048 
# Size of image on sky, directional cosines
theta = 0.0125
# Dataset path
dataset_path = "./example_simulation.zarr"
# Output image name
image_name = "mpi_test_time_jit.png"

# Open the Zarr dataset
dataset = xr.open_zarr(dataset_path)
#dataset = dataset.isel(time=slice(0, 128))

UVW = dataset.UVW.compute().to_numpy()
VIS = dataset.VISIBILITY.compute().to_numpy()
FRQ = dataset.frequency.compute().to_numpy()

num_baselines = len(dataset.baseline_id)
num_timestamps = len(dataset.time)

# Determine the chunk size for each process
chunk_size = num_timestamps // size
start_idx = rank * chunk_size
end_idx = (rank + 1) * chunk_size if rank < size - 1 else num_timestamps

# Initialize grid for this process
grid_local = np.zeros((image_size, image_size), dtype=complex)

@jit(nopython=True)
def gridding_chunked_time(UVW, VIS, FRQ, grid_local):
    for i in range(start_idx, end_idx):
        uvws = UVW[i]
        visss = VIS[i]
        for (uvw, viss) in zip(uvws, visss):
            for (freq, vis) in zip(FRQ, viss):
                iu = round(theta * uvw[0] * freq / c)
                iv = round(theta * uvw[1] * freq / c)
                iu_global = iu + image_size // 2
                iv_global = iv + image_size // 2
                if 0 <= iu_global < image_size and 0 <= iv_global < image_size:
                    grid_local[iu_global, iv_global] += vis
    return grid_local

grid_local = gridding_chunked_time(UVW, VIS, FRQ, grid_local)
# Gather local grids from all processes
grid_global = comm.gather(grid_local, root=0)

# Process 0 collects and combines grids from all processes
if rank == 0:
    # Combine grids from all processes
    grid_combined = np.sum(grid_global, axis=0)

    # Perform inverse Fourier transform and get the real part of the result
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_combined))).real
    time_end = time.perf_counter()
    print(f"time: {time_end - time_start}")

    # Plot and save the image
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

