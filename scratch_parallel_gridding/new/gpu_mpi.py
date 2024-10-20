#!/usr/bin/env python3

import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from mpi4py import MPI
import cupy as cp

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

# Open the Zarr dataset
dataset = xr.open_zarr(dataset_path)

# Get the number of available GPUs
num_gpus = cp.cuda.runtime.getDeviceCount()

# Determine the chunk size for each GPU
chunk_size_per_gpu = len(dataset.UVW) // (num_gpus * size)
start_idx = rank * num_gpus * chunk_size_per_gpu
end_idx = (rank + 1) * num_gpus * chunk_size_per_gpu

# Initialize grids on GPUs
grids = [cp.zeros((image_size, image_size), dtype=cp.complex64, device=i % num_gpus) for i in range(start_idx, end_idx)]

# Iterate over the assigned portion of the dataset
for i in range(start_idx, end_idx):
    uvw = dataset.UVW[i].compute().data
    viss = dataset.VISIBILITY[i].compute().data
    gpu_idx = i % num_gpus
    # Iterate over frequencies
    for (freq, vis) in zip(dataset.frequency.data, viss):
        # Calculate position in wavelengths
        iu = cp.round(theta * uvw[0] * freq / c)
        iv = cp.round(theta * uvw[1] * freq / c)
        # Accumulate visibility at approximate location in the grid
        iu_global = iu + image_size // 2
        iv_global = iv + image_size // 2
        if 0 <= iu_global < image_size and 0 <= iv_global < image_size:
            grids[i - start_idx][iu_global, iv_global] += vis

# Gather grids from all processes
grid_global = comm.allgather(grids)

# Combine grids from all processes
grid_combined = sum(grid_global, [])

# Transfer grid back to host memory
grid_host = cp.asnumpy(grid_combined)

# Perform inverse Fourier transform and get the real part of the result
image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_host))).real

# Plot and save the image
if rank == 0:
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image, cmap='gray')
    plt.show()

