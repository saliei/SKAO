#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import xarray as xr

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Speed of light
c = 299792458
# Size of image in pixels
image_size = 2048 
# Size of image on sky, directional cosines
theta = 0.0125
# Dataset path
dataset_path = "./example_simulation.zarr"
# Output image name
image_name = "image.png"

# Open the Zarr dataset
dataset = xr.open_zarr(dataset_path)

# Determine the chunk size for each process
chunk_size = len(dataset.UVW) // size
start_idx = rank * chunk_size
end_idx = (rank + 1) * chunk_size if rank < size - 1 else len(dataset.UVW)

# Initialize grid for this process
grid_local = np.zeros((image_size, image_size), dtype=complex)

# Iterate over the assigned portion of the dataset
for i in range(start_idx, end_idx):
    uvw = dataset.UVW[i].compute().data
    viss = dataset.VISIBILITY[i].compute().data
    # Iterate over frequencies
    for (freq, vis) in zip(dataset.frequency.data, viss):
        # Calculate position in wavelengths
        iu = np.round(theta * uvw[0] * freq / c)
        iv = np.round(theta * uvw[1] * freq / c)
        # Accumulate visibility at approximate location in the local grid
        iu_global = iu + image_size // 2
        iv_global = iv + image_size // 2
        # if 1 <= iu_global < image_size and 0 <= iv_global < image_size:
        grid_local[iu_global, iv_global] += vis

# Gather local grids from all processes
grid_global = comm.gather(grid_local, root=0)

# Process 0 collects and combines grids from all processes
if rank == 0:
    # Combine grids from all processes
    grid_combined = np.sum(grid_global, axis=0)

    # Perform inverse Fourier transform and get the real part of the result
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_combined))).real

    # Plot and save the image
    plt.figure(figsize=(16, 16))
    plt.imsave(image, cmap='gray')

