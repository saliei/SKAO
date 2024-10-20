#!/usr/bin/env python3

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
image_name = "v8.png"

print("**v8**")
dataset = open_dataset(dataset_path)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

uvwt = dataset.UVW
vist = dataset.VISIBILITY
freq = dataset.frequency.data

def open_dataset(dataset_path):
    dataset = xr.open_zarr(dataset_path)
    return dataset

def fourier_transform(grid):
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    return image
    
def plot_image(image):
    plt.figure(figsize=(8,8)) 
    plt.imsave(image_name, image)

def gridding_mpi_single_timestep_v7(grid, uvwt, vist, freq):
    uvw0 = uvwt[:,0] 
    uvw1 = uvwt[:,1]

    uvw0 = np.expand_dims(uvw0, axis=0) # (1, 351)
    uvw1 = np.expand_dims(uvw1, axis=0) # (1, 351)

    iu = np.round(theta * uvw0 * freq / c).astype(int)
    iv = np.round(theta * uvw1 * freq / c).astype(int)
    iu_idx = iu + image_size // 2
    iv_idx = iv + image_size // 2
    
    vist = np.swapaxes(vist, 0, 1)

    np.add.at(grid, (iu_idx, iv_idx), vist)
    
    return grid

def gridding_v7(uvwt, vist, freq, size, rank):
    # number of time steps for each process
    chunk_size = uvwt.shape[0]// size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank < size - 1 else uvwt.shape[0]
    
    grid_local = np.zeros((image_size, image_size), dtype=np.complex128)
    freq = np.expand_dims(freq, axis=1) # (256, 1)

    for t in range(start_idx, end_idx):
        grid_local = gridding_mpi_single_timestep_v7(grid_local, uvwt[t], vist[t].compute().data, freq)

    return grid_local


start_gridding_time = MPI.Wtime()
grid_local = gridding_v7(uvwt, vist, freq, size, rank)
grid_global = comm.gather(grid_local, root=0)
end_gridding_time = MPI.Wtime()
print(f"rank: {rank}, gridding: {end_gridding_time - start_dataset_time}s")

if rank == 0:
    start_fft_time = MPI.Wtime()
    grid = np.sum(grid_global, axis=0)
    image = fourier_transform(grid)
    end_fft_time = MPI.Wtime()
    print(f"rank: {rank}, fourier transform: {end_fft_time - start_fft_time}s")

    plot_image(image)
    
