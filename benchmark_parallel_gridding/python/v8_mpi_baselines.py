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
dataset_path = "../example_simulation.zarr"
# cutoff time step, used for testing, 512 is the max
image_name = "v8.png"

dataset = xr.open_zarr(dataset_path)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

uvwt = dataset.UVW
vist = dataset.VISIBILITY
freq = dataset.frequency.data

def fourier_transform(grid):
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    return image
    
def plot_image(image):
    plt.figure(figsize=(8,8)) 
    plt.imsave(image_name, image)

def gridding_v8(uvwt, vist, freq, size, rank):
    # number of baselines to process for each process
    chunk_size = uvwt.shape[1]// size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank < size - 1 else uvwt.shape[1]
    
    grid_local = np.zeros((image_size, image_size), dtype=np.complex128)
    freq = np.expand_dims(freq, axis=1) # (256, 1)

    for t in range(uvwt.shape[0]):
        uvwt_local = uvwt[t][start_idx:end_idx].compute().data
        vist_local = vist[t][start_idx:end_idx].compute().data
        uvw0 = uvwt_local[:,0]
        uvw1 = uvwt_local[:,1]

        uvw0 = np.expand_dims(uvw0, axis=0) # (1, 351)
        uvw1 = np.expand_dims(uvw1, axis=0) # (1, 351)

        iu = np.round(theta * uvw0 * freq / c).astype(int)
        iv = np.round(theta * uvw1 * freq / c).astype(int)
        iu_idx = iu + image_size // 2
        iv_idx = iv + image_size // 2
    
        vist_local = np.swapaxes(vist_local, 0, 1)

        np.add.at(grid_local, (iu_idx, iv_idx), vist_local)

    return grid_local


start_gridding_time = MPI.Wtime()
grid_local = gridding_v8(uvwt, vist, freq, size, rank)
grid_global = comm.gather(grid_local, root=0)
end_gridding_time = MPI.Wtime()
print(f"rank: {rank}, gridding: {end_gridding_time - start_gridding_time}s")

if rank == 0:
    start_fft_time = MPI.Wtime()
    grid = np.sum(grid_global, axis=0)
    image = fourier_transform(grid)
    end_fft_time = MPI.Wtime()
    print(f"rank: {rank}, fourier transform: {end_fft_time - start_fft_time}s")

    plot_image(image)
    
