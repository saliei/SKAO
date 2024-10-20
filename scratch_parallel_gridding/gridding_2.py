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
# output image name
image_name = "image.png"

def open_dataset(filepath):
    dataset = xr.open_zarr(filepath)
    return dataset

def gridding_single(uvw, viss):
    grid = np.zeros((image_size, image_size), dtype=complex)
    for (freq, vis) in zip(uvw, viss):
        iu = round(theta * uvw[0] * freq / c)
        iv = round(theta * uvw[1] * freq / c)
        grid[iu + image_size//2, iv + image_size//2] += vis
    return grid

def gridding(dataset, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    uvws = dataset.UVW
    visss = dataset.VISIBILITY
    
    chunk_size = len(uvws) // size
    start = rank * chunk_size
    end = start + chunk_size if rank < size - 1 else len(uvws)
    
    local_results = []
    for i in range(start, end):
        local_results.append(gridding_single(uvws[i], visss[i]))
    
    global_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        grid = np.sum(global_results, axis=0)
    else:
        grid = None
    
    return grid

def fourier_transform(grid):
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    return image

def save_image(image):
    plt.figure(figsize=(16, 16))
    plt.imshow(image, cmap='hot')
    plt.colorbar()
    plt.savefig(image_name)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        dataset = open_dataset(dataset_path)
    else:
        dataset = None
    
    dataset = comm.bcast(dataset, root=0)
    grid = gridding(dataset, comm)
    
    if rank == 0:
        image = fourier_transform(grid)
        save_image(image)

if __name__ == "__main__":
    main()
