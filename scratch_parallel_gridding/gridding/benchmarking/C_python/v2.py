
#!/usr/bin/env python3

import time
import ctypes
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpi4py import MPI

image_size = 2048
image_name = "v2_C.png"
dataset_path = "./example_simulation.zarr"

libgrid = ctypes.CDLL("./libgrid.so")

libgrid.gridding_mpi_omp.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
]
libgrid.gridding_mpi_omp.restype = None


def open_dataset(dataset_path):
    dataset = xr.open_zarr(dataset_path)
    return dataset

dataset = open_dataset(dataset_path)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

grid = np.zeros((image_size, image_size), dtype=np.complex128)
grid_flat = grid.ravel()

chunk_size = len(dataset.time) // size
start_t = rank * chunk_size
end_t = (rank + 1) * chunk_size if rank < size - 1 else len(dataset.time)

uvwt = dataset.UVW[start_t:end_t]
vist = dataset.VISIBILITY[start_t:end_t]
freq = dataset.frequency

uvwt_flat = uvwt.compute().values.ravel()
vist_flat = vist.compute().values.ravel()
freq_flat = freq.compute().values.ravel()

start_gridding_time = MPI.Wtime()
libgrid.gridding_mpi_omp(grid_flat, uvwt_flat, vist_flat, freq_flat, start_t, end_t)
end_gridding_time = MPI.Wtime()
gridding_time = end_gridding_time - start_gridding_time
print(f"rank: {rank}, gridding time: {gridding_time:10.8f}")

if rank == 0:
    grid = grid_flat.reshape((image_size, image_size))
    start_fft_time = time.perf_counter()
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    end_fft_time = time.perf_counter()
    fft_time = end_fft_time - start_fft_time
    print(f"rank: {rank}, fft time: {fft_time:15.8f}")

    plt.figure(figsize=(8,8)) 
    plt.imsave(image_name, image)
