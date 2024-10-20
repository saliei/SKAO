#!/usr/bin/env python3

import time
import ctypes
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

image_size = 2048
image_name = "v1_C.png"
dataset_path = "../example_simulation.zarr"

libgrid = ctypes.CDLL("./libgrid.so")

libgrid.gridding_omp.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
]
libgrid.gridding_omp.restype = None


def open_dataset(dataset_path):
    dataset = xr.open_zarr(dataset_path)
    return dataset

dataset = open_dataset(dataset_path)
uvwt = dataset.UVW
vist = dataset.VISIBILITY
freq = dataset.frequency

grid = np.zeros((image_size, image_size), dtype=np.complex128)
grid_flat = grid.ravel()

# load the data as flat numpy arrays
uvwt_flat = uvwt.compute().values.ravel()
vist_flat = vist.compute().values.ravel()
freq_flat = freq.compute().values.ravel()

start_gridding_time = time.perf_counter()
libgrid.gridding_omp(grid_flat, uvwt_flat, vist_flat, freq_flat)
end_gridding_time = time.perf_counter()
gridding_time = end_gridding_time - start_gridding_time
print(f"gridding time: {gridding_time:10.8f}")

grid = grid_flat.reshape((image_size, image_size))
start_fft_time = time.perf_counter()
image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
end_fft_time = time.perf_counter()
fft_time = end_fft_time - start_fft_time
print(f"fft time: {fft_time:10.8f}")

plt.figure(figsize=(8,8)) 
plt.imsave(image_name, image)
