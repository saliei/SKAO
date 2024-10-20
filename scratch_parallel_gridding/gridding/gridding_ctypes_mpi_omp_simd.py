#!/usr/bin/env python3

import time
import ctypes
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpi4py import MPI

image_size = 2048
dataset_path = "./example_simulation.zarr"

lib = ctypes.CDLL("./grid.so")

lib.gridding_omp.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
]
lib.gridding_omp.restype = None

lib.gridding_mpi_omp.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
]
lib.gridding_mpi_omp.restype = None

lib.gridding_simd.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
]
lib.gridding_simd.restype = None

lib.gridding_simd_mpi.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
]
lib.gridding_simd_mpi.restype = None

def open_dataset(dataset_path):
    dataset = xr.open_zarr(dataset_path)
    return dataset

def gridding_omp(dataset):
    grid = np.zeros((image_size, image_size), dtype=np.complex128)
    grid_flat = grid.ravel()

    uvwt = dataset.UVW
    vist = dataset.VISIBILITY
    freq = dataset.frequency

    # load the data as flat numpy arrays
    uvwt_flat = uvwt.compute().values.ravel()
    vist_flat = vist.compute().values.ravel()
    freq_flat = freq.compute().values.ravel()

    start_gridding_time = time.perf_counter()
    lib.gridding_omp(grid_flat, uvwt_flat, vist_flat, freq_flat)
    end_gridding_time = time.perf_counter()
    gridding_time = end_gridding_time - start_gridding_time
    print(f"gridding time: {gridding_time:10.8f}")

    grid = grid_flat.reshape((image_size, image_size))
    start_fft_time = time.perf_counter()
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    end_fft_time = time.perf_counter()
    fft_time = end_fft_time - start_fft_time
    print(f"fft time: {fft_time:10.8f}")

    plt.figure(figsize=(4,4)) 
    plt.imsave("skao.png", image)

def gridding_mpi(dataset):
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
    lib.gridding_mpi_omp(grid_flat, uvwt_flat, vist_flat, freq_flat, start_t, end_t)
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

        plt.figure(figsize=(4,4)) 
        plt.imsave("skao.png", image)

def gridding_simd(dataset):
    grid = np.zeros((image_size, image_size), dtype=np.complex128)
    grid_flat = grid.ravel()

    uvwt = dataset.UVW
    vist = dataset.VISIBILITY
    freq = dataset.frequency

    # load the data as flat numpy arrays
    uvwt_flat = uvwt.compute().values.ravel()
    vist_flat = vist.compute().values.ravel()
    freq_flat = freq.compute().values.ravel()

    start_gridding_time = time.perf_counter()
    lib.gridding_simd(grid_flat, uvwt_flat, vist_flat, freq_flat)
    end_gridding_time = time.perf_counter()
    gridding_time = end_gridding_time - start_gridding_time
    print(f"gridding time: {gridding_time:10.8f}")

    grid = grid_flat.reshape((image_size, image_size))
    start_fft_time = time.perf_counter()
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    end_fft_time = time.perf_counter()
    fft_time = end_fft_time - start_fft_time
    print(f"fft time: {fft_time:10.8f}")

    plt.figure(figsize=(4,4)) 
    plt.imsave("skao.png", image)

def gridding_simd_mpi(dataset):
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
    lib.gridding_mpi_simd(grid_flat, uvwt_flat, vist_flat, freq_flat, start_t, end_t)
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

        plt.figure(figsize=(4,4)) 
        plt.imsave("skao.png", image)

def main():
    dataset = open_dataset(dataset_path)

    # gridding_omp(dataset)
    gridding_mpi(dataset)
    # gridding_simd(dataset)

if __name__ == "__main__":
    main()
