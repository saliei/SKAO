#!/usr/bin/env python3

import time
import functools
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import jit, njit, vectorize, prange

# Speed of light
c = 299792458
# Size of image in pixels
image_size = 2048 
# Size of image on sky, directional cosines
theta = 0.0125
# Dataset path
dataset_path = "./example_simulation.zarr"
# Output image name
image_name = "numpy_no_loop.png"

def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"in function: {func.__name__} ...")
        start = time.perf_counter()
        return_value = func(*args, **kwargs)
        end = time.perf_counter()
        spent_time = end - start
        print(f"{func.__name__}: {spent_time:10.8f}")
        print("")
        return return_value
    return wrapper


@profile
def grid_visibilities_all_timesteps(uvwt, vist, frequencies):
    grid = np.zeros((image_size, image_size), dtype=complex)

    uvw_x = uvwt[:,:,0]
    uvw_y = uvwt[:,:,1]

    uvw_x = np.expand_dims(uvw_x, axis=1)
    uvw_y = np.expand_dims(uvw_y, axis=1)
    freq  = np.expand_dims(frequencies, axis=1)
    

    iu = np.round(theta * uvw_x * freq / c).astype(int)
    iv = np.round(theta * uvw_y * freq / c).astype(int)

    iu_global = iu + image_size // 2
    iv_global = iv + image_size // 2

    # def add_at_block(grid_block, iu_block, iv_block, vis_block):
        # vis_block = np.swapaxes(vis_block, 1, -1)
        # np.add.at(grid_block, (iu_block, iv_block), vis_block)
        # return grid_block


    # grid = np.array(list(map(add_at_block, grid, iu_global, iv_global, vist)))
    vist = np.swapaxes(vist, 1, -1)
    grid = np.add.at(grid, (iu_global, iv_global), vist)

    return grid

def add_at_block(grid_block, iu_block, iv_block, vis_block):
    for i in range(iu_block.shape[0]):
        np.add.at(grid_block, (iu_block[i], iv_block[i]), vis_block[i])
    return grid_block

def grid_visibilities_all_timesteps_threads(uvwt, vist, frequencies, n_workers=4):
    grid = np.zeros((image_size, image_size), dtype=complex)

    uvw_x = uvwt[:,:,0]
    uvw_y = uvwt[:,:,1]

    # Adjust expand_dims to match the shape of vis_block
    uvw_x = np.expand_dims(uvw_x, axis=1)  # Shape becomes (512, 351, 1)
    uvw_y = np.expand_dims(uvw_y, axis=1)  # Shape becomes (512, 351, 1)
    freq  = np.expand_dims(frequencies, axis=1)  # Shape becomes (1, 256)

    # Use broadcasting to achieve the desired shape
    iu = np.round(theta * uvw_x * freq / c).astype(int)  # Shape becomes (512, 351, 256)
    iv = np.round(theta * uvw_y * freq / c).astype(int)  # Shape becomes (512, 351, 256)

    # Transpose the shapes to match vis_block (512, 256, 351)
    # iu = iu.transpose(0, 2, 1)  # Shape becomes (512, 256, 351)
    # iv = iv.transpose(0, 2, 1)  # Shape becomes (512, 256, 351)

    iu_global = iu + image_size // 2
    iv_global = iv + image_size // 2

    vist = np.swapaxes(vist, 1, -1)

    # Split the data into chunks for parallel processing
    def chunk_data(data, n_chunks):
        chunk_size = data.shape[0] // n_chunks
        return [data[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]

    iu_chunks = chunk_data(iu_global, n_workers)
    iv_chunks = chunk_data(iv_global, n_workers)
    vis_chunks = chunk_data(vist, n_workers)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for iu_chunk, iv_chunk, vis_chunk in zip(iu_chunks, iv_chunks, vis_chunks):
            futures.append(executor.submit(add_at_block, grid.copy(), iu_chunk, iv_chunk, vis_chunk))

        for future in as_completed(futures):
            grid += future.result()

    return grid


def grid_visibilities_single_timestep(uvwt, vist, frequencies):
    grid = da.zeros((image_size, image_size), dtype=complex)

    uvw_x = uvwt[:,0] 
    uvw_y = uvwt[:,1]

    uvw_x = da.expand_dims(uvw_x, axis=0) # (1, 351)
    uvw_y = da.expand_dims(uvw_y, axis=1) # (1, 351)
    freq = da.expand_dims(frequencies, axis=1) # (256, 1)

    iu = (theta * uvw_x * freq / c).round().astype(int)
    iv = (theta * uvw_y * freq / c).round().astype(int)

    iu_global = iu + image_size // 2
    iv_global = iv + image_size // 2

    def add_at_block(grid, iu, iv, vis):
        vis = vis.T
        np.add.at(grid, (iu, iv), vis)
        return grid

    grid = grid.map_blocks(add_at_block, iu_global, iv_global, vist, dtype=grid.dtype)

    return grid

if __name__ == "__main__":

    # Open the Zarr dataset using Dask
    dataset = xr.open_zarr(dataset_path)
    # dataset = dataset.isel(time=slice(0, 10))
    # dataset = dataset.chunk({"time": 1})
    # dataset = client.persist(dataset)

    # uvws = da.from_array(dataset.UVW.values, chunks=(16, 351, 3))
    # visss = da.from_array(dataset.VISIBILITY.values, chunks=(16, 351, 256))
    # visss = da.from_array(dataset.VISIBILITY.values, chunks='auto')

    # uvws_dask = da.from_array(dataset.UVW.values, chunks=(1, 351, 3))
    # viss_dask = da.from_array(dataset.VISIBILITY.values, chunks=(1, 351, 256))

    # uvws_dask = da.from_array(dataset.UVW.values, chunks="auto")
    # viss_dask = da.from_array(dataset.VISIBILITY.values, chunks="auto")

    # uvws_dask = da.from_array(dataset.UVW.values)
    # viss_dask = da.from_array(dataset.VISIBILITY.values)

    # frequencies = da.from_array(dataset.frequency.values, chunks=(256,))
    # frequencies = da.from_array(dataset.frequency.values)

    uvws = dataset.UVW.compute().values
    visss = dataset.VISIBILITY.compute().values
    frequencies = dataset.frequency.compute().values

    # grid = grid_visibilities_dataset(uvws, visss, frequencies)


    # grids = []
    # for t in range(len(dataset.time)):
        # grid = grid_visibilities_single_timestep(uvws_dask[t], viss_dask[t], dataset.frequency.values)
        # grid = grid.compute()
        # grids.append(grid)

    # grid = grid_visibilities_all_timesteps(uvws_dask, viss_dask, dataset.frequency.values)
    # grid = grid_visibilities_all_timesteps(uvws, visss, frequencies)
    grid = grid_visibilities_all_timesteps_threads(uvws, visss, frequencies, 4)
    # grid = grid.compute()

    # print(grid)
    # print(grid.shape)

    # Compute the result
    # grids = np.array(grids)
    # grid = grids.sum(axis=0)

    # Perform inverse Fourier transform and get the real part of the result
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    # image = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(grid))).real

    # Plot and save the image
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)
