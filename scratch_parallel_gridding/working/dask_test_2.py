#!/usr/bin/env python3

import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster

# Speed of light
c = 299792458
# Size of image in pixels
image_size = 2048 
# Size of image on sky, directional cosines
theta = 0.0125
# Dataset path
dataset_path = "./example_simulation.zarr"
# Output image name
image_name = "dask_test_2.png"

def grid_visibilities_all_timestep(uvwt, vist, frequencies):
    grid = da.zeros((image_size, image_size), dtype=complex)

    uvw_x = uvwt[:,:,0]
    uvw_y = uvwt[:,:,1]

    uvw_x = da.expand_dims(uvw_x, axis=1)
    uvw_y = da.expand_dims(uvw_y, axis=1)
    freq = da.expand_dims(frequencies, axis=1)

    iu = (theta * uvw_x * freq / c).round().astype(int)
    iv = (theta * uvw_y * freq / c).round().astype(int)

    iu_global = iu + image_size // 2
    iv_global = iv + image_size // 2

    # mask = (iu_global >= 0) & (iu_global < image_size) & (iv_global >= 0) & (iv_global < image_size)
    def add_at_block(grid, iu, iv, vis):
        vis = vis.T
        np.add.at(grid, (iu, iv), vis)
        return grid

    # grid = da.map_blocks(lambda vis: np.where(mask, np.real(vis), 0), vist.T, dtype=grid.dtype)
    grid = grid.map_blocks(add_at_block, iu_global, iv_global, vist, dtype=grid.dtype)

    return grid


def grid_visibilities_single_timestep(uvwt, vist, frequencies):
    grid = da.zeros((image_size, image_size), dtype=complex)

    uvw_x = uvwt[:,0] 
    uvw_y = uvwt[:,1] 

    uvw_x = da.expand_dims(uvw_x, axis=0) # (1, 351)
    uvw_y = da.expand_dims(uvw_y, axis=0) # (1, 351)
    freq = da.expand_dims(frequencies, axis=1) # (256, 1)

    iu = (theta * uvw_x * freq / c).round().astype(int)
    iv = (theta * uvw_y * freq / c).round().astype(int)

    iu_global = iu + image_size // 2
    iv_global = iv + image_size // 2

    # mask = (iu_global >= 0) & (iu_global < image_size) & (iv_global >= 0) & (iv_global < image_size)
    def add_at_block(grid, iu, iv, vis):
        vis = vis.T
        np.add.at(grid, (iu, iv), vis)
        return grid

    # grid = da.map_blocks(lambda vis: np.where(mask, np.real(vis), 0), vist.T, dtype=grid.dtype)
    grid = grid.map_blocks(add_at_block, iu_global, iv_global, vist, dtype=grid.dtype)

    return grid



# @dask.delayed
def grid_visibilities_dataset(uvws, visss, frequencies):
    # Initialize the grid using dask.array
    grid = da.zeros((image_size, image_size), dtype=complex)

    # uvws_x = da.expand_dims(uvws[:, :, 0], axis=2)
    # uvws_y = da.expand_dims(uvws[:, :, 1], axis=2)
    #n frequencies = da.expand_dims(frequencies, axis=(0, 1))

    # uvws_x = uvws[:, :, 0].reshape((1, uvws.shape[1], 1))
    # uvws_y = uvws[:, :, 1].reshape((1, uvws.shape[1], 1))
    # frequencies = frequencies.reshape((1, 1, frequencies.shape[0]))
    
    # Calculate positions in wavelengths
    uvws_x = uvws[:,:,0]
    uvws_y = uvws[:,:,1]
    frq = frequencies[:,None]
    iu = (theta * uvws_x[:256,:] * frq / c).round().astype(int)
    iv = (theta * uvws_y[:256,:] * frq / c).round().astype(int)
    # iu = (theta * uvws[:,:,0] * frequencies / c).round().astype(int)
    # iv = (theta * uvws[:,:,0] * frequencies / c).round().astype(int)
    
    # Calculate global positions
    iu_global = iu + image_size // 2
    iv_global = iv + image_size // 2

    # grid[iu_global, iv_global] = 10
    
    # Mask for valid positions
    # mask = (iu_global >= 0) & (iu_global < image_size) & (iv_global >= 0) & (iv_global < image_size)
    
    # def accumulate(grid, iu_global, iv_global, visss, mask):
        # for i in range(frequencies.shape[2]):
            # masked_vis = da.where(mask[:, :, i], visss[:, :, i], 0)
            # grid = grid.at[iu_global[:, :, i], iv_global[:, :, i]].add(masked_vis)
            # grid = grid[iu_global[:, :, i], iv_global[:, :, i]].add(masked_vis)
        # return grid
    
    # Use dask's map_blocks to apply the accumulation function across blocks
    # grid = da.map_blocks(accumulate, grid, iu_global, iv_global, visss, mask, dtype=grid.dtype)
    # grid = da.map_blocks(lambda vis: np.where(mask, np.real(vis), 0), visss, dtype=grid.dtype)

    return grid

if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)

    # Open the Zarr dataset using Dask
    dataset = xr.open_zarr(dataset_path)
    # dataset = dataset.isel(time=slice(0, 10))

    # uvws = da.from_array(dataset.UVW.values, chunks=(16, 351, 3))
    # visss = da.from_array(dataset.VISIBILITY.values, chunks=(16, 351, 256))
    # visss = da.from_array(dataset.VISIBILITY.values, chunks='auto')

    # uvws_dask = da.from_array(dataset.UVW.values, chunks=(512, 351, 3))
    # viss_dask = da.from_array(dataset.VISIBILITY.values, chunks=(512, 351, 256))

    # uvws_dask = da.from_array(dataset.UVW.values, chunks="auto")
    # viss_dask = da.from_array(dataset.VISIBILITY.values, chunks="auto")

    uvws_dask = da.from_array(dataset.UVW.values)
    viss_dask = da.from_array(dataset.VISIBILITY.values)

    # frequencies = da.from_array(dataset.frequency.values, chunks=(256,))
    # frequencies = da.from_array(dataset.frequency.values)

    # uvws = dataset.UVW.values
    # visss = dataset.VISIBILITY.values
    frequencies = dataset.frequency.values

    # grid = grid_visibilities_dataset(uvws, visss, frequencies)


    # grids = []
    # for t in range(len(dataset.time)):
        # grid = grid_visibilities_single_timestep(uvws_dask[t], viss_dask[t], dataset.frequency.values)
        # grid = grid.compute()
        # grids.append(grid)

    grid = grid_visibilities_all_timestep(uvws_dask, viss_dask, dataset.frequency.values)
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

    client.close()
    cluster.close()
