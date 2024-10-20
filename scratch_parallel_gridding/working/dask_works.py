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
image_name = "numpy_no_loop.png"

@dask.delayed
def grid_visibilities_all_timestep(uvwt, vist, frequencies):
    grid = da.zeros((image_size, image_size), dtype=complex)

    uvw_x = uvwt[:,:,0]
    uvw_y = uvwt[:,:,1]

    uvw_x = uvw_x.expand_dims(dim="new_dim", axis=1)
    uvw_y = uvw_y.expand_dims(dim="new_dim", axis=1)
    freq  = da.expand_dims(frequencies, axis=1)

    iu = (theta * uvw_x * freq / c).round().astype(int)
    iv = (theta * uvw_y * freq / c).round().astype(int)

    iu_global = iu + image_size // 2
    iv_global = iv + image_size // 2
    # iu_global = iu_global.values
    # iv_global = iv_global.values

    def add_at_block(grid_block, iu_block, iv_block, vis_block):
        vis_block = da.swapaxes(vis_block, 1, -1)
        vis_block = vis_block.compute()
        iu_block = iu_block.compute()
        iv_block = iv_block.compute()
        np.add.at(grid_block, (iu_block, iv_block), vis_block)
        return grid_block


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

    def add_at_block(grid, iu, iv, vis):
        vis = vis.T
        np.add.at(grid, (iu, iv), vis)
        return grid

    grid = grid.map_blocks(add_at_block, iu_global, iv_global, vist, dtype=grid.dtype)

    return grid

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    # Open the Zarr dataset using Dask
    dataset = xr.open_zarr(dataset_path)
    dataset = dataset.isel(time=slice(0, 10))
    # dataset = dataset.chunk({"time": 1})
    # dataset = client.persist(dataset)
    dataset = dataset.compute()

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

    # uvws = dataset.UVW.values
    # visss = dataset.VISIBILITY.values
    frequencies = dataset.frequency.values

    # grid = grid_visibilities_dataset(uvws, visss, frequencies)


    # grids = []
    # for t in range(len(dataset.time)):
        # grid = grid_visibilities_single_timestep(uvws_dask[t], viss_dask[t], dataset.frequency.values)
        # grid = grid.compute()
        # grids.append(grid)

    # grid = grid_visibilities_all_timestep(uvws_dask, viss_dask, dataset.frequency.values)
    grid = grid_visibilities_all_timestep(dataset.UVW, dataset.VISIBILITY, dataset.frequency.values)
    grid = grid.compute()

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
