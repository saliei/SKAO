#!/usr/bin/env python3

import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client, progress, LocalCluster

# speed of light
c = 299792458
# size of image in pixels
image_size = 2048 
# size of image on sky, directional cosines
theta = 0.0125
# dataset path
dataset_path = "./example_simulation.zarr"
# output image name
image_name = "dask_dist.png"

def grid_visibilities(uvw_chunk, viss_chunk):
    grid = np.zeros((image_size, image_size), dtype=complex)
    uvws = uvw_chunk.compute()
    visss = viss_chunk.compute()
    # Iterate over individual baselines (i.e. pair of antennas)
    for uvw, viss in zip(uvws, visss):
        # Iterate over frequencies
        for freq, vis in zip(dataset.frequency.data, viss):
            # Calculate position in wavelengths
            iu = round(theta * uvw[0] * freq / c)
            iv = round(theta * uvw[1] * freq / c)
            # Accumulate visibility at approximate location in the grid
            #iu_global = iu + image_size // 2
            #iv_global = iv + image_size // 2
            #if 0 <= iu_global < image_size and 0 <= iv_global < image_size:
                #   grid[iu_global, iv_global] += vis
    return grid

if __name__ == "__main__":

    # Connect to Dask cluster
    cluster = LocalCluster()
    #client = Client(n_workers=4, threads_per_worker=2)
    client = Client(cluster)

    # Open the Zarr dataset using Dask
    dataset = xr.open_zarr(dataset_path, chunks={'UVW': 'auto', 'VISIBILITY': 'auto'})

    # Define function for gridding


    # Map function over chunks of data and aggregate results
    grids = []
    for uvw_chunk, viss_chunk in zip(dataset.UVW.data, dataset.VISIBILITY.data):
        grid = client.submit(grid_visibilities, uvw_chunk, viss_chunk)
        grids.append(grid)

    # Compute the result
    # grid = da.stack(grids).sum(axis=0).compute()

    # Perform inverse Fourier transform and get the real part of the result
    # image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real

    # Plot and save the image
    # plt.figure(figsize=(16, 16))
    # plt.imsave(image_name, image)
