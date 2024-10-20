#!/usr/bin/env python3

import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt
from numba import jit

# speed of light
c = 299792458
# size of image in pixels
image_size = 2048 
# size of image on sky, directional cosines
theta = 0.0125
# dataset path
dataset_path = "./example_simulation.zarr"
# output image name
image_name = "jitting.png"

dataset = xr.open_zarr(dataset_path)
dataset = dataset.isel(time=slice(0, 100))

UVW = dataset.UVW.compute().to_numpy()
VIS = dataset.VISIBILITY.compute().to_numpy()
FRQ = dataset.frequency.compute().to_numpy()

grid = np.zeros((image_size, image_size), dtype=complex)

#@jit(nopython=True)
def gridding_freq(uvw, viss, freq, grid):
    for (f, v) in zip(freq, viss):
        iu = round(theta * uvw[0] * f / c)
        iv = round(theta * uvw[1] * f / c)
        grid[iu + image_size//2, iv + image_size//2] += v
    return grid

def gridding(UVW, VIS, FRQ, grid):
    for (uvws, visss) in zip(UVW, VIS):
        for (uvw, viss) in zip(uvws, visss):
            grid = gridding_freq(uvw, viss, FRQ, grid)
    return grid

grid = gridding(UVW, VIS, FRQ, grid)
image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real

plt.figure(figsize=(16, 16))
plt.imsave(image_name, image)
