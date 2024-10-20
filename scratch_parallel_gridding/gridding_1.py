#!/usr/bin/env python3

import time
import functools
import numpy as np
import xarray as xr
import zarr as zr
import matplotlib.pyplot as plt

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
def open_dataset(filepath):
    dataset = xr.open_zarr(filepath)
    return dataset

@profile
def gridding(dataset):
    grid = np.zeros((image_size, image_size), dtype=complex)
    
    # iterate over time (baseline UW coordinates rotate with earth)
    for (uvws, visss) in zip(dataset.UVW, dataset.VISIBILITY):
        # iterate over individual baselines (i.e. pair of antennas)
        for (uvw, viss) in zip(uvws.compute().data, visss.compute().data):
            # iterate over frequencies
            for (freq, vis) in zip(dataset.frequency.data, viss):
                # calculate position in wavelengths
                iu = round(theta * uvw[0] * freq / c)
                iv = round(theta * uvw[1] * freq / c)
                # accumulate visibility at approximate location in the grid
                grid[iu + image_size//2, iv + image_size//2] += vis
    return grid

@profile
def fourier_transform(grid):
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    return image

@profile
def save_image(image):
    plt.figure(figsize=(16, 16))
    plt.imsave(image_name, image)

@profile
def main():
    dataset = open_dataset(dataset_path)
    grid = gridding(dataset)
    image = fourier_transform(grid)
    save_image(image)

if __name__ == "__main__":
    main()

