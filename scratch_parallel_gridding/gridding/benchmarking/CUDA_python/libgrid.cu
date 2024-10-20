#include <cuda_runtime.h>
#include <complex>
#include <cmath>
#include <cstdio>

#include "libgrid.h"


__global__ void gridding_cuda_kernel(double complex *grid, double *uvw_data, double complex *visibility_data, double *frequency_data) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k;

    for (k = 0; k < FREQUENCS; k++) {
        double complex vis = visibility_data[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k];
        double freq = frequency_data[k];

        int iu = (int)round(THETA_OVER_C * uvw_data[(i * BASELINES * 3) + (j * 3) + 0] * freq);
        int iv = (int)round(THETA_OVER_C * uvw_data[(i * BASELINES * 3) + (j * 3) + 1] * freq);
        int iu_idx = iu + IMAGE_SIZE / 2;
        int iv_idx = iv + IMAGE_SIZE / 2;

        atomicAdd(&grid[iu_idx * IMAGE_SIZE + iv_idx], vis);
    }
}

extern "C" void gridding_cuda(double complex *grid, double *uvw_data, double complex *visibility_data, double *frequency_data) {
    double complex *d_grid;
    double *d_uvw_data;
    double complex *d_visibility_data;
    double *d_frequency_data;

    cudaMalloc(&d_grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(double complex));
    cudaMalloc(&d_uvw_data, TIMESTEPS * BASELINES * 3 * sizeof(double));
    cudaMalloc(&d_visibility_data, TIMESTEPS * BASELINES * FREQUENCS * sizeof(double complex));
    cudaMalloc(&d_frequency_data, FREQUENCS * sizeof(double));

    cudaMemcpy(d_grid, grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(double complex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uvw_data, uvw_data, TIMESTEPS * BASELINES * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visibility_data, visibility_data, TIMESTEPS * BASELINES * FREQUENCS * sizeof(double complex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frequency_data, frequency_data, FREQUENCS * sizeof(double), cudaMemcpyHostToDevice);

    gridding_cuda_kernel<<<TIMESTEPS, BASELINES>>>(d_grid, d_uvw_data, d_visibility_data, d_frequency_data);

    cudaMemcpy(grid, d_grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(double complex), cudaMemcpyDeviceToHost);

    cudaFree(d_grid);
    cudaFree(d_uvw_data);
    cudaFree(d_visibility_data);
    cudaFree(d_frequency_data);
}

