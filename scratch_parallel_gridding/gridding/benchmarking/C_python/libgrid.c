#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <immintrin.h> // simd instructions

#include "libgrid.h"


void gridding_omp(double complex *grid ,double *uvwt, double complex *vist, double *freq) {
    int i, j, k;

#pragma omp parallel for collapse(2) schedule(static) shared(grid, uvwt, vist, freq)
    for (i = 0; i < TIMESTEPS; i++) {
        for (j = 0; j < BASELINES; j++) {
            for (k = 0; k < FREQUENCS; k++) {
                double complex vis = vist[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k];
                double fq = freq[k];

                int iu = (int)round(THETA_OVER_C * uvwt[(i * BASELINES * 3) + (j * 3) + 0] * fq);
                int iv = (int)round(THETA_OVER_C * uvwt[(i * BASELINES * 3) + (j * 3) + 1] * fq);
                int iu_idx = iu + IMAGE_SIZE / 2;
                int iv_idx = iv + IMAGE_SIZE / 2;

                grid[iu_idx * IMAGE_SIZE + iv_idx] += vis;
            }
        }
    }
}

void gridding_mpi_omp(double complex *grid, double *uvwt, double complex *vist, double *freq) {
    int i, j, k;
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int TIMESLICE = (TIMESTEPS / size);

    double complex *grid_local = (double complex*)calloc(IMAGE_SIZE * IMAGE_SIZE, sizeof(double complex));

#pragma omp parallel for collapse(2) schedule(static) shared(grid, uvwt, vist, freq)
    for (i = 0; i < TIMESLICE; i++) {
        for (j = 0; j < BASELINES; j++) {
            for (k = 0; k < FREQUENCS; k++) {
                double complex vis = vist[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k];
                double fq = freq[k];
                int iu = (int)round(THETA_OVER_C * uvwt[(i * BASELINES * 3) + (j * 3) + 0] * fq);
                int iv = (int)round(THETA_OVER_C * uvwt[(i * BASELINES * 3) + (j * 3) + 1] * fq);
                int iu_idx = iu + IMAGE_SIZE / 2;
                int iv_idx = iv + IMAGE_SIZE / 2;

                grid_local[iu_idx * IMAGE_SIZE + iv_idx] += vis;
            }
        }
    }

    MPI_Reduce(grid_local, grid, IMAGE_SIZE * IMAGE_SIZE, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

    free(grid_local);
}

void gridding_simd(double complex *grid, double *uvwt, double complex *vist, double *freq) {
    int i, j, k;

#pragma omp parallel for collapse(2) private(i, j, k) schedule(static)
    for (i = 0; i < TIMESTEPS; i++) {
        for (j = 0; j < BASELINES; j++) {
            // vectorize the innermost loop using SIMD instructions
            for (k = 0; k < FREQUENCS; k += 4) {
                // load the data to registers
                __m256d vis_real = _mm256_loadu_pd((double*)&vist[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k]);
                __m256d vis_imag = _mm256_loadu_pd((double*)&vist[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k] + 4);

                __m256d fq = _mm256_loadu_pd(&freq[k]);

                double u = uvwt[(i * BASELINES * 3) + (j * 3) + 0];
                double v = uvwt[(i * BASELINES * 3) + (j * 3) + 1];

                // compute the indices
                __m256d iu = _mm256_mul_pd(_mm256_set1_pd(u * THETA_OVER_C), fq);
                __m256d iv = _mm256_mul_pd(_mm256_set1_pd(v * THETA_OVER_C), fq);

                iu = _mm256_round_pd(iu, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                iv = _mm256_round_pd(iv, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                __m256i iu_idx = _mm256_castsi128_si256(_mm256_cvtpd_epi32(iu));
                __m256i iv_idx = _mm256_castsi128_si256(_mm256_cvtpd_epi32(iv));


                iu_idx = _mm256_add_epi32(iu_idx, _mm256_set1_epi32(IMAGE_SIZE_HALF));
                iv_idx = _mm256_add_epi32(iv_idx, _mm256_set1_epi32(IMAGE_SIZE_HALF));

                // accumulate the visibilities
                int iu_scalars[8];
                int iv_scalars[8];
                _mm256_storeu_si256((__m256i*)iu_scalars, iu_idx);
                _mm256_storeu_si256((__m256i*)iv_scalars, iv_idx);

                for (int idx = 0; idx < 4; idx++) {
                    int iu_scalar = iu_scalars[idx];
                    int iv_scalar = iv_scalars[idx];
                    grid[iu_scalar * IMAGE_SIZE + iv_scalar] += vis_real[idx] + vis_imag[idx] * I;
                }
            }
        }
    }
}

void gridding_simd_mpi(double complex *grid, double *uvwt, double complex *vist, double *freq) {
    int i, j, k;
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int TIMESLICE = (TIMESTEPS / size);

    double complex *grid_local = (double complex*)calloc(IMAGE_SIZE * IMAGE_SIZE, sizeof(double complex));

#pragma omp parallel for collapse(2) private(i, j, k) schedule(static)
    for (i = 0; i < TIMESLICE; i++) {
        for (j = 0; j < BASELINES; j++) {
            // vectorize the innermost loop using SIMD instructions
            for (k = 0; k < FREQUENCS; k += 4) {
                // load the data to registers
                __m256d vis_real = _mm256_loadu_pd((double*)&vist[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k]);
                __m256d vis_imag = _mm256_loadu_pd((double*)&vist[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k] + 4);

                __m256d fq = _mm256_loadu_pd(&freq[k]);

                double u = uvwt[(i * BASELINES * 3) + (j * 3) + 0];
                double v = uvwt[(i * BASELINES * 3) + (j * 3) + 1];

                // compute the indices
                __m256d iu = _mm256_mul_pd(_mm256_set1_pd(u * THETA_OVER_C), fq);
                __m256d iv = _mm256_mul_pd(_mm256_set1_pd(v * THETA_OVER_C), fq);

                iu = _mm256_round_pd(iu, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                iv = _mm256_round_pd(iv, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                __m256i iu_idx = _mm256_castsi128_si256(_mm256_cvtpd_epi32(iu));
                __m256i iv_idx = _mm256_castsi128_si256(_mm256_cvtpd_epi32(iv));


                iu_idx = _mm256_add_epi32(iu_idx, _mm256_set1_epi32(IMAGE_SIZE_HALF));
                iv_idx = _mm256_add_epi32(iv_idx, _mm256_set1_epi32(IMAGE_SIZE_HALF));

                // accumulate the visibilities
                int iu_scalars[8];
                int iv_scalars[8];
                _mm256_storeu_si256((__m256i*)iu_scalars, iu_idx);
                _mm256_storeu_si256((__m256i*)iv_scalars, iv_idx);

                for (int idx = 0; idx < 4; idx++) {
                    int iu_scalar = iu_scalars[idx];
                    int iv_scalar = iv_scalars[idx];
                    grid_local[iu_scalar * IMAGE_SIZE + iv_scalar] += vis_real[idx] + vis_imag[idx] * I;
                }
            }
        }
    }

    MPI_Reduce(grid_local, grid, IMAGE_SIZE * IMAGE_SIZE, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

    free(grid_local);
}
