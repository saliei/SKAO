#ifndef LIBGRID_H
#define LIBGRID_H

#define IMAGE_SIZE 2048
#define IMAGE_SIZE_HALF 1024
#define THETA 0.0125
#define C 299792458
#define THETA_OVER_C 4.16955e-11

#define TIMESTEPS 512
#define BASELINES 351
#define FREQUENCS 256

void gridding_omp(double complex *grid ,double *uvwt, double complex *vist, double *freq);
void gridding_mpi_omp(double complex *grid, double *uvwt, double complex *vist, double *freq);
void gridding_simd(double complex *grid, double *uvwt, double complex *vist, double *freq);
void gridding_simd_mpi(double complex *grid, double *uvwt, double complex *vist, double *freq);

#endif // LIBGRID_H
