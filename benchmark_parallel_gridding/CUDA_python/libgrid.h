#ifndef LIBGRID_H
#define LIBGRID_H

#include <complex>

#define IMAGE_SIZE 2048
#define IMAGE_SIZE_HALF 1024
#define THETA 0.0125
#define C 299792458
#define THETA_OVER_C 4.16955e-11

#define TIMESTEPS 512
#define BASELINES 351
#define FREQUENCS 256

#ifdef __cplusplus
extern "C" {
#endif

void gridding_cuda(std::complex<double> *grid, double *uvwt, std::complex<double> *vist, double *freq);
void gridding_cuda_mpi(std::complex<double> *grid, double *uvwt, std::complex<double> *vist, double *freq);

#ifdef __cplusplus
}
#endif

#endif // LIBGRID_H
