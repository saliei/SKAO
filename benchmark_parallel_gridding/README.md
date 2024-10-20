# Gridding

These are the sources for the gridding problem.
The sources are as follow:

- [python](python): Optimizations on the original code itself, parallelization using mpi4py.
- [C_python](C_python): Offload of the gridding function to C and parallelization with hybrid MPI/OpenMP.
- [CUDA_python](CUDA_python): Offload of the gridding fuinction to multi-GPUs with CUDA and MPI.

For a summary of the results and scaling and bottlenecks see the jupyter notebook.

**Notes:** 

- To make the C shared library, `libgrid.so`, issue the `make` in the `C_python` directory, 
note that an MPI compiler is needed, it is tested with GNU `mpicc`.

- To make the CUDA shared libraries, by default `make` will compile the one 
without the MPI communications,`libgrid_cuda.so`, for the one with MPI calls, `libgrid_cuda_mpi.so`, 
issue the `make mpi`.
