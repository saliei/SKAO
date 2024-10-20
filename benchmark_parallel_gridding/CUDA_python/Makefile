all:
	nvcc -shared -O3 -arch=sm_70 -o libgrid_cuda.so -Xcompiler -fPIC libgrid.cu

mpi:
	nvcc -shared -O3 -arch=sm_70 -o libgrid_cuda_mpi.so -Xcompiler -fPIC libgrid_mpi.cu -I/opt/programs/openMPI/4.1.5/include/ -L/opt/programs/openMPI/4.1.5/lib/ -lmpi


clean:
	rm libgrid_cuda.so
	rm libgrid_cuda_mpi.so

