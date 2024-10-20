all:
	#mpicc -shared -fopenmp -O3 -march=native -o libgrid.so -fPIC libgrid.c -Wall -lm
	nvcc -shared -O3 -o libgrid_cuda.so -Xcompiler -fPIC libgrid.cu -Wall

clean:
	rm libgrid.so

