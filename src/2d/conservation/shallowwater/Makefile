CC=nvcc
CFLAGS=-arch=sm_13

all: dgshallowtest

dgshallowtest: test/test.cu shallowwater.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../quadrature.cu ../basis.cu 
	$(CC) $(CFLAGS) test/test.cu -o dgshallowtest

