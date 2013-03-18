CC=nvcc
CFLAGS=-arch=sm_20 

all: dgadvecexact dgadvecspin dgadvecrotating dgadvecsquare

dgadvecexact: exact/exact.cu advection.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../basis.cu 
	$(CC) $(CFLAGS) exact/exact.cu -o dgadvecexact

dgadvecspin: spin/spin.cu advection.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../quadrature.cu ../basis.cu 
	$(CC) $(CFLAGS) spin/spin.cu -o dgadvecspin

dgadvecrotating: rotatinghill/rotatinghill.cu advection.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../quadrature.cu ../basis.cu 
	$(CC) $(CFLAGS) rotatinghill/rotatinghill.cu -o dgadvecrotating

dgadvecsquare: rotatinghill/squarehill.cu advection.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../quadrature.cu ../basis.cu 
	$(CC) $(CFLAGS) rotatinghill/squarehill.cu -o dgadvecsquare

