CC=nvcc
CFLAGS=-arch=sm_20

all: dgcylinder dgsupersonic dgdoublemach dgairfoil dglshock dgtrumpet

dgcylinder: cylinder/cylinder.cu euler.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../quadrature.cu ../basis.cu 
	$(CC) $(CFLAGS) cylinder/cylinder.cu -o dgcylinder

dgsupersonic: supersonic/supersonic.cu euler.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../quadrature.cu ../basis.cu 
	$(CC) $(CFLAGS) supersonic/supersonic.cu -o dgsupersonic

dgdoublemach: doublemach/doublemach.cu euler.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../quadrature.cu ../basis.cu 
	$(CC) $(CFLAGS) doublemach/doublemach.cu -o dgdoublemach

dgairfoil: airfoil/airfoil.cu euler.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../quadrature.cu ../basis.cu 
	$(CC) $(CFLAGS) airfoil/airfoil.cu -o dgairfoil

dglshock: lshock/lshock.cu euler.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../quadrature.cu ../basis.cu 
	$(CC) $(CFLAGS) lshock/lshock.cu -o dglshock

dgtrumpet: trumpet/trumpet.cu euler.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../quadrature.cu ../basis.cu 
	$(CC) $(CFLAGS) trumpet/trumpet.cu -o dgtrumpet

dgtube: tube/tube.cu euler.cu ../main.cu ../conserv_kernels.cu ../time_integrator.cu ../quadrature.cu ../basis.cu 
	$(CC) $(CFLAGS) tube/tube.cu -o dgtube

