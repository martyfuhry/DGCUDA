CC=gcc
CFLAGS=-O2
LFLAGS=-lm 

all: dgcylinder dgsupersonic dgthreepoint dgairfoil dglshock

dgcylinder: cylinder/cylinder.c euler.c ../main.c ../conserv_kernels.c ../time_integrator.c ../quadrature.c ../basis.c 
	$(CC) $(CFLAGS) cylinder/cylinder.c -o dgcylinder $(LFLAGS)

dgsupersonic: supersonic/supersonic.c euler.c ../main.c ../conserv_kernels.c ../time_integrator.c ../quadrature.c ../basis.c 
	$(CC) $(CFLAGS) supersonic/supersonic.c -o dgsupersonic $(LFLAGS)

dgthreepoint: threepoint/threepoint.c euler.c ../main.c ../conserv_kernels.c ../time_integrator.c ../quadrature.c ../basis.c 
	$(CC) $(CFLAGS) threepoint/threepoint.c -o dgthreepoint $(LFLAGS)

dgairfoil: airfoil/airfoil.c euler.c ../main.c ../conserv_kernels.c ../time_integrator.c ../quadrature.c ../basis.c 
	$(CC) $(CFLAGS) airfoil/airfoil.c -o dgairfoil $(LFLAGS)

dglshock: lshock/lshock.c euler.c ../main.c ../conserv_kernels.c ../time_integrator.c ../quadrature.c ../basis.c 
	$(CC) $(CFLAGS) lshock/lshock.c -o dglshock $(LFLAGS)
