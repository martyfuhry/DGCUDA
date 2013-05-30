#include "../euler.cu"

/* supersonic.cu
 *
 * Supersonic flow around a cylinder.
 *
 */

#define PI 3.14159
#define GAMMA 1.4
#define MACH 2.25

int limiter = LIMITER;  // use a limiter or not
int time_integrator = RK4; // time integrator to use

/***********************
 *
 * INITIAL CONDITIONS
 *
 ***********************/

/* initial condition function
 *
 * returns the value of the intial condition at point x,y
 */

__device__ void U0(double *U, double x, double y) {
    double r = sqrt(x*x + y*y);

    U[0] = pow(1+1.0125*(1.- 1./(r * r)),2.5);
    U[1] = U[0] * sin(atan(y / x)) * MACH / r;
    U[2] = U[0] * -cos(atan(y / x)) * MACH / r;

    double p = (1.0 / GAMMA) * pow(U[0], GAMMA);

    U[3] = 0.5 * U[0] * (MACH*MACH/(r * r)) + p * (1./(GAMMA - 1.));
}

/***********************
*
* INFLOW CONDITIONS
*
************************/

__device__ void U_inflow(double *U, double x, double y, double t) {
    U0(U, x, y);
}

/***********************
*
* OUTFLOW CONDITIONS
*
************************/

__device__ void U_outflow(double *U, double x, double y, double t) {
    U0(U, x, y);
}

/***********************
*
* REFLECTING CONDITIONS
*
************************/

__device__ void U_reflection(double *U_left, double *U_right,
                             double x, double y, double t,
                             double nx, double ny) {
    double Nx, Ny, dot;

    // set rho and E to be the same in the ghost cell
    U_right[0] = U_left[0];
    U_right[3] = U_left[3];

    // taken from algorithm 2 from lilia's code
    dot = sqrt(x*x + y*y);
    Nx = x / dot;
    Ny = y / dot;

    if (Nx * nx + Ny * ny < 0) {
        Nx *= -1;
        Ny *= -1;
    }

    // set the velocities to reflect
    U_right[1] =  (U_left[1] * Ny - U_left[2] * Nx)*Ny;
    U_right[2] = -(U_left[1] * Ny - U_left[2] * Nx)*Nx;
}

/***********************
*
* REFLECTING CONDITIONS
*
************************/

__device__ void U_exact(double *U, double x, double y, double t) {
    U0(U, x, y);
}

/***********************
 *
 * MAIN FUNCTION
 *
 ***********************/

__device__ double get_GAMMA() {
    return GAMMA;
}

int main(int argc, char *argv[]) {
    run_dgcuda(argc, argv);
}
