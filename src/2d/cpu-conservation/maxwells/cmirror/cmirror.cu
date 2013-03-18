#include "../maxwells.cu"

/* test.cu
 *
 * maxwells equations
 *
 */

int limiter = NO_LIMITER;  // no limiter
int time_integrator = RK4; // time integrator to use

#define PI 3.141592653589793

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

    U[0] = 0;
    U[1] = 0;
    U[2] = 0;
}

/***********************
*
* INFLOW CONDITIONS
*
************************/
__device__ double get_mu() {
    return 1.;
}
__device__ double get_eps() {
    return 1;
}
__device__ void U_inflow(double *U, double x, double y, double t) {
    U[0] = 0.;
    U[1] = 0.;
    if (y > .95 && y < 1.05 && t <= 1./20) {
        U[2] = sinpi(20*t);
    } else {
        U[2] = 0.;
    }
}

/***********************
*
* OUTFLOW CONDITIONS
*
************************/

__device__ void U_outflow(double *U, double x, double y, double t) {
    U[0] = 0;
    U[1] = 0;
    U[2] = 0;
}

/***********************
*
* REFLECTING CONDITIONS
*
************************/

__device__ void U_reflection(double *U_left, double *U_right, 
                             double x, double y, double t,
                             double nx, double ny) {
    double dot;

    // set h to be the same
    U_right[2] = U_left[2];

    // and reflect the velocities
    dot = U_left[0] * nx + U_left[1] * ny;

    U_right[1] = -U_left[1] + 2*dot*nx;
    U_right[2] = -U_left[2] + 2*dot*ny;
}

__device__ void U_exact(double *U, double x, double y, double t) {
}

/***********************
 *
 * MAIN FUNCTION
 *
 ***********************/

int main(int argc, char *argv[]) {
    run_dgcuda(argc, argv);
}
