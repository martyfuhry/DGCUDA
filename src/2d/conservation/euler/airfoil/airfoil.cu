#include "../euler.cu"

/* airfoil.cu
 *
 * Flow around an airfoil.
 */

#define PI 3.14159
#define GAMMA 1.4
#define MACH .78
#define ALPHA 5.

int limiter = LIMITER;  
int time_integrator = RK4; 

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
    U[0] = GAMMA;
    U[1] = U[0] * MACH * cospi(ALPHA / 180.);
    U[2] = U[0] * MACH * sinpi(ALPHA / 180.);
    U[3] = 0.5 * U[0] * MACH * MACH + 1./ (GAMMA - 1.0);
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

    double dot;

    // set rho and E to be the same in the ghost cell
    U_right[0] = U_left[0];
    U_right[3] = U_left[3];

    // normal reflection
    dot = U_left[1] * nx + U_left[2] * ny;

    U_right[1] = U_left[1] - 2*dot*nx;
    U_right[2] = U_left[2] - 2*dot*ny;
}


/***********************
 *
 * EXACT SOLUTION
 *
 ***********************/
__device__ void U_exact(double *U, double x, double y, double t) {
    // no exact solution
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
