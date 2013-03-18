#include "../euler.cu"

/* lshock.cu
 *
 * Shock flowing through an "L" tube
 */

#define PI 3.14159
#define GAMMA 1.4
#define MACH 2

int limiter = NO_LIMITER;  // no limiter
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
    U[0] = GAMMA;
    U[1] = U[0] * MACH;
    U[2] = U[0] * 0.;
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
                             double x, double y, 
                             double nx, double ny) {

    //double dot, vx, vy;
    U_right[0] = U_left[0];
    U_right[3] = U_left[3];
    double n = -(nx * U_left[1] + ny * U_left[2]);
    double t = ny * U_left[1] - nx * U_left[2];
    U_right[1] = n*nx + t*ny;
    U_right[2] = n*ny - t*nx;
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
