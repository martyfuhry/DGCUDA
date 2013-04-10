#include "../shallowwater.cu"

/* test1.cu
 *
 * simple flow with exact boundary conditions
 *
 */

#define PI 3.14159

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
    double x0, y0, r;

    x0 = 0.5;
    y0 = 0.5;
    r  = 0.1;

    U[0] = 10 + exp(-(pow(x - x0,2) + pow(y - y0,2))/(2*r*r));
    U[1] = 0.;
    U[2] = 0.;
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
    // there are no outflow boundaries in this problem 
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
    U_right[0] = U_left[0];

    // and reflect the velocities
    dot = U_left[1] * nx + U_left[2] * ny;

    U_right[1] = U_left[1] - 2*dot*nx;
    U_right[2] = U_left[2] - 2*dot*ny;

}
/***********************
*
* EXACT SOLUTION
*
************************/

__device__ void U_exact(double *U, double x, double y, double t) {
    // no exact solution
}


/***********************
 *
 * MAIN FUNCTION
 *
 ***********************/

int main(int argc, char *argv[]) {
    run_dgcuda(argc, argv);
}
