#include "../advection.cu"

/* test1.cu
 *
 * simple flow with exact boundary conditions
 *
 */

#define PI 3.14159265358979323

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
   U[0] = 5.477225575051629E+00-1.095445115010309E+01*x-3.286335345030997E+01*y + 5.477225575051381E+00*x*x + 3.286335345031001E+01*x*y + 3.286335345030994E+01*y*y;
}

/***********************
*
* INFLOW CONDITIONS
*
************************/
__device__ void get_velocity(double *A, double x, double y, double t) {
    A[0] = 0;
    A[1] = 1;
}

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
                             double x, double y, 
                             double nx, double ny) {
    // there are no reflecting boundaries in this problem
}

/***********************
*
* EXACT SOLUTION
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

int main(int argc, char *argv[]) {
    run_dgcuda(argc, argv);
}
