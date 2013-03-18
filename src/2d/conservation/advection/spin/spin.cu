#include "../advection.cu"

/* test1.cu
 *
 * simple flow with exact boundary conditions
 *
 */

#define PI 3.14159

int limiter = NO_LIMITER;  // no limiter
int time_integrator = RK2; // time integrator to use

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
    U[0] = exp(-((x-x0)*(x-x0) + (y-y0)*(y-y0))/(2*r*r));

    //if (x < .4 && x > .3
     //&& y < .4 && y > .3) {
        //U[0] = 10;
        //U[1] = 10;
    //} else {
        //U[0] = 0;
        //U[1] = 0;
    //}
}

/***********************
*
* INFLOW CONDITIONS
*
************************/
__device__ void get_velocity(double *A, double x, double y, double t) {
    A[0] = -2*PI*y;
    A[1] =  2*PI*x;
}

__device__ void U_inflow(double *U, double x, double y, double t) {
    U[0] = 0;
    U[1] = 0;
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
