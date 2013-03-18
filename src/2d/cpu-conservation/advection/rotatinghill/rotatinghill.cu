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
    double x0, y0 ,r;

    //if (x < .4 && x > .2 && y < .1 && y > -.1) {
        //U[0] = 1;
    //} else {
        //U[0] = 0;
    //}
        
    x0 = 0.2;
    y0 = 0.0;
    r  = 0.15;
    U[0] = 5*exp(-(pow(x - x0,2) + 
                  pow(y - y0,2))/(2*r*r));
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

    //U[0] = 0;

    double x0, y0, r;
    x0 = 0.2;
    y0 = 0.0;
    r  = 0.15;
    U[0] = 5*exp(-(pow(x*cospi(2*t) + y*sinpi(2*t) - x0,2) + 
                   pow(-x*sinpi(2*t) + y*cospi(2*t) - y0,2))/(2*r*r));
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
    // there are no reflecting boundaries in this problem
}

/***********************
*
* EXACT SOLUTION
*
************************/

__device__ void U_exact(double *U, double x, double y, double t) {
    //double A[2];
    //get_velocity(A, x, y, t);
    //U0(U, x - A[0]*t, y - A[1]*t);
    U[0] = 5*exp(-20*pow((pow(x*cos(2*PI*t) + y*sin(2*PI*t)-.6,2) +
                      -x*sin(2*PI*t) + y*cos(2*PI*t)),2));
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
