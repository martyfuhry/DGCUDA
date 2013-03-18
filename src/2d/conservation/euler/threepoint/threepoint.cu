#include "../euler.cu"

/* threepoint.cu
 *
 * Use the Euler equations to solve the three point paradox.
 *
 * Create a weak shock and run it into a thin wedge with angle THETA at mach MACH.
 * Use p = 0 to use weak shocks and don't use a limiter.
 * 
 */

#define GAMMA 1.4
#define PI 3.14159265358979323846
#define MACH 10.
#define X0 0.16666666666666666
#define THETA 60.

int limiter = LIMITER;  
int time_integrator = RK4; 

/***********************
 *
 * SHOCK CONDITIONS
 *
 ***********************/

__device__ void U_shock(double *U, double x, double y) {
    double p;

    p = 116.5;
    U[0] = 8.;
    U[1] =  8.25 * cospi(1. / 6.) * U[0];
    U[2] = -8.25 * sinpi(1. / 6.) * U[0];
    U[3] = 0.5 * U[0] * 8.25 * 8.25 + p / (GAMMA - 1.0);
}

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
    double angle = atan(y / (x - X0));
    if (angle > THETA * PI / 180 || x < X0) {
        U_shock(U, x, y);
    } else {
        U[0] = GAMMA;
        U[1] = 0.;
        U[2] = 0.;
        U[3] = 1./ (GAMMA - 1.0);
    }
}

/***********************
*
* INFLOW CONDITIONS
*
************************/

__device__ void U_inflow(double *U, double x, double y, double t) {
    double shock_position;

    shock_position = X0 + (1. + 20.*t)/(sqrt(3.));

    if (x < shock_position) {
        U_shock(U, x, y);
    } else {
        U[0] = GAMMA;
        U[1] = 0.;
        U[2] = 0.;
        U[3] = 1./ (GAMMA - 1.0);
    }
}

/***********************
*
* OUTFLOW CONDITIONS
*
************************/

__device__ void U_outflow(double *U, double x, double y, double t) {
    U_inflow(U, x, y, t);
}

/***********************
*
* REFLECTING CONDITIONS
*
************************/
__device__ void U_reflection(double *U_left, double *U_right,
                             double x, double y, double t,
                             double nx, double ny) {

    if (x < X0) {
        U_shock(U_right, x, y);
    } else {
        double dot;
        // set rho and E to be the same in the ghost cell
        U_right[0] = U_left[0];
        U_right[3] = U_left[3];

        // normal reflection
        dot = U_left[1] * nx + U_left[2] * ny;

        U_right[1] = U_left[1] - 2*dot*nx;
        U_right[2] = U_left[2] - 2*dot*ny;
    }
}

/***********************
 *
 * EXACT SOLUTION
 *
 ***********************/

__device__ void U_exact(double *U, double x, double y, double t) {
    // there is no exact solution
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
