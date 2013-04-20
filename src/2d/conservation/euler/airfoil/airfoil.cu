#include "../euler.cu"

/* airfoil.cu
 *
 * Flow around an airfoil.
 */

#define PI 3.14159
#define GAMMA 1.4
#define MACH .98
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

    //double dot;

    // set rho and E to be the same in the ghost cell
    //U_right[0] = U_left[0];
    //U_right[3] = U_left[3];

    // normal reflection
    //dot = U_left[1] * nx + U_left[2] * ny;

    //U_right[1] = U_left[1] - 2*dot*nx;
    //U_right[2] = U_left[2] - 2*dot*ny;
    // set rho and E to be the same in the ghost cell
    U_right[0] = U_left[0];
    U_right[3] = U_left[3];
    //double Nx, Ny, dot;
    double u_N, u_T;

    u_N = -(nx * U_left[1] + ny * U_left[2]);
    u_T = ny * U_left[1] - nx * U_left[2];

    U_right[1] = u_N * nx + u_T * ny;
    U_right[2] = u_N * ny - u_T * nx;


    // taken from algorithm 2 from lilia's code
    //dot = sqrt(x*x + y*y);
    //Nx = x / dot;
    //Ny = y / dot;
//
    //if (Nx * nx + Ny * ny < 0) {
        //Nx *= -1;
        //Ny *= -1;
    //}
//
    // set the velocities to reflect
    //U_right[1] =  (U_left[1] * Ny - U_left[2] * Nx)*Ny;
    //U_right[2] = -(U_left[1] * Ny - U_left[2] * Nx)*Nx;
//}

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
