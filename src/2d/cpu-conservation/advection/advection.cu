#include "../main.cu"

/* advection.cu
 *
 * This file contains the relevant information for making a system to solve
 *
 * d_t [ u ] + d_x [ au ] + d_y [ bu ] = 0
 * d_t [ v ] + d_x [ av ] + d_y [ bv ] = 0
 *
 */

__device__ void get_velocity(double *, double, double, double);

/* size of the system */
int local_N = 1;

/***********************
 *
 * INITIAL CONDITIONS
 *
 ***********************/

__device__ void evalU0(double *U, double *V, int i) {
    int j;
    double u0[1];
    double X[2];

    U[0] = 0.;

    for (j = 0; j < n_quad; j++) {

        // get the 2d point on the mesh
        get_coordinates_2d(X, V, j);

        // evaluate U0 here
        U0(u0, X[0], X[1]);

        // evaluate U at the integration point
        U[0] += w[j] * u0[0] * basis[i * n_quad + j];
    }
}

/***********************
 *
 * ADVECTION FLUX
 *
 ***********************/
 /*
  * sets the flux for advection
 */
__device__ void eval_flux(double *U, double *flux_x, double *flux_y, 
                          double *V, double t, int j, int left_side) {

    double A[2];
    double X[2];

    // get the grid points on the mesh
    if (left_side >= 0) {
        get_coordinates_1d(X, V, j, left_side);
    } else {
        get_coordinates_2d(X, V, j);
    }

    get_velocity(A, X[0], X[1], t);

    // flux_1 
    flux_x[0] = A[0] * U[0];

    // flux_2
    flux_y[0] = A[1] * U[0];
}

/***********************
 *
 * RIEMAN SOLVER
 *
 ***********************/
/* finds the max absolute value of the jacobian for F(u).
 */
__device__ double eval_lambda(double *U_left, double *U_right,
                              double *V,      double t,
                              double nx,      double ny,
                              int j,          int left_side) {
                              
    double X[2];
    double A[2];

    get_coordinates_1d(X, V, j, left_side);
    // the speed of the wave
    get_velocity(A, X[0], X[1], t);
    return sqrt(A[0]*A[0] + A[1]*A[1]);
}

__device__ void riemann_solver_upwind(double *F_n, double *U_left, double *U_right,
                                      double *V, double t, 
                                      double nx, double ny, 
                                      int j, int left_side) {
    double flux_x_l[1], flux_x_r[1], flux_y_l[1], flux_y_r[1];
    double A[2];
    double X[2];

    // calculate the left and right fluxes
    eval_flux(U_left , flux_x_l, flux_y_l, V, t, j, left_side);
    eval_flux(U_right, flux_x_r, flux_y_r, V, t, j, left_side);

    // get velocity
    get_coordinates_1d(X, V, j, left_side);
    get_velocity(A, X[0], X[1], t);

    // get the direction of the wave wrt the normal
    if (nx * A[0] + ny * A[1] > 0) {
        F_n[0] = flux_x_l[0] * nx + flux_y_l[0] * ny;
    } else {
        F_n[0] = flux_x_r[0] * nx + flux_y_r[0] * ny;
    }
}
/* local lax-friedrichs riemann solver
 */
__device__ void riemann_solver(double *F_n, double *U_left, double *U_right,
                               double *V, double t,
                               double nx, double ny, 
                               int j, int left_side) {
    /*
    double flux_x_l[2], flux_x_r[2];
    double flux_y_l[2], flux_y_r[2];
    double lambda;
    int n;

    // calculate the left and right fluxes
    eval_flux(U_left , flux_x_l, flux_y_l, V, t, j, left_side);
    eval_flux(U_right, flux_x_r, flux_y_r, V, t, j, left_side);

    lambda = eval_lambda(U_left, U_right, V, t, nx, ny, j, left_side);

    // calculate the riemann problem at this integration point
    for (n = 0; n < N; n++) {
        F_n[n] = 0.5 * ((flux_x_l[n] + flux_x_r[n]) * nx + (flux_y_l[n] + flux_y_r[n]) * ny 
                    + lambda * (U_left[n] - U_right[n]));
    }
    */
    riemann_solver_upwind(F_n, U_left, U_right, V, t, nx, ny, j, left_side);
}

__device__ void check_physical(double *C_global, double *C, double *U, int idx) { 
    // do nothing
}

/***********************
 *
 * CFL CONDITION
 *
 ***********************/
/* global lambda evaluation
 *
 * computes the max eigenvalue of |u + c|, |u|, |u - c|.
 */
__global__ void eval_global_lambda(double *C, double *lambda, 
                                   double *V1x, double *V1y,
                                   double *V2x, double *V2y,
                                   double *V3x, double *V3y,
                                   double t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elem) { 

        double A[2];
        double X[2];
        double V[6];

        V[0] = V1x[idx];
        V[1] = V1y[idx];
        V[2] = V2x[idx];
        V[3] = V2y[idx];
        V[4] = V3x[idx];
        V[5] = V3y[idx];

        // get velocity at the center integration point in the cell
        get_coordinates_2d(X, V, n_quad / 2);
        get_velocity(A, X[0], X[1], t);

        // speed of the wave
        lambda[idx] = sqrt(A[0]*A[0] + A[1]*A[1]);
    }
}
