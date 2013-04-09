#include "../main.cu"

/* euler_system.cu
 *
 * This file contains the relevant information for making a system to solve
 *
 * d_t [   rho   ] + d_x [     rho * u    ] + d_y [    rho * v     ] = 0
 * d_t [ rho * u ] + d_x [ rho * u^2 + p  ] + d_y [   rho * u * v  ] = 0
 * d_t [ rho * v ] + d_x [  rho * u * v   ] + d_y [  rho * v^2 + p ] = 0
 * d_t [    E    ] + d_x [ u * ( E +  p ) ] + d_y [ v * ( E +  p ) ] = 0
 *
 */

__device__ double get_mu();
__device__ double get_eps();

/* size of the system */
int local_N = 3;

/***********************
 *
 * INITIAL CONDITIONS
 *
 ***********************/

__device__ void evalU0(double *U, double *V, int i) {
    int j;
    double u0[3];
    double X[2];

    U[0] = 0.;
    U[1] = 0.;
    U[2] = 0.;

    for (j = 0; j < n_quad; j++) {

        // get the actual point on the mesh
        get_coordinates_2d(X, V, j);

        // evaluate U0 here
        U0(u0, X[0], X[1]);

        // evaluate U at the integration point
        U[0] += w[j] * u0[0] * basis[i * n_quad + j];
        U[1] += w[j] * u0[1] * basis[i * n_quad + j];
        U[2] += w[j] * u0[2] * basis[i * n_quad + j];
    }
}

__device__ double eval_c(double *U) {
    return 1./sqrt(get_eps() * get_mu());
}


__device__ void check_physical(double *C_global, double *C, double *U, int idx) { 
}

/***********************
 *
 * MAXWELLS FLUX
 *
 ***********************/
 /*
  * sets the flux for advection
  * U = (H_x, H_y, E_z)
 */
__device__ void eval_flux(double *U, double *flux_x, double *flux_y,
                          double *V, double t, int j, int left_side) {
    double mu, eps;

    mu  = get_mu();
    eps = get_eps();

    // flux_1 
    flux_x[0] = 0.;
    flux_x[1] = 1./mu * U[2];
    flux_x[2] = 1./eps * U[1];

    // flux_2
    flux_y[0] = -1./mu * U[2];
    flux_y[1] = 0.;
    flux_y[2] = -1./eps * U[0];

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
                              int j, int left_side) {
                              
    double c_left, c_right;

    // get c for both sides
    c_left  = abs(eval_c(U_left));
    c_right = abs(eval_c(U_right));

    return (c_left > c_right) ? c_left : c_right;
}
/* upwinding riemann solver
 */
__device__ void riemann_solver(double *F_n, double *U_left, double *U_right,
                                      double *V, double t, 
                                      double nx, double ny, 
                                      int j, int left_side) {
    /*
    double flux_x_l[3], flux_x_r[3], flux_y_l[3], flux_y_r[3];
    int n;

    // calculate the left and right fluxes
    eval_flux(U_left , flux_x_l, flux_y_l, V, t, j, left_side);
    eval_flux(U_right, flux_x_r, flux_y_r, V, t, j, left_side);

    // get the direction of the wave wrt the normal
    if (nx * U_left[0] + ny * U_left[1] > 0) {
        // use left element
        for (n = 0; n < N; n++) {
            F_n[n] = flux_x_l[n] * nx + flux_y_l[n] * ny;
        }
    } else {
        // use right element
        for(n = 0; n < N; n++) {
            F_n[n] = flux_x_r[n] * nx + flux_y_r[n] * ny;
        }
    }

    double UR[3];
    double flux_x[3];
    double flux_y[3];

    UR[0] = (U_left[0]+U_right[0])*(nx*ny/2 + ny*ny/2) + (U_left[1]+U_right[1])*(nx*nx/2-nx*ny/2) + (U_left[2]-U_right[2])*ny/2;
    UR[1] = (U_left[0]+U_right[0])*(-nx*ny/2 + ny*ny/2) + (U_left[1]+U_right[1])*(nx*nx/2 + nx*ny/2) + (U_left[2]-U_right[2])*(-nx)/2;
    UR[2] = ny*(U_left[0]-U_right[0])/2 + nx*(U_right[1]-U_left[1])/2 + (U_left[2]+U_right[2])/2;

    eval_flux(UR, flux_x, flux_y, V, t, j, left_side);

    F_n[0] = flux_x[0] * nx + flux_y[0] * ny;
    F_n[1] = flux_x[1] * nx + flux_y[1] * ny;
    F_n[2] = flux_x[2] * nx + flux_y[2] * ny;
    */
    int n;

    double flux_x_l[3], flux_x_r[3];
    double flux_y_l[3], flux_y_r[3];

    // calculate the left and right fluxes
    eval_flux(U_left, flux_x_l, flux_y_l, V, t, j, left_side);
    eval_flux(U_right, flux_x_r, flux_y_r, V, t, j, left_side);

    double lambda = eval_lambda(U_left, U_right, V, t, nx, ny, j, left_side);

    // calculate the riemann problem at this integration point
    for (n = 0; n < N; n++) {
        F_n[n] = 0.5 * ((flux_x_l[n] + flux_x_r[n]) * nx + (flux_y_l[n] + flux_y_r[n]) * ny 
                    + lambda * (U_left[n] - U_right[n]));
    }
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
        double U[3];

        lambda[idx] = abs(eval_c(U));
    }
}
