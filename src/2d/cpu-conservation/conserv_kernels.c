/* conserv_kernels.cu
 *
 * contains the GPU kernel functions for pre-evaluation of mesh variables
 * and the integration functions to solve the DG formulation for systems of
 * hyperbolic conservation laws in two-dimensions.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "conserv_headers.h"

#define N_MAX 4
#define NP_MAX 21

/***********************
 *
 * DEVICE VARIABLES
 *
 ***********************/
/* These are always prefixed with d_ for "device" */
 double *d_c;                 // coefficients for [rho, rho * u, rho * v, E]
 double *d_c_prev;            // coefficients for [rho, rho * u, rho * v, E]
 double *d_rhs_volume;          // the right hand side containing the quadrature contributions
 double *d_rhs_surface_left;  // the right hand side containing the left riemann contributions
 double *d_rhs_surface_right; // the right hand side containing the right riemann contributions

// TODO: switch to low storage runge-kutta
// runge kutta variables
 double *d_k1;
 double *d_k2;
 double *d_k3;
 double *d_k4;

// precomputed basis functions 
// TODO: maybe making these 2^n makes sure the offsets are cached more efficiently? who knows...
// precomputed basis functions ordered like so
//
// [phi_1(r1, s1), phi_1(r2, s2), ... , phi_1(r_nq, s_nq)   ]
// [phi_2(r1, s1), phi_2(r2, s2), ... , phi_2(r_nq, s_nq)   ]
// [   .               .           .            .           ]
// [   .               .           .            .           ]
// [   .               .           .            .           ]
// [phi_np(r1, s1), phi_np(r2, s2), ... , phi_np(r_nq, s_nq)]
//
 double *basis;
// note: these are the precomupted gradients of the basis functions
//       multiplied by the weights at that integration point
 double *basis_grad_x;
 double *basis_grad_y;

 int N;
 int n_p;
 int num_elem;
 int num_sides;
 int n_quad;
 int n_quad1d;

// precomputed basis functions evaluated along the sides. ordered
// similarly to basis and basis_grad_{x,y} but with one "matrix" for each side
// starting with side 0. to get to each side, offset with:
//      side_number * n_p * num_quad1d.
 double *basis_side;
 double *basis_vertex;

// weights for 2d and 1d quadrature rules
 double *w;
 double *w_oned;

// integration ponits for 2d and 1d quadrature rules
 double *r1;
 double *r2;
 double *r_oned;

void set_N(int value) {
    N = value;
}
void set_n_p(int value) {
    n_p = value;
}
void set_num_elem(int value) {
    num_elem = value;
}
void set_num_sides(int value) {
    num_sides = value;
}
void set_n_quad(int value) {
    n_quad = value;
}
void set_n_quad1d(int value) {
    n_quad1d = value;
}
void set_basis(void *value, int size) {
    basis = (double *)malloc(size * sizeof(double));
    memcpy(basis, value, size * sizeof(double));
}
void set_basis_grad_x(void *value, int size) {
    basis_grad_x = (double *)malloc(size * sizeof(double));
    memcpy(basis_grad_x, value, size * sizeof(double));
}
void set_basis_grad_y(void *value, int size) {
    basis_grad_y = (double *)malloc(size * sizeof(double));
    memcpy(basis_grad_y, value, size * sizeof(double));
}
void set_basis_side(void *value, int size) {
    basis_side = (double *)malloc(size * sizeof(double));
    memcpy(basis_side, value, size * sizeof(double));
}
void set_basis_vertex(void *value, int size) {
    basis_vertex = (double *)malloc(size * sizeof(double));
    memcpy(basis_vertex, value, size * sizeof(double));
}
void set_w(void *value, int size) {
    w = (double *)malloc(size * sizeof(double));
    memcpy(w, value, size * sizeof(double));
}
void set_w_oned(void *value, int size) {
    w_oned = (double *)malloc(size * sizeof(double));
    memcpy(w_oned, value, size * sizeof(double));
}
void set_r1(void *value, int size) {
    r1 = (double *)malloc(size * sizeof(double));
    memcpy(r1, value, size * sizeof(double));
}
void set_r2(void *value, int size) {
    r2 = (double *)malloc(size * sizeof(double));
    memcpy(r2, value, size * sizeof(double));
}
void set_r_oned(void *value, int size) {
    r_oned = (double *)malloc(size * sizeof(double));
    memcpy(r_oned, value, size * sizeof(double));
}

// tells which side (1, 2, or 3) to evaluate this boundary integral over
 int *d_left_side_number;
 int *d_right_side_number;

 double *d_J;         // jacobian determinant 
 double *d_reduction; // for the min / maxes in the reductions 
 double *d_lambda;    // stores computed lambda values for each element
 double *d_s_length;  // length of sides

// the num_elem values of the x and y coordinates for the two vertices defining a side
// TODO: can i delete these after the lengths are precomputed?
//       maybe these should be in texture memory?
 double *d_s_V1x;
 double *d_s_V1y;
 double *d_s_V2x;
 double *d_s_V2y;

// the num_elem values of the x and y partials
 double *d_xr;
 double *d_yr;
 double *d_xs;
 double *d_ys;

// the K indices of the sides for each element ranged 0->H-1
 int *d_elem_s1;
 int *d_elem_s2;
 int *d_elem_s3;

// vertex x and y coordinates on the mesh which define an element
// TODO: can i delete these after the jacobians are precomputed?
//       maybe these should be in texture memory?
 double *d_V1x;
 double *d_V1y;
 double *d_V2x;
 double *d_V2y;
 double *d_V3x;
 double *d_V3y;

// stores computed values at three vertices
 double *d_Uv1;
 double *d_Uv2;
 double *d_Uv3;

// for computing the error
 double *d_error;

// normal vectors for the sides
 double *d_Nx;
 double *d_Ny;

// index lists for sides
 int *d_left_elem;  // index of left  element for side idx
 int *d_right_elem; // index of right element for side idx

/* initial conditions
 *
 * computes the coefficients for the initial conditions
 * THREADS: num_elem
 */
 void init_conditions(double *c, double *J,
                                double *V1x, double *V1y,
                                double *V2x, double *V2y,
                                double *V3x, double *V3y) {
    int idx;
    for (idx = 0; idx < num_elem; idx++) {
        int i, n;
        double U[4];
        double V[6];

        if (idx < num_elem) {
            V[0] = V1x[idx];
            V[1] = V1y[idx];
            V[2] = V2x[idx];
            V[3] = V2y[idx];
            V[4] = V3x[idx];
            V[5] = V3y[idx];

            for (i = 0; i < n_p; i++) {
                // evaluate U times the i'th basis function
                evalU0(U, V, i);

                // store the coefficients
                for (n = 0; n < 4; n++) {
                    //GPU: c[num_elem * n_p * n + i * num_elem + idx] = U[n];
                    c[num_elem * n_p * n + idx * n_p + i] = U[n];
                }
            } 
        }
    }
}

/* gets the grid coordinates at the j'th integration ponit for 2d */
 void get_coordinates_2d(double *x, double *V, int j) {
    // x = x2 * r + x3 * s + x1 * (1 - r - s)
    x[0] = V[2] * r1[j] + V[4] * r2[j] + V[0] * (1 - r1[j] - r2[j]);
    x[1] = V[3] * r1[j] + V[5] * r2[j] + V[1] * (1 - r1[j] - r2[j]);
}

 /* gets the grid cooridinates at the j'th integration point for 1d */
 void get_coordinates_1d(double *x, double *V, int j, int left_side) {

    double r1_eval, r2_eval;

    // we need the mapping back to the grid space
    switch (left_side) {
        case 0: 
            r1_eval = 0.5 + 0.5 * r_oned[j];
            r2_eval = 0.;
            break;
        case 1: 
            r1_eval = (1. - r_oned[j]) / 2.;
            r2_eval = (1. + r_oned[j]) / 2.;
            break;
        case 2: 
            r1_eval = 0.;
            r2_eval = 0.5 + 0.5 * r_oned[n_quad1d - 1 - j];
            break;
    }

    // x = x2 * r + x3 * s + x1 * (1 - r - s)
    x[0] = V[2] * r1_eval + V[4] * r2_eval + V[0] * (1 - r1_eval - r2_eval);
    x[1] = V[3] * r1_eval + V[5] * r2_eval + V[1] * (1 - r1_eval - r2_eval);
}
 
/***********************
 *
 * PRECOMPUTING
 *
 ***********************/

/* side length computer
 *
 * precomputes the length of each side.
 * THREADS: num_sides
 */ 
 void preval_side_length(double *s_length, 
                              double *s_V1x, double *s_V1y, 
                              double *s_V2x, double *s_V2y) {
    int idx;

    for (idx = 0; idx < num_sides; idx++) {
        // compute and store the length of the side
        s_length[idx] = sqrt(pow(s_V1x[idx] - s_V2x[idx],2) + pow(s_V1y[idx] - s_V2y[idx],2));
    }
}

/* inscribed circle radius computing
 *
 * computes the radius of each inscribed circle. stores in d_J to find the minumum,
 * then we reuse d_J.
 */
 void preval_inscribed_circles(double *J,
                                    double *V1x, double *V1y,
                                    double *V2x, double *V2y,
                                    double *V3x, double *V3y) {
    int idx;

    for (idx = 0; idx < num_elem; idx++) {
        double a, b, c, k;
        a = sqrt(pow(V1x[idx] - V2x[idx], 2) + pow(V1y[idx] - V2y[idx], 2));
        b = sqrt(pow(V2x[idx] - V3x[idx], 2) + pow(V2y[idx] - V3y[idx], 2));
        c = sqrt(pow(V1x[idx] - V3x[idx], 2) + pow(V1y[idx] - V3y[idx], 2));

        k = 0.5 * (a + b + c);

        // for the diameter, we multiply by 2
        J[idx] = 2 * sqrt(k * (k - a) * (k - b) * (k - c)) / k;
    }
}

/* jacobian computing
 *
 * precomputes the jacobian determinant for each element.
 * THREADS: num_elem
 */
 void preval_jacobian(double *J, 
                           double *V1x, double *V1y, 
                           double *V2x, double *V2y, 
                           double *V3x, double *V3y) {
    int idx;

    for (idx = 0; idx < num_elem; idx++) {
        double x1, y1, x2, y2, x3, y3;

        // read vertex points
        x1 = V1x[idx];
        y1 = V1y[idx];
        x2 = V2x[idx];
        y2 = V2y[idx];
        x3 = V3x[idx];
        y3 = V3y[idx];

        // calculate jacobian determinant
        // x = x2 * r + x3 * s + x1 * (1 - r - s)
        J[idx] = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    }
}

/* evaluate normal vectors
 *
 * computes the normal vectors for each element along each side.
 * THREADS: num_sides
 *
 */
 void preval_normals(double *Nx, double *Ny, 
                          double *s_V1x, double *s_V1y, 
                          double *s_V2x, double *s_V2y,
                          double *V1x, double *V1y, 
                          double *V2x, double *V2y, 
                          double *V3x, double *V3y,
                          int *left_side_number) {

    int idx;

    for (idx = 0; idx < num_sides; idx++) {
        double x, y, length;
        double sv1x, sv1y, sv2x, sv2y;
    
        sv1x = s_V1x[idx];
        sv1y = s_V1y[idx];
        sv2x = s_V2x[idx];
        sv2y = s_V2y[idx];
    
        // lengths of the vector components
        x = sv2x - sv1x;
        y = sv2y - sv1y;
    
        // normalize
        length = sqrt(pow(x, 2) + pow(y, 2));

        // store the result
        Nx[idx] = -y / length;
        Ny[idx] =  x / length;
    }
}

 void preval_normals_direction(double *Nx, double *Ny, 
                          double *V1x, double *V1y, 
                          double *V2x, double *V2y, 
                          double *V3x, double *V3y,
                          int *left_elem, int *left_side_number) {

    int idx;

    for (idx = 0; idx < num_sides; idx++) {
        double new_x, new_y, dot;
        double initial_x, initial_y, target_x, target_y;
        double x, y;
        int left_idx, side;

        // get left side's vertices
        left_idx = left_elem[idx];
        side     = left_side_number[idx];

        // get the normal vector
        x = Nx[idx];
        y = Ny[idx];
    
        // make it point the correct direction by learning the third vertex point
        switch (side) {
            case 0: 
                target_x = V3x[left_idx];
                target_y = V3y[left_idx];
                initial_x = (V1x[left_idx] + V2x[left_idx]) / 2.;
                initial_y = (V1y[left_idx] + V2y[left_idx]) / 2.;
                break;
            case 1:
                target_x = V1x[left_idx];
                target_y = V1y[left_idx];
                initial_x = (V2x[left_idx] + V3x[left_idx]) / 2.;
                initial_y = (V2y[left_idx] + V3y[left_idx]) / 2.;
                break;
            case 2:
                target_x = V2x[left_idx];
                target_y = V2y[left_idx];
                initial_x = (V1x[left_idx] + V3x[left_idx]) / 2.;
                initial_y = (V1y[left_idx] + V3y[left_idx]) / 2.;
                break;
        }

        // create the vector pointing towards the third vertex point
        new_x = target_x - initial_x;
        new_y = target_y - initial_y;

        // find the dot product between the normal and new vectors
        dot = x * new_x + y * new_y;
        
        if (dot > 0) {
            Nx[idx] *= -1;
            Ny[idx] *= -1;
        }
    }
}

 void preval_partials(double *V1x, double *V1y,
                                double *V2x, double *V2y,
                                double *V3x, double *V3y,
                                double *xr,  double *yr,
                                double *xs,  double *ys) {
    int idx;
    for (idx = 0; idx < num_elem; idx++) {
        // evaulate the jacobians of the mappings for the chain rule
        // x = x2 * r + x3 * s + x1 * (1 - r - s)
        xr[idx] = V2x[idx] - V1x[idx];
        yr[idx] = V2y[idx] - V1y[idx];
        xs[idx] = V3x[idx] - V1x[idx];
        ys[idx] = V3y[idx] - V1y[idx];
    }
}

/***********************
 *
 * BOUNDARY CONDITIONS
 *
 ***********************/
/* Put the boundary conditions for the problem in here.
*/
 void inflow_boundary(double *U_left, double *U_right,
                                double *V, double nx, double ny,
                                int j, int left_side, double t) {

    double X[2];

    // get x, y coordinates
    get_coordinates_1d(X, V, j, left_side);

    U_inflow(U_right, X[0], X[1], t);
}

 void outflow_boundary(double *U_left, double *U_right,
                                 double *V, double nx, double ny,
                                 int j, int left_side, double t) {
    double X[2];

    // get x, y coordinates
    get_coordinates_1d(X, V, j, left_side);
    
    U_outflow(U_right, X[0], X[1], t);
}

 void reflecting_boundary(double *U_left, double *U_right,
                                    double *V, double nx, double ny, 
                                    int j, int left_side, double t) {
    double X[2];

    // get x, y coordinates
    get_coordinates_1d(X, V, j, left_side);
    
    U_reflection(U_left, U_right, X[0], X[1], t, nx, ny);
}

/***********************
 *
 * MAIN FUNCTIONS
 *
 ***********************/

/* limiter
 *
 * the standard Barth-Jespersen limiter for p = 1 
 *
 * THREADS: num_elem
 void limit_c(double *C,
                        int *elem_s1, int *elem_s2, int *elem_s3,
                        int *left_side_idx, int *right_side_idx) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elem) { 
        int i, n;
        double phi, min_phi;
        double diff, lim, umin, umax;
        int s1, s2, s3;
        int U1_idx, U2_idx, U3_idx;
        int vertex;

        double Uavg, U;
        //double U1max, U2max, U3max;
        //double U1min, U2min, U3min;
        double U1, U2, U3;
        double delmin, delmax;

        // get side index
        s1 = elem_s1[idx];
        s2 = elem_s2[idx];
        s3 = elem_s3[idx];

        // get neighboring element index
        U1_idx = (left_side_idx[s1] == idx) ? right_side_idx[s1] : left_side_idx[s1];
        U2_idx = (left_side_idx[s2] == idx) ? right_side_idx[s2] : left_side_idx[s2];
        U3_idx = (left_side_idx[s3] == idx) ? right_side_idx[s3] : left_side_idx[s3];

        // evaluate U_left 
        for (n = 0; n < N; n++) {

            // make sure Ui_idx isn't a boundary
            U1_idx = (U1_idx < 0) ? idx : U1_idx;
            U2_idx = (U2_idx < 0) ? idx : U2_idx;
            U3_idx = (U3_idx < 0) ? idx : U3_idx;

            // evaluate the centroid values
            Uavg = C[num_elem * n_p * n + idx]    * basis[0]; // the cell
            U1   = C[num_elem * n_p * n + U1_idx] * basis[0]; // neighbor 1
            U2   = C[num_elem * n_p * n + U2_idx] * basis[0]; // neighbor 2
            U3   = C[num_elem * n_p * n + U3_idx] * basis[0]; // neighbor 3

            // get minimum centroid value
            if (Uavg <= U1 && Uavg <= U2 && Uavg <= U3) {
                umin = Uavg;
            } else if (U1 <= Uavg && U1 <= U2 && U1 <= U3) {
                umin = U1;
            } else if (U2 <= Uavg && U2 <= U1 && U2 <= U3) {
                umin = U2;
            } else {
                umin = U3;
            }

            // get maximum centroid value
            if (Uavg >= U1 && Uavg >= U2 && Uavg >= U3) {
                umax = Uavg;
            } else if (U1 >= Uavg && U1 >= U2 && U1 >= U3) {
                umax = U1;
            } else if (U2 >= Uavg && U2 >= U1 && U2 >= U3) {
                umax = U2;
            } else {
                umax = U3;
            }

            // compute del u
            delmin = umin - Uavg;
            delmax = umax - Uavg;

            // use this as the min
            min_phi = 1.;

            // at each integration point
            for (vertex = 0; vertex < 3; vertex++) {
                // evaluate U
                U = 0.;
                for (i = 0; i < n_p; i++) {
                    U += C[num_elem * n_p * n + i * num_elem + idx] 
                         * basis_vertex[i * 3 + vertex];
                }

                // evaluate the difference
                diff = U - Uavg;

                // pick the min correct phi
                if (diff > 0) {
                    lim = delmax / diff;
                    phi = (1. < lim) ? 1. : lim;
                } else if (diff < 0) {
                    lim = delmin / diff;
                    phi = (1. < lim) ? 1. : lim;
                } else {
                    phi = 1.;
                    lim = 1.;
                }
                
                // venkatakrishnan
                phi = (lim*lim + 2*lim) / (lim*lim + lim + 2);

                // find min_phi
                min_phi = (phi < min_phi) ? phi : min_phi;
            }

            // limit the coefficients
            //C[num_elem * n_p * n + 0 * num_elem + idx] = min_phi / basis[0];
            //C[num_elem * n_p * n + 1 * num_elem + idx] = 0;
            //C[num_elem * n_p * n + 2 * num_elem + idx] = 0;
            C[num_elem * n_p * n + 1 * num_elem + idx] *= min_phi;
            C[num_elem * n_p * n + 2 * num_elem + idx] *= min_phi;
        }
    }
}
*/

 void eval_boundary(double *U_left, double *U_right, 
                              double *V, double nx, double ny,
                              int j, int left_side, double t, int right_idx) {
    switch (right_idx) {
        // reflecting 
        case -1: 
            reflecting_boundary(U_left, U_right,
                V, nx, ny,
                j, left_side, t);
            break;
        // outflow 
        case -2: 
            outflow_boundary(U_left, U_right,
                V, nx, ny,
                j, left_side, t);
            break;
        // inflow 
        case -3: 
            inflow_boundary(U_left, U_right,
                V, nx, ny, 
                j, left_side, t);
            break;
    }
}

/* left & right evaluator
 * 
 * calculate U_left and U_right at the integration point,
 * using boundary conditions when necessary.
 */
  void eval_left_right(double *C, double *C_left, double *C_right, 
                             double *U_left, double *U_right,
                             double nx, double ny,
                             double *V, int j, 
                             int left_side, int right_side,
                             int left_idx, int right_idx,
                             double t) { 

    int i, n;

    // set U to 0
    for (n = 0; n < N; n++) {
        U_left[n]  = 0.;
        U_right[n] = 0.;
    }

    //evaluate U at the integration points
    for (i = 0; i < n_p; i++) {
        for (n = 0; n < N; n++) {
            U_left[n] += C_left[n*n_p + i] * 
                         basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
        }
    }

    // make sure U_left is physical
    check_physical(C, C_left, U_left, left_idx);

    // boundaries are sorted to avoid warp divergence
    if (right_idx < 0) {
        eval_boundary(U_left, U_right, V, nx, ny, j, left_side, t, right_idx);
    } else {
        // evaluate the right side at the integration point
        for (i = 0; i < n_p; i++) {
            for (n = 0; n < N; n++) {
                U_right[n] += C_right[n*n_p + i] * 
                              basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - j - 1];
            }
        }

        // make sure U_right is physical
        check_physical(C, C_right, U_right, right_idx);
    }
}

/* surface integral evaluation
 *
 * evaluate all the riemann problems for each element.
 * THREADS: num_sides
 */
 void eval_surface(double *C, 
                             double *rhs_surface_left, double *rhs_surface_right, 
                             double *side_len, 
                             double *V1x, double *V1y,
                             double *V2x, double *V2y,
                             double *V3x, double *V3y,
                             int *left_elem,  int *right_elem,
                             int *left_side_number, int *right_side_number,
                             double *Nx, double *Ny, double t) {
    int idx;

    for (idx = 0; idx < num_sides; idx++) {
        int i, j, n;
        int left_idx, right_idx, left_side, right_side;
        double len, nx, ny;
        double C_left [N_MAX * NP_MAX];
        double C_right[N_MAX * NP_MAX];
        double U_left[N_MAX], U_right[N_MAX];
        double F_n[N_MAX];
        double V[6];

        // read edge data
        len = side_len[idx];
        nx  = Nx[idx];
        ny  = Ny[idx];
        left_idx   = left_elem[idx];
        right_idx  = right_elem[idx];
        left_side  = left_side_number[idx];
        right_side = right_side_number[idx];

        // get verticies
        V[0] = V1x[left_idx];
        V[1] = V1y[left_idx];
        V[2] = V2x[left_idx];
        V[3] = V2y[left_idx];
        V[4] = V3x[left_idx];
        V[5] = V3y[left_idx];

        // read coefficients
        if (right_idx > -1) {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    // GPU: C_left[n*n_p + i]  = C[num_elem * n_p * n + i * num_elem + left_idx];
                    // GPU: C_right[n*n_p + i] = C[num_elem * n_p * n + i * num_elem + right_idx];
                    C_left[n*n_p + i]  = C[num_elem * n_p * n + left_idx  * n_p + i];
                    C_right[n*n_p + i] = C[num_elem * n_p * n + right_idx * n_p + i];
                }
            }
        } else {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    // GPU: C_left[n*n_p + i]  = C[num_elem * n_p * n + i * num_elem + left_idx];
                    C_left[n*n_p + i]  = C[num_elem * n_p * n + left_idx  * n_p + i];
                }
            }
        }


        // set RHS to 0
        for (i = 0; i < n_p; i++) {
            for (n = 0; n < N; n++) {
                // GPU: rhs_surface_left[num_sides * n_p * n + i * num_sides + idx]  = 0.;
                // GPU: rhs_surface_right[num_sides * n_p * n + i * num_sides + idx] = 0.;
                rhs_surface_left[num_sides * n_p * n + idx * n_p + i] = 0.;;
                rhs_surface_right[num_sides * n_p * n + idx * n_p + i] = 0.;;
            }
        }

        // at each integration point
        for (j = 0; j < n_quad1d; j++) {

            // calculate the left and right values along the surface
            eval_left_right(C, C_left, C_right,
                            U_left, U_right,
                            nx, ny,
                            V, j, left_side, right_side,
                            left_idx, right_idx, t);

            // compute F_n(U_left, U_right)
            riemann_solver(F_n, U_left, U_right, V, t, nx, ny, j, left_side);

            // multiply across by phi_i at this integration point
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    // GPU: rhs_surface_left[num_sides * n_p * n + i * num_sides + idx]  += -len / 2 * (w_oned[j] * F_n[n] * basis_side[left_side  * n_p * n_quad1d + i * n_quad1d + j]);
                    // GPU: rhs_surface_right[num_sides * n_p * n + i * num_sides + idx] +=  len / 2 * (w_oned[j] * F_n[n] * basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - 1 - j]);
                    rhs_surface_left[num_sides * n_p * n + idx * n_p + i]  += -len / 2 * (w_oned[j] * F_n[n] * basis_side[left_side  * n_p * n_quad1d + i * n_quad1d + j]);
                    rhs_surface_right[num_sides * n_p * n + idx * n_p + i] +=  len / 2 * (w_oned[j] * F_n[n] * basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - 1 - j]);
                }
            }
        }
    }
}

/* volume integrals
 *
 * evaluates and adds the volume integral to the rhs vector
 * THREADS: num_elem
 */
 void eval_volume(double *C, double *rhs_volume, 
                            double *Xr, double *Yr, double *Xs, double *Ys,
                            double *V1x, double *V1y,
                            double *V2x, double *V2y,
                            double *V3x, double *V3y,
                            double t) {
    int idx;

    for (idx = 0; idx < num_elem; idx++) {
        double V[6];
        double C_left[N_MAX * NP_MAX];
        int i, j, k, n;
        double U[N_MAX];
        double flux_x[N_MAX], flux_y[N_MAX];
        double xr, yr, xs, ys;

        // read coefficients
        for (i = 0; i < n_p; i++) {
            for (n = 0; n < N; n++) {
                // GPU: C_left[n*n_p + i]  = C[num_elem * n_p * n + i * num_elem + idx];
                C_left[n*n_p + i]  = C[num_elem * n_p * n + idx  * n_p + i];
            }
        }

        // get element data
        xr = Xr[idx];
        yr = Yr[idx];
        xs = Xs[idx];
        ys = Ys[idx];
        
        // get verticies
        V[0] = V1x[idx];
        V[1] = V1y[idx];
        V[2] = V2x[idx];
        V[3] = V2y[idx];
        V[4] = V3x[idx];
        V[5] = V3y[idx];

        // set to 0
        for (i = 0; i < n_p; i++) {
            for (n = 0; n < N; n++) {
                // GPU: rhs_volume[num_elem * n_p * n + i * num_elem + idx] = 0.;
                rhs_volume[num_elem * n_p * n + idx * n_p + i] = 0.;
            }
        }

        // for each integration point
        for (j = 0; j < n_quad; j++) {

            // initialize to zero
            for (n = 0; n < N; n++) {
                U[n] = 0.;
            }

            // calculate at the integration point
            for (k = 0; k < n_p; k++) {
                for (n = 0; n < N; n++) {
                    U[n] += C_left[n*n_p + k] * basis[n_quad * k + j];
                }
            }

            // make sure U is physical
            check_physical(C, C_left, U, idx);

            // evaluate the flux
            eval_flux(U, flux_x, flux_y, V, t, j, -1);

            // multiply across by phi_i
            for (i = 0; i < n_p; i++) {
                // compute the sum
                //     [fx fy] * [y_s, -y_r; -x_s, x_r] * [phi_x phi_y]
                for (n = 0; n < N; n++) {
                    // GPU: rhs_volume[num_elem * n_p * n + i * num_elem + idx] += 
                    rhs_volume[num_elem * n_p * n + idx * n_p + i] += 
                              flux_x[n] * ( basis_grad_x[n_quad * i + j] * ys
                                           -basis_grad_y[n_quad * i + j] * yr)
                            + flux_y[n] * (-basis_grad_x[n_quad * i + j] * xs 
                                          + basis_grad_y[n_quad * i + j] * xr);
                }
            }
        }
    }
}

/*
 *
 * This was an attempt to measure the performance difference between putting
 * the entirity of the RHS calulation into a single kernel.
 * Here was the result:
 * n = 1, p = 4, 1000 timesteps of the rotating hill problem
 * Single Kernel:    24.42 seconds
 * Seperate Kernels: 15.28 seconds
 *
 * Certainly, seperate kernels is the way to go.
 *
 *
typedef struct {
    int idx;
    int left_elem;
    int right_elem;
    int left_side;
    int right_side;
    double len;
    double nx;
    double ny;
} edge;

typedef struct {
    int idx;
    double xr;
    double yr;
    double xs;
    double ys;
    double J;
    int elem_s1;
    int elem_s2;
    int elem_s3;
} element;

 void eval_volumenew(double *C, double *RHS, double *C_elem, 
                               double *V, element E, double t, double dt) {
    int i, j, k, n;
    double flux_x[4], flux_y[4];
    double U[4];

    // for each integration point
    for (j = 0; j < n_quad; j++) {

        // initialize to zero
        for (n = 0; n < N; n++) {
            U[n] = 0.;
        }

        // calculate at the integration point
        for (k = 0; k < n_p; k++) {
            for (n = 0; n < N; n++) {
                U[n] += C_elem[n*n_p + k] * basis[n_quad * k + j];
            }
        }

        // make sure U is physical
        check_physical(C, C_elem, U, E.idx);

        // evaluate the flux
        eval_flux(U, flux_x, flux_y, V, t, j, -1);

        // multiply across by phi_i
        for (i = 0; i < n_p; i++) {
            // compute the sum
            //     [fx fy] * [y_s, -y_r; -x_s, x_r] * [phi_x phi_y]
            for (n = 0; n < N; n++) {
                RHS[num_elem * n_p * n + i * num_elem + E.idx] += dt / E.J *(
                          flux_x[n] * ( basis_grad_x[n_quad * i + j] * E.ys
                                       -basis_grad_y[n_quad * i + j] * E.yr)
                        + flux_y[n] * (-basis_grad_x[n_quad * i + j] * E.xs 
                                      + basis_grad_y[n_quad * i + j] * E.xr));
            }
        }
    }
}
 void eval_surfacenew(double *C, double *RHS,
                                double *C_left, double *C_right,
                                double *V, element E, edge e, 
                                double t, double dt) {
    int i, j, n;
    double U_left[4], U_right[4];
    double F_n[4];

    // at each integration point
    for (j = 0; j < n_quad1d; j++) {

        // calculate the left and right values along the surface
        eval_left_right(C, C_left, C_right,
                        U_left, U_right,
                        e.nx, e.ny,
                        V, j, e.left_elem, e.right_elem,
                        e.left_side, e.right_side, t);

        // compute F_n(U_left, U_right)
        riemann_solver(F_n, U_left, U_right, V, t, e.nx, e.ny, j, e.left_side);

        // swap sides if we're on the right side
        e.len *= (e.left_side == E.idx) ? 1. : -1.;

        // multiply across by phi_i at this integration point
        for (i = 0; i < n_p; i++) {
            for (n = 0; n < N; n++) {
                RHS[num_elem * n_p * n + i * num_elem + E.idx] += dt / E.J * e.len / 2. * w_oned[j] * F_n[n] * basis_side[e.left_side  * n_p * n_quad1d + i * n_quad1d + j];
            }
        }
    }
}

 void eval_rhsnew(double *C, double *RHS,
                            int *elem_s1, int *elem_s2, int *elem_s3,
                            double *Xr, double *Yr, double *Xs, double *Ys,
                            double *J,
                            double *s_length,
                            double *V1x, double *V1y,
                            double *V2x, double *V2y,
                            double *V3x, double *V3y,
                            int *left_elem, int *right_elem,
                            int *left_side_number, int *right_side_number,
                            double *Nx, double *Ny,
                            double t, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elem) {
        int i, n, side;
        double C_left [MAX_N * MAX_P];
        double C_right[MAX_N * MAX_P];
        double V[6];

        V[0] = V1x[idx];
        V[1] = V1y[idx];
        V[2] = V2x[idx];
        V[3] = V2y[idx];
        V[4] = V3x[idx];
        V[5] = V3y[idx];

        edge e;
        element E;

        // create element
        E.xr = Xr[idx];
        E.yr = Yr[idx];
        E.xs = Xs[idx];
        E.ys = Ys[idx];
        E.J  = J[idx];
        E.elem_s1 = elem_s1[idx];
        E.elem_s2 = elem_s2[idx];
        E.elem_s3 = elem_s3[idx];

        // set to 0
        for (i = 0; i < n_p; i++) {
            for (n = 0; n < N; n++) {
                RHS[num_elem * n_p * n + i * num_elem + E.idx] = 0.;
            }
        }

        // read element coefficients
        for (i = 0; i < n_p; i++) {
            for (n = 0; n < N; n++) {
                C_left[n*n_p + i] = C[num_elem*n_p*n + i*num_elem + E.idx];
            }
        }

        // evaluate volume 
        eval_volumenew(C, RHS, C_left, V, E, t, dt);

        int side_idx[3] = {E.elem_s1, E.elem_s2, E.elem_s3};

        for (side = 0; side < 3; side++) {
            // create edge
            e.left_elem  = left_elem[side_idx[side]];
            e.right_elem = right_elem[side_idx[side]];
            e.left_side  = right_side_number[side_idx[side]];
            e.right_side = left_side_number[side_idx[side]];
            e.len        = s_length[side_idx[side]];
            e.nx         = Nx[side_idx[side]];
            e.ny         = Ny[side_idx[side]];

            // read right element coefficients
            if (e.right_elem > -1) {
                for (i = 0; i < n_p; i++) {
                    for (n = 0; n < N; n++) {
                        C_right[n*n_p + i] = C[num_elem*n_p*n + i*num_elem + e.right_elem];
                    }
                }
            }

            // evaluate surface integral
            eval_surfacenew(C, RHS, C_left, C_right, V, E, e, t, dt);
        }
    }
}
*/
 
/* evaluate u
 * 
 * evaluates rho and E at the three vertex points for output
 * THREADS: num_elem
 */
 void eval_u(double *C, 
                      double *Uv1, double *Uv2, double *Uv3, 
                      int n) {
    int idx;
    for (idx = 0; idx < num_elem; idx++) {
        int i;
        double uv1, uv2, uv3;

        // calculate values at the integration points
        uv1 = 0.;
        uv2 = 0.;
        uv3 = 0.;
        for (i = 0; i < n_p; i++) {
            // GPU: uv1 += C[num_elem * n_p * n + i * num_elem + idx] * basis_vertex[i * 3 + 0];
            // GPU: uv2 += C[num_elem * n_p * n + i * num_elem + idx] * basis_vertex[i * 3 + 1];
            // GPU: uv3 += C[num_elem * n_p * n + i * num_elem + idx] * basis_vertex[i * 3 + 2];
            
            uv1 += C[num_elem * n_p * n + idx * n_p + i] * basis_vertex[i * 3 + 0];
            uv2 += C[num_elem * n_p * n + idx * n_p + i] * basis_vertex[i * 3 + 1];
            uv3 += C[num_elem * n_p * n + idx * n_p + i] * basis_vertex[i * 3 + 2];
        }

        // store result
        Uv1[idx] = uv1;
        Uv2[idx] = uv2;
        Uv3[idx] = uv3;
    }
}


/* plot exact
 * 
 * evaluates rho and E at the three vertex points for output
 * THREADS: num_elem
 */
 void plot_exact(double *C, 
                           double *V1x, double *V1y,
                           double *V2x, double *V2y,
                           double *V3x, double *V3y,
                           double *Uv1, double *Uv2, double *Uv3, 
                           double t, int n) {
    int idx;
    for (idx = 0; idx < num_elem; idx++) {
        double U[4];

        // calculate values at the integration points
        U_exact(U, V1x[idx], V1y[idx], t);
        Uv1[idx] = U[n];
        U_exact(U, V2x[idx], V2y[idx], t);
        Uv2[idx] = U[n];
        U_exact(U, V3x[idx], V3y[idx], t);
        Uv3[idx] = U[n];
    }
}


/* plot error
 * 
 * evaluates rho and E at the three vertex points for output
 * THREADS: num_elem
 */
 void plot_error(double *C, 
                           double *V1x, double *V1y,
                           double *V2x, double *V2y,
                           double *V3x, double *V3y,
                           double *Uv1, double *Uv2, double *Uv3, 
                           double t, int n) {
    int idx;
    for (idx = 0; idx < num_elem; idx++) {
        int i;
        double uv1, uv2, uv3;
        double U[4];

        // calculate values at the integration points
        uv1 = 0.;
        uv2 = 0.;
        uv3 = 0.;
        for (i = 0; i < n_p; i++) {
            // GPU: uv1 += C[num_elem * n_p * n + i * num_elem + idx] * basis_vertex[i * 6 + 0];
            // GPU: uv2 += C[num_elem * n_p * n + i * num_elem + idx] * basis_vertex[i * 6 + 1];
            // GPU: uv3 += C[num_elem * n_p * n + i * num_elem + idx] * basis_vertex[i * 6 + 2];
            uv1 += C[num_elem * n_p * n + idx * n_p + i] * basis_vertex[i * 3 + 0];
            uv2 += C[num_elem * n_p * n + idx * n_p + i] * basis_vertex[i * 3 + 1];
            uv3 += C[num_elem * n_p * n + idx * n_p + i] * basis_vertex[i * 3 + 2];
        }

        // store the difference of error and exact
        U_exact(U, V1x[idx], V1y[idx], t);
        Uv1[idx] = uv1 - U[n];
        U_exact(U, V2x[idx], V2y[idx], t);
        Uv2[idx] = uv2 - U[n];
        U_exact(U, V3x[idx], V3y[idx], t);
        Uv3[idx] = uv3 - U[n];
    }
}

/* evaluate error
 * 
 * evaluates rho and E at the three vertex points for output
 * THREADS: num_elem
 */
 void eval_error_L1(double *C, double *error,
                              double *V1x, double *V1y,
                              double *V2x, double *V2y,
                              double *V3x, double *V3y,
                              double *J,
                              int n, double t) {
    int idx;
    for (idx = 0; idx < num_elem; idx++) {
        int i, j;
        double e;
        double X[2];
        double U;
        double u[4];
        double V[6];
        double detJ;

        V[0] = V1x[idx];
        V[1] = V1y[idx];
        V[2] = V2x[idx];
        V[3] = V2y[idx];
        V[4] = V3x[idx];
        V[5] = V3y[idx];

        detJ = J[idx];
        e = 0.;
        for (j = 0; j < n_quad; j++) {

            // get the grid points on the mesh
            get_coordinates_2d(X, V, j);
            
            // evaluate U at the integration point
            U = 0.;
            for (i = 0; i < n_p; i++) {
                // GPU: U += C[num_elem * n_p * n + i * num_elem + idx] * basis[i * n_quad + j];
                U += C[num_elem * n_p * n + idx * n_p + i] * basis[i * n_quad + j];
            }

            // get the exact solution
            U_exact(u, X[0], X[1], t);

            // evaluate the L1 error
            e += w[j] * fabs(u[n] - U) * detJ;
        }

        // store the result
        error[idx] = e;
    }
}

/* evaluate error
 * 
 * evaluates rho and E at the three vertex points for output
 * THREADS: num_elem
 */
 void eval_error_L2(double *C, double *error,
                              double *V1x, double *V1y,
                              double *V2x, double *V2y,
                              double *V3x, double *V3y,
                              double *J,
                              int n, double t) {
    int idx;
    for (idx = 0; idx < num_elem; idx++) {
        int i, j;
        double e;
        double X[2];
        double U;
        double u[4];
        double V[6];
        double detJ;

        V[0] = V1x[idx];
        V[1] = V1y[idx];
        V[2] = V2x[idx];
        V[3] = V2y[idx];
        V[4] = V3x[idx];
        V[5] = V3y[idx];

        detJ = J[idx];
        e = 0.;
        for (j = 0; j < n_quad; j++) {

            // get the grid points on the mesh
            get_coordinates_2d(X, V, j);
            
            // evaluate U at the integration point
            U = 0.;
            for (i = 0; i < n_p; i++) {
                U += C[num_elem * n_p * n + idx * n_p + i] * basis[i * n_quad + j];
            }

            // get the exact solution
            U_exact(u, X[0], X[1], t);

            // evaluate the L2 error
            e += w[j] * (u[n] - U) * (u[n] - U) * detJ;
        }

        // store the result
        error[idx] = e;
    }
}

/* check for convergence
 *
 * see if the difference in coefficients is less than the tolerance
 */
 void check_convergence(double *c_prev, double *c) {
    int idx;

    for (idx = 0; idx < num_elem; idx++) {
        c_prev[idx] = fabs(c[idx] - c_prev[idx]);
    }
}
