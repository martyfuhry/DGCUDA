/* time_integrator.cu
 *
 * time integration functions.
 */
#ifndef TIMEINTEGRATOR_H_GUARD
#define TIMEINTEGRATOR_H_GUARD
#endif

extern int local_N;
extern int limiter;

void write_U(int, int, int);


/* right hand side
 *
 * computes the sum of the quadrature and the riemann flux for the 
 * coefficients for each element
 * THREADS: num_elem
 */
void eval_rhs(double *c, double *rhs_volume, double *rhs_surface_left, double *rhs_surface_right, 
              int *elem_s1, int *elem_s2, int *elem_s3,
              int *left_elem, double *J, double dt) {
    int idx;
    double j;
    int i, s1_idx, s2_idx, s3_idx;
    int n;

    for (idx = 0; idx < num_elem; idx++) {

        // set to 0
        for (i = 0; i < n_p; i++) { 
            for (n = 0; n < N; n++) {
                c[num_elem * n_p * n + idx * n_p + i] = 0.;
            }
        }

        // read jacobian determinant
        j = J[idx];

        // get the indicies for the riemann contributions for this element
        s1_idx = elem_s1[idx];
        s2_idx = elem_s2[idx];
        s3_idx = elem_s3[idx];

        // add volume integral
        for (i = 0; i < n_p; i++) {
            for (n = 0; n < N; n++) {
                c[num_elem * n_p * n + idx * n_p + i] += dt / j * rhs_volume[num_elem * n_p * n + idx * n_p + i];
            }
        }

        // for the first edge, add either left or right surface integral
        if (idx == left_elem[s1_idx]) {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + idx * n_p + i] += dt / j * rhs_surface_left[num_sides * n_p * n + s1_idx * n_p + i];
                }
            }
        } else {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + idx * n_p + i] += dt / j * rhs_surface_right[num_sides * n_p * n + s1_idx * n_p + i];
                }
            }
        }
        // for the second edge, add either left or right surface integral
        if (idx == left_elem[s2_idx]) {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + idx * n_p + i] += dt / j * rhs_surface_left[num_sides * n_p * n + s2_idx * n_p + i];
                }
            }
        } else {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + idx * n_p + i] += dt / j * rhs_surface_right[num_sides * n_p * n + s2_idx * n_p + i];
                }
            }
        }
        // for the third edge, add either left or right surface integral
        if (idx == left_elem[s3_idx]) {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + idx * n_p + i] += dt / j * rhs_surface_left[num_sides * n_p * n + s3_idx * n_p + i];
                }
            }
        } else {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + idx * n_p + i] += dt / j * rhs_surface_right[num_sides * n_p * n + s3_idx * n_p + i];
                }
            }
        }
    }
}

/* tempstorage for RK
 * 
 * I need to store u + alpha * k_i into some temporary variable called k*.
 */
void rk_tempstorage(double *c, double *kstar, double*k, double alpha) {
    int idx;

    for (idx = 0; idx < N * n_p * num_elem; idx++) {
        kstar[idx] = c[idx] + alpha * k[idx];
    }
}

/***********************
 * RK4 
 ***********************/

/* rk4
 *
 * computes the runge-kutta solution 
 * u_n+1 = u_n + k1/6 + k2/3 + k3/3 + k4/6
 */
void rk4(double *c, double *k1, double *k2, double *k3, double *k4) {
    int idx;

    for (idx = 0; idx < N * n_p * num_elem; idx++) {
        c[idx] += k1[idx]/6. + k2[idx]/3. + k3[idx]/3. + k4[idx]/6.;
    }
}

double time_integrate_rk4(int local_num_elem, int local_num_sides, 
                          int local_n, int local_n_p,
                          double endtime, int total_timesteps, double min_r, 
                          int verbose, int convergence, int video, double tol) {
    int i, vidnum;
    double dt, t;
    double *c;
    double conv;

    double max_l;

    t = 0;
    int timestep = 0;

    // limit before stage 1
    if (limiter) {
        //limit_c(d_c, d_elem_s1, d_elem_s2, d_elem_s3, d_left_elem, d_right_elem);
                                              
    }

    // write initial conditions if video
    vidnum = 0;
    if (video > 0) {
        if (timestep % video == 0) {
            write_U(local_num_elem, vidnum, total_timesteps);
            vidnum++;
        }
    }

    conv = 1.;
    printf("Computing...\n");
    while (t < endtime || (timestep < total_timesteps && total_timesteps != -1) || (convergence && conv > tol)) {
        // compute all the lambda values over each cell
        eval_global_lambda(d_c, d_lambda, 
                           d_V1x, d_V1y,
                           d_V2x, d_V2y,
                           d_V3x, d_V3y,
                           t);

        // find min lambda
        max_l = d_lambda[0];
        for (i = 0; i < local_num_elem; i++) {
            max_l = (d_lambda[i] > max_l) ? d_lambda[i] : max_l;
        }

        timestep++;

        //cfl condition
        dt = 0.7 * min_r / max_l /  (2. * local_n + 1.);

        // panic
        if (isnan(dt)) {
            printf("Error: dt is NaN. Dumping...\n");
            return t;
        }

        // get next timestep
        if (t + dt > endtime && total_timesteps == -1 && convergence != 1) {
            dt = endtime - t;
            t = endtime;
        } else {
            t += dt;
        }

        if (verbose == 1) {
            printf("(%i) t = %lf, dt = %lf, max_l = %lf\n", timestep, t, dt, max_l);
        } else if (convergence == 1)  {
            printf("\r(%i) t = %lf, convergence = %.015lf", timestep, t, conv);
        }
        else {
            printf("\r(%i) t = %lf", timestep, t);
        }

        // stage 1
        eval_surface (d_c, d_rhs_surface_left, d_rhs_surface_right, 
                      d_s_length, 
                      d_V1x, d_V1y,
                      d_V2x, d_V2y,
                      d_V3x, d_V3y,
                      d_left_elem, d_right_elem,
                      d_left_side_number, d_right_side_number,
                      d_Nx, d_Ny, t);


        eval_volume (d_c, d_rhs_volume, 
                     d_xr, d_yr, d_xs, d_ys,
                     d_V1x, d_V1y,
                     d_V2x, d_V2y,
                     d_V3x, d_V3y,
                     t);


        eval_rhs(d_k1, d_rhs_volume, d_rhs_surface_left, d_rhs_surface_right, 
                 d_elem_s1, d_elem_s2, d_elem_s3, 
                 d_left_elem, d_J, dt);

        if (limiter) {
            //limit_c(d_k1, d_elem_s1, d_elem_s2, d_elem_s3, d_left_elem, d_right_elem);
                                                  
        }

        rk_tempstorage(d_c, d_k2, d_k1, 0.5);

        // stage 2
        eval_surface (d_k2, d_rhs_surface_left, d_rhs_surface_right, 
                      d_s_length, 
                      d_V1x, d_V1y,
                      d_V2x, d_V2y,
                      d_V3x, d_V3y,
                      d_left_elem, d_right_elem,
                      d_left_side_number, d_right_side_number,
                      d_Nx, d_Ny, 
                      t + 0.5*dt);

        eval_volume (d_k2, d_rhs_volume, 
                     d_xr, d_yr, d_xs, d_ys,
                     d_V1x, d_V1y,
                     d_V2x, d_V2y,
                     d_V3x, d_V3y,
                     t + 0.5*dt);

        eval_rhs(d_k2, d_rhs_volume, d_rhs_surface_left, d_rhs_surface_right,
                 d_elem_s1, d_elem_s2, d_elem_s3, 
                 d_left_elem, d_J, dt);

        if (limiter) {
            //limit_c(d_k2, d_elem_s1, d_elem_s2, d_elem_s3, d_left_elem, d_right_elem);
        }

        rk_tempstorage(d_c, d_k3, d_k2, 0.5);

        // stage 3
        eval_surface (d_k3, d_rhs_surface_left, d_rhs_surface_right, 
                      d_s_length, 
                      d_V1x, d_V1y,
                      d_V2x, d_V2y,
                      d_V3x, d_V3y,
                      d_left_elem, d_right_elem,
                      d_left_side_number, d_right_side_number,
                      d_Nx, d_Ny, 
                      t + 0.5*dt);

        eval_volume (d_k3, d_rhs_volume, 
                     d_xr, d_yr, d_xs, d_ys,
                     d_V1x, d_V1y,
                     d_V2x, d_V2y,
                     d_V3x, d_V3y,
                     t + 0.5*dt);

        eval_rhs(d_k3, d_rhs_volume, d_rhs_surface_left, d_rhs_surface_right, 
                 d_elem_s1, d_elem_s2, d_elem_s3, 
                 d_left_elem, d_J, dt);

        if (limiter) {
            //limit_c(d_k3, d_elem_s1, d_elem_s2, d_elem_s3, d_left_elem, d_right_elem);
        }

        rk_tempstorage(d_c, d_k4, d_k3, 1.0);


        // stage 4
        eval_surface (d_k4, d_rhs_surface_left, d_rhs_surface_right, 
                      d_s_length, 
                      d_V1x, d_V1y,
                      d_V2x, d_V2y,
                      d_V3x, d_V3y,
                      d_left_elem, d_right_elem,
                      d_left_side_number, d_right_side_number,
                      d_Nx, d_Ny, 
                      t + dt);

        eval_volume (d_k4, d_rhs_volume, 
                     d_xr, d_yr, d_xs, d_ys,
                     d_V1x, d_V1y,
                     d_V2x, d_V2y,
                     d_V3x, d_V3y,
                     t + dt);

        eval_rhs(d_k4, d_rhs_volume, d_rhs_surface_left, d_rhs_surface_right, 
                 d_elem_s1, d_elem_s2, d_elem_s3, 
                 d_left_elem, d_J, dt);

        if (limiter) {
            //limit_c(d_k4, d_elem_s1, d_elem_s2, d_elem_s3, d_left_elem, d_right_elem);
        }

        
        // final stage
        rk4(d_c, d_k1, d_k2, d_k3, d_k4);

        if (limiter) {
            //limit_c(d_c, d_elem_s1, d_elem_s2, d_elem_s3, d_left_elem, d_right_elem);
        }

    }

    printf("\n");
    return t;
}
