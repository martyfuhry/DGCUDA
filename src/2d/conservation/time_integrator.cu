/* time_integrator.cu
 *
 * time integration functions.
 */
#ifndef TIMEINTEGRATOR_H_GUARD
#define TIMEINTEGRATOR_H_GUARD
void checkCudaError(const char*);
#endif

extern int local_N;
extern int limiter;

void write_U(int, int, int);

/***********************
 * ASSEMBLE RHS FUNCTIONS
 ***********************/

/* right hand side
 *
 * computes the sum of the quadrature and the riemann flux for the 
 * coefficients for each element
 * THREADS: num_elem
 */
__global__ void eval_rhs(double *c, double *rhs_volume, double *rhs_surface_left, double *rhs_surface_right, 
                         int *elem_s1, int *elem_s2, int *elem_s3,
                         int *left_elem, double *J, double dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    double j;
    int i, s1_idx, s2_idx, s3_idx;
    int n;

    if (idx < num_elem) {

        // set to 0
        for (i = 0; i < n_p; i++) { 
            for (n = 0; n < N; n++) {
                c[num_elem * n_p * n + i * num_elem + idx] = 0.;
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
                c[num_elem * n_p * n + i * num_elem + idx] += dt / j * rhs_volume[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        // for the first edge, add either left or right surface integral
        if (idx == left_elem[s1_idx]) {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + i * num_elem + idx] += dt / j * rhs_surface_left[num_sides * n_p * n + i * num_sides + s1_idx];
                }
            }
        } else {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + i * num_elem + idx] += dt / j * rhs_surface_right[num_sides * n_p * n + i * num_sides + s1_idx];
                }
            }
        }
        // for the second edge, add either left or right surface integral
        if (idx == left_elem[s2_idx]) {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + i * num_elem + idx] += dt / j * rhs_surface_left[num_sides * n_p * n + i * num_sides + s2_idx];
                }
            }
        } else {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + i * num_elem + idx] += dt / j * rhs_surface_right[num_sides * n_p * n + i * num_sides + s2_idx];
                }
            }
        }
        // for the third edge, add either left or right surface integral
        if (idx == left_elem[s3_idx]) {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + i * num_elem + idx] += dt / j * rhs_surface_left[num_sides * n_p * n + i * num_sides + s3_idx];
                }
            }
        } else {
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    c[num_elem * n_p * n + i * num_elem + idx] += dt / j * rhs_surface_right[num_sides * n_p * n + i * num_sides + s3_idx];
                }
            }
        }
    }
}

/* tempstorage for RK
 * 
 * I need to store u + alpha * k_i into some temporary variable called kstar
 */
__global__ void rk_tempstorage(double *c, double *kstar, double*k, double alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N * n_p * num_elem) {
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
__global__ void rk4(double *c, double *k1, double *k2, double *k3, double *k4) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N * n_p * num_elem) {
        c[idx] += k1[idx]/6. + k2[idx]/3. + k3[idx]/3. + k4[idx]/6.;
    }
}



/* time integrate rk4
 *
 * uses fourth order runge-kutta time integration to solve the RHS.
 * returns the final time this runs to.
 */
double time_integrate_rk4(int local_num_elem, int local_num_sides, 
                          int local_n, int local_n_p,
                          double endtime, int total_timesteps, double min_r, 
                          int verbose, int convergence, int video, double tol) {
    int n_threads = 256;
    int i, vidnum;
    double dt, t;
    double *c;
    double conv;

    if (convergence) {
        c = (double *) malloc(local_num_elem * local_n_p * local_N * sizeof(double));
    }
    double *max_lambda = (double *) malloc(local_num_elem * sizeof(double));
    double max_l;

    int n_blocks_elem  = (local_num_elem  / n_threads) 
                       + ((local_num_elem  % n_threads) ? 1 : 0);
    int n_blocks_sides = (local_num_sides / n_threads) 
                       + ((local_num_sides % n_threads) ? 1 : 0);
    int n_blocks_rk    = ((local_N * local_n_p * local_num_elem) / n_threads) 
                       + (((local_N * local_n_p * local_num_elem) % n_threads) ? 1 : 0);

    t = 0;
    int timestep = 0;

    // limit before stage 1
    if (limiter) {
        limit_c<<<n_blocks_elem, n_threads>>>(d_c, d_elem_s1, d_elem_s2, d_elem_s3,
                                              d_left_elem, d_right_elem);
                                              
        cudaThreadSynchronize();
    }

    // write initial conditions if video
    vidnum = 0;
    if (video > 0) {
        if (timestep % video == 0) {
            write_U(local_num_elem, vidnum, total_timesteps);
            cudaThreadSynchronize();
            vidnum++;
        }
    }

    conv = 1.;
    printf("Computing...\n");
    while (t < endtime || (timestep < total_timesteps && total_timesteps != -1) || (convergence && conv > tol)) {
        // compute all the lambda values over each cell
        eval_global_lambda<<<n_blocks_elem, n_threads>>>(d_c, d_lambda, 
                                                         d_V1x, d_V1y,
                                                         d_V2x, d_V2y,
                                                         d_V3x, d_V3y,
                                                         t);

        // grab all the lambdas off the GPU and find the min one
        cudaMemcpy(max_lambda, d_lambda, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
        max_l = max_lambda[0];
        for (i = 0; i < local_num_elem; i++) {
            max_l = (max_lambda[i] > max_l) ? max_lambda[i] : max_l;
        }
        checkCudaError("error after eval_lambda");

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
        cudaThreadSynchronize();
        checkCudaError("error before stage 1: eval_surface");
        eval_surface<<<n_blocks_sides, n_threads>>>
                      (d_c, d_rhs_surface_left, d_rhs_surface_right, 
                       d_s_length, 
                       d_V1x, d_V1y,
                       d_V2x, d_V2y,
                       d_V3x, d_V3y,
                       d_left_elem, d_right_elem,
                       d_left_side_number, d_right_side_number,
                       d_Nx, d_Ny, t);

        checkCudaError("error after stage 1: eval_surface");

        eval_volume<<<n_blocks_elem, n_threads>>>
                        (d_c, d_rhs_volume, 
                         d_xr, d_yr, d_xs, d_ys,
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         t);
        cudaThreadSynchronize();

        checkCudaError("error after stage 1: eval_volume");

        eval_rhs<<<n_blocks_elem, n_threads>>>(d_k1, d_rhs_volume, d_rhs_surface_left, d_rhs_surface_right, 
                                              d_elem_s1, d_elem_s2, d_elem_s3, 
                                              d_left_elem, d_J, dt);
        cudaThreadSynchronize();

        if (limiter) {
            limit_c<<<n_blocks_elem, n_threads>>>(d_k1, d_elem_s1, d_elem_s2, d_elem_s3,
                                                  d_left_elem, d_right_elem);
                                                  
            cudaThreadSynchronize();
        }

        rk_tempstorage<<<n_blocks_rk, n_threads>>>(d_c, d_k2, d_k1, 0.5);
        cudaThreadSynchronize();
        checkCudaError("error after stage 1.");

        // stage 2
        eval_surface<<<n_blocks_sides, n_threads>>>
                        (d_k2, d_rhs_surface_left, d_rhs_surface_right, 
                         d_s_length, 
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, 
                         t + 0.5*dt);

        eval_volume<<<n_blocks_elem, n_threads>>>
                        (d_k2, d_rhs_volume, 
                         d_xr, d_yr, d_xs, d_ys,
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         t + 0.5*dt);
        cudaThreadSynchronize();

        eval_rhs<<<n_blocks_elem, n_threads>>>(d_k2, d_rhs_volume, d_rhs_surface_left, d_rhs_surface_right,
                                              d_elem_s1, d_elem_s2, d_elem_s3, 
                                              d_left_elem, d_J, dt);
        cudaThreadSynchronize();

        if (limiter) {
            limit_c<<<n_blocks_elem, n_threads>>>(d_k2, d_elem_s1, d_elem_s2, d_elem_s3,
                                                  d_left_elem, d_right_elem);
            cudaThreadSynchronize();
        }


        rk_tempstorage<<<n_blocks_rk, n_threads>>>(d_c, d_k3, d_k2, 0.5);
        cudaThreadSynchronize();

        checkCudaError("error after stage 2.");

        // stage 3
        eval_surface<<<n_blocks_sides, n_threads>>>
                        (d_k3, d_rhs_surface_left, d_rhs_surface_right, 
                         d_s_length, 
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, 
                         t + 0.5*dt);

        eval_volume<<<n_blocks_elem, n_threads>>>
                        (d_k3, d_rhs_volume, 
                         d_xr, d_yr, d_xs, d_ys,
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         t + 0.5*dt);
        cudaThreadSynchronize();

        eval_rhs<<<n_blocks_elem, n_threads>>>(d_k3, d_rhs_volume, d_rhs_surface_left, d_rhs_surface_right, 
                                              d_elem_s1, d_elem_s2, d_elem_s3, 
                                              d_left_elem, d_J, dt);
        cudaThreadSynchronize();

        if (limiter) {
            limit_c<<<n_blocks_elem, n_threads>>>(d_k3, d_elem_s1, d_elem_s2, d_elem_s3,
                                                  d_left_elem, d_right_elem);
            cudaThreadSynchronize();
        }

        rk_tempstorage<<<n_blocks_rk, n_threads>>>(d_c, d_k4, d_k3, 1.0);
        cudaThreadSynchronize();

        checkCudaError("error after stage 3.");

        // stage 4
        eval_surface<<<n_blocks_sides, n_threads>>>
                        (d_k4, d_rhs_surface_left, d_rhs_surface_right, 
                         d_s_length, 
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, 
                         t + dt);

        eval_volume<<<n_blocks_elem, n_threads>>>
                        (d_k4, d_rhs_volume, 
                         d_xr, d_yr, d_xs, d_ys,
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         t + dt);
        cudaThreadSynchronize();

        eval_rhs<<<n_blocks_elem, n_threads>>>(d_k4, d_rhs_volume, d_rhs_surface_left, d_rhs_surface_right, 
                                              d_elem_s1, d_elem_s2, d_elem_s3, 
                                              d_left_elem, d_J, dt);
        cudaThreadSynchronize();

        if (limiter) {
            limit_c<<<n_blocks_elem, n_threads>>>(d_k4, d_elem_s1, d_elem_s2, d_elem_s3,
                                                  d_left_elem, d_right_elem);
            cudaThreadSynchronize();
        }

        checkCudaError("error after stage 4.");
        
        // final stage
        rk4<<<n_blocks_rk, n_threads>>>(d_c, d_k1, d_k2, d_k3, d_k4);
        cudaThreadSynchronize();

        if (limiter) {
            limit_c<<<n_blocks_elem, n_threads>>>(d_c, d_elem_s1, d_elem_s2, d_elem_s3,
                                                  d_left_elem, d_right_elem);
            cudaThreadSynchronize();
        }


        // check the convergence
        if (convergence && timestep > 0) {
            check_convergence<<<n_blocks_rk, n_threads>>>(d_c_prev, d_c);
            cudaMemcpy(c, d_c_prev, local_num_elem * local_N * sizeof(double), cudaMemcpyDeviceToHost);

            conv = c[0];
            for (i = 1; i < local_num_elem * local_N; i++) {
                if (c[i] > conv) {
                    conv = c[i];
                }
            }
        }
        if (convergence) {
            cudaMemcpy(d_c_prev, d_c, local_num_elem * local_n_p * local_N * sizeof(double), cudaMemcpyDeviceToDevice);
        }


        cudaThreadSynchronize();
        checkCudaError("error after final stage.");

        // evaluate and write the solution
        if (video > 0) {
            if (timestep % video == 0) {
                write_U(local_num_elem, vidnum, total_timesteps);
                cudaThreadSynchronize();
                vidnum++;
            }
        }
    }

    printf("\n");
    free(max_lambda);
    if (convergence) {
        free(c);
    }
    return t;
}


/***********************
 * RK2 
 ***********************/

/* tempstorage for RK2
 * 
 * I need to store u + alpha * k_i into some temporary variable called k*.
 */
__global__ void rk2_tempstorage(double *c, double *kstar, double*k, double alpha, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N * n_p * num_elem) {
        kstar[idx] = c[idx] + alpha * k[idx];
    }
}

/* rk2
 *
 * computes the runge-kutta solution 
 * u_n+1 = u_n + k1/6 + k2/3 + k3/3 + k4/6
 */
__global__ void rk2(double *c, double *k) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N * n_p * num_elem) {
        c[idx] += k[idx];
    }
}

double time_integrate_rk2(int local_num_elem, int local_num_sides, 
                          int local_n, int local_n_p,
                          double endtime, int total_timesteps, double min_r, 
                          int verbose, int convergence, int video, double tol) {
    int n_threads = 512;
    int i, timestep;
    double *c;
    double dt, t;

    double *max_lambda = (double *) malloc(local_num_elem * sizeof(double));
    double max_l;
    double conv;

    int n_blocks_elem  = (local_num_elem  / n_threads) 
                       + ((local_num_elem  % n_threads) ? 1 : 0);
    int n_blocks_sides = (local_num_sides / n_threads) 
                       + ((local_num_sides % n_threads) ? 1 : 0);
    int n_blocks_rk    = ((local_N * local_n_p * local_num_elem) / n_threads) 
                       + (((local_N * local_n_p * local_num_elem) % n_threads) ? 1 : 0);

    if (convergence) {
        c = (double *) malloc(local_num_elem * local_n_p * local_N * sizeof(double));
    }

    t = 0;
    timestep = 0;

    conv = 1;
    printf("Computing...\n");
    while (t < endtime || (timestep < total_timesteps && total_timesteps != -1)) {
        // compute all the lambda values over each cell
        eval_global_lambda<<<n_blocks_elem, n_threads>>>(d_c, d_lambda, 
                                                         d_V1x, d_V1y,
                                                         d_V2x, d_V2y,
                                                         d_V3x, d_V3y, t);

        // just grab all the lambdas and sort them since there are so few of them
        cudaMemcpy(max_lambda, d_lambda, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
        max_l = max_lambda[0];
        for (i = 0; i < local_num_elem; i++) {
            max_l = (max_lambda[i] > max_l) ? max_lambda[i] : max_l;
        }

        timestep++;

        // cfl condition
        dt = 0.7 * min_r / max_l /  (2. * local_n + 1.);

        // panic
        if (isnan(dt)) {
            printf("Error: dt is NaN. Dumping...\n");
            return t;
        }

        if (t + dt > endtime && total_timesteps == -1) {
            dt = endtime - t;
            t = endtime;
        } else {
            t += dt;
        }

        if (verbose == 1) {
            printf("t = %lf, dt = %lf, max_l = %lf\n", t, dt, max_l);
        } else {
            printf("\rt = %lf", t);
        }

        // stage 1
        cudaThreadSynchronize();
        checkCudaError("error before stage 1: eval_surface");
        eval_surface<<<n_blocks_sides, n_threads>>>
                        (d_c, d_rhs_surface_left, d_rhs_surface_right, 
                         d_s_length, 
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, t);

        cudaThreadSynchronize();
        checkCudaError("error after stage 1: eval_surface");

        eval_volume<<<n_blocks_elem, n_threads>>>
                        (d_c, d_rhs_volume, 
                         d_xr, d_yr, d_xs, d_ys,
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y, t);
        cudaThreadSynchronize();

        checkCudaError("error after stage 1: eval_volume");

        eval_rhs<<<n_blocks_elem, n_threads>>>(d_k1, d_rhs_volume, d_rhs_surface_left, d_rhs_surface_right, 
                                              d_elem_s1, d_elem_s2, d_elem_s3, 
                                              d_left_elem, d_J, dt);
        cudaThreadSynchronize();

        if (limiter) {
            limit_c<<<n_blocks_elem, n_threads>>>(d_k1, d_elem_s1, d_elem_s2, d_elem_s3,
                                                  d_left_elem, d_right_elem);
        }

        rk_tempstorage<<<n_blocks_rk, n_threads>>>(d_c, d_k1, d_k1, 0.5);
        cudaThreadSynchronize();
        checkCudaError("error after stage 1.");

        // stage 2
        eval_surface<<<n_blocks_sides, n_threads>>>
                        (d_k1, d_rhs_surface_left, d_rhs_surface_right, 
                         d_s_length, 
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, t + 0.5*dt);

        eval_volume<<<n_blocks_elem, n_threads>>>
                        (d_k1, d_rhs_volume, 
                         d_xr, d_yr, d_xs, d_ys,
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         t + 0.5*dt);
        cudaThreadSynchronize();

        eval_rhs<<<n_blocks_elem, n_threads>>>(d_k1, d_rhs_volume, d_rhs_surface_left, d_rhs_surface_right,
                                              d_elem_s1, d_elem_s2, d_elem_s3, 
                                              d_left_elem, d_J, dt);
        cudaThreadSynchronize();

        if (limiter) {
            limit_c<<<n_blocks_elem, n_threads>>>(d_k1, d_elem_s1, d_elem_s2, d_elem_s3,
                                                  d_left_elem, d_right_elem);
        }
        checkCudaError("error after stage 2.");

        // final stage
        rk2<<<n_blocks_rk, n_threads>>>(d_c, d_k1);
        cudaThreadSynchronize();

        // check the convergence
        if (convergence && timestep > 0) {
            check_convergence<<<n_blocks_rk, n_threads>>>(d_c_prev, d_c);
            cudaMemcpy(c, d_c_prev, local_num_elem * local_N * sizeof(double), cudaMemcpyDeviceToHost);

            conv = c[0];
            for (i = 1; i < local_num_elem * local_N; i++) {
                if (c[i] > conv) {
                    conv = c[i];
                }
            }
        }
        if (convergence) {
            cudaMemcpy(d_c_prev, d_c, local_num_elem * local_n_p * local_N * sizeof(double), cudaMemcpyDeviceToDevice);
        }

        cudaThreadSynchronize();
        checkCudaError("error after final stage.");

    }

    printf("\n");
    free(max_lambda);
    //free(c);
    return t;
}
