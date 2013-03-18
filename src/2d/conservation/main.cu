#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "conserv_kernels.cu"
#include "time_integrator.cu"
#include "quadrature.cu"
#include "basis.cu"

extern int local_N;
extern int limiter;
extern int time_integrator;

// limiter optoins
#define NO_LIMITER 0
#define LIMITER 1

// time integration options
#define RK4 1
#define RK2 2

// riemann solver options
#define LLF 1

/* 2dadvec_euler.cu
 * 
 * This file calls the kernels in 2dadvec_kernels_euler.cu for the 2D advection
 * DG method.
 */

void write_U(int local_num_elem, int num, int total_timesteps) {
    double *Uv1, *Uv2, *Uv3;
    double *V1x, *V1y, *V2x, *V2y, *V3x, *V3y;
    int i, n;
    int n_threads     = 512;
    int n_blocks_elem = (local_num_elem  / n_threads) + ((local_num_elem  % n_threads) ? 1 : 0);
    FILE *out_file;
    char out_filename[100];

    cudaMalloc((void **) &d_Uv1, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_Uv2, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_Uv3, local_num_elem * sizeof(double));

    // evaluate at the vertex points and copy over data
    Uv1 = (double *) malloc(local_num_elem * sizeof(double));
    Uv2 = (double *) malloc(local_num_elem * sizeof(double));
    Uv3 = (double *) malloc(local_num_elem * sizeof(double));
    V1x = (double *) malloc(local_num_elem * sizeof(double));
    V1y = (double *) malloc(local_num_elem * sizeof(double));
    V2x = (double *) malloc(local_num_elem * sizeof(double));
    V2y = (double *) malloc(local_num_elem * sizeof(double));
    V3x = (double *) malloc(local_num_elem * sizeof(double));
    V3y = (double *) malloc(local_num_elem * sizeof(double));

    cudaMemcpy(V1x, d_V1x, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V1y, d_V1y, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V2x, d_V2x, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V2y, d_V2y, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V3x, d_V3x, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V3y, d_V3y, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);

    // evaluate and write to file
    for (n = 0; n < local_N; n++) {
        eval_u<<<n_blocks_elem, n_threads>>>(d_c, d_Uv1, d_Uv2, d_Uv3, n);
        cudaMemcpy(Uv1, d_Uv1, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Uv2, d_Uv2, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Uv3, d_Uv3, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);

        if (num == total_timesteps) {
            sprintf(out_filename, "output/U%d-final.pos", n, num);
        } else {
            sprintf(out_filename, "output/video/U%d-%d.pos", n, num);
        }
        out_file  = fopen(out_filename , "w");
        fprintf(out_file, "View \"U%i \" {\n", n);
        for (i = 0; i < local_num_elem; i++) {
            fprintf(out_file, "ST (%.015lf,%.015lf,0,%.015lf,%.015lf,0,%.015lf,%.015lf,0) {%.015lf,%.015lf,%.015lf};\n", 
                                   V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                                   Uv1[i], Uv2[i], Uv3[i]);
        }
        fprintf(out_file,"};");
        fclose(out_file);
    }

    free(Uv1);
    free(Uv2);
    free(Uv3);
    free(V1x);
    free(V1y);
    free(V2x);
    free(V2y);
    free(V3x);
    free(V3y);
}


/* set quadrature 
 *
 * sets the 1d quadrature integration points and weights for the boundary integrals
 * and the 2d quadrature integration points and weights for the volume intergrals.
 */
void set_quadrature(int n,
                    double **r1_local, double **r2_local, double **w_local,
                    double **s_r, double **oned_w_local, 
                    int *local_n_quad, int *local_n_quad1d) {
    int i;
    /*
     * The sides are mapped to the canonical element, so we want the integration points
     * for the boundary integrals for sides s1, s2, and s3 as shown below:

     s (r2) |\
     ^      | \
     |      |  \
     |      |   \
     |   s3 |    \ s2
     |      |     \
     |      |      \
     |      |       \
     |      |________\
     |         s1
     |
     ------------------------> r (r1)

    *
    */
    switch (n) {
        case 0: *local_n_quad = 1;
                *local_n_quad1d = 1;
                break;
        case 1: *local_n_quad = 3;
                *local_n_quad1d = 2;
                break;
        case 2: *local_n_quad = 6;
                *local_n_quad1d = 3;
                break;
        case 3: *local_n_quad = 12 ;
                *local_n_quad1d = 4;
                break;
        case 4: *local_n_quad = 16;
                *local_n_quad1d = 5;
                break;
        case 5: *local_n_quad = 25;
                *local_n_quad1d = 6;
                break;
    }
    // allocate integration points
    *r1_local = (double *)  malloc(*local_n_quad * sizeof(double));
    *r2_local = (double *)  malloc(*local_n_quad * sizeof(double));
    *w_local  = (double *) malloc(*local_n_quad * sizeof(double));

    *s_r = (double *) malloc(*local_n_quad1d * sizeof(double));
    *oned_w_local = (double *) malloc(*local_n_quad1d * sizeof(double));

    // set 2D quadrature rules
    for (i = 0; i < *local_n_quad; i++) {
        if (n > 0) {
            (*r1_local)[i] = quad_2d[2 * n - 1][3*i];
            (*r2_local)[i] = quad_2d[2 * n - 1][3*i+1];
            (*w_local) [i] = quad_2d[2 * n - 1][3*i+2] / 2.; //weights are 2 times too big for some reason
        } else {
            (*r1_local)[i] = quad_2d[0][3*i];
            (*r2_local)[i] = quad_2d[0][3*i+1];
            (*w_local) [i] = quad_2d[0][3*i+2] / 2.; //weights are 2 times too big for some reason
        }
    }

    // set 1D quadrature rules
    for (i = 0; i < *local_n_quad1d; i++) {
        (*s_r)[i] = quad_1d[n][2*i];
        (*oned_w_local)[i] = quad_1d[n][2*i+1];
    }
}

void checkCudaError(const char *message)
{
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
        exit(-1);
    }
}

void read_mesh(FILE *mesh_file, 
              int *local_num_sides,
              int *local_num_elem,
              double **V1x, double **V1y,
              double **V2x, double **V2y,
              double **V3x, double **V3y,
              int **left_side_number, int **right_side_number,
              double **sides_x1, double **sides_y1,
              double **sides_x2, double **sides_y2,
              int **elem_s1,  int **elem_s2, int **elem_s3,
              int **left_elem, int **right_elem) {

    int i, items;
    char line[1000];
    // stores the number of sides this element has.

    fgets(line, 1000, mesh_file);
    sscanf(line, "%i", local_num_elem);
    *V1x = (double *) malloc(*local_num_elem * sizeof(double));
    *V1y = (double *) malloc(*local_num_elem * sizeof(double));
    *V2x = (double *) malloc(*local_num_elem * sizeof(double));
    *V2y = (double *) malloc(*local_num_elem * sizeof(double));
    *V3x = (double *) malloc(*local_num_elem * sizeof(double));
    *V3y = (double *) malloc(*local_num_elem * sizeof(double));

    *elem_s1 = (int *) malloc(*local_num_elem * sizeof(int));
    *elem_s2 = (int *) malloc(*local_num_elem * sizeof(int));
    *elem_s3 = (int *) malloc(*local_num_elem * sizeof(int));

    // read the elements from the mesh
    for (i = 0; i < *local_num_elem; i++) {
        fgets(line, sizeof(line), mesh_file);
        items = sscanf(line, "%lf %lf %lf %lf %lf %lf %i %i %i", &(*V1x)[i], &(*V1y)[i], 
                                         &(*V2x)[i], &(*V2y)[i], 
                                         &(*V3x)[i], &(*V3y)[i], 
                                         &(*elem_s1)[i], &(*elem_s2)[i], &(*elem_s3)[i]);

        if (items != 9) {
            printf("error: not enough items (%i) while reading elements from mesh.\n", items);
            exit(0);
        }

    }

    fgets(line, 1000, mesh_file);
    sscanf(line, "%i", local_num_sides);

    *left_side_number  = (int *)   malloc(*local_num_sides * sizeof(int));
    *right_side_number = (int *)   malloc(*local_num_sides * sizeof(int));

    *sides_x1    = (double *) malloc(*local_num_sides * sizeof(double));
    *sides_x2    = (double *) malloc(*local_num_sides * sizeof(double));
    *sides_y1    = (double *) malloc(*local_num_sides * sizeof(double));
    *sides_y2    = (double *) malloc(*local_num_sides * sizeof(double)); 

    *left_elem   = (int *) malloc(*local_num_sides * sizeof(int));
    *right_elem  = (int *) malloc(*local_num_sides * sizeof(int));

    // read the edges from the mesh
    for (i = 0; i < *local_num_sides; i++) {
        fgets(line, sizeof(line), mesh_file);
        items = sscanf(line, "%lf %lf %lf %lf %i %i %i %i", &(*sides_x1)[i], &(*sides_y1)[i],
                                            &(*sides_x2)[i], &(*sides_y2)[i],
                                            &(*left_elem)[i], &(*right_elem)[i],
                                            &(*left_side_number)[i],
                                            &(*right_side_number)[i]);

        if (items != 8) {
            printf("error: not enough items (%i) while reading edges from mesh.\n", items);
            exit(0);
        }
    }
}

void init_gpu(int local_num_elem, int local_num_sides, int local_n_p,
              double *V1x, double *V1y, 
              double *V2x, double *V2y, 
              double *V3x, double *V3y, 
              int *left_side_number, int *right_side_number,
              double *sides_x1, double *sides_y1,
              double *sides_x2, double *sides_y2,
              int *elem_s1, int *elem_s2, int *elem_s3,
              int *left_elem, int *right_elem,
              int convergence, int eval_error) {

    checkCudaError("error before init.");
    cudaDeviceReset();

    double total_memory = local_num_elem*12*sizeof(double)  +
                   local_num_elem*3*sizeof(int)      +
                   local_num_sides*3*sizeof(double) + 
                   local_num_sides*4*sizeof(int)     +
                   local_N*local_num_elem*local_n_p*2*sizeof(double) +
                   local_N*local_num_sides*local_n_p*2*sizeof(double);

    switch (time_integrator) {
        case RK4: total_memory += 4*local_N*local_num_elem*local_n_p*sizeof(double);
                  break;
        case RK2: total_memory += 1*local_N*local_num_elem*local_n_p*sizeof(double);
                  break;
    }

    if (convergence) {
        total_memory += local_N*local_num_elem*local_n_p*sizeof(double);
    }

    if (total_memory < 1e3) {
        printf("Total memory required: %lf B\n", total_memory);
    } else if (total_memory >= 1e3 && total_memory < 1e6) {
        printf("Total memory required: %lf KB\n", total_memory * 1e-3);
    } else if (total_memory >= 1e6 && total_memory < 1e9) {
        printf("Total memory required: %lf MB\n", total_memory * 1e-6);
    } else {
        printf("Total memory required: %lf GB\n", total_memory * 1e-9);
    }

    cudaMalloc((void **) &d_c, local_N * local_num_elem * local_n_p * sizeof(double));
    if (convergence) {
        cudaMalloc((void **) &d_c_prev, local_N * local_num_elem * local_n_p * sizeof(double));
    }
    cudaMalloc((void **) &d_rhs_volume, local_N * local_num_elem * local_n_p * sizeof(double));
    cudaMalloc((void **) &d_rhs_surface_left,  local_N * local_num_sides * local_n_p * sizeof(double));
    cudaMalloc((void **) &d_rhs_surface_right, local_N * local_num_sides * local_n_p * sizeof(double));

    switch (time_integrator) {
        case RK4: 
            cudaMalloc((void **) &d_k1, local_N * local_num_elem * local_n_p * sizeof(double));
            cudaMalloc((void **) &d_k2, local_N * local_num_elem * local_n_p * sizeof(double));
            cudaMalloc((void **) &d_k3, local_N * local_num_elem * local_n_p * sizeof(double));
            cudaMalloc((void **) &d_k4, local_N * local_num_elem * local_n_p * sizeof(double));
            checkCudaError("error after gpu malloc");
            break;
        case RK2: 
            cudaMalloc((void **) &d_k1, local_N * local_num_elem * local_n_p * sizeof(double));
            checkCudaError("error after gpu malloc");
            break;
        default:
            printf("Error selecting time integrator.\n");
            exit(0);
    }

    cudaMalloc((void **) &d_J        , local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_lambda   , local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_s_length , local_num_sides * sizeof(double));

    cudaMalloc((void **) &d_s_V1x, local_num_sides * sizeof(double));
    cudaMalloc((void **) &d_s_V2x, local_num_sides * sizeof(double));
    cudaMalloc((void **) &d_s_V1y, local_num_sides * sizeof(double));
    cudaMalloc((void **) &d_s_V2y, local_num_sides * sizeof(double));

    cudaMalloc((void **) &d_elem_s1, local_num_elem * sizeof(int));
    cudaMalloc((void **) &d_elem_s2, local_num_elem * sizeof(int));
    cudaMalloc((void **) &d_elem_s3, local_num_elem * sizeof(int));

    cudaMalloc((void **) &d_V1x, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_V1y, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_V2x, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_V2y, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_V3x, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_V3y, local_num_elem * sizeof(double));

    cudaMalloc((void **) &d_xr, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_yr, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_xs, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_ys, local_num_elem * sizeof(double));

    cudaMalloc((void **) &d_left_side_number , local_num_sides * sizeof(int));
    cudaMalloc((void **) &d_right_side_number, local_num_sides * sizeof(int));

    cudaMalloc((void **) &d_Nx, local_num_sides * sizeof(double));
    cudaMalloc((void **) &d_Ny, local_num_sides * sizeof(double));

    cudaMalloc((void **) &d_right_elem, local_num_sides * sizeof(int));
    cudaMalloc((void **) &d_left_elem , local_num_sides * sizeof(int));
    checkCudaError("error after gpu malloc");

    // copy over data
    cudaMemcpy(d_s_V1x, sides_x1, local_num_sides * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V1y, sides_y1, local_num_sides * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V2x, sides_x2, local_num_sides * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V2y, sides_y2, local_num_sides * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_left_side_number , left_side_number , local_num_sides * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_side_number, right_side_number, local_num_sides * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_elem_s1, elem_s1, local_num_elem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elem_s2, elem_s2, local_num_elem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elem_s3, elem_s3, local_num_elem * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError("error after gpu copy");

    cudaMemcpy(d_V1x, V1x, local_num_elem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V1y, V1y, local_num_elem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2x, V2x, local_num_elem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2y, V2y, local_num_elem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V3x, V3x, local_num_elem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V3y, V3y, local_num_elem * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_left_elem , left_elem , local_num_sides * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_elem, right_elem, local_num_sides * sizeof(int), cudaMemcpyHostToDevice);
}

void free_gpu(int free, int convergence, int eval_error) {
    switch (free) {
        // free precomputed stuff
        case 1:
            cudaFree(d_s_V1x);
            cudaFree(d_s_V2x);
            cudaFree(d_s_V1y);
            cudaFree(d_s_V2y);
            break;

        // free everything but the coefficients and verticies
        case 2:
            cudaFree(d_s_length);
            cudaFree(d_lambda);
            switch (time_integrator) {
                case RK4: 
                    cudaFree(d_k1);
                    cudaFree(d_k2);
                    cudaFree(d_k3);
                    cudaFree(d_k4);
                    break;
                case RK2:
                    cudaFree(d_k1);
                    break;
             }
            cudaFree(d_rhs_volume);
            cudaFree(d_rhs_surface_left);
            cudaFree(d_rhs_surface_right);
            cudaFree(d_elem_s1);
            cudaFree(d_elem_s2);
            cudaFree(d_elem_s3);
            cudaFree(d_xr);
            cudaFree(d_yr);
            cudaFree(d_xs);
            cudaFree(d_ys);

            cudaFree(d_left_side_number);
            cudaFree(d_right_side_number);

            cudaFree(d_Nx);
            cudaFree(d_Ny);

            cudaFree(d_right_elem);
            cudaFree(d_left_elem);
            break;

        // free everything else
        case 3:
            cudaFree(d_c);
            cudaFree(d_J);
            if (convergence) {
                cudaFree(d_c_prev);
            }

            if (eval_error) {
                cudaFree(d_error);
            }

            cudaFree(d_Uv1);
            cudaFree(d_Uv2);
            cudaFree(d_Uv3);
            cudaFree(d_V1x);
            cudaFree(d_V1y);
            cudaFree(d_V2x);
            cudaFree(d_V2y);
            cudaFree(d_V3x);
            cudaFree(d_V3y);
            break;
    }
}

void usage_error() {
    printf("\nUsage: dgcuda [OPTIONS] [MESH] [OUTFILE]\n");
    printf(" Options: [-n] Order of polynomial approximation.\n");
    printf("          [-t] Number of timesteps.\n");
    printf("          [-T] End time.\n");
    printf("          [-v] Verbose.\n");
    printf("          [-c] Run to convergence of tolerance TOL\n");
    printf("          [-V] Print out every N timesteps.\n");
    printf("          [-e] Evaluate the error. Requires an exact solution.\n");
    printf("          [-b] Benchmark.\n");
    printf("          [-h] Display this message.\n");
}

int get_input(int argc, char *argv[],
               int *n, int *total_timesteps, 
               double *endtime,
               int *verbose,
               int *video,
               int *convergence,
               double *tol,
               int *benchmark,
               int *eval_error,
               char **mesh_filename) {

    int i;

    *total_timesteps = -1;
    *verbose         = 0;
    *convergence     = 0;
    *eval_error      = 0;
    *benchmark       = 0;
    *video           = 0;
    
    // read command line input
    if (argc < 5) {
        usage_error();
        return 1;
    }
    for (i = 0; i < argc; i++) {
        // order of polynomial
        if (strcmp(argv[i], "-n") == 0) {
            if (i + 1 < argc) {
                *n = atoi(argv[i+1]);
                if (*n < 0 || *n > 5) {
                    usage_error();
                    return 1;
                }
            } else {
                usage_error();
                return 1;
            }
        }
        // number of total_timesteps
        if (strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                *total_timesteps = atoi(argv[i+1]);
                if (*total_timesteps < 0) {
                    usage_error();
                    return 1;
                }
            } else {
                usage_error();
                return 1;
            }
        }
        if (strcmp(argv[i], "-T") == 0) {
            if (i + 1 < argc) {
                *endtime = atof(argv[i+1]);
                if (*endtime < 0) {
                    usage_error();
                    return 1;
                }
            } else {
                usage_error();
                return 1;
            }
        }
        if (strcmp(argv[i], "-V") == 0) {
            if (i + 1 < argc) {
                *video = atof(argv[i+1]);
                if (*video < 0) {
                    usage_error();
                    return 1;
                }
            } else {
                usage_error();
                return 1;
            }
        }
        if (strcmp(argv[i], "-c") == 0) {
            if (i + 1 < argc) {
                *convergence = 1;
                *tol         = atof(argv[i+1]);
                if (*tol < 0) {
                    usage_error();
                    return 1;
                }
            }
        }
        if (strcmp(argv[i], "-b") == 0) {
            *benchmark = 1;
        }
        if (strcmp(argv[i], "-e") == 0) {
            *eval_error = 1;
        }
        if (strcmp(argv[i], "-v") == 0) {
            *verbose = 1;
        }
        if (strcmp(argv[i], "-V") == 0) {
            if (i + 1 < argc) {
                *video = atof(argv[i+1]);
                printf("video = %i\n", *video);
                if (*video < 0) {
                    usage_error();
                    return 1;
                }
            } else {
                usage_error();
                return 1;
            }
        }
        if (strcmp(argv[i], "-h") == 0) {
                usage_error();
                return 1;
        }
    } 

    // second last argument is filename
    *mesh_filename = argv[argc - 1];

    return 0;
}
int run_dgcuda(int argc, char *argv[]) {
    checkCudaError("error before start.");
    int local_num_elem, local_num_sides;
    int n_threads, n_blocks_elem, n_blocks_sides;
    int i, n, local_n_p, total_timesteps, local_n_quad, local_n_quad1d;
    int verbose, convergence, video, eval_error, benchmark;

    double endtime, t;
    double tol, total_error, max_error;
    double *min_radius;
    double min_r;
    double *V1x, *V1y, *V2x, *V2y, *V3x, *V3y;
    double *sides_x1, *sides_x2;
    double *sides_y1, *sides_y2;

    double *r1_local, *r2_local, *w_local;

    double *s_r, *oned_w_local;

    int *left_elem, *right_elem;
    int *elem_s1, *elem_s2, *elem_s3;
    int *left_side_number, *right_side_number;

    FILE *mesh_file, *out_file;

    char out_filename[100];
    char *mesh_filename;

    double *Uv1, *Uv2, *Uv3;
    double *error;

    clock_t start, end;
    double elapsed;

    // get input 
    endtime = -1;
    if (get_input(argc, argv, &n, &total_timesteps, &endtime, 
                              &verbose, &video, &convergence, &tol, 
                              &benchmark, &eval_error, 
                              &mesh_filename)) {
        return 1;
    }

    // set the order of the approximation & timestep
    local_n_p = (n + 1) * (n + 2) / 2;

    // sanity check on limiter
    if (limiter && n != 1) {
        printf("Error: limiter only enabled for p = 1\n");
        exit(0);
    }

    // open the mesh to get local_num_elem for allocations
    mesh_file = fopen(mesh_filename, "r");
    if (!mesh_file) {
        printf("\nERROR: mesh file not found.\n");
        return 1;
    }

    // read in the mesh and make all the mappings
    read_mesh(mesh_file, &local_num_sides, &local_num_elem,
                         &V1x, &V1y, &V2x, &V2y, &V3x, &V3y,
                         &left_side_number, &right_side_number,
                         &sides_x1, &sides_y1, 
                         &sides_x2, &sides_y2, 
                         &elem_s1, &elem_s2, &elem_s3,
                         &left_elem, &right_elem);

    // close the file
    fclose(mesh_file);

    // initialize the gpu
    init_gpu(local_num_elem, local_num_sides, local_n_p,
             V1x, V1y, V2x, V2y, V3x, V3y,
             left_side_number, right_side_number,
             sides_x1, sides_y1,
             sides_x2, sides_y2, 
             elem_s1, elem_s2, elem_s3,
             left_elem, right_elem,
             convergence, eval_error);

    // get the correct quadrature rules for this scheme
    set_quadrature(n, &r1_local, &r2_local, &w_local, 
                   &s_r, &oned_w_local, &local_n_quad, &local_n_quad1d);

    // set constant data
    set_N(local_N);
    set_n_p(local_n_p);
    set_num_elem(local_num_elem);
    set_num_sides(local_num_sides);
    set_n_quad(local_n_quad);
    set_n_quad1d(local_n_quad1d);

    checkCudaError("error after gpu init.");
    n_threads          = 512;
    n_blocks_elem      = (local_num_elem  / n_threads) + ((local_num_elem  % n_threads) ? 1 : 0);
    n_blocks_sides     = (local_num_sides / n_threads) + ((local_num_sides % n_threads) ? 1 : 0);

    // find the min inscribed circle
    preval_inscribed_circles<<<n_blocks_elem, n_threads>>>
                (d_J, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y);
    min_radius = (double *) malloc(local_num_elem * sizeof(double));

    /*
    // find the min inscribed circle. do it on the gpu if there are at least 256 elements
    if (local_num_elem >= 256) {
        //min_reduction<<<n_blocks_reduction, 256>>>(d_J, d_reduction, local_num_elem);
        cudaThreadSynchronize();
        checkCudaError("error after min_jacobian.");

        // each block finds the smallest value, so need to sort through n_blocks_reduction
        min_radius = (double *) malloc(n_blocks_reduction * sizeof(double));
        cudaMemcpy(min_radius, d_reduction, n_blocks_reduction * sizeof(double), cudaMemcpyDeviceToHost);
        min_r = min_radius[0];
        for (i = 1; i < n_blocks_reduction; i++) {
            min_r = (min_radius[i] < min_r) ? min_radius[i] : min_r;
        }
        free(min_radius);

    } else {
        */
        // just grab all the radii and sort them since there are so few of them
        min_radius = (double *) malloc(local_num_elem * sizeof(double));
        cudaMemcpy(min_radius, d_J, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
        min_r = min_radius[0];
        for (i = 1; i < local_num_elem; i++) {
            min_r = (min_radius[i] < min_r) ? min_radius[i] : min_r;
            // report problem
            if (min_radius[i] == 0) {
                printf("%i\n", i);
                printf("%.015lf, %.015lf, %.015lf, %.015lf, %.015lf, %.015lf\n", 
                                                         V1x[i], V1y[i],
                                                         V2x[i], V2y[i],
                                                         V3x[i], V3y[i]);
            }
        }
        free(min_radius);
    //}

    // pre computations
    preval_jacobian<<<n_blocks_elem, n_threads>>>(d_J, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y); 
    checkCudaError("error after preval_jacobian.");

    cudaThreadSynchronize();

    preval_side_length<<<n_blocks_sides, n_threads>>>(d_s_length, d_s_V1x, d_s_V1y, d_s_V2x, d_s_V2y);
                                                      
    preval_normals<<<n_blocks_sides, n_threads>>>(d_Nx, d_Ny, 
                                                  d_s_V1x, d_s_V1y, d_s_V2x, d_s_V2y,
                                                  d_V1x, d_V1y, 
                                                  d_V2x, d_V2y, 
                                                  d_V3x, d_V3y, 
                                                  d_left_side_number);
    cudaThreadSynchronize();


    preval_normals_direction<<<n_blocks_sides, n_threads>>>(d_Nx, d_Ny, 
                                                  d_V1x, d_V1y, 
                                                  d_V2x, d_V2y, 
                                                  d_V3x, d_V3y, 
                                                  d_left_elem, d_left_side_number);

    preval_partials<<<n_blocks_elem, n_threads>>>(d_V1x, d_V1y,
                                                  d_V2x, d_V2y,
                                                  d_V3x, d_V3y,
                                                  d_xr,  d_yr,
                                                  d_xs,  d_ys);
    cudaThreadSynchronize();
    checkCudaError("error after prevals.");

    // free computed variables
    free_gpu(1, convergence, eval_error);

   // evaluate the basis functions at those points and store on GPU
    preval_basis(r1_local, r2_local, s_r, w_local, oned_w_local, local_n_quad, local_n_quad1d, local_n_p);
    cudaThreadSynchronize();

    // no longer need any of these CPU variables
    free(elem_s1);
    free(elem_s2);
    free(elem_s3);
    free(sides_x1);
    free(sides_x2);
    free(sides_y1);
    free(sides_y2);
    free(left_elem);
    free(right_elem);
    free(left_side_number);
    free(right_side_number);
    free(r1_local);
    free(r2_local);
    free(w_local);
    free(s_r);
    free(oned_w_local);

    // initial conditions
    init_conditions<<<n_blocks_elem, n_threads>>>(d_c, d_J, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y);
    checkCudaError("error after initial conditions.");

    printf(" ? %i degree polynomial interpolation (local_n_p = %i)\n", n, local_n_p);
    printf(" ? %i precomputed basis points\n", local_n_quad * local_n_p);
    printf(" ? %i elements\n", local_num_elem);
    printf(" ? %i sides\n", local_num_sides);
    printf(" ? min radius = %.015lf\n", min_r);

    if (endtime == -1 && convergence != 1) {
        printf(" ? total_timesteps = %i\n", total_timesteps);
    } else if (endtime != -1 && convergence != 1) {
        printf(" ? endtime = %lf\n", endtime);
    } else if (convergence == 1) {
        printf(" ? convergence = %lf\n", tol);
    }

    checkCudaError("error before time integration.");

    if (benchmark) {
        start = clock();
    }
    switch (time_integrator) {
        case RK4:
            t = time_integrate_rk4(local_num_elem, local_num_sides, 
                                   n, local_n_p,
                                   endtime, total_timesteps, min_r,
                                   verbose, convergence, video, tol);
            break;
        case RK2:
            t = time_integrate_rk2(local_num_elem, local_num_sides, 
                                   n, local_n_p,
                                   endtime, total_timesteps, min_r,
                                   verbose, convergence, video, tol);
            break;
        default:
            printf("Error: no time integrator selected.\n");
            exit(0);
    }

    if (benchmark) {
        end = clock();
        elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Runtime: %lf seconds\n", elapsed);
    }

    // free everything but the coefficients and verticies
    free_gpu(2, convergence, eval_error);

    // evaluate the error
    if (eval_error) {
        cudaMalloc((void **) &d_error, local_num_elem * sizeof(double));
        error = (double *) malloc(local_num_elem * sizeof(double));
        // L_2
        for (n = 0; n < local_N; n++) {
            eval_error_L2<<<n_blocks_elem, n_threads>>>(d_c, d_error, 
                            d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y, d_J,
                            n, t);

            // copy over error
            cudaMemcpy(error, d_error, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);

            // sum total L2 error
            total_error = 0.;
            max_error   = -1;
            for (i = 0; i < local_num_elem; i++) {
                total_error += error[i];
                max_error = (error[i] > max_error) ? error[i] : max_error;
            }
            printf("L2 U%i error     = %e\n", n, sqrt(total_error));
            //printf("log L2 U%i error = %.015lf\n", n, 0.5 * log(total_error));
            //printf("inf L2 U%i error = %.019lf\n", n, max_error);
        }

        // L_1
        for (n = 0; n < local_N; n++) {
            eval_error_L1<<<n_blocks_elem, n_threads>>>(d_c, d_error, 
                            d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y, d_J,
                            n, t);

            // copy over error
            cudaMemcpy(error, d_error, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);

            // sum total L1 error
            total_error = 0.;
            max_error   = -1;
            for (i = 0; i < local_num_elem; i++) {
                total_error += error[i];
                max_error = (error[i] > max_error) ? error[i] : max_error;
            }
            //printf("L1     U%i error = %.015lf\n", n, total_error);
            //printf("inf L1 U%i error = %.019lf\n", n, max_error);
        }

        free(error);
    }

    // evaluate and write U to file
    write_U(local_num_elem, total_timesteps, total_timesteps);

    cudaMalloc((void **) &d_Uv1, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_Uv2, local_num_elem * sizeof(double));
    cudaMalloc((void **) &d_Uv3, local_num_elem * sizeof(double));
    Uv1 = (double *) malloc(local_num_elem * sizeof(double));
    Uv2 = (double *) malloc(local_num_elem * sizeof(double));
    Uv3 = (double *) malloc(local_num_elem * sizeof(double));

    // evaluate and write to file
    if (eval_error) {
        for (n = 0; n < local_N; n++) {
            plot_error<<<n_blocks_elem, n_threads>>>(d_c, 
                                                     d_V1x, d_V1y,
                                                     d_V2x, d_V2y,
                                                     d_V3x, d_V3y,
                                                     d_Uv1, d_Uv2, d_Uv3, 
                                                     t, n);
            cudaMemcpy(Uv1, d_Uv1, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(Uv2, d_Uv2, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(Uv3, d_Uv3, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
            sprintf(out_filename, "output/U%derror.msh", n);
            printf("Writing to %s...\n", out_filename);
            out_file  = fopen(out_filename , "w");
            fprintf(out_file, "View \"U%i \" {\n", n);
            for (i = 0; i < local_num_elem; i++) {
                fprintf(out_file, "ST (%.015lf,%.015lf,0,%.015lf,%.015lf,0,%.015lf,%.015lf,0) {%.015lf,%.015lf,%.015lf};\n", 
                                       V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                                       Uv1[i], Uv2[i], Uv3[i]);
            }
            fprintf(out_file,"};");
            fclose(out_file);
        }        
        
        for (n = 0; n < local_N; n++) {
            plot_exact<<<n_blocks_elem, n_threads>>>(d_c, 
                                                     d_V1x, d_V1y,
                                                     d_V2x, d_V2y,
                                                     d_V3x, d_V3y,
                                                     d_Uv1, d_Uv2, d_Uv3, 
                                                     t, n);
            cudaMemcpy(Uv1, d_Uv1, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(Uv2, d_Uv2, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(Uv3, d_Uv3, local_num_elem * sizeof(double), cudaMemcpyDeviceToHost);
            sprintf(out_filename, "output/U%dexact.msh", n);
            printf("Writing to %s...\n", out_filename);
            out_file  = fopen(out_filename , "w");
            fprintf(out_file, "View \"U%i \" {\n", n);
            for (i = 0; i < local_num_elem; i++) {
                fprintf(out_file, "ST (%.015lf,%.015lf,0,%.015lf,%.015lf,0,%.015lf,%.015lf,0) {%.015lf,%.015lf,%.015lf};\n", 
                                       V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                                       Uv1[i], Uv2[i], Uv3[i]);
            }
            fprintf(out_file,"};");
            fclose(out_file);
        }
    }

    // free everything else
    free_gpu(3, convergence, eval_error);

    // free CPU variables
    free(Uv1);
    free(Uv2);
    free(Uv3);

    free(V1x);
    free(V1y);
    free(V2x);
    free(V2y);
    free(V3x);
    free(V3y);

    return 0;
}
