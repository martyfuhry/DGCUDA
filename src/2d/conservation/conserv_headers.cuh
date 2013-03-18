/* conserv_headers.cuh
 *
 * Header file storing user implemented problem-specific functions
 */
__device__ void evalU0(double *U, double *V, int i);
__device__ void eval_flux(double *U, double *flux_x, double *flux_y, 
                          double *, double, int, int);
__device__ void riemann_solver(double *, double *, double *, 
                               double *, double, 
                               double, double, int, int);
__device__ void inflow_boundary(double *U_left, double *U_right, 
                                double *, double nx, double ny, 
                                int j, int left_side, double t);
__device__ void outflow_boundary(double *U_left, double *U_right, 
                                 double *, 
                                 double nx, double ny, 
                                 int j, int left_side, double t);
__device__ void reflecting_boundary(double *U_left, double *U_right, 
                                    double *, double nx, double ny, 
                                    int j, int left_side, double t);
__global__ void eval_global_lambda(double *C, double *lambda, 
                                   double *, double *, 
                                   double *, double *, 
                                   double *, double *, 
                                   double);
__device__ void check_physical(double *, double *, double *, int);
__device__ bool is_physical(double *);
__device__ void U0(double *, double, double);
__device__ void U_inflow(double *, double, double, double);
__device__ void U_outflow(double *, double, double, double);
__device__ void U_reflection(double *, double *, double, double, double, double, double);
__device__ void U_exact(double *, double, double, double);
