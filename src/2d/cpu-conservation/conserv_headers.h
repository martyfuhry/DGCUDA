void evalU0(double *U, double *V, int i);
void eval_flux(double *U, double *flux_x, double *flux_y, 
                          double *, double, int, int);
void riemann_solver(double *, double *, double *, 
                               double *, double, 
                               double, double, int, int);
void inflow_boundary(double *U_left, double *U_right, 
                                double *, double nx, double ny, 
                                int j, int left_side, double t);
void outflow_boundary(double *U_left, double *U_right, 
                                 double *, 
                                 double nx, double ny, 
                                 int j, int left_side, double t);
void reflecting_boundary(double *U_left, double *U_right, 
                                    double *, double nx, double ny, 
                                    int j, int left_side, double t);
void eval_global_lambda(double *C, double *lambda, 
                                   double *, double *, 
                                   double *, double *, 
                                   double *, double *, 
                                   double);
void check_physical(double *, double *, double *, int);
int is_physical(double *);
void U0(double *, double, double);
void U_inflow(double *, double, double, double);
void U_outflow(double *, double, double, double);
void U_reflection(double *, double *, double, double, double, double, double);
void U_exact(double *, double, double, double);
