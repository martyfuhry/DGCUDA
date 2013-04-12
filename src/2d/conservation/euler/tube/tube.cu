#include "../euler.cu"

/* cylinder.cu
 *
 * Flow around a cylinder.
 */

#define PI 3.14159
#define GAMMA 1.4
#define MACH .38

int limiter = NO_LIMITER;  // no limiter
int time_integrator = RK4; // time integrator to use

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
    double h, p;
    double center = 0.5;
    double a = 1;
    double b = 10;
    h = 1. / b;

    if (x >= center - h && x <= center + h) {
        p = 1. + (a / 2) * (1. + cospi(b * (x-center)));  
    } else { 
        p = 1.;
    }

    U[0] = GAMMA;
    U[1] = U[0] * 0.;
    U[2] = U[0] * 0.;
    U[3] = 0.5 * U[0] * 0. + p/ (GAMMA - 1.0);
}

/***********************
*
* INFLOW CONDITIONS
*
************************/

__device__ void U_inflow(double *U, double x, double y, double t) {
    double rho, u, v, p, h;

    /*
    t += .9197;
    if( x > 0-10.e-10      && x < 0+10.e-10 && 
        y > -0.007         && y < 0.007) {

        p=1.0+ ((2*0.0310778152197599) *cos(1.42429852485657*t+3.11072301864624))
             + ((2*0.00720852287486196)*cos(2.84859704971313*t+2.72678995132446))
             + ((2*0.00503001408651471)*cos(4.27289581298828*t+1.9487407207489))
             + ((2*0.000847975315991789)*cos(5.69719409942627*t+1.16978299617767))
             + ((2*0.00129160704091191)*cos(7.12149286270142*t+-2.02217507362366))
             + ((2*0.00072062574326992)*cos(8.54579162597656*t+2.19518876075745))
             + ((2*0.000242143694777042)*cos(9.97008991241455*t+1.39034509658813))
             + ((2*1.72300478880061e-005)*cos(11.3943881988525*t+2.81903266906738))
             + ((2*6.1697639466729e-005)*cos(12.8186864852905*t+-1.70821964740753))
             + ((2*7.94851002865471e-005)*cos(14.2429857254028*t+-1.57708919048309))
             + ((2*2.84207799268188e-005)*cos(15.6672840118408*t+-1.59401333332062))
             + ((2*3.26190129271708e-005)*cos(17.0915832519531*t+0.797450065612793))
             + ((2*4.33758286817465e-005)*cos(18.5158805847168*t+0.733162939548492))
             + ((2*6.36465774732642e-005)*cos(19.9401798248291*t+0.374574363231659))
             + ((2*3.47004934155848e-005)*cos(21.3644771575928*t+-0.500430643558502));

        rho = GAMMA * pow(p, (1. / GAMMA));
        u   = ((p - 1) / GAMMA);
        v   = 0.0;

        U[0] = rho;
        U[1] = u * rho; 
        U[2] = v * rho; 
        U[3] = 0.5*rho*(u*u+v*v) + (p / (GAMMA-1.)); 
    } else {
        U0(U, x, y);
    }
    */
    p = 1;
    U[0] = GAMMA;
    U[1] = 0;
    U[2] = 0;
    U[3] = 0.5*U[0]*0 + (p / (GAMMA-1.)); 

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

    double dot, vx, vy;
    // set rho and E to be the same in the ghost cell
    U_right[0] = U_left[0];
    U_right[3] = U_left[3];

    // taken from algorithm 2 from lilia's code
    dot = sqrt(x*x + y*y);

    // normal reflection
    dot = U_left[1] * nx + U_left[2] * ny;
    U_right[1] = U_left[1] - 2*dot*nx;
    U_right[2] = U_left[2] - 2*dot*ny;

    //vx = x / dot;
    //vy = y / dot;

    // see which direction (Nx, Ny) faces
    //if (vx * nx + vy * ny < 0) {
        //vx *= -1;
        //vy *= -1;
    //}

    // set the velocities to reflect
    //U_right[1] =  (U_left[1] * vy - U_left[2] * vx)*vy;
    //U_right[2] = -(U_left[1] * vy - U_left[2] * vx)*vx;

    // normal reflection
    //double n = -(nx * U_left[1] + ny * U_left[2]);
    //double t = ny * U_left[1] - nx * U_left[2];
    //U_right[1] = n*nx + t*ny;
    //U_right[2] = n*ny - t*nx;
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
