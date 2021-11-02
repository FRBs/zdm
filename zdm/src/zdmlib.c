#include <math.h>

/* testlib.c */
double lognormal_dlog_c(int n, double *xx){
    return xx[3]*exp(-0.5* pow((xx[0]-xx[1])/xx[2],2));
}