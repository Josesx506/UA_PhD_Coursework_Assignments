#ifndef UTIL_H
#define UTIL_H

void nrerror(char error_text[]);
int *ivector(int nl, int nh);
void free_ivector(int *v, int nl, int nh);
unsigned int *lvector(int nl, int nh);
float *vector(int nl, int nh);
double *dvector(int nl, int nh);
void free_dvector(double *v, int nl, int nh);
int **imatrix(int nrl, int nrh, int ncl, int nch);
double **dmatrix(int nrl, int nrh, int ncl, int nch);
void indexx(int n, double arr[], int indx[]);
double zbrent(double (*func)(double), double x1, double x2, double tol);

#endif /* UTIL_H*/