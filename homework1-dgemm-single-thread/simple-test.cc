
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <strings.h>
#include <mkl.h>

extern "C" const char* dgemm_desc;
extern "C" void square_dgemm (int, double*, double*, double*);

int main(int argc, char *argv[]) {
    int dim = 4;
    if (argc >= 2) {
        dim = atoi(argv[1]);
    }

    int size = dim * dim;

    double *buf = (double *) malloc(sizeof(double) * 4 * size);
    double *A = buf, *B = buf + size, *C = buf + 2 * size, *D = buf + 3 * size;
    bzero(buf, sizeof(double) * 4 * size);

    srand48(time(0));

    for (int i = 0; i < size; ++i) {
        A[i] = drand48();
        B[i] = drand48();
    }

    square_dgemm(dim, A, B, C);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, A, dim, B, dim, 0.0, D, dim);

    puts("square_dgemm: ");
    for (int i = 0; i < size; ++i) {
        printf("%.3lf%c", C[i], (i + 1) % dim == 0 ? '\n': '\t');
    }

    puts("cblas_dgemm: ");
    for (int i = 0; i < size; ++i) {
        printf("%.3lf%c", D[i], (i + 1) % dim == 0 ? '\n': '\t');
    }

    return 0;
}
