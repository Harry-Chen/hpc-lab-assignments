#include <stdio.h>
#include <strings.h>
#include <mkl.h>

extern const char* dgemm_desc;
extern void square_dgemm (int, double*, double*, double*);

int main(int argc, char *argv[]) {
    int dim = 4;
    if (argc >= 2) {
        dim = atoi(argv[1]);
    }

    int size = dim * dim;

    double *buf = (double *) malloc(sizeof(double) * 4 * size);
    double *A = buf, *B = buf + size, *C = buf + 2 * size, *D = buf + 3 * size;
    bzero(buf, sizeof(double) * 4 * size);

    for (int i = 0; i < size; ++i) {
        A[i] = B[i] = i + 1;
    }

    square_dgemm(dim, A, B, C);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, A, dim, B, dim, 0.0, D, dim);

    puts("square_dgemm: ");
    for (int i = 0; i < size; ++i) {
        printf("%.0lf%c", C[i], (i + 1) % dim == 0 ? '\n': '\t');
    }

    puts("cblas_dgemm: ");
    for (int i = 0; i < size; ++i) {
        printf("%.0lf%c", D[i], (i + 1) % dim == 0 ? '\n': '\t');
    }

    return 0;
}
