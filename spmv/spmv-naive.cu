#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include "common.h"


const char* version_name = "naive base-line";\

void preprocess(dist_matrix_t *mat) {
}

void destroy_additional_info(void *additional_info) {
}

__global__ void spmv_naive_kernel(int m, const index_t *r_pos, \
    const index_t *c_idx, const data_t *values, const data_t *x, data_t *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < m) {
        int p, begin = r_pos[i], end = r_pos[i+1];
        data_t s = y[i];
        for(p = begin; p < end; ++p) {
            int j = c_idx[p];
            s += values[p] * x[j];
        }
        y[i] = s;
    }
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

void spmv(dist_matrix_t *mat, const data_t* x, data_t* y) {
    int m = mat->global_m;
    dim3 grid_size (ceiling(m, 512), 1, 1);
    dim3 block_size (512, 1, 1);
    spmv_naive_kernel<<<grid_size, block_size>>>(m, \
        mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, x, y);
}
