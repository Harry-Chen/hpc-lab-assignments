#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

#include "common.h"
#include "utils.h"

#ifndef GRID_SIZE
#define GRID_SIZE 256
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif


const char* version_name = "optimized version";

void preprocess(dist_matrix_t *mat) {
}

void destroy_additional_info(void *additional_info) {
}


__global__ void spmv_merge_based_kernel(int m, int nnz, int ntasks_per_thread, 
    const index_t *__restrict__ r_pos, const index_t *__restrict__ c_idx, const data_t *__restrict__ values,
    const data_t *__restrict__ x, data_t *__restrict__ y) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = i * ntasks_per_thread;

    index_t curr_row = 0, count = m;

    while (count > 0) {
        index_t pos = curr_row;
        index_t step = count >> 1;
        pos += step;
        if (r_pos[pos + 1] <= k - pos - 1) {
            curr_row = ++pos;
            count -= step + 1;
        } else {
            count = step;
        }
    }

    index_t curr_index = k - curr_row;

    int ntasks = MIN(ntasks_per_thread, m + nnz - curr_row - curr_index); // task number for each thread
    bool self_row = curr_index == r_pos[curr_row];

    data_t res = 0.0;
 
    if (curr_row < m && curr_index < nnz) {
#pragma unroll
        for (int t = 0; t < ntasks; ++t) {
            if (curr_index == r_pos[curr_row + 1]) {
                // end of a row, aggregate
                if (self_row) { // current thread fully calculates this row
                    y[curr_row] = res;
                } else {
                    atomicAddDouble(&y[curr_row], res);
                }
                curr_row++;
                self_row = true;
                res = 0.0;
            } else {
                res += x[c_idx[curr_index]] * values[curr_index];
                curr_index++;
            }
        }
        if (curr_row < m) atomicAddDouble(&y[curr_row], res);
    }
}

void spmv(dist_matrix_t *mat, const data_t *__restrict__ x, data_t *__restrict__ y) {

    int m = mat->global_m;
    int n = mat->global_nnz;

    dim3 grid_size(GRID_SIZE, 1, 1);
    dim3 block_size(BLOCK_SIZE, 1, 1);

    int ntasks_per_thread = ceiling(m + n, GRID_SIZE * BLOCK_SIZE);

    spmv_merge_based_kernel<<<grid_size, block_size>>>(m, n, ntasks_per_thread, mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, x, y);
}
