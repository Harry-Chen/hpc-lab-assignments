#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

#include "common.h"
#include "utils.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

const char* version_name = "optimized version";

void preprocess(dist_matrix_t *mat) {
    auto info = new sptrsv_info_t;
    CUDA_CHECK(cudaMalloc(&info->finished, mat->global_m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&info->curr_row, sizeof(int)));
    mat->additional_info = info;
}

void destroy_additional_info(void *additional_info) {}


__global__ void sptrsv_capellini_kernel(
    const index_t *__restrict__ r_pos, const index_t *__restrict__ c_idx, const data_t *__restrict__ values,
    const int m, const int nnz, const data_t *__restrict__ b, data_t *__restrict__ x, volatile int *__restrict__ finished, int *curr_row
) {
    
    // allocate thread id by scheduling order
    const int i = atomicAdd(curr_row, 1);
    if (i >= m) return;


    // begin index of current warp (must be contiguous 32 numbers in a row)
    const int warp_begin = (i >> 5) << 5;

    data_t left_sum = 0;

    const int begin = r_pos[i], end = r_pos[i + 1];

    int j = begin;

    // for (; j < end; ++j) {
    //     int col = c_idx[j];
    //     if (col < warp_begin) {
    //         while (finished[col] != 1) __threadfence_block();
    //         left_sum += values[j] * x[col];
    //     } else {
    //         break;
    //     }
    // }

    // go through all numbers on current row
    while (j < end) {
    // for (int k = 0; k < 16; ++k) {
        int col = c_idx[j];
        // iterate over all finished numbers
        while (finished[col] == 1) {
            // __threadfence();
            left_sum += values[j] * x[col];
            col = c_idx[++j];
        }
        // last number (on diagonal)
        if (col == i) {
            x[i] = (b[i] - left_sum) / values[end - 1];
            // __threadfence(); // ensure x[i] can be read properly by other threads
            finished[i] = 1;
            // printf("%d\n", i);
            ++j;
        }
    }
}


void sptrsv(dist_matrix_t *mat, const data_t *__restrict__ b, data_t *__restrict__ x) {
    int m = mat->global_m;
    int nnz = mat->global_nnz;

    auto info = (sptrsv_info_t *) mat->additional_info;
    auto finished = info->finished;
    auto curr_row = info->curr_row;
    CUDA_CHECK(cudaMemset(finished, 0, m * sizeof(int)));
    CUDA_CHECK(cudaMemset(curr_row, 0, sizeof(int)));

    sptrsv_capellini_kernel<<<ceiling(m, BLOCK_SIZE), BLOCK_SIZE>>>(mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, m, nnz, b, x, finished, curr_row);
    CUDA_CHECK(cudaGetLastError());
}
