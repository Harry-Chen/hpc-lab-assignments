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
    CUDA_CHECK(cudaMalloc(&info->get_value, mat->global_m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&info->id_getter, sizeof(int)));
    mat->additional_info = info;
}

void destroy_additional_info(void *additional_info) {}


__global__ void sptrsv_capellini_kernel(
    const index_t *__restrict__ r_pos, const index_t *__restrict__ c_idx, const data_t *__restrict__ values,
    const int m, const int nnz, const data_t *__restrict__ b, data_t * x, int * get_value, int *id_getter
) {

    // const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = atomicAdd(id_getter, 1);
    if (i >= m) return;

    data_t xi;
    data_t left_sum = 0;
    // const int warp_begin = (i >> 5) << 5; // begin index of this warp

    const int begin = r_pos[i], end = r_pos[i + 1];

    int j = begin;

    while (j < end) {
        int col = c_idx[j];
        while (get_value[col] == 1) {
            left_sum += values[j] * x[col];
            j++;
            col = c_idx[j];
        }
        if (col == i) {
            xi = (b[i] - left_sum) / values[end - 1];
            x[i] = xi;
            __threadfence();
            get_value[i] = 1;
            ++j;
        }
    }
}


void sptrsv(dist_matrix_t *mat, const data_t *__restrict__ b, data_t *__restrict__ x) {
    int m = mat->global_m;
    int nnz = mat->global_nnz;

    auto info = (sptrsv_info_t *) mat->additional_info;
    auto get_value = info->get_value;
    auto id_getter = info->id_getter;
    CUDA_CHECK(cudaMemset(get_value, 0, m * sizeof(int)));
    CUDA_CHECK(cudaMemset(id_getter, 0, sizeof(int)));

    sptrsv_capellini_kernel<<<ceiling(m, BLOCK_SIZE), BLOCK_SIZE>>>(mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, m, nnz, b, x, get_value, id_getter);
    CUDA_CHECK(cudaGetLastError());

    // auto nums = new int[m];
    // CUDA_CHECK(cudaMemcpy(nums, get_value, m * sizeof(int), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < m; ++i) {
    //     printf("%d\n", nums[i]);
    // }

    // exit(0);
}
