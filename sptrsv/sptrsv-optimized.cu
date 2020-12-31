#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#include "common.h"
#include "utils.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

#ifndef AVG_NUM_THRESHOLD
#define AVG_NUM_THRESHOLD 1000000
#endif

#ifndef FORCE_USE_WARP
#define FORCE_USE_WARP true
#endif

const char* version_name = "optimized version";

void preprocess(dist_matrix_t *mat) {

    int m = mat->global_m;
    int nnz = mat->global_nnz;
    int curr_row = 0;

    auto row_offset = new int[m + 1];

    row_offset[0] = 0;
    int k = 1;

    for (int i = 0; i < m; i += 32) {
        int row_end = min(i + 32, m);
        int rows = row_end - i;
        int elements = mat->r_pos[row_end] - mat->r_pos[i];
        auto avg_per_row = (double) elements / rows;
        bool use_warp = avg_per_row >= AVG_NUM_THRESHOLD;
        if (FORCE_USE_WARP || use_warp) {
            // one warp for each row
            for (int j = 0; j < rows; ++j) {
                row_offset[k++] = ++curr_row;
            }
        } else {
            // one thread for each row
            curr_row += rows;
            row_offset[k++] = curr_row;
        }
    }

    auto info = new sptrsv_info_t;
    info->warp_count = k - 1;
    CUDA_CHECK(cudaMalloc(&info->finished, m * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&info->row_offset, (m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&info->curr_id, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(info->row_offset, row_offset, k * sizeof(int), cudaMemcpyHostToDevice));
    mat->additional_info = info;

    delete[] row_offset;
}

void destroy_additional_info(void *additional_info) {}


__global__ void sptrsv_capellini_thread_kernel(
    const index_t *__restrict__ r_pos, const index_t *__restrict__ c_idx, const data_t *__restrict__ values,
    const int m, const int nnz, const data_t *__restrict__ b, data_t *__restrict__ x, volatile char *__restrict__ finished, int *curr_id
) {
    
    // allocate thread id by scheduling order
    const int i = atomicAdd(curr_id, 1);
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
        volatile char *finished_col = finished + col;
        while (finished[col] != 1) {}
            left_sum += values[j] * x[col];
            col = c_idx[++j];
        // }
        // last number (on diagonal)
        if (col == i) {
            x[i] = (b[i] - left_sum) / values[end - 1];
            __threadfence(); // ensure x[i] can be read properly by other threads
            finished[i] = 1;
            ++j;
        }
    }
}


__global__ void sptrsv_capellini_warp_kernel(
    const index_t *__restrict__ r_pos, const index_t *__restrict__ c_idx, const data_t *__restrict__ values, const int *__restrict__ row_offset,
    const int warp_count, const int m, const int nnz, const data_t *__restrict__ b, data_t *__restrict__ x, volatile char *__restrict__ finished, int *curr_id
) {
    
    // allocate thread id by scheduling order
    const int id = atomicAdd(curr_id, 1);
    // const int id = blockDim.x * blockIdx.x + threadIdx.x;
    const int w = id >> 5;
    const int lane_id = id & 31;
    if (w >= warp_count) return;

    bool use_thread = row_offset[w + 1] > row_offset[w] + 1;

    if (!FORCE_USE_WARP && use_thread) {
        // one thread for current row
        int i = row_offset[w] + lane_id;
        if (i >= m) return;

        data_t left_sum = 0;
        const int begin = r_pos[i], end = r_pos[i + 1];

        int j = begin;
        while (j < end) {
            int col = c_idx[j];
            while (finished[col]) {
                __threadfence();
                left_sum += values[j] * x[col];
                col = c_idx[++j];
            }
            if (col == i) {
                x[i] = (b[i] - left_sum) / values[end - 1];
                __threadfence();
                finished[i] = 1;
                ++j;
            }
        }
    } else {
        // one warp for current row
        // int i = row_offset[w];
        int i = w;
        if (i >= m) return;

        data_t left_sum = 0;
        const int begin = r_pos[i], end = r_pos[i + 1];
    
        // calculate sum of previous columns
        for (int j = begin + lane_id; j < end - 1; j += 32) {
            int col = c_idx[j];
            while (!finished[col]) {
                __threadfence();
            }
            // volatile data_t *x_col = x + col;
            left_sum += values[j] * x[col];
        }
    
        // reduce within warp
        for (int offset = 16; offset > 0; offset >>= 1) {
            left_sum += __shfl_down_sync(0xFFFFFFFF, left_sum, offset);
        }
    
        if (lane_id == 0) {
            x[i] = (b[i] - left_sum) / values[end - 1];
            __threadfence();
            finished[i] = 1;
        }
    }
}


void sptrsv(dist_matrix_t *mat, const data_t *__restrict__ b, data_t *__restrict__ x) {
    int m = mat->global_m;
    int nnz = mat->global_nnz;

    auto info = (sptrsv_info_t *) mat->additional_info;
    auto finished = info->finished;
    auto curr_id = info->curr_id;
    CUDA_CHECK(cudaMemset(finished, 0, m * sizeof(char)));
    CUDA_CHECK(cudaMemset(curr_id, 0, sizeof(int)));

    // sptrsv_capellini_thread_kernel<<<ceiling(m, BLOCK_SIZE), BLOCK_SIZE>>>(mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, m, nnz, b, x, finished, curr_id);
    sptrsv_capellini_warp_kernel<<<ceiling(m * 32, BLOCK_SIZE), BLOCK_SIZE>>>(mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, info->row_offset, info->warp_count, m, nnz, b, x, finished, curr_id);
    CUDA_CHECK(cudaGetLastError());
}
