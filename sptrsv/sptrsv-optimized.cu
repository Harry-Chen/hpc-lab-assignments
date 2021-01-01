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
#define AVG_NUM_THRESHOLD 32
#endif

#ifndef FORCE_USE_THREAD
#define FORCE_USE_THREAD false
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
        if (!FORCE_USE_THREAD && (FORCE_USE_WARP || use_warp)) {
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

    auto r_pos = new index_t[m + 1];
    auto c_idx = new index_t[nnz - m];
    auto values = new data_t[nnz - m];
    auto values_diag_inv = new data_t[m];

    auto levels = new int[m];
    levels[0] = 0;
    int max_level = 0;

    // int pos = 0;
    // r_pos[0] = 0;
    for (int i = 0; i < m; ++i) {
        int begin = mat->r_pos[i], end = mat->r_pos[i + 1];
        // level for current row
        int l = -1;
        for (int j = begin; j < end - 1; j++) {
            int col = mat->c_idx[j];
            l = max(levels[col], l);
        }
        levels[i] = l + 1;
        max_level = max(max_level, l + 1);
        int count = end - begin - 1;
        // r_pos[i + 1] = pos + count; // set r_pos
        values_diag_inv[i] = 1 / mat->values[end - 1]; // copy diag
        // memcpy(c_idx + pos, mat->c_idx + begin, count * sizeof(index_t)); // copy c_idx
        // memcpy(values + pos, mat->values + begin, count * sizeof(data_t)); // copy values
        // pos += count;
    }

    // for (int i = 0; i < 100; ++i) {
    //     printf("%d ", levels[i]);
    // }
    // puts("");
    // printf("Max Level: %d\n", max_level);

    // count number of each levels (counting sort)
    auto level_offsets = new int[max_level + 2](), level_counts = new int[max_level + 1]();
    auto row_orders = new index_t[m];

    for (int i = 0; i < m; ++i) {
        level_offsets[levels[i] + 1]++;
    }

    for (int i = 0; i < max_level + 1; ++i) {
        level_offsets[i + 1] += level_offsets[i];
    }

    // for (int i = 0; i < min(max_level + 1, 100); ++i) {
    //     printf("%d ", level_offsets[i]);
    // }
    // puts("");

    for (int i = 0; i < m; ++i) {
        int level = levels[i];
        int new_order = level_offsets[level] + (level_counts[level]++);
        // if (i < 100) printf("%d ", new_order);
        row_orders[new_order] = i;
    }
    // puts("");

    // for (int i = 0; i < 100; ++i) {
    //     printf("%d ", row_orders[i]);
    // }
    // puts("");

    // CUDA_CHECK(cudaMalloc(&info->r_pos_aligned, (m + 1) * sizeof(index_t)));
    // CUDA_CHECK(cudaMalloc(&info->c_idx_aligned, (nnz - m) * sizeof(index_t)));
    // CUDA_CHECK(cudaMalloc(&info->values_aligned, (nnz - m) * sizeof(data_t)));
    CUDA_CHECK(cudaMalloc(&info->values_diag_inv, m * sizeof(data_t)));
    CUDA_CHECK(cudaMalloc(&info->finished, m * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&info->row_offset, (m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&info->curr_id, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&info->row_orders, m * sizeof(index_t)));


    CUDA_CHECK(cudaMemcpy(info->row_offset, row_offset, k * sizeof(int), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(info->r_pos_aligned, r_pos, (m + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(info->c_idx_aligned, c_idx, (nnz - m) * sizeof(index_t), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(info->values_aligned, values, (nnz - m) * sizeof(data_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(info->values_diag_inv, values_diag_inv, m * sizeof(data_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(info->row_orders, row_orders, m * sizeof(index_t), cudaMemcpyHostToDevice));
    mat->additional_info = info;

    delete[] row_offset;
    delete[] r_pos;
    delete[] c_idx;
    delete[] values;
    delete[] values_diag_inv;
    delete[] row_orders;
    delete[] level_offsets;
    delete[] level_counts;
}

void destroy_additional_info(void *additional_info) {}


__global__ void sptrsv_capellini_adaptive_kernel(
    const index_t *__restrict__ r_pos, const index_t *__restrict__ c_idx, const data_t *__restrict__ values, const data_t *__restrict__ values_diag_inv, const index_t *__restrict__ row_orders,
    const int *__restrict__ row_offset, const int warp_count, const int m, const data_t *__restrict__ b, data_t *__restrict__ x, volatile char *__restrict__ finished, int *curr_id
) {
    
    // allocate thread id by scheduling order
    // const int id = blockDim.x * blockIdx.x + threadIdx.x;    
    const int lane_id = threadIdx.x & 31;
    int id = 0;
    if (lane_id == 0) {
        id = atomicAdd(curr_id, 1) << 5;
    }
    id = __shfl_sync(0xFFFFFFFF, id, 0) + lane_id;
    const int w = id >> 5;
    
#if !FORCE_USE_THREAD
    if (w >= warp_count) return;
#endif

    bool use_thread = FORCE_USE_THREAD || (!FORCE_USE_WARP && row_offset[w + 1] > row_offset[w] + 1);

    if (FORCE_USE_THREAD || (!FORCE_USE_WARP && use_thread)) {
        // one thread for current row
#if FORCE_USE_THREAD
        int i = id;
#else
        int i = row_offset[w] + lane_id;
#endif

#if FORCE_USE_WARP
        assert(false);
#endif
        if (i >= m) return;
        i = row_orders[i];

        const int begin = r_pos[i], end = r_pos[i + 1];
        data_t bi = b[i], diag_inv = values_diag_inv[i];
        
        for (int j = begin; j < end;) {
            int col = c_idx[j];
            // volatile char *finished_col = finished + col;
            if (col == i) {
                x[i] = bi * diag_inv;
                __threadfence();
                finished[col] = 1;
                break;
                // j++;
            }
            while (finished[col]) {
                // __threadfence();
                bi -= values[j] * x[col];
                col = c_idx[++j];
            }
        }
    } else {
        // one warp for current row
#if FORCE_USR_WARP
        int i = w;
#else
        int i = row_offset[w];
#endif

#if FORCE_USE_THREAD
        assert(false);
#endif
        if (i >= m) return;
        i = row_orders[i];

        data_t left_sum = 0;
        const int begin = r_pos[i], end = r_pos[i + 1];
        data_t bi = b[i], diag_inv = values_diag_inv[i];
        bi *= diag_inv;
            
        // calculate sum of previous columns
        for (int j = begin + lane_id; j < end - 1; j += 32) {
            data_t value = values[j];
            int col = c_idx[j];
            while (finished[col] == 0) {
                __threadfence();
            }
            // volatile data_t *x_col = x + col;
            left_sum += value * x[col];
        }

        left_sum *= diag_inv;
    
        // reduce within warp
        for (int offset = 16; offset > 0; offset >>= 1) {
            left_sum += __shfl_down_sync(0xFFFFFFFF, left_sum, offset);
        }
    
        if (lane_id == 0) {
            x[i] = bi - left_sum;
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

    sptrsv_capellini_adaptive_kernel<<<ceiling(m * 32, BLOCK_SIZE), BLOCK_SIZE>>>(mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, info->values_diag_inv, info->row_orders, info->row_offset, info->warp_count, m, b, x, finished, curr_id);
    CUDA_CHECK(cudaGetLastError());
}
