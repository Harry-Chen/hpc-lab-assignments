#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#include <algorithm>
#include <utility>
#include <functional>

#include "common.h"
#include "utils.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

#ifndef AVG_NUM_THRESHOLD
#define AVG_NUM_THRESHOLD 50
#endif

#ifndef FORCE_USE_THREAD
#define FORCE_USE_THREAD false
#endif

#ifndef FORCE_USE_WARP
#define FORCE_USE_WARP true
#endif

#if FORCE_USE_THREAD && FORCE_USE_WARP
#error "cannot specify thread-only & warp-only simultaneously"
#endif

#ifndef REORDER_ROW
#define REORDER_ROW true
#endif

#ifndef SORT_COLUMN
#define SORT_COLUMN true
#endif


const char *version_name = "optimized version";

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

    CUDA_CHECK(cudaStreamCreate(&info->copy_stream));
    CUDA_CHECK(cudaMalloc(&info->c_idx_sorted, nnz * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&info->values_sorted, nnz * sizeof(data_t)));
    CUDA_CHECK(cudaMalloc(&info->values_diag_inv, m * sizeof(data_t)));
    if (REORDER_ROW) {
        CUDA_CHECK(cudaMalloc(&info->row_orders, m * sizeof(index_t)));
    }
    CUDA_CHECK(cudaMalloc(&info->finished, m * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&info->curr_id, sizeof(int)));
    if (!FORCE_USE_THREAD && !FORCE_USE_WARP) {
        CUDA_CHECK(cudaMalloc(&info->row_offset, (m + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(info->row_offset, row_offset, k * sizeof(int), cudaMemcpyHostToDevice, info->copy_stream));
    }


    auto values_diag_inv = new data_t[m];

    auto levels = new int[m];
    levels[0] = 0;
    int max_level = 0;

    using sort_data_t = std::pair<int, std::pair<index_t, data_t>>;
    index_t *c_idx_sorted = new index_t[nnz];
    data_t *values_sorted = new data_t[nnz];
    sort_data_t *data_sort;

    memcpy(c_idx_sorted, mat->c_idx, sizeof(index_t) * nnz);
    memcpy(values_sorted, mat->values, sizeof(data_t) * nnz);
    
    if (SORT_COLUMN) {
        data_sort = new sort_data_t[m];
    }

    for (int i = 0; i < m; ++i) {
        int begin = mat->r_pos[i], end = mat->r_pos[i + 1];
        // level for current row
        int l = -1;
        for (int j = begin; j < end - 1; j++) {
            int col = mat->c_idx[j];
            l = max(levels[col], l);
            if (SORT_COLUMN) data_sort[j - begin] = std::make_pair(levels[col], std::make_pair(col, mat->values[j]));
        }
        if (SORT_COLUMN && end > begin + 1) {
            std::sort(data_sort, data_sort + end - begin - 1, [&](const sort_data_t &i, const sort_data_t &j) { return i.first < j.first; });
            for (int j = begin; j < end - 1; j++) {
                const auto &data = data_sort[j - begin].second;
                c_idx_sorted[j] = data.first;
                values_sorted[j] = data.second;
            }
        }
        levels[i] = l + 1;
        max_level = max(max_level, l + 1);
        int count = end - begin - 1;
        values_diag_inv[i] = 1 / mat->values[end - 1]; // copy diag
    }

    // copy sorted values to
    CUDA_CHECK(cudaMemcpyAsync(info->values_diag_inv, values_diag_inv, m * sizeof(data_t), cudaMemcpyHostToDevice, info->copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(info->c_idx_sorted, c_idx_sorted, nnz * sizeof(index_t), cudaMemcpyHostToDevice, info->copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(info->values_sorted, values_sorted, nnz * sizeof(data_t), cudaMemcpyHostToDevice, info->copy_stream));

    // count number of each levels then reorder according to levels
    if (REORDER_ROW) {
        auto level_offsets = new int[max_level + 2](), level_counts = new int[max_level + 1]();
        auto row_orders = new index_t[m];

        // counting sort
        for (int i = 0; i < m; ++i) {
            level_offsets[levels[i] + 1]++;
        }
        for (int i = 0; i < max_level + 1; ++i) {
            level_offsets[i + 1] += level_offsets[i];
        }
        for (int i = 0; i < m; ++i) {
            int level = levels[i];
            int new_order = level_offsets[level] + (level_counts[level]++);
            row_orders[new_order] = i;
        }

        // copy new row orders to GPU
        CUDA_CHECK(cudaMemcpyAsync(info->row_orders, row_orders, m * sizeof(index_t), cudaMemcpyHostToDevice, info->copy_stream));
    }

    mat->additional_info = info;
}

void destroy_additional_info(void *additional_info) {
    cudaStreamDestroy(((sptrsv_info_t *)additional_info)->copy_stream);
}


template <bool FORCE_THREAD = FORCE_USE_THREAD, bool FORCE_WARP = FORCE_USE_WARP, bool USE_REORDER_ROW = REORDER_ROW>
__global__ void sptrsv_capellini_adaptive_kernel(
    const index_t *__restrict__ r_pos, const index_t *__restrict__ c_idx, const data_t *__restrict__ values, const data_t *__restrict__ values_diag_inv, const index_t *__restrict__ row_orders,
    const int *__restrict__ row_offset, const int warp_count, const int m, const data_t *__restrict__ b, data_t *__restrict__ x, volatile char *__restrict__ finished, int *curr_id
) {
    
    // allocate thread id by scheduling order
    const int lane_id = threadIdx.x & 31;
    int id = 0;
    if (lane_id == 0) {
        id = atomicAdd(curr_id, 1) << 5;
    }
    id = __shfl_sync(0xFFFFFFFF, id, 0) + lane_id;
    const int w = id >> 5;
    
    if (!FORCE_THREAD && w >= warp_count) {
        return;
    }

    // whether use thread (thread-only or hybrid mode)
    bool use_thread = FORCE_THREAD || (!FORCE_WARP && row_offset[w + 1] > row_offset[w] + 1);

    if (FORCE_THREAD || (!FORCE_WARP && use_thread)) {
        // assign one thread for current row

        int i; // row id
        if (FORCE_THREAD) {
            i = id;
        } else {
            i = row_offset[w] + lane_id;
        }

        if (i >= m) return;
        if (USE_REORDER_ROW) i = row_orders[i];

        const int begin = r_pos[i], end = r_pos[i + 1];
        data_t bi = b[i], diag_inv = values_diag_inv[i];
        
        for (int j = begin; j < end;) {
            int col = c_idx[j];
            if (col == i) {
                x[i] = bi * diag_inv;
                __threadfence();
                finished[col] = 1;
                break;
            }
            while (finished[col]) {
                // __threadfence();
                bi -= values[j] * x[col];
                col = c_idx[++j];
            }
        }
    } else {
        // assign one warp for current row

        int i; // row id
        if (FORCE_WARP) {
            i = w;
        } else {
            i = row_offset[w];
        }

        if (i >= m) return;
        if (USE_REORDER_ROW) i = row_orders[i];

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

    sptrsv_capellini_adaptive_kernel<<<ceiling(m * 32, BLOCK_SIZE), BLOCK_SIZE>>>(mat->gpu_r_pos, info->c_idx_sorted, info->values_sorted, info->values_diag_inv, info->row_orders, info->row_offset, info->warp_count, m, b, x, finished, curr_id);
    CUDA_CHECK(cudaGetLastError());
}

