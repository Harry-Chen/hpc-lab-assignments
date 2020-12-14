#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

#include <algorithm>

#include "common.h"
#include "utils.h"

#ifndef GRID_SIZE
#define GRID_SIZE 256
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

#ifndef ENABLE_SORT_BASED
#define ENABLE_SORT_BASED 1
#endif

const char* version_name = "optimized version";

void preprocess(dist_matrix_t *mat) {
#if ENABLE_SORT_BASED
    int m = mat->global_m;
    int n = mat->global_nnz;
    auto info = new csr_info_t;

    // sort tasks by desceding order
    auto tasks = new task_info_t[m];
    for (int i = 0; i < m; ++i) {
        tasks[i].task_num = mat->r_pos[i + 1] - mat->r_pos[i];
        tasks[i].row = i;
    }
    std::sort(tasks, tasks + m, [](const task_info_t &a, const task_info_t &b){ return a.task_num > b.task_num; });

    auto row_index = new index_t[m], row_task_num = new index_t[m]();

    int warp_num = ceiling(m, 32);
    index_t *sorted_offset = new index_t[warp_num](), sorted_size = 0;
    
    // merge data in a warp together
    for (int i = 0; i < m; i += 32) {
        int warp_nnz = tasks[i].task_num;
        if (warp_nnz == 0) break; // nothing to do afterwards
        sorted_offset[i / 32] = sorted_size;
        sorted_size += warp_nnz * 32;
    }

    // sort arrays
    auto c_idx_sorted = new index_t[sorted_size];
    auto values_sorted = new data_t[sorted_size];

    for (int i = 0; i < m; ++i) {
        int row_number = tasks[i].row, task_num = tasks[i].task_num;
        int row_offset = mat->r_pos[row_number];
        row_index[i] = row_number;
        row_task_num[i] = task_num;
        if (task_num == 0) break; // all tasks following will be zero
        for (int j = 0; j < task_num; ++j) {
            c_idx_sorted[sorted_offset[i / 32] + j * 32 + i % 32] = mat->c_idx[row_offset + j];
            values_sorted[sorted_offset[i / 32] + j * 32 + i % 32] = mat->values[row_offset + j];
        }
    }


    CUDA_CHECK(cudaMalloc(&info->sorted_offset, warp_num * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&info->row_index, m * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&info->row_task_num, m * sizeof(index_t)));

    CUDA_CHECK(cudaMemcpy(info->sorted_offset, sorted_offset, warp_num * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(info->row_index, row_index, m * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(info->row_task_num, row_task_num, m * sizeof(index_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&info->c_idx_sorted, sorted_size * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&info->values_sorted, sorted_size * sizeof(data_t)));

    CUDA_CHECK(cudaMemcpy(info->c_idx_sorted, c_idx_sorted, sorted_size * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(info->values_sorted, values_sorted, sorted_size * sizeof(data_t), cudaMemcpyHostToDevice));

    csr_info_t *gpu_info;
    CUDA_CHECK(cudaMalloc(&gpu_info, sizeof(csr_info_t)));
    CUDA_CHECK(cudaMemcpy(gpu_info, info, sizeof(csr_info_t), cudaMemcpyHostToDevice));
    mat->additional_info = gpu_info;

    cudaDeviceSynchronize();
#endif
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
                    ATOMIC_ADD_DOUBLE(&y[curr_row], res);
                }
                curr_row++;
                self_row = true;
                res = 0.0;
            } else {
                res += x[c_idx[curr_index]] * values[curr_index];
                curr_index++;
            }
        }
        if (curr_row < m) ATOMIC_ADD_DOUBLE(&y[curr_row], res);
    }
}

__global__ void spmv_sort_based_kernel(int m, int nnz, const data_t *__restrict__ values,
    const data_t *__restrict__ x, data_t *__restrict__ y, csr_info_t *__restrict__ info) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int warp_offset = info->sorted_offset[i / 32] + i % 32;
    int row_index = info->row_index[i];
    int row_task_num = info->row_task_num[i];
    int thread_offset = warp_offset;

    if (i < m && row_task_num > 0) {
        double sum = 0.0;
        for (int j = 0; j < row_task_num; ++j) {
            sum += info->values_sorted[thread_offset] * x[info->c_idx_sorted[thread_offset]];
            thread_offset += 32;
        }
        y[row_index] = sum;
    }

}

void spmv(dist_matrix_t *mat, const data_t *__restrict__ x, data_t *__restrict__ y) {

    int m = mat->global_m;
    int n = mat->global_nnz;

#if ENABLE_SORT_BASED
    dim3 grid_size(ceiling(m, 1024), 1, 1);
    dim3 block_size(1024, 1, 1);
    spmv_sort_based_kernel<<<grid_size, block_size>>>(m, n, mat->gpu_values, x, y, (csr_info_t *)mat->additional_info);
    CUDA_CHECK(cudaGetLastError());
#else
    dim3 grid_size(GRID_SIZE, 1, 1);
    dim3 block_size(BLOCK_SIZE, 1, 1);
    int ntasks_per_thread = ceiling(m + n, GRID_SIZE * BLOCK_SIZE);
    spmv_merge_based_kernel<<<grid_size, block_size>>>(m, n, ntasks_per_thread, mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, x, y);
    CUDA_CHECK(cudaGetLastError());
#endif

}
