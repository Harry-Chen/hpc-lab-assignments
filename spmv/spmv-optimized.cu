#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

#include "common.h"
#include "utils.h"

#define GRID_SIZE 16
#define BLOCK_SIZE 1024
#define THREAD_NUM (GRID_SIZE * BLOCK_SIZE)

int my_abort(int line, int code);

#define MY_ABORT(ret) my_abort(__LINE__, ret)
#define ABORT_IF_ERROR(ret) CHECK_ERROR(ret, MY_ABORT(ret))

const char* version_name = "optimized version";

void preprocess(dist_matrix_t *mat) {
    // preprocess (i, j) for each thread
    int m = mat->global_m;
    int n = mat->global_nnz;
    int total_tasks = m + n;
    int ntasks_per_thread = ceiling(total_tasks, THREAD_NUM); // work item count for each thread

    auto info = new csr_info_t;
    mat->additional_info = info;

    CUDA_CHECK(cudaMalloc(&info->row_offsets, sizeof(index_t) * THREAD_NUM));
    CUDA_CHECK(cudaMalloc(&info->index_offsets, sizeof(index_t) * THREAD_NUM));

    for (int i = 0; i < 10; ++i) {
        // fprintf(stderr, "r_pos[%d] = [%d]\n", i, mat->r_pos[i]);
    }

    int last_i = 0;

    // calculate offset for each thread
    auto row_offsets_cpu = new index_t[THREAD_NUM], index_offsets_cpu = new index_t[THREAD_NUM];
    for (int t = 0; t < THREAD_NUM; t++) {
        int k = t * ntasks_per_thread;
        // find first (i, j) that i + j == k && row_offset[i] > j - 1
        while (mat->r_pos[last_i + 1] <= k - last_i - 1) {
            last_i++;
        }
        row_offsets_cpu[t] = last_i;
        index_offsets_cpu[t] = k - last_i;
        // if (last_i <= 1000)
        // fprintf(stderr, "Thread %d: row %d index %d\n", t, last_i, k - last_i);
    }

    // copy preprocessed information to GPU
    CUDA_CHECK(cudaMemcpy(info->row_offsets, row_offsets_cpu, sizeof(index_t) * THREAD_NUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(info->index_offsets, index_offsets_cpu, sizeof(index_t) * THREAD_NUM, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
}

void destroy_additional_info(void *additional_info) {
}


__global__ void spmv_merge_based_kernel(int m, int nnz, int ntasks_per_thread, 
    index_t *__restrict__ row_offsets, index_t *__restrict__ index_offsets, const index_t *__restrict__ r_pos,
    const index_t *__restrict__ c_idx, const data_t *__restrict__ values, const data_t *__restrict__ x, data_t *__restrict__ y) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    index_t curr_row = row_offsets[i], curr_index = index_offsets[i];
    int ntasks = MIN(ntasks_per_thread, m + nnz - curr_row - curr_index); // task number for each thread
 
    if (curr_row < m && curr_index < nnz) {
        data_t res = 0.0;
        for (int t = 0; t < ntasks; ++t) {
            // int remain_tasks = ntasks - t;
            if (curr_index == r_pos[curr_row + 1]) {
                // end of a row, aggregate
                // if (full_row) { // current thread fully calculates this row
                //     y[curr_row] = res;
                // } else {
                //     atomicAdd(y[curr_row], res); 
                // }
                // if (curr_row >= 32 && curr_row <= 35) printf("thread %d runs to row %d index %d\n", i, curr_row, curr_index);
                atomicAddDouble(&y[curr_row], res);
                curr_row++;
                res = 0.0;
            } else {
                res += x[c_idx[curr_index]] * values[curr_index];
                curr_index++;
            }
        }
        atomicAddDouble(&y[curr_row], res);
    }

}

void spmv(dist_matrix_t *mat, const data_t *__restrict__ x, data_t *__restrict__ y) {

    int m = mat->global_m;
    int n = mat->global_nnz;
    int ntasks_per_thread = ceiling(m + n, THREAD_NUM);
    csr_info_t *info = (csr_info_t*) mat->additional_info;

    dim3 grid_size(GRID_SIZE, 1, 1);
    dim3 block_size(BLOCK_SIZE, 1, 1);

    spmv_merge_based_kernel<<<grid_size, block_size>>>(m, n, ntasks_per_thread,
        info->row_offsets, info->index_offsets, mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, x, y);
}
