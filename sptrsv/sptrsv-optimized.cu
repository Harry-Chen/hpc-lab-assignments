#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include "common.h"
#include "utils.h"

const char* version_name = "optimized version";\

void preprocess(dist_matrix_t *mat) {
    
    /*
    int m = mat->global_m;
    int nnz = mat->global_nnz;
    int size = (m + 1) * sizeof(index_t);
    cudaMalloc((void**) &mat->gpu_r_pos, size);
    cudaMemcpy(mat->gpu_r_pos, mat->r_pos, size, cudaMemcpyHostToDevice);
    size = (nnz + 1) * sizeof(index_t);
    cudaMalloc((void**) &mat->gpu_c_idx, size);
    cudaMemcpy(mat->gpu_c_idx, mat->c_idx, size, cudaMemcpyHostToDevice);
    size = (nnz + 1) * sizeof(data_t);
    cudaMalloc((void**) &mat->gpu_values, size);
    cudaMemcpy(mat->gpu_values, mat->values, size, cudaMemcpyHostToDevice);
    */
}

void destroy_additional_info(void *additional_info) {
}


void sptrsv(dist_matrix_t *mat, const data_t* b, data_t* x) {

}
