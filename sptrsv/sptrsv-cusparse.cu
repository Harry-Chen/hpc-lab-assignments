#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cusparse.h>
#include "common.h"

const char* version_name = "cuSPARSE SpTRSV";\

#define CHECK_CUSPARSE(ret) if(ret != CUSPARSE_STATUS_SUCCESS) { fprintf(stderr, "error in line %d\n", __LINE__);}

typedef struct {
    cusparseHandle_t handle;
    cusparseMatDescr_t descrA;
    csrsv2Info_t info;
    void *pBuffer;
} additional_info_t;

typedef additional_info_t *info_ptr_t;

void preprocess(dist_matrix_t *mat) {
    info_ptr_t p = (info_ptr_t)malloc(sizeof(additional_info_t));
    int pBufferSize;
    cusparseCreate(&p->handle);
    cusparseSetPointerMode(p->handle, CUSPARSE_POINTER_MODE_HOST);
    cusparseCreateMatDescr(&p->descrA);
    cusparseSetMatFillMode(p->descrA, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(p->descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
    //cusparseSetMatIndexBase(p->descrA, CUSPARSE_INDEX_BASE_ZERO);
    
    cusparseCreateCsrsv2Info(&p->info);
    cusparseDcsrsv2_bufferSize(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, mat->global_m, \
                               mat->global_nnz, p->descrA, mat->gpu_values, mat->gpu_r_pos, \
                               mat->gpu_c_idx, p->info, &pBufferSize);
    cudaMalloc(&p->pBuffer, pBufferSize);
    cusparseDcsrsv2_analysis(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, mat->global_m, \
                             mat->global_nnz, p->descrA, mat->gpu_values, mat->gpu_r_pos, \
                             mat->gpu_c_idx, p->info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, p->pBuffer);
    mat->additional_info = p;
    mat->perm = (int*) malloc(sizeof(int) * mat->global_m);
    for(int i = 0; i < mat->global_m; ++i) {
        mat->perm[i] = i;
    }
}

void destroy_additional_info(void *additional_info) {
    info_ptr_t p = (info_ptr_t)additional_info;
    cudaFree(p->pBuffer);
    cusparseDestroyCsrsv2Info(p->info);
    cusparseDestroyMatDescr(p->descrA);
    cusparseDestroy(p->handle);
    free(p);
}

void sptrsv(dist_matrix_t *mat, const data_t* b, data_t* x) {
    int m = mat->global_m, nnz = mat->global_nnz;
    const data_t alpha = 1.0;
    info_ptr_t p = (info_ptr_t)mat->additional_info;
    cusparseDcsrsv2_solve(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnz, &alpha, \
                          p->descrA, mat->gpu_values, mat->gpu_r_pos, mat->gpu_c_idx, \
                          p->info, b, x, CUSPARSE_SOLVE_POLICY_USE_LEVEL, p->pBuffer);
}
