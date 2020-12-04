#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cusparse.h>
#include "common.h"

const char* version_name = "cuSPARSE SpMV";\

#define CHECK_CUSPARSE(ret) if(ret != CUSPARSE_STATUS_SUCCESS) { fprintf(stderr, "error in line %d\n", __LINE__);}

typedef struct {
    cusparseHandle_t handle;
    cusparseMatDescr_t descrA;
} additional_info_t;

typedef additional_info_t *info_ptr_t;

void preprocess(dist_matrix_t *mat) {
    info_ptr_t p = (info_ptr_t)malloc(sizeof(additional_info_t));
    cusparseCreate(&p->handle);
    cusparseSetPointerMode(p->handle, CUSPARSE_POINTER_MODE_HOST);
    cusparseCreateMatDescr(&p->descrA);
    //cusparseSetMatIndexBase(p->descrA, CUSPARSE_INDEX_BASE_ZERO);
    //cusparseSetMatType(p->descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    mat->additional_info = p;
}

void destroy_additional_info(void *additional_info) {
    info_ptr_t p = (info_ptr_t)additional_info;
    cusparseDestroy(p->handle);
    cusparseDestroyMatDescr(p->descrA);
    free(p);
}

void spmv(dist_matrix_t *mat, const data_t* x, data_t* y) {
    int m = mat->global_m, nnz = mat->global_nnz;
    const data_t alpha = 1.0, beta = 1.0;
    info_ptr_t p = (info_ptr_t)mat->additional_info;

    cusparseDcsrmv(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, m, nnz, 
        &alpha, p->descrA, mat->gpu_values, mat->gpu_r_pos, mat->gpu_c_idx, x, &beta, y);
}
