#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED 1

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#define UNREACHABLE (-1)
typedef int32_t index_t;
typedef double data_t;
#define MPI_DATA MPI_DOUBLE
typedef void (*free_func_t)(void*);

typedef struct {
    int *finished;
    int *curr_id;
    int *row_offset;
    int warp_count;
} sptrsv_info_t;

/* 
 * global_m: number of rows in the whole input matrix
 * global_n: number of columns in the whole input matrix
 * global_nnz: number of non-zeros in the whole input matrix
 * local_m: number of rows in the current process
 * offset_i: number of rows in previous processes
 * local_nnz: number of non-zeros in the current process
 */
typedef struct {
    int global_m, global_nnz;             /* do not modify */
    index_t *perm;                        /* do not modify after preprocess */
    
    index_t *r_pos;
    index_t *c_idx;
    data_t *values;
    index_t *flag;
    free_func_t CPU_free;

    index_t *__restrict__ gpu_r_pos;
    index_t *__restrict__ gpu_c_idx;
    data_t *__restrict__ gpu_values;
    free_func_t GPU_free;

    void *additional_info;         /* any information you want to attach */
} dist_matrix_t;

#ifdef __cplusplus
extern "C" {
#endif

void preprocess(dist_matrix_t *mat);
void destroy_additional_info(void *additional_info);
void sptrsv(dist_matrix_t *mat, const data_t *__restrict__ b, data_t *__restrict__ x);

#ifdef __cplusplus
}
#endif

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

#endif