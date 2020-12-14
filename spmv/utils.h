#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED 1

#include <stdio.h>
#include <assert.h>

#include "common.h"

#define CHECK(err, err_code) if(err) { return err_code; }
#define CHECK_ERROR(ret, err_code) CHECK(ret != 0, err_code)
#define CHECK_NULL(ret, err_code) CHECK(ret == NULL, err_code)

#ifdef __cplusplus
extern "C" {
#endif

int fatal_error(int code);

int read_matrix_default(dist_matrix_t *mat, const char* filename);
void destroy_dist_matrix(dist_matrix_t *mat);

int read_vector(dist_matrix_t *mat, const char* filename, const char* suffix, int n, data_t* x);
int check_answer(dist_matrix_t *mat, const char* filename, data_t* y);

#ifdef __cplusplus
}
#endif

#define SUCCESS cudaSuccess
#define NO_MEM cudaErrorMemoryAllocation
#define IO_ERR 3

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = (condition); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d: %s\n", error, cudaGetErrorString(error)); \
        assert(error == cudaSuccess); \
    } \
  } while (0)

#endif

#if __CUDA_ARCH__ < 600
#define ATOMIC_ADD_DOUBLE atomicAddDouble
__device__ inline double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#else
#define ATOMIC_ADD_DOUBLE atomicAdd
#endif
