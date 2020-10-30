
#include <immintrin.h>
#include <sched.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

#define MAX_N 2500

#ifndef ENABLE_STRASSEN
#define ENABLE_STRASSEN 0
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

#if !ENABLE_STRASSEN
const char *dgemm_desc = "Simple blocked dgemm.";
#else
const char *dgemm_desc = "Simple blocked dgemm (with Strassen algorithm).";
#endif


// use AVX2 to calculate C += or = A * B, row major
template<bool aligned, bool override, int M, int N = M, int K = M, int UNROLL = N / 4>
static inline __attribute__((always_inline)) void avx_kernel(
    int lda, int ldb, int ldc, const double *__restrict__ const A,
    const double *__restrict__ const B, double *__restrict__ const C) {

  // copy whole A to L1
  static double A_block[M * K];

  for (int i = 0; i < M; ++i) {
    memcpy(A_block + i * K, A + i * lda, sizeof(double) * K);
  }

  // calculate using AVX intrinsics
  for (int j = 0; j < N; j += 4 * UNROLL) {
#pragma ivdep
    for (int i = 0; i < M; i++) {
      __m256d ymm[UNROLL];
#pragma unroll(UNROLL)
        for (int x = 0; x < UNROLL; x++) {
          if constexpr (override) {
            ymm[x] = _mm256_setzero_pd();
          } else if constexpr (aligned) {
            ymm[x] = _mm256_loadu_pd(C + i * ldc + j + x * 4);
          } else {
            ymm[x] = _mm256_load_pd(C + i * ldc + j + x * 4);
          }
        }

#pragma unroll(K)
      for (int k = 0; k < K; k++) {
#pragma unroll(UNROLL)
        for (int x = 0; x < UNROLL; x++) {
          // gcc cannot inline fmadd, so weak
          // ymm[x] =
          //     _mm256_fmadd_pd(_mm256_load_pd(B + k * lda + j + x * 4),
          //                     _mm256_broadcast_sd(A + i * lda + k), ymm[x]);
          __m256d B_block;
          if constexpr (aligned) {
            B_block = _mm256_load_pd(B + k * ldb + j + x * 4);
          } else {
            B_block = _mm256_loadu_pd(B + k * ldb + j + x * 4);
          }
          ymm[x] = _mm256_add_pd(ymm[x], _mm256_mul_pd(B_block,
                              _mm256_broadcast_sd(A_block + i * K + k)));
        }
      }

#pragma unroll(UNROLL)
      for (int x = 0; x < UNROLL; x++) {
        if constexpr (aligned) {
          _mm256_store_pd(C + i * ldc + j + x * 4, ymm[x]);
        } else {
          _mm256_storeu_pd(C + i * ldc + j + x * 4, ymm[x]);
        }
      }
    }
  }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline __attribute__((always_inline)) void do_block_naive(
    int lda, int ldb, int ldc, int M, int N, int K, const double *__restrict__ const A,
    const double *__restrict__ const B, double *__restrict__ const C) {

  // fprintf(stderr, "Naive %d %d %d %p %p %p %d %d %d\n", M, N, K, A, B, C, lda, ldb, ldc);

  /* For each row i of A */
#pragma ivdep
  for (int i = 0; i < M; ++i) {
    /* For each column j of B */
    __builtin_prefetch(A + i * lda, 0);
    __builtin_prefetch(C + i * ldc, 1);
#pragma ivdep
    for (int j = 0; j < N; ++j) {
      /* Compute C(i,j) */
#pragma ivdep
      for (int k = 0; k < K; ++k) C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
    }
  }
}


// buffer for DGEMM padding
double *A_buf, *B_buf, *C_buf;

template <bool aligned, int BLOCK_SIZE_M, int BLOCK_SIZE_N = BLOCK_SIZE_M, int BLOCK_SIZE_K = BLOCK_SIZE_M>
static inline __attribute__((always_inline)) void do_block_simd(bool pad, int dim, int whole_width, int lda, const double *__restrict__ const A,
    const double *__restrict__ const B, double *__restrict__ const C) {
  /* For each block-row of A */
  for (int i = 0; i < dim; i += BLOCK_SIZE_M) {
    /* For each block-column of B */
    for (int j = 0; j < dim; j += BLOCK_SIZE_N) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < dim; k += BLOCK_SIZE_K) {
        // if in the "whole blocks" region
        if (likely(i < whole_width && j < whole_width && k < whole_width)) {
#pragma forceinline
          avx_kernel<aligned, false, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(lda, lda, lda, A + i * lda + k, B + k * lda + j, C + i * lda + j);
        } else {
          int M = min(BLOCK_SIZE_M, dim - i);
          int N = min(BLOCK_SIZE_N, dim - j);
          int K = min(BLOCK_SIZE_K, dim - k);
          // printf("SIMD %d %d %d %d %d\n", i, j, k, lda, dim);
          if (likely(pad)) {
            // padded, the remaining numbers are all M * N * K
            // use SIMD kernel to calculate remaining
            // need to judge whether source / dest data resides in buf or original array
            int stride_a = (i < whole_width && k < whole_width) ? lda : MAX_N;
            int stride_b = (k < whole_width && j < whole_width) ? lda : MAX_N;
            int stride_c = (i < whole_width && j < whole_width) ? lda : MAX_N;
            const double *__restrict__ const A_ = (i < whole_width && k < whole_width) ? A : A_buf;
            const double *__restrict__ const B_ = (k < whole_width && j < whole_width) ? B : B_buf;
            double *__restrict__ const C_ = (i < whole_width && j < whole_width) ? C : C_buf;
#pragma forceinline
            avx_kernel<aligned, false, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(stride_a, stride_b, stride_c, A_ + i * stride_a + k, B_ + k * stride_b + j,
                C_ + i * stride_c + j);
          } else {
            // printf("naive %d %d %d\n", M, N, K);
            // use naive implementation to calculate remaining numbers
#pragma forceinline
            do_block_naive(lda, lda, lda, M, N, K, A + i * lda + k, B + k * lda + j,
                          C + i * lda + j);
          }
        }
      }
    }
  }
}


#include "dgemm-blocked-strassen-kernel.hh"
#define STRASSEN_DEGRADE_SIZE 32

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
extern "C" void square_dgemm(int lda, const double *__restrict__ A, const double *__restrict__ B,
                  double *__restrict__ const C) {

  // (A*B)^T = B^T * A^T, so we can treat A, B, C in row-major format and
  // calculate C = C + B * A swap A and B for simplicity
  const double *__restrict__ temp = A;
  A = B;
  B = temp;


  // if everything is aligned, we can use the aligned kernel!
  bool aligned = ((size_t) A & 0b11111) == 0 && ((size_t) B & 0b11111) == 0 && ((size_t) C & 0b11111) == 0 && (lda % 4) == 0;

  bool pad = false;
  int dim = lda;

#if ENABLE_STRASSEN
  if (likely((dim & (dim - 1)) == 0 && dim >= STRASSEN_DEGRADE_SIZE)) {
    // power of 2 - use strassen
    // C_s = A * B
    do_block_strassen<STRASSEN_DEGRADE_SIZE>(dim, dim, MAX_N, dim, A, B, C_strassen, st_im_1, st_im_2, st_im_3, st_im_4, st_p_1, st_p_2);
    // C += C_s
    matrix_add_single(MAX_N, dim, dim, C_strassen, C);
    return;
  }
#endif

  int size = lda;

  if (lda % 32 == 31) {
    size++;
  } else if (lda % 32 == 1) {
    size--;
  }

  int padding_dim = 32;

  if (size % 40 == 0) {
    padding_dim = 40;
  }

  
  int whole_count = lda / padding_dim;
  int whole_width = whole_count * padding_dim;
  int remain = lda % padding_dim; // AVX registers

  // pad if remaining numbers are relatively many
  if (remain > padding_dim / 2 + 1) {
    pad = true;
    dim = (lda / padding_dim + 1) * padding_dim;
#pragma ivdep
    for (int i = 0; i < whole_width; ++i) {
      double *__restrict__ const A_buf_pos = A_buf + i * MAX_N + whole_width;
      double *__restrict__ const B_buf_pos = B_buf + i * MAX_N + whole_width;
      double *__restrict__ const C_buf_pos = C_buf + i * MAX_N + whole_width;
      const double *__restrict__ const A_pos = A + i * lda + whole_width;
      const double *__restrict__ const B_pos = B + i * lda + whole_width;
      const double *__restrict__ const C_pos = C + i * lda + whole_width;
      __builtin_prefetch(B_buf_pos, 1);
      __builtin_prefetch(B_pos, 0);
      __builtin_prefetch(C_buf_pos, 1);
      __builtin_prefetch(C_pos, 0);
      memcpy(A_buf_pos, A_pos, sizeof(double) * remain);
      memcpy(B_buf_pos, B_pos, sizeof(double) * remain);
      memcpy(C_buf_pos, C_pos, sizeof(double) * remain);
    }
#pragma ivdep
    for (int i = whole_width; i < lda; ++i) {
      double *__restrict__ const A_buf_pos = A_buf + i * MAX_N;
      double *__restrict__ const B_buf_pos = B_buf + i * MAX_N;
      double *__restrict__ const C_buf_pos = C_buf + i * MAX_N;
      const double *__restrict__ const A_pos = A + i * lda;
      const double *__restrict__ const B_pos = B + i * lda;
      const double *__restrict__ const C_pos = C + i * lda;
      __builtin_prefetch(B_buf_pos, 1);
      __builtin_prefetch(B_pos, 0);
      __builtin_prefetch(C_buf_pos, 1);
      __builtin_prefetch(C_pos, 0);
      memcpy(A_buf_pos, A_pos, sizeof(double) * lda);
      memcpy(B_buf_pos, B_pos, sizeof(double) * lda);
      memcpy(C_buf_pos, C_pos, sizeof(double) * lda);
    }
  }

  if (padding_dim == 32) {
    if (aligned) {
      do_block_simd<true, 32>(pad, dim, whole_width, lda, A, B, C);
    } else {
      do_block_simd<false, 32>(pad, dim, whole_width, lda, A, B, C);
    }
  } else if (padding_dim == 40) {
    if (aligned) {
      do_block_simd<true, 40>(pad, dim, whole_width, lda, A, B, C);
    } else {
      do_block_simd<false, 40>(pad, dim, whole_width, lda, A, B, C);
    }
  } else {
    __builtin_unreachable();
  }

  // copy data back
  if (pad) {
#pragma ivdep
    for (int i = 0; i < whole_width; ++i) {
      const double *__restrict__ const C_buf_pos = C_buf + i * MAX_N + whole_width;
      double *__restrict__ const C_pos = C + i * lda + whole_width;
      __builtin_prefetch(C_buf_pos + MAX_N, 0);
      __builtin_prefetch(C_pos + lda, 1);
      memcpy(C_pos, C_buf_pos, sizeof(double) * remain);
    }
#pragma ivdep
    for (int i = whole_width; i < lda; ++i) {
      const double *__restrict__ const C_buf_pos = C_buf + i * MAX_N;
      double *__restrict__ const C_pos = C + i * lda;
      __builtin_prefetch(C_buf_pos + MAX_N, 0);
      __builtin_prefetch(C_pos + lda, 1);
      memcpy(C_pos, C_buf_pos, sizeof(double) * lda);
    }
  }
  
}

// bind this process to CPU 1
__attribute__((constructor)) void bind_core() {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(1, &set);
  sched_setaffinity(0, sizeof(set), &set);
  size_t buf_size = MAX_N * MAX_N * sizeof(double);
  posix_memalign((void **)&A_buf, 64, buf_size);
  posix_memalign((void **)&B_buf, 64, buf_size);
  posix_memalign((void **)&C_buf, 64, buf_size);
  bzero(A_buf, buf_size);
  bzero(B_buf, buf_size);
  bzero(C_buf, buf_size);
}
