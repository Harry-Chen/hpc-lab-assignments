
#include <immintrin.h>
#include <sched.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <omp.h>
#include <numa.h>

#include <algorithm>

#include "dgemm-openmp-threadnum.hh"

#ifndef _OPENMP
#error This file must be compiled with OpenMP
#endif

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

#define MAX_N 2500

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

const char *dgemm_desc = "Simple blocked dgemm with OpenMP.";

// use AVX2 to calculate C += or = A * B, row major
template<bool aligned, bool override, int M, int N = M, int K = M, int UNROLL = N / 4>
static inline __attribute__((always_inline)) void do_block_simd(
    int lda, int ldb, int ldc, const double *__restrict__ const A,
    const double *__restrict__ const B, double *__restrict__ const C) {

  // copy whole A to L1
  static double A_block[M * K];
#pragma omp threadprivate(A_block)

  if constexpr (!aligned) {
    for (int i = 0; i < M; ++i) {
      memcpy(A_block + i * K, A + i * lda, sizeof(double) * K);
    }
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
          } else {
            ymm[x] = _mm256_loadu_pd(C + i * ldc + j + x * 4);
          }
        }

#pragma unroll(K)
      for (int k = 0; k < K; k++) {
#pragma unroll(UNROLL)
        for (int x = 0; x < UNROLL; x++) {
          auto B_block = _mm256_loadu_pd(B + k * ldb + j + x * 4);
          __m256d A_num;
          if constexpr (aligned) {
            A_num = _mm256_broadcast_sd(A + i * lda + k);
          } else {
            A_num = _mm256_broadcast_sd(A_block + i * K + k);
          }
          ymm[x] = _mm256_fmadd_pd(A_num, B_block, ymm[x]);
        }
      }

#pragma unroll(UNROLL)
      for (int x = 0; x < UNROLL; x++) {
        _mm256_storeu_pd(C + i * ldc + j + x * 4, ymm[x]);
      }
    }
  }
}

template<int M, int K, int ROUND16 = K / 16, int ROUND8 = (K - ROUND16 * 16) / 8>
static inline __attribute__((always_inline)) void do_block_simd_N_1(
  int lda, int ldb, int ldc, const double *__restrict__ const A,
    const double *__restrict__ const B, double *__restrict__ const C
) {

  static_assert(ROUND16 * 16 + ROUND8 * 8 == K);

  // buffer to store B
  __m256d y[K / 4];
#pragma unroll(K / 4)
  for (int i = 0; i < K; i += 4) {
    y[i / 4] = _mm256_set_pd(B[(i + 3) * ldb], B[(i + 2) * ldb], B[(i + 1) * ldb], B[i * ldb]);
  }

  for (int i = 0; i < M; ++i) {

    // interemdiate results
    __m256d dot_product[2];

#pragma unroll(ROUND16)
    // do product of 16 numbers to 2 vectors
    for (int x = 0; x < ROUND16; ++x) {
      auto xy0 = _mm256_mul_pd(_mm256_loadu_pd(A + i * lda + x * 16 + 0), y[x * 4 + 0]);
      auto xy1 = _mm256_mul_pd(_mm256_loadu_pd(A + i * lda + x * 16 + 4), y[x * 4 + 1]);
      auto xy2 = _mm256_mul_pd(_mm256_loadu_pd(A + i * lda + x * 16 + 8), y[x * 4 + 2]);
      auto xy3 = _mm256_mul_pd(_mm256_loadu_pd(A + i * lda + x * 16 + 12), y[x * 4 + 3]);
      // low to high: xy00+xy01 xy10+xy11 xy02+xy03 xy12+xy13
      auto temp01 = _mm256_hadd_pd(xy0, xy1);   
      // low to high: xy20+xy21 xy30+xy31 xy22+xy23 xy32+xy33
      auto temp23 = _mm256_hadd_pd(xy2, xy3);
      // low to high: xy02+xy03 xy12+xy13 xy20+xy21 xy30+xy31
      auto swapped = _mm256_permute2f128_pd(temp01, temp23, 0x21);
      // low to high: xy00+xy01 xy10+xy11 xy22+xy23 xy32+xy33
      auto blended = _mm256_blend_pd(temp01, temp23, 0b1100);
      dot_product[x] = _mm256_add_pd(swapped, blended);
    }
    // accumalate product sums
    auto sum = _mm256_hadd_pd(dot_product[0], dot_product[1]);
    // add upper 128 bits of sum to its lower 128 bits
    auto result = _mm_add_pd(_mm256_extractf128_pd(sum, 1), _mm256_castpd256_pd128(sum));
    // do product of 8 numbers and accumulate to result

#pragma unroll(ROUND8)
    for (int k = ROUND16 * 16; k < K; k += 8) {
      auto xy = _mm256_mul_pd(_mm256_loadu_pd(A + i * lda + k), y[k / 8 * 2]);
      auto zw = _mm256_mul_pd(_mm256_loadu_pd(A + i * lda + k + 4), y[k / 8 * 2 + 1]);
      auto temp = _mm256_hadd_pd(xy, zw);
      auto product = _mm_add_pd(_mm256_castpd256_pd128(temp), _mm256_extractf128_pd(temp, 1));
      result = _mm_add_pd(result, product);
    }
    C[i * lda] += ((double*)&result)[0] + ((double*)&result)[1];
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
#pragma vector vecremainder
    for (int j = 0; j < N; ++j) {
      /* Compute C(i,j) */
#pragma ivdep
      for (int k = 0; k < K; ++k) C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
    }
  }
}


// buffers for DGEMM padding
double *A_buf, *B_buf, *C_buf;


// calculate square DGEMM with certain block size 
template <bool aligned, int BLOCK_SIZE_M, int BLOCK_SIZE_N = BLOCK_SIZE_M, int BLOCK_SIZE_K = BLOCK_SIZE_M>
static inline __attribute__((always_inline)) void square_gemm_simd(bool pad, int dim, int whole_width, int lda, const double *__restrict__ const A,
    const double *__restrict__ const B, double *__restrict__ const C) {
  /* For each block-row of A */
#pragma omp parallel for schedule(static)
  for (int i = 0; i < dim; i += BLOCK_SIZE_M) {
    /* For each block-column of B */
    for (int j = 0; j < dim; j += BLOCK_SIZE_N) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < dim; k += BLOCK_SIZE_K) {
        // if in the "whole blocks" region
        if (likely(i < whole_width && j < whole_width && k < whole_width)) {
#pragma forceinline
          do_block_simd<aligned, false, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(lda, lda, lda, A + i * lda + k, B + k * lda + j, C + i * lda + j);
        } else {
          int M = min(BLOCK_SIZE_M, dim - i);
          int N = min(BLOCK_SIZE_N, dim - j);
          int K = min(BLOCK_SIZE_K, dim - k);
          // printf("SIMD %d %d %d %d %d\n", i, j, k, lda, dim);
          if (pad) {
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
            do_block_simd<aligned, false, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(stride_a, stride_b, stride_c, A_ + i * stride_a + k, B_ + k * stride_b + j,
                C_ + i * stride_c + j);
          } else {
            // printf("naive %d %d %d\n", M, N, K);
            // remaining large blocks
            if constexpr (BLOCK_SIZE_N == 32) {
              // use some specialized kernels to speed up
              if (M == 1 && N == BLOCK_SIZE_N && K == BLOCK_SIZE_K) {
#pragma forceinline
                do_block_simd<aligned, false, 1, BLOCK_SIZE_N, BLOCK_SIZE_K>(lda, lda, lda, A + i * lda + k, B + k * lda + j, C + i * lda + j);
              } else if (M == BLOCK_SIZE_M && N == BLOCK_SIZE_N && K == 1) {
#pragma forceinline
                do_block_simd<aligned, false, BLOCK_SIZE_M, BLOCK_SIZE_N, 1>(lda, lda, lda, A + i * lda + k, B + k * lda + j, C + i * lda + j);
              } else if (M == BLOCK_SIZE_M && N == 1 && K == BLOCK_SIZE_K) {
#pragma forceinline
                do_block_simd_N_1<BLOCK_SIZE_M, BLOCK_SIZE_K>(lda, lda, lda, A + i * lda + k, B + k * lda + j, C + i * lda + j);
              } else {
#pragma forceinline
                do_block_naive(lda, lda, lda, M, N, K, A + i * lda + k, B + k * lda + j, C + i * lda + j);
              }
            } else {
#pragma forceinline
              do_block_naive(lda, lda, lda, M, N, K, A + i * lda + k, B + k * lda + j, C + i * lda + j);
            }
          }
        }
      }
    }
  }
}

static int last_lda = -1, max_threads;

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

  bool pad = false; // when the matricies are padded
  int dim = lda; // the processed dimension

  // choose appropriate openmp thread number
  if (unlikely(lda != last_lda)) {
    int nodes = numa_num_task_nodes();
    int cpus = numa_num_task_cpus();
    last_lda = lda;
    auto opt_threads = OPENMP_THREADS_TUNED.find(lda);
    if (opt_threads != OPENMP_THREADS_TUNED.end()) {
        for (const auto t : opt_threads->second) {
            if (t <= max_threads) {
                // fprintf(stderr, "Set OMP_NUM_THREADS to %d for LDA %d OMP_NUM_THREADS %d\n", t, lda, max_threads);
                omp_set_num_threads(t);
                auto cpu_mask = numa_allocate_nodemask();
                numa_bitmask_setbit(cpu_mask, 0);
                if (t > cpus / nodes) {
                  numa_bitmask_setbit(cpu_mask, 1);
                }
                numa_bind(cpu_mask);
                break;
            }
        }
    }
  }

  // round to power of 2
  int size = lda;

  if (lda % 32 == 31) {
    size++;
  } else if (lda % 32 == 1) {
    size--;
  }

  // block size if needed to pad
  int padding_dim = 32;

  if (size % 40 == 0) {
    padding_dim = 40;
  }

  
  int whole_count = lda / padding_dim;
  int whole_width = whole_count * padding_dim;
  int remain = lda % padding_dim;

  // pad if remaining numbers are relatively many
  // copy only numbers that are out of whole_width * whole_width
  if (remain > padding_dim / 2 + 1) {
    pad = true;
    dim = (lda / padding_dim + 1) * padding_dim;
  }

  // copy on memory of last block (needs to be padded)
  if (pad) {
#pragma ivdep
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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

  // use different kernels for different block size / alignments
  if (padding_dim == 32) {
    if (aligned) {
      square_gemm_simd<true, 32>(pad, dim, whole_width, lda, A, B, C);
    } else {
      square_gemm_simd<false, 32>(pad, dim, whole_width, lda, A, B, C);
    }
  } else if (padding_dim == 40) {
    if (aligned) {
      square_gemm_simd<true, 40>(pad, dim, whole_width, lda, A, B, C);
    } else {
      square_gemm_simd<false, 40>(pad, dim, whole_width, lda, A, B, C);
    }
  } else {
    __builtin_unreachable();
  }

  // copy data back to original matrix (only modified part)
  if (pad) {
#pragma ivdep
#pragma omp parallel for schedule(static)
    for (int i = 0; i < whole_width; ++i) {
      const double *__restrict__ const C_buf_pos = C_buf + i * MAX_N + whole_width;
      double *__restrict__ const C_pos = C + i * lda + whole_width;
      __builtin_prefetch(C_buf_pos + MAX_N, 0);
      __builtin_prefetch(C_pos + lda, 1);
      memcpy(C_pos, C_buf_pos, sizeof(double) * remain);
    }
#pragma ivdep
#pragma omp parallel for schedule(static)
    for (int i = whole_width; i < lda; ++i) {
      const double *__restrict__ const C_buf_pos = C_buf + i * MAX_N;
      double *__restrict__ const C_pos = C + i * lda;
      __builtin_prefetch(C_buf_pos + MAX_N, 0);
      __builtin_prefetch(C_pos + lda, 1);
      memcpy(C_pos, C_buf_pos, sizeof(double) * lda);
    }
  }
  
}


// run before main()
__attribute__((constructor)) void preprocess() {
  // allocate aligned buffers and clear them
  size_t buf_size = MAX_N * MAX_N * sizeof(double);
  posix_memalign((void **)&A_buf, 64, buf_size);
  posix_memalign((void **)&B_buf, 64, buf_size);
  posix_memalign((void **)&C_buf, 64, buf_size);
  bzero(A_buf, buf_size);
  bzero(B_buf, buf_size);
  bzero(C_buf, buf_size);
  max_threads = omp_get_max_threads();
}

