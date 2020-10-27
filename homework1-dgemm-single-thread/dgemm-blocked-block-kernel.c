  /* For each block-row of A */
  for (int i = 0; i < dim; i += BLOCK_SIZE) {
    /* For each block-column of B */
    for (int j = 0; j < dim; j += BLOCK_SIZE) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < dim; k += BLOCK_SIZE) {
        // if in the "whole blocks" region
        if (likely(i < whole_width && j < whole_width && k < whole_width)) {
#pragma forceinline
          AVX_KERNEL_NAME_(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)(lda, lda, lda, A + i * lda + k, B + k * lda + j, C + i * lda + j, false);
        } else {
          int M = min(BLOCK_SIZE, dim - i);
          int N = min(BLOCK_SIZE, dim - j);
          int K = min(BLOCK_SIZE, dim - k);
          // printf("SIMD %d %d %d %d %d\n", i, j, k, lda, dim);
          if (likely(pad)) {
            // padded, the remaining numbers are all 32x32 blocks
            // use SIMD kernel to calculate remaining
            // need to judge whether source / dest data resides in buf or original array
            int stride_a = (i < whole_width && k < whole_width) ? lda : MAX_N;
            int stride_b = (k < whole_width && j < whole_width) ? lda : MAX_N;
            int stride_c = (i < whole_width && j < whole_width) ? lda : MAX_N;
            const double *__restrict__ const A_ = (i < whole_width && k < whole_width) ? A : A_buf;
            const double *__restrict__ const B_ = (k < whole_width && j < whole_width) ? B : B_buf;
            double *__restrict__ const C_ = (i < whole_width && j < whole_width) ? C : C_buf;
#pragma forceinline
            AVX_KERNEL_NAME_(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)(stride_a, stride_b, stride_c, A_ + i * stride_a + k, B_ + k * stride_b + j,
                C_ + i * stride_c + j, false);
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
