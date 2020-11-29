#include <cmath>
#include <omp.h>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <cstdio>

// #include "../../dbg-macro/dbg.h"

#include "common.h"

const char* version_name = "Optimized version (OpenMP)";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    if(grid_info->p_id == 0) {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        grid_info->local_size_z = grid_info->global_size_z;
    } else {
        grid_info->local_size_x = 0;
        grid_info->local_size_y = 0;
        grid_info->local_size_z = 0;
    }
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

#if !defined BX
#define BX 16
#endif
#if !defined BY
#define BY 16
#endif
#if !defined BZ
#define BZ 16
#endif
#if !defined BT
#define BT 4
#endif

#define BUF_DIM_X (BX + 2 * BT)
#define BUF_DIM_Y (BY + 2 * BT)
#define BUF_DIM_Z (BZ + 2 * BT)
#define BUF_SIZE (BUF_DIM_X * BUF_DIM_Y * BUF_DIM_Z)


// copy data from buffer to origin matrix
static inline __attribute__((always_inline)) void copy_from_buffer(
    ptr_t a, ptr_t b, ptr_t c,
    cptr_t a_buf, cptr_t b_buf, cptr_t c_buf,
    int x, int y, int z,
    int x_start, int y_start, int z_start,
    int ldx, int ldy, int ldz
) {
    for (int zz = z; zz < z + BZ; ++zz) {
        for (int yy = y; yy < y + BY; ++yy) {
            int buf_z = (zz - z_start) % BZ + BT;
            int buf_y = (yy - y_start) % BY + BT;
            int buf_off = INDEX(BT, buf_y, buf_z, BUF_DIM_X, BUF_DIM_Y);
            int src_off = INDEX(x, yy, zz, ldx, ldy);
            memcpy(a + src_off, a_buf + buf_off, sizeof(data_t) * BX);
            memcpy(b + src_off, b_buf + buf_off, sizeof(data_t) * BX);
            memcpy(c + src_off, c_buf + buf_off, sizeof(data_t) * BX);
        }
    }
}


// print thread local buffer, debug purpose
void debug_buffer(ptr_t buf, int dim, int lda, int start=0) {
    fprintf(stderr, "\n\nstart ");
    for (int z = start; z < start + dim; ++z) {
        for (int y = start; y < start + dim; ++y) {
            for (int x = start; x < start + dim; ++x) {
                fprintf(stderr, "%.3lf ", buf[INDEX(x, y, z, lda, lda)]);
            }
        }
    }
    fprintf(stderr, "end\n\n");
    fflush(stderr);
}

template <int BUFFER_CLEAR_EXT = 3>
static inline __attribute__((always_inline)) void clear_buffers(ptr_t a) {
    for (int zz = BT - BUFFER_CLEAR_EXT; zz < BZ + BT + BUFFER_CLEAR_EXT; ++zz) {
        for (int yy = BT - BUFFER_CLEAR_EXT; yy < BY + BT + BUFFER_CLEAR_EXT; ++yy) {
            bzero(a + INDEX(BT - BUFFER_CLEAR_EXT, yy, zz, BUF_DIM_X, BUF_DIM_Y), sizeof(data_t) * (BX + 2 * BUFFER_CLEAR_EXT));
        }
    }
}


ptr_t stencil_7(ptr_t A0, ptr_t A1, ptr_t B0, ptr_t B1, ptr_t C0, ptr_t C1, const dist_grid_info_t *grid_info, int nt) {
    ptr_t bufferx[2] = {A0, A1};
    ptr_t buffery[2] = {B0, B1};
    ptr_t bufferz[2] = {C0, C1};

    // check constraints
    assert(grid_info->local_size_x % BX == 0);
    assert(grid_info->local_size_y % BY == 0);
    assert(grid_info->local_size_z % BZ == 0);
    assert(BT % 2 == 0);


    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    // actual stencil computation
    auto stencil_inner_loop = [](cptr_t a0, ptr_t a1, cptr_t b0, ptr_t b1, cptr_t c0, ptr_t c1, int x_start, int x_end, int y, int z, int ldx, int ldy, int ldz) __attribute__((always_inline)) {
        #pragma omp simd
        for (int x = x_start; x < x_end; ++x) {
            data_t a7,b7,c7;
            a7 \
                = ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] \
                + ALPHA_NZZ * a0[INDEX(x-1, y, z, ldx, ldy)] \
                + ALPHA_PZZ * a0[INDEX(x+1, y, z, ldx, ldy)] \
                + ALPHA_ZNZ * a0[INDEX(x, y-1, z, ldx, ldy)] \
                + ALPHA_ZPZ * a0[INDEX(x, y+1, z, ldx, ldy)] \
                + ALPHA_ZZN * a0[INDEX(x, y, z-1, ldx, ldy)] \
                + ALPHA_ZZP * a0[INDEX(x, y, z+1, ldx, ldy)];
            b7 \
                = ALPHA_PNZ * b0[INDEX(x, y, z, ldx, ldy)] \
                + ALPHA_NPZ * b0[INDEX(x-1, y, z, ldx, ldy)] \
                + ALPHA_PPZ * b0[INDEX(x+1, y, z, ldx, ldy)] \
                + ALPHA_NZN * b0[INDEX(x, y-1, z, ldx, ldy)] \
                + ALPHA_PZN * b0[INDEX(x, y+1, z, ldx, ldy)] \
                + ALPHA_PZP * b0[INDEX(x, y, z-1, ldx, ldy)] \
                + ALPHA_NZP * b0[INDEX(x, y, z+1, ldx, ldy)];
            c7 \
                = ALPHA_PNN * c0[INDEX(x, y, z, ldx, ldy)] \
                + ALPHA_PPN * c0[INDEX(x-1, y, z, ldx, ldy)] \
                + ALPHA_PPN * c0[INDEX(x+1, y, z, ldx, ldy)] \
                + ALPHA_NNP * c0[INDEX(x, y-1, z, ldx, ldy)] \
                + ALPHA_PNP * c0[INDEX(x, y+1, z, ldx, ldy)] \
                + ALPHA_NPP * c0[INDEX(x, y, z-1, ldx, ldy)] \
                + ALPHA_PPP * c0[INDEX(x, y, z+1, ldx, ldy)];
            a1[INDEX(x, y, z, ldx, ldy)] = a7  +  (b7 * c7) / (b7 + c7); //sqrt
            b1[INDEX(x, y, z, ldx, ldy)] = b7  +  (a7 * c7) / (a7 + c7); //sqrt
            c1[INDEX(x, y, z, ldx, ldy)] = c7  +  (a7 * b7) / (a7 + b7); //sqrt
        }
    };


    ptr_t ret = A0;
    // fused rounds (after each round, a1 will store BT rounds of stencil on a0, and a0 will be garbage)
    int t_fused = (nt - 1) / BT;

    // main stencil loop
    for (int t = 0; t < t_fused; t++) {

        ptr_t a0 = bufferx[t % 2];
        ptr_t a1 = bufferx[(t + 1) % 2];

        ptr_t b0 = buffery[t % 2];
        ptr_t b1 = buffery[(t + 1) % 2];

        ptr_t c0 = bufferz[t % 2];
        ptr_t c1 = bufferz[(t + 1) % 2];
        
#pragma omp parallel for collapse(2) schedule(static)
        for (int z = z_start; z < z_end; z += BZ) {
            for (int y = y_start; y < y_end; y += BY) {
                for (int x = x_start; x < x_end; x += BX) {
                    int z_off = z - z_start, y_off = y - y_start, x_off = x - x_start;
                    using std::min;
                    using std::max;
                    int z_begin = max(z_off - BT, 0), z_stop = min(z + BZ + BT, z_end) - z_start, buf_z_start = BT - (z_off - z_begin), buf_z_end = BUF_DIM_Z - (z_off + BZ + BT - z_stop);
                    int y_begin = max(y_off - BT, 0), y_stop = min(y + BY + BT, y_end) - y_start, buf_y_start = BT - (y_off - y_begin), buf_y_end = BUF_DIM_Y - (y_off + BY + BT - y_stop);
                    int x_begin = max(x_off - BT, 0), x_stop = min(x + BX + BT, x_end) - x_start, buf_x_start = BT - (x_off - x_begin), buf_x_end = BUF_DIM_X - (x_off + BX + BT - x_stop);
                    // allocate contiguous buffer
                    data_t a_buf_0[BUF_SIZE] = {}, b_buf_0[BUF_SIZE] = {}, c_buf_0[BUF_SIZE] = {};
                    data_t a_buf_1[BUF_SIZE] = {}, b_buf_1[BUF_SIZE] = {}, c_buf_1[BUF_SIZE] = {};
                    // clear buffer to avoid errors
                    // clear_buffers(a_buf_0); clear_buffers(b_buf_0); clear_buffers(c_buf_0);
                    // clear_buffers(a_buf_1); clear_buffers(b_buf_1); clear_buffers(c_buf_1);
                    // data needed to be copied in each loop
                    size_t copy_size = sizeof(data_t) * (x_stop - x_begin);
                    // pack a0, b0, c0 (and BT level of neighbours) to buffer
                    for (int z_ = z_begin; z_ < z_stop; ++z_) {
                        int src_z = z_ + z_start;
                        int buf_z = z_ - z_off + BT;
                        for (int y_ = y_begin; y_ < y_stop; ++y_) {
                            int src_y = y_ + y_start;
                            int src_x = x_begin + x_start;
                            int buf_y = y_ - y_off + BT;
                            // dbg(buf_x, buf_y, buf_z, src_x, src_y, src_z);
                            int buf_off = INDEX(buf_x_start, buf_y, buf_z, BUF_DIM_X, BUF_DIM_Y);
                            int src_off = INDEX(src_x, src_y, src_z, ldx, ldy);
                            memcpy(a_buf_0 + buf_off, a0 + src_off, copy_size);
                            memcpy(b_buf_0 + buf_off, b0 + src_off, copy_size);
                            memcpy(c_buf_0 + buf_off, c0 + src_off, copy_size);
                        }
                    }
                    // run stencil kernel on current block for BT rounds
                    // in each round, the dimension shrinks by 1
                    // a_buf_0 -> a_buf_1 -> a_buf_0 -> a_buf_1 -> a_buf_0
                    ptr_t A0 = a_buf_0, A1 = a_buf_1, B0 = b_buf_0, B1 = b_buf_1, C0 = c_buf_0, C1 = c_buf_1;
#pragma unroll(BT)
                    for (int t = BT - 1; t >= 0; --t) {
                        int x_start = max(BT - t, buf_x_start), y_start = max(BT - t, buf_y_start), z_start = max(BT - t, buf_z_start);
                        int x_end = min(BT + BX + t, buf_x_end), y_end = min(BT + BY + t, buf_y_end), z_end = min(BT + BZ + t, buf_z_end);
                        for (int zz = z_start; zz < z_end; ++zz) {
                            for (int yy = y_start; yy < y_end; ++yy) {
                                stencil_inner_loop(A0, A1, B0, B1, C0, C1, x_start, x_end, yy, zz, BUF_DIM_X, BUF_DIM_Y, BUF_DIM_Z);
                            }
                        }
                        // swap buffers for next round
                        std::swap(A0, A1);
                        std::swap(B0, B1);
                        std::swap(C0, C1);
                    }
                    // copy back from buffer to a1, b1, c1
                    copy_from_buffer(a1, b1, c1, a_buf_0, b_buf_0, c_buf_0, x, y, z, x_start, y_start, z_start, ldx, ldy, ldz);
                    ret = a1;
                }
            }
        }
    }


    // deal with remaining steps (because we need information from both A0 and A1)
    for (int t = 0; t < nt - t_fused * BT; ++t) {
        int t_ = t_fused + t; // actual rounds
        cptr_t a0 = bufferx[t_ % 2];
        ptr_t a1 = bufferx[(t_ + 1) % 2];

        cptr_t b0 = buffery[t_ % 2];
        ptr_t b1 = buffery[(t_ + 1) % 2];

        cptr_t c0 = bufferz[t_ % 2];
        ptr_t c1 = bufferz[(t_ + 1) % 2];
        
#pragma omp parallel for collapse(2) schedule(static)
        for (int z = z_start; z < z_end; ++z) {
            for (int y = y_start; y < y_end; ++y) {
                stencil_inner_loop(a0, a1, b0, b1, c0, c1, x_start, x_end, y, z, ldx, ldy, ldz);
            }
        }
        ret = a1;
    }

    return ret;
}
