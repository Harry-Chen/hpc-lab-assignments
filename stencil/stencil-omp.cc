#include <cmath>
#include <omp.h>
#include <cassert>
#include <algorithm>
#include <cstring>

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

#define BUF_DIM_X (BX + 2 * BT + 2)
#define BUF_DIM_Y (BY + 2 * BT + 2)
#define BUF_DIM_Z (BZ + 2 * BT + 2)
#define BUF_SIZE (BUF_DIM_X * BUF_DIM_Y * BUF_DIM_Z)


inline __attribute__((always_inline)) void stencil_7_kernel(
    ptr_t a0, ptr_t a1, ptr_t b0, ptr_t b1, ptr_t c0, ptr_t c1, 
    int x_start, int x_end,
    int y_start, int y_end,
    int z_start, int z_end
) {

    int ldx = BUF_DIM_X;
    int ldy = BUF_DIM_Y;
    int ldz = BUF_DIM_Z;

    for (int z = z_start; z < z_end; ++z) {
        for (int y = y_start; y < y_end; ++y) {
#pragma omp simd
#pragma ivdep
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
        }
    }
}


static inline __attribute__((always_inline)) void do_stencil_7_block(ptr_t A0, ptr_t A1, ptr_t B0, ptr_t B1, ptr_t C0, ptr_t C1) {
    for (int t = BT - 1; t >= 0; ++t) {
        int dim_x = BX + 2 * t, dim_y = BY + 2 * t, dim_z = BZ + 2 * t;
        int x_start = 1 + (BT - t), y_start = 1 + (BT - t), z_start = 1 + (BT - t);
        stencil_7_kernel(A0, A1, B0, B1, C0, C1, x_start, x_start + dim_x, y_start, y_start + dim_y, z_start, z_start + dim_z);
        std::swap(A0, A1);
        std::swap(B0, B1);
        std::swap(C0, C1);
    }
}

static inline __attribute__((always_inline)) void copy_to_buffer(
    cptr_t a, cptr_t b, cptr_t c,
    ptr_t a_buf, ptr_t b_buf, ptr_t c_buf,
    int x_off, int y_off, int z_off,
    int x_start, int y_start, int z_start,
    int x_end, int y_end, int z_end,
    int ldx, int ldy, int ldz
) {
    using std::min;
    using std::max;
    // begin index of x, y, z on buf
    int z_begin = max(z_off - BT, 0), z_stop = min(z_off + BZ + BT, z_end - z_start);
    int y_begin = max(y_off - BT, 0), y_stop = min(y_off + BY + BT, y_end - y_start);
    int x_begin = max(x_off - BT, 0), x_stop = min(x_off + BX + BT, x_end - x_start);
    for (int z = z_begin; z < z_stop; ++z) {
        int src_z = z + z_start;
        int buf_z = 1 + BT + z % BZ - z_begin;
        for (int y = y_begin; y < y_stop; ++y) {
            int src_y = y + y_start;
            int src_x = x_begin + x_start;
            int buf_y = 1 + BT + y % BY - y_begin;
            int buf_x = 1 + BT - x_begin;
            size_t copy_size = sizeof(data_t) * (x_stop - x_begin);
            int buf_off = INDEX(buf_x, buf_y, buf_z, BUF_DIM_X, BUF_DIM_Y);
            int src_off = INDEX(src_x, src_y, src_z, ldx, ldy);
            memcpy(a_buf + buf_off, a + src_off, copy_size);
            memcpy(b_buf + buf_off, b + src_off, copy_size);
            memcpy(c_buf + buf_off, c + src_off, copy_size);
        }
    }
}

static inline __attribute__((always_inline)) void copy_from_buffer(
    ptr_t a, ptr_t b, ptr_t c,
    cptr_t a_buf, cptr_t b_buf, cptr_t c_buf,
    int x, int y, int z,
    int ldx, int ldy, int ldz
) {
    for (int z_ = z; z_ < z + BZ; ++z_) {
        for (int y_ = y; y_ < y + BY; ++y_) {
            int buf_off = INDEX(1 + BT, 1 + BT + y_, 1 + BT + z_, BUF_DIM_X, BUF_DIM_Y);
            int src_off = INDEX(x, y, z, ldx, ldy);
            memcpy(a + src_off, a_buf + buf_off, sizeof(data_t) * BX);
            memcpy(b + src_off, b_buf + buf_off, sizeof(data_t) * BX);
            memcpy(c + src_off, c_buf + buf_off, sizeof(data_t) * BX);
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


    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    int t, k;
    for (t = 0, k = 0; t < nt; t += BT, k++) {

        ptr_t a0 = bufferx[k % 2];
        ptr_t a1 = bufferx[(k + 1) % 2];

        ptr_t b0 = buffery[k % 2];
        ptr_t b1 = buffery[(k + 1) % 2];

        ptr_t c0 = bufferz[k % 2];
        ptr_t c1 = bufferz[(k + 1) % 2];
        
#pragma omp parallel for collapse(2) schedule(static)
        for(int z = z_start; z < z_end; z += BZ) {
            for(int y = y_start; y < y_end; y += BY) {
                for(int x = x_start; x < x_end; x += BX) {
                    int z_off = z - z_start, y_off = y - y_start, x_off = x - x_start;
                    // allocate contiguous buffer
                    data_t a0_buf[BUF_SIZE] = {}, b0_buf[BUF_SIZE] = {}, c0_buf[BUF_SIZE] = {}; // make sure it is zero initialized
                    data_t a1_buf[BUF_SIZE], b1_buf[BUF_SIZE], c1_buf[BUF_SIZE];
                    // pack a0, b0, c0 (and BT level of neighbours) to buffer
                    //copy_to_buffer(a0, b0, c0, a0_buf, b0_buf, c0_buf, x_off, y_off, z_off, x_start, y_start, z_start, x_end, y_end, z_end, ldx, ldy, ldz);
                    // run stencil kernel on current block for BT rounds
                    do_stencil_7_block(a0_buf, a1_buf, b0_buf, b1_buf, c0_buf, c1_buf);
                    // copy back from buffer to a1, b1, c1
                    //copy_from_buffer(a1, b1, c1, a1_buf, b1_buf, c1_buf, x, y, z, ldx, ldy, ldz);
                }
            }
        }
    }

    for (int t_ = k; t_ < nt - t + k; ++t_) {
        cptr_t a0 = bufferx[t_ % 2];
        ptr_t a1 = bufferx[(t_ + 1) % 2];

        cptr_t b0 = buffery[t_ % 2];
        ptr_t b1 = buffery[(t_ + 1) % 2];

        cptr_t c0 = bufferz[t_ % 2];
        ptr_t c1 = bufferz[(t_ + 1) % 2];
        
        for(int z = z_start; z < z_end; ++z) {
            for(int y = y_start; y < y_end; ++y) {
                for(int x = x_start; x < x_end; ++x) {

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
            }
        }
    }

    return bufferx[nt % 2];
}
