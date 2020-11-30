#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>

#include "stencil-common.hh"

extern "C" const char *version_name = "Optimized version (OpenMP)";

#ifndef _OPENMP
#error This file must be compiled with OpenMP
#endif

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {

    if (grid_info->p_id == 0) {
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

    // check constraints
    REQUIRE(grid_info->p_num == 1, "OpenMP version must run with single process.");
    assert(grid_info->local_size_x % BX == 0);
    assert(grid_info->local_size_y % BY == 0);
    assert(grid_info->local_size_z % BZ == 0);

    load_stencil_coefficients();
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {}


ptr_t inline __attribute__((always_inline)) stencil_trivial(
    int x_start, int x_end, int y_start, int y_end, int z_start, int z_end, 
    int nt, int ldx, int ldy, int ldz,
    ptr_t bufferx[2], ptr_t buffery[2], ptr_t bufferz[2]
) {
    
    // return array
    ptr_t ret = bufferx[0];
    // fused rounds (after each round, a1 will store BT rounds of stencil on a0, and a0 will be garbage)
    int t_fused = nt / BT;

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
                    int z_begin = max(z_off - BT, 0), z_stop = min(z + BZ + BT, z_end) - z_start, buf_z_start = BT - (z_off - z_begin), buf_z_end = BUF_DIM_Z - (z_off + BZ + BT - z_stop);
                    int y_begin = max(y_off - BT, 0), y_stop = min(y + BY + BT, y_end) - y_start, buf_y_start = BT - (y_off - y_begin), buf_y_end = BUF_DIM_Y - (y_off + BY + BT - y_stop);
                    int x_begin = max(x_off - BT, 0), x_stop = min(x + BX + BT, x_end) - x_start, buf_x_start = BT - (x_off - x_begin), buf_x_end = BUF_DIM_X - (x_off + BX + BT - x_stop);
                    // allocate contiguous buffer
                    data_t a_buf_0[BUF_SIZE] = {}, b_buf_0[BUF_SIZE] = {}, c_buf_0[BUF_SIZE] = {};
                    data_t a_buf_1[BUF_SIZE] = {}, b_buf_1[BUF_SIZE] = {}, c_buf_1[BUF_SIZE] = {};
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
                    // run stencil kernel on block buffer for BT rounds
                    // in each round, the dimension shrinks by 1
                    // a_buf_0 -> a_buf_1 -> a_buf_0 -> ...
                    ptr_t A0 = a_buf_0, A1 = a_buf_1, B0 = b_buf_0, B1 = b_buf_1, C0 = c_buf_0, C1 = c_buf_1;
#pragma unroll(BT)
                    for (int t = BT - 1; t >= 0; --t) {
                        int xx_start = max(BT - t, buf_x_start), yy_start = max(BT - t, buf_y_start), zz_start = max(BT - t, buf_z_start);
                        int xx_end = min(BT + BX + t, buf_x_end), yy_end = min(BT + BY + t, buf_y_end), zz_end = min(BT + BZ + t, buf_z_end);
                        for (int zz = zz_start; zz < zz_end; ++zz) {
                            for (int yy = yy_start; yy < yy_end; ++yy) {
                                stencil_inner_loop(A0, A1, B0, B1, C0, C1, xx_start, xx_end, yy, zz, BUF_DIM_X, BUF_DIM_Y, BUF_DIM_Z);
                            }
                        }
                        // swap buffers for next round
                        std::swap(A0, A1);
                        std::swap(B0, B1);
                        std::swap(C0, C1);
                    }
                    // copy back from buffer to a1, b1, c1
                    copy_from_buffer(a1, b1, c1, A0, B0, C0, x, y, z, x_start, y_start, z_start, ldx, ldy, ldz);
                    ret = a1;
                }
            }
        }
    }
    // deal with remaining steps
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


// benchmark entrypoint
ptr_t stencil_7(ptr_t A0, ptr_t A1, ptr_t B0, ptr_t B1, ptr_t C0, ptr_t C1, const dist_grid_info_t *grid_info, int nt) {

    ptr_t bufferx[2] = {A0, A1};
    ptr_t buffery[2] = {B0, B1};
    ptr_t bufferz[2] = {C0, C1};

    // calculate local size
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    if (grid_info->global_size_x < TRIVIAL_METHOD_THRESHOLD_OMP) {
        return stencil_time_skew(x_start, x_end, y_start, y_end, z_start, z_end, nt, ldx, ldy, ldz, bufferx, buffery, bufferz);
    } else {
        return stencil_trivial(x_start, x_end, y_start, y_end, z_start, z_end, nt, ldx, ldy, ldz, bufferx, buffery, bufferz);
    }
}
