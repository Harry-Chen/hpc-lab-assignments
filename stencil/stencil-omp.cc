#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>

#include "stencil-common.hh"

extern "C" const char *version_name = "Optimized version (OpenMP)";

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

#define TX 16
#define TY 16
#define TZ 16

ptr_t stencil_7(ptr_t A0, ptr_t A1, ptr_t B0, ptr_t B1, ptr_t C0, ptr_t C1,
                const dist_grid_info_t *grid_info, int nt) {
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

    using std::max;
    using std::min;

    for (int y = y_start; y < y_end; y += TY) {

        // slope in time skewing
        int neg_y_slope = y == 1 ? 0 : 1;
        int pos_y_slope = y == y_end - TY ? 0 : -1;

        // do a full stencil
        for (int t = 0; t < nt; t++) {

            // blocking on y dimension
            int y_begin = max(y_start, y - t * neg_y_slope);
            int y_stop = max(y_start, y + TY + t * pos_y_slope);

            cptr_t a0 = bufferx[t % 2];
            ptr_t a1 = bufferx[(t + 1) % 2];

            cptr_t b0 = buffery[t % 2];
            ptr_t b1 = buffery[(t + 1) % 2];

            cptr_t c0 = bufferz[t % 2];
            ptr_t c1 = bufferz[(t + 1) % 2];

#pragma omp parallel for collapse(1) schedule(static)
            for (int zz = z_start; zz < z_end; zz++) {
                for (int yy = y_begin; yy < y_stop; yy++) {
                    stencil_inner_loop<true>(a0, a1, b0, b1, c0, c1, x_start, x_end, yy, zz, ldx, ldy, ldz);
                }
            }
        }
    }

    return bufferx[nt % 2];
}
