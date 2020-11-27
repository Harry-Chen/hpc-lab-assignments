#include "common.h"
#include <math.h>
const char* version_name = "A naive base-line";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    /* Naive implementation uses Process 0 to do all computations */
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

ptr_t stencil_7(ptr_t A0, ptr_t A1,ptr_t B0, ptr_t B1,ptr_t C0, ptr_t C1, const dist_grid_info_t *grid_info, int nt) {
    ptr_t bufferx[2] = {A0, A1};
    ptr_t buffery[2] = {B0, B1};
    ptr_t bufferz[2] = {C0, C1};


    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    for(int t = 0; t < nt; ++t) {

        cptr_t a0 = bufferx[t % 2];
        ptr_t a1 = bufferx[(t + 1) % 2];

        cptr_t b0 = buffery[t % 2];
        ptr_t b1 = buffery[(t + 1) % 2];

        cptr_t c0 = bufferz[t % 2];
        ptr_t c1 = bufferz[(t + 1) % 2];
        
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
