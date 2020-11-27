#include <stdio.h>
#include <stdlib.h>
#include "common.h"

const char* version_name = "Optimized version";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    fprintf(stderr, "unimplemented\n");
    exit(1);
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

ptr_t stencil_7(ptr_t A0, ptr_t A1,ptr_t B0, ptr_t B1,ptr_t C0, ptr_t C1, const dist_grid_info_t *info, int nt) {
    fprintf(stderr, "unimplemented\n");
    exit(1);
    return A0;
}

