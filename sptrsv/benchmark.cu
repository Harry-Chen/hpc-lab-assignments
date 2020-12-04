#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <cuda.h>
#include <sys/time.h>
#include "common.h"
#include "utils.h"

extern const char* version_name;

int parse_args(int* reps, int p_id, int argc, char **argv);
int my_abort(int line, int code);

#define MY_ABORT(ret) my_abort(__LINE__, ret)
#define ABORT_IF_ERROR(ret) CHECK_ERROR(ret, MY_ABORT(ret))
#define ABORT_IF_NULL(ret) CHECK_NULL(ret, MY_ABORT(NO_MEM))
#define INDENT "    "
#define TIME_DIFF(start, stop) 1.0 * (stop.tv_sec - start.tv_sec) + 1e-6 * (stop.tv_usec - start.tv_usec)

int main(int argc, char **argv) {
    int reps, i, ret;
    dist_matrix_t mat;
    double compute_time = 0, pre_time;
    data_t *x;
    data_t *b;
    data_t *cpu_buffer;
    struct timeval start, stop;
    double gflops, compute_time_per_run;

    ret = parse_args(&reps, 0, argc, argv);
    ABORT_IF_ERROR(ret)
    ret = read_matrix_default(&mat, argv[2]);
    ABORT_IF_ERROR(ret)

    printf("Benchmarking %s on %s.\n", version_name, argv[2]);
    printf(INDENT"%d x %d, %d non-zeros, %d run(s)\n", \
            mat.global_m, mat.global_m, mat.global_nnz, reps);
    printf(INDENT"Preprocessing.\n");
    
    gettimeofday(&start, NULL);
    preprocess(&mat);
    gettimeofday(&stop, NULL);
    pre_time = TIME_DIFF(start, stop);

    cpu_buffer = (data_t*) malloc(sizeof(data_t) * 2 * mat.global_m);
    ABORT_IF_NULL(cpu_buffer)
    ret = read_vector(&mat, argv[2], "_b.vec", 1, cpu_buffer, cpu_buffer + mat.global_m);
    ABORT_IF_ERROR(ret)
    ret = cudaMalloc(&x, sizeof(data_t) * mat.global_m);
    ABORT_IF_ERROR(ret)
    ret = cudaMalloc(&b, sizeof(data_t) * mat.global_m);
    ABORT_IF_ERROR(ret)

    cudaMemcpy(b, cpu_buffer, sizeof(data_t) * mat.global_m, cudaMemcpyHostToDevice);

    /* warm up */
    printf(INDENT"Warming up.\n");
    sptrsv(&mat, b, x);
    

    printf(INDENT"Testing.\n");
    for(i = 0; i < reps; ++i) {
        cudaMemset(x, 0, sizeof(double) * mat.global_m);

        gettimeofday(&start, NULL);
        sptrsv(&mat, b, x);
        cudaDeviceSynchronize();
        gettimeofday(&stop, NULL);
        compute_time += TIME_DIFF(start, stop);
    }
    cudaMemcpy(cpu_buffer, x, sizeof(data_t) * mat.global_m, cudaMemcpyDeviceToHost);

    printf(INDENT"Checking.\n");
    ret = check_answer(&mat, argv[2], cpu_buffer);
    if(ret == 0) {
        printf("\e[1;32m"INDENT"Result validated.\e[0m\n");
    } else {
        fprintf(stderr, "\e[1;31m"INDENT"Result NOT validated.\e[0m\n");
        MY_ABORT(ret);
    }
    destroy_dist_matrix(&mat);
    free(cpu_buffer);
    cudaFree(b);
    cudaFree(x);

    gflops = 2e-9 * mat.global_nnz * reps / compute_time;
    compute_time_per_run = compute_time / reps;
    printf(INDENT INDENT"preprocess time = %lf s, compute time = %lf s per run\n", \
                    pre_time, compute_time_per_run);
    printf("\e[1;34m"INDENT INDENT"Performance: %lf Gflop/s\e[0m\n", gflops);

    return 0;
}

void print_help(const char *argv0, int p_id) {
    if(p_id == 0) {
        printf("\e[1;31mUSAGE: %s <repetitions> <input-file>\e[0m\n", argv0);
    }
}

int parse_args(int* reps, int p_id, int argc, char **argv) {
    int r;
    if(argc < 3) {
        print_help(argv[0], p_id);
        return 1;
    }
    r = atoi(argv[1]);
    if(r <= 0) {
        print_help(argv[0], p_id);
        return 1;
    }
    *reps = r;
    return SUCCESS;
}

int my_abort(int line, int code) {
    fprintf(stderr, "\e[1;33merror at line %d, error code = %d\e[0m\n", line, code);
    return fatal_error(code);
}

int fatal_error(int code) {
    exit(code);
    return code;
}
