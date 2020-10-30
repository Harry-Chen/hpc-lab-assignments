# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = ./icc-wrapper
OPT_COMMON = -Ofast -ffast-math
OPT = $(OPT_COMMON) -fast -xHost -ip -ipo
OPT_GCC = $(OPT_COMMON) -march=native -mtune=native
CFLAGS_COMMON = -Wall -DGETTIMEOFDAY -std=c99 -g
CFLAGS = $(CFLAGS_COMMON) $(OPT)
CFLAGS_GCC = $(CFLAGS_COMMON) $(OPT_GCC)
LDFLAGS = -Wall
# mkl is needed for blas implementation
LDLIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

targets = benchmark-test benchmark-naive benchmark-blocked benchmark-blas benchmark-strassen benchmark-perf
objects = benchmark-test.o benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o dgemm-strassen.o benchmark-strassen.o benchmark-perf.o simple-test.o

.PHONY : all
all : $(targets)

# to test correctness via address sanitizer
simple-test: simple-test.c dgemm-blocked.c
	gcc  -o $@ $^ $(CFLAGS_GCC) -fsanitize=address $(LDLIBS)

benchmark-test: benchmark-test.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)
benchmark-perf: benchmark-perf.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)

benchmark-naive: benchmark.o dgemm-naive.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)
benchmark-blocked: benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)
benchmark-blas: benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)


# for Strassen
benchmark-strassen : benchmark-strassen.o dgemm-strassen.o
	$(CC) -o $@ $^ $(LDLIBS)
dgemm-blocked.o: dgemm-blocked.c dgemm-blocked-block-kernel.c dgemm-blocked-avx-kernel.c
	$(CC) -c $(CFLAGS) -o $@ $<
dgemm-strassen.o: dgemm-blocked.c dgemm-blocked-strassen-kernel.h
	$(CC) -c $(CFLAGS) -DENABLE_STRASSEN=1 -o $@ $<
benchmark-strassen.o: benchmark.c
	$(CC) -c $(CFLAGS) -DENABLE_STRASSEN=1 -o $@ $<
benchmark-test.o: benchmark.c
	$(CC) -c $(CFLAGS) -DBENCHMARK_TEST=1 -o $@ $<
benchmark-perf.o: benchmark.c
	$(CC) -c $(CFLAGS) -DBENCHMARK_PERF=1 -o $@ $<

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) simple-test $(objects)
