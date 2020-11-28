# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = ./icc-wrapper icc
CXX = ./icc-wrapper icpc
OPT_COMMON = -Ofast -ffast-math
OPT = $(OPT_COMMON) -fast -xHost -ip 
OPT_GCC = $(OPT_COMMON) -march=native -mtune=native
FLAGS_COMMON = -Wall -DGETTIMEOFDAY -g
CFLAGS = -std=c99 $(FLAGS_COMMON) $(OPT)
CXXFLAGS = -std=c++17 $(FLAGS_COMMON) $(OPT)
CXXFLAGS_GCC = -std=c++17 $(FLAGS_COMMON) $(OPT_GCC)
LDFLAGS = -Wall
# mkl is needed for blas implementation
LDLIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

targets = benchmark-test benchmark-naive benchmark-blocked benchmark-blas benchmark-strassen benchmark-perf benchmark-openmp
objects = benchmark-test.o benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o dgemm-strassen.o benchmark-strassen.o benchmark-perf.o  benchmark-openmp.o simple-test.o

.PHONY : all
all : $(targets)

# to test correctness via address sanitizer
simple-test: simple-test.cc dgemm-blocked.cc
	g++ -o $@ $^ $(CXXFLAGS_GCC) -fsanitize=address $(LDLIBS)

benchmark-test: benchmark-test.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)
benchmark-perf: benchmark-perf.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)

benchmark-naive: benchmark.o dgemm-naive.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)
benchmark-blocked: benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)
benchmark-openmp: benchmark.o dgemm-openmp.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(LDLIBS) -qopenmp
benchmark-blas: benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)


# for Strassen
benchmark-strassen : benchmark-strassen.o dgemm-strassen.o
	$(CC) -o $@ $^ $(LDLIBS)
dgemm-blocked.o: dgemm-blocked.cc
	$(CXX) -c $(CXXFLAGS) -o $@ $<
dgemm-strassen.o: dgemm-blocked.cc dgemm-blocked-strassen-kernel.hh
	$(CXX) -c $(CXXFLAGS) -DENABLE_STRASSEN=1 -o $@ $<
dgemm-openmp.o: dgemm-openmp.cc dgemm-openmp-threadnum.hh
	$(CXX) -c $(CXXFLAGS) -qopenmp -o $@ $<
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
