CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

CUDAOPT = -O3
CUDAFLAGS = $(CUDAOPT) -g -std=c++14

CUDALIB = $(CUDA_HOME)/lib64
LDFLAGS = 
LDLIBS = $(LDFLAGS) -L$(CUDALIB) -L$(CUDALIB)/stubs -lcuda -lcudart

targets =  benchmark-cusparse benchmark-naive benchmark-optimized
commonobj = utils.o benchmark.o
objects = $(commonobj) spmv-cusparse.o spmv-naive.o spmv-optimized.o

.PHONY : default
default : all

.PHONY : all
all : $(targets)

benchmark.o : benchmark.cu common.h utils.h
	$(NVCC) -c $(CUDAFLAGS) $< -o $@
utils.o : utils.cu common.h
	$(NVCC) -c $(CUDAFLAGS) $< -o $@

spmv-cusparse.o : spmv-cusparse.cu common.h
	$(NVCC) -c $(CUDAFLAGS) $< -o $@
benchmark-cusparse : $(commonobj) spmv-cusparse.o
	$(NVCC) -o $@ $^ $(LDLIBS) -lcusparse

spmv-naive.o : spmv-naive.cu common.h
	$(NVCC) -c $(CUDAFLAGS) $< -o $@
benchmark-naive : $(commonobj) spmv-naive.o
	$(NVCC) -o $@ $^ $(LDLIBS)

spmv-optimized.o : spmv-optimized.cu common.h
	$(NVCC) -c $(CUDAFLAGS) $< -o $@
benchmark-optimized : $(commonobj) spmv-optimized.o
	$(NVCC) -o $@ $^ $(LDLIBS)

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)
