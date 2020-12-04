CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

CUDAOPT = -O3
CUDAFLAGS = $(CUDAOPT) -g -std=c++14

CUDALIB = $(CUDA_HOME)/lib64
LDFLAGS = 
LDLIBS = $(LDFLAGS) -L$(CUDALIB) -L$(CUDALIB)/stubs -lcuda -lcudart

targets =  benchmark-cusparse benchmark-optimized
commonobj = utils.o benchmark.o
objects = $(commonobj) sptrsv-cusparse.o sptrsv-optimized.o

.PHONY : default
default : all

.PHONY : all
all : $(targets)

benchmark.o : benchmark.cu common.h utils.h
	$(NVCC) -c $(CUDAFLAGS) $< -o $@
utils.o : utils.cu common.h
	$(NVCC) -c $(CUDAFLAGS) $< -o $@

sptrsv-cusparse.o : sptrsv-cusparse.cu common.h
	$(NVCC) -c $(CUDAFLAGS) $< -o $@
benchmark-cusparse : $(commonobj) sptrsv-cusparse.o
	$(NVCC) -o $@ $^ $(LDLIBS) -lcusparse

sptrsv-optimized.o : sptrsv-optimized.cu common.h
	$(NVCC) -c $(CUDAFLAGS) $< -o $@
benchmark-optimized : $(commonobj) sptrsv-optimized.o
	$(NVCC) -o $@ $^ $(LDLIBS)

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)
