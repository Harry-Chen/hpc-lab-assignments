include ../gpu.makefrag

targets =  benchmark-cusparse benchmark-naive benchmark-optimized
commonobj = utils.o benchmark.o
objects = $(commonobj) spmv-cusparse.o spmv-naive.o spmv-optimized.o
headers = $(wildcard *.h)

.PHONY : default
default : all

.PHONY : all
all : $(targets)

benchmark.o : benchmark.cu $(headers)
	$(NVCC) -c $(CUDAFLAGS) $< -o $@
utils.o : utils.cu common.h
	$(NVCC) -c $(CUDAFLAGS) $< -o $@

spmv-cusparse.o : spmv-cusparse.cu $(headers)
	$(NVCC) -c $(CUDAFLAGS) $< -o $@
benchmark-cusparse : $(commonobj) spmv-cusparse.o
	$(NVCC) -o $@ $^ $(LDLIBS) -lcusparse

spmv-naive.o : spmv-naive.cu $(headers)
	$(NVCC) -c $(CUDAFLAGS) $< -o $@
benchmark-naive : $(commonobj) spmv-naive.o
	$(NVCC) -o $@ $^ $(LDLIBS)

spmv-optimized.o : spmv-optimized.cu $(headers)
	$(NVCC) -c $(CUDAFLAGS) $< -o $@
benchmark-optimized : $(commonobj) spmv-optimized.o
	$(NVCC) -o $@ $^ $(LDLIBS)

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)
