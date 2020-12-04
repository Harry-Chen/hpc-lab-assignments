CUDAC = nvcc

CUDAOPT =

CUDAFLAGS = $(CUDAOPT)

CUDAPATH = /usr/local/cuda/lib64
LDFLAGS = 
LDLIBS = $(LDFLAGS) -L$(CUDAPATH) -L$(CUDAPATH)/stubs -lcuda -lcudart

targets =  benchmark-cusparse benchmark-naive benchmark-optimized
commonobj = utils.o benchmark.o
objects = $(commonobj) spmv-cusparse.o spmv-naive.o spmv-optimized.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark.o : benchmark.cu common.h utils.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
utils.o : utils.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@

spmv-cusparse.o : spmv-cusparse.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
benchmark-cusparse : $(commonobj) spmv-cusparse.o
	$(CUDAC) -o $@ $^ $(LDLIBS) -lcusparse

spmv-naive.o : spmv-naive.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
benchmark-naive : $(commonobj) spmv-naive.o
	$(CUDAC) -o $@ $^ $(LDLIBS)

spmv-optimized.o : spmv-optimized.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
benchmark-optimized : $(commonobj) spmv-optimized.o
	$(CUDAC) -o $@ $^ $(LDLIBS)

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)
