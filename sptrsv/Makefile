CUDAC = nvcc

CUDAOPT =

CUDAFLAGS = $(CUDAOPT)

CUDAPATH = /usr/local/cuda/lib64
LDFLAGS = 
LDLIBS = $(LDFLAGS) -L$(CUDAPATH) -L$(CUDAPATH)/stubs -lcuda -lcudart

targets =  benchmark-cusparse benchmark-optimized
commonobj = utils.o benchmark.o
objects = $(commonobj) sptrsv-cusparse.o sptrsv-optimized.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark.o : benchmark.cu common.h utils.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
utils.o : utils.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@

sptrsv-cusparse.o : sptrsv-cusparse.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
benchmark-cusparse : $(commonobj) sptrsv-cusparse.o
	$(CUDAC) -o $@ $^ $(LDLIBS) -lcusparse

sptrsv-optimized.o : sptrsv-optimized.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
benchmark-optimized : $(commonobj) sptrsv-optimized.o
	$(CUDAC) -o $@ $^ $(LDLIBS)

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)
