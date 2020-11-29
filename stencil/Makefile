
CC = ../icc-wrapper icc
CXX = ../icc-wrapper icpc
MPICC = ../icc-wrapper mpiicc
MPICXX = ../icc-wrapper mpiicpc
OPT = -Ofast -ffast-math -fast -xHost -ip
FLAGS_COMMON = -Wall -g -qopenmp -no-multibyte-chars
CFLAGS = -std=c99 $(FLAGS_COMMON) $(OPT)
CXXFLAGS = -std=c++17 $(FLAGS_COMMON) $(OPT)
LDFLAGS = -Wall -qopenmp -g
LDLIBS = $(LDFLAGS) -lnuma

targets = benchmark-naive benchmark-omp benchmark-mpi
objects = check.o benchmark.o stencil-naive.o stencil-omp.o stencil-mpi.o

.PHONY : default
default : all

.PHONY : all
all : $(targets)

benchmark-% : check.o benchmark.o stencil-%.o
	$(MPICXX) -o $@ $^ $(LDFLAGS) $(LDLIBS)

check.o : check.c common.h
	$(MPICC) -c $(CFLAGS) $< -o $@
benchmark.o : benchmark.c common.h
	$(MPICC) -c $(CFLAGS) $< -o $@
stencil-naive.o : stencil-naive.c common.h
	$(MPICC) -c $(CFLAGS) $< -o $@
%.o : %.cc common.h stencil-common.hh
	$(MPICXX) -c $(CXXFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)
