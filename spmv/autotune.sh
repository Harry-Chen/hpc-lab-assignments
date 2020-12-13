#!/bin/bash

GRID_SIZE="8 16 32 64 128 256 512 1024 2048 4096 8192"
BLOCK_SIZE="128 256 512 1024"

mkdir -p result/autotune

for g in $GRID_SIZE; do
    for b in $BLOCK_SIZE; do
        echo Grid size $g, block size $b | tee -a result/autotune/result.txt
        export CXXFLAGS="-DGRID_SIZE=$g -DBLOCK_SIZE=$b"
        make clean && make -j benchmark-optimized
        ./run_all.sh ./benchmark-optimized |& tee result/autotune/g${g}_b${b}.txt
        python average.py result/autotune/g${g}_b${b}.txt >> result/autotune/result.txt
    done
done
