#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <executable>" >&2
    exit 1
fi

export DAPL_DBG_TYPE=0

if [ -d "/mnt/ssd/harry/sptrsv_data" ]; then
  DATAPATH="/mnt/ssd/harry/sptrsv_data"
else
  DATAPATH="/home/course/HW/sptrsv_data"
fi

SRUN=$(which srun)

if [ -x "$SRUN" ]; then
    EXEC_PREFIX="$SRUN -N 1 --partition=gpu --exclusive --pty"
fi

export EXEC_PREFIX
export DATAPATH
export EXECUTABLE=$1
export REP=64
