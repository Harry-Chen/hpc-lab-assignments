#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <executable> <number of nodes> <number of threads>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0
export NODES=$2
export THREADS=$3
export EXE=$1

if [ -d "/mnt/ssd/harry/stencil_data" ]; then
  BASEDIR="/mnt/ssd/harry/stencil_data"
else
  BASEDIR="/home/course/HW/stencil_data"
fi

export BASEDIR

MPIRUN=$(which mpirun)
SRUN=$(which srun)

if [ -x "$SRUN" ]; then
  EXEC_PREFIX="$SRUN -N $NODES-$NODES --partition=cpu --exclusive --pty"
else
  EXEC_PREFIX="$MPIRUN -n $NODES"
fi

export EXEC="$EXEC_PREFIX $EXE"

# OpenMP core binding
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$THREADS

echo "Running command \"$EXEC\" on $NODES nodes * $THREADS threads"
