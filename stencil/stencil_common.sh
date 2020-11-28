#!/bin/bash

export DAPL_DBG_TYPE=0
export NODES=$2
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

echo Running command $EXEC on $NODES nodes
