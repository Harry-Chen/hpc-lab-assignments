#!/bin/bash

export DAPL_DBG_TYPE=0
export NODES=$2
export EXE=$1

if [ -d "/mnt/ssd/harry/stencil_data" ]; then
  BASEDIR="/mnt/ssd/harry/stencil_data"
else
  BASEDIR="/home/course/HW/stencil_data"
fi

export DATAPATH=${BASEDIR}/ylm_ans/

MPIRUN=$(which mpirun)
SRUN=$(which srun)

if [ -x "$MPIRUN" ]; then
  EXEC_PREFIX="$MPIRUN -n $NODES"
else
  EXEC_PREFIX="$SRUN -N $NODES --nodelist=cn00[2-5] --exclusive --pty"
fi

export EXEC="$EXEC_PREFIX $EXE"

echo Running command $EXEC on $NODES nodes
