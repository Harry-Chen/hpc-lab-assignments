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
  case $NODES in
    1)
      NODELIST="cn002"
    ;;
    2)
      NODELIST="cn00[2-3]"
    ;;
    4)
      NODELIST="cn00[2-5]"
    ;;
    *)
      echo "Node number $NODES not supportted"
      exit 1
    ;;
  esac
  if [[ $string == *"mpi" ]]; then
    TASK_PER_NODE=2 # SMP
  else
    TASK_PER_NODE=1 # OMP
  fi
  EXEC_PREFIX="$EXEC_PREFIX --nodelist=$NODELIST --ntasks-per-node=${TASKS_PER_NODE}"
else
  EXEC_PREFIX="$MPIRUN -n $NODES" # just for test purpose
fi

export EXEC="$EXEC_PREFIX $EXE"

# OpenMP core binding
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$THREADS

echo "Running command \"$EXEC\" on $NODES ranks * $THREADS threads"
