# !/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <executable>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/course/HW/spmv_data

EXECUTABLE=$1
REP=64

srun -p gpu ${EXECUTABLE} ${REP} ${DATAPATH}/parabolic_fem.csr
