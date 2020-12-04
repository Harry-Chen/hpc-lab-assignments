# !/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <executable>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/course/HW/sptrsv_data

EXECUTABLE=$1
REP=64

srun -p gpu ${EXECUTABLE} ${REP} ${DATAPATH}/2cubes_sphere.nd_chol.csr
