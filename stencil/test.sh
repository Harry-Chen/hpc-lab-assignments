# !/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <executable> <number of nodes>" >&2
  exit 1
fi

export  DAPL_DBG_TYPE=0

DATAPATH=/home/course/HW/stencil_data/ylm_ans/

srun -N $2 $1 7 256 256 256 16 ${DATAPATH}/stencil_data_256x256x256 ${DATAPATH}/stencil_answer_7_256x256x256_16steps
srun -N $2 $1 7 512 512 512 16 ${DATAPATH}/stencil_data_512x512x512 ${DATAPATH}/stencil_answer_7_512x512x512_16steps

# srun -N $2 $1 27 256 256 256 16 ${DATAPATH}/stencil_data_256x256x256 ${DATAPATH}/stencil_answer_27_256x256x256_16steps
# srun -N $2 $1 27 512 512 512 16 ${DATAPATH}/stencil_data_512x512x512 ${DATAPATH}/stencil_answer_27_512x512x512_16steps