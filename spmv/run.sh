#!/bin/bash

source ./spmv_common.sh $*

DATA=$2

if [ -z $DATA ]; then
    DATA=parabolic_fem.csr
fi

${EXEC_PREFIX} ${EXECUTABLE} ${REP} ${DATAPATH}/${DATA}
