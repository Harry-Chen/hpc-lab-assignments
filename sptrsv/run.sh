#!/bin/bash

source ./sptrsv_common.sh $*

DATA=$2

if [ -z $DATA ]; then
    DATA=2cubes_sphere.nd_chol.csr
fi

${EXEC_PREFIX} ${EXECUTABLE} ${REP} ${DATAPATH}/${DATA}
