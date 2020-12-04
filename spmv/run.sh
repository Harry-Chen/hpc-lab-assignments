#!/bin/bash

source ./spmv_common.sh $*

${EXEC_PREFIX} ${EXECUTABLE} ${REP} ${DATAPATH}/parabolic_fem.csr
