#!/bin/bash

source ./sptrsv_common.sh $*

${EXEC_PREFIX} ${EXECUTABLE} ${REP} ${DATAPATH}/2cubes_sphere.nd_chol.csr
