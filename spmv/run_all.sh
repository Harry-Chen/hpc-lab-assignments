#!/bin/bash

source ./spmv_common.sh $*

FILELIST=`ls -Sr ${DATAPATH} | grep "\.csr"`

for FILE in ${FILELIST}; do
    FILEPATH=${DATAPATH}/${FILE}
    if test -f ${FILEPATH}; then
        ${EXEC_PREFIX} ${EXECUTABLE} ${REP} ${FILEPATH}
    fi
done
