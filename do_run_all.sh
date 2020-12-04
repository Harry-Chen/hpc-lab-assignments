#!/bin/bash

source $*
shift

FILELIST=`ls -Sr ${DATAPATH} | grep "\.csr"`

for FILE in ${FILELIST}; do
    FILEPATH=${DATAPATH}/${FILE}
    if test -f ${FILEPATH}; then
        ${EXECUTABLE} ${REP} ${FILEPATH}
    fi
done
