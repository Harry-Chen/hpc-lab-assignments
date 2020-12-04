# !/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <executable> " >&2
    exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/course/HW/sptrsv_data

EXECUTABLE=$1
REP=64

FILELIST=`ls -Sr ${DATAPATH} | grep "\.csr"`

for FILE in ${FILELIST}; do
    FILEPATH=${DATAPATH}/${FILE}
    if test -f ${FILEPATH}; then
        srun -p gpu ${EXECUTABLE} ${REP} ${FILEPATH}
    fi
done
