#!/bin/bash

source ./stencil_common.sh $*

DATAPATH=${BASEDIR}/ylm_ans

echo Running 256x256x256
$EXEC 7 256 256 256 16 ${DATAPATH}/stencil_data_256x256x256 ${DATAPATH}/stencil_answer_7_256x256x256_16steps

echo Running 512x512x512
$EXEC 7 512 512 512 16 ${DATAPATH}/stencil_data_512x512x512 ${DATAPATH}/stencil_answer_7_512x512x512_16steps
