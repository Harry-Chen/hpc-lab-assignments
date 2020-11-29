#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --exclusive

source ./stencil_common.sh $*

DATAPATH=${BASEDIR}

$EXEC 7 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256
$EXEC 7 384 384 384 100 ${DATAPATH}/stencil_data_384x384x384
$EXEC 7 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512
$EXEC 7 768 768 768 100 ${DATAPATH}/stencil_data_768x768x768
