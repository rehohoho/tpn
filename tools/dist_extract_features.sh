#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29502}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/extract_features.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
