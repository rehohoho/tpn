#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=$(pwd)

CONFIG_FILE=config_files/sthv2/tsm_tpn_hdf5.py
CHECKPOINT_FILE=pretrained/sthv2_tpn.pth
RESULT_FILE=pretrained/sthv2_tpn.pkl
NUM_GPU=1

./tools/dist_test_recognizer.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPU} \
    --out ${RESULT_FILE} \
    --ignore_cache \
    --fcn_testing \
2>&1 | tee ${RESULT_FILE}.log
