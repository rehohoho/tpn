#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2

export PYTHONPATH=$(pwd)

CONFIG_FILE=config_files/sthv2/tsm_tpn_hdf5.py
CHECKPOINT_FILE=pretrained/sthv2_tpn.pth
RESULT_FILE=pretrained/sthv2_tpn.pkl
EVAL_METRICS=top_k_accuracy
NUM_GPU=3

./tools/dist_test_recognizer.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPU} \
    --out ${RESULT_FILE} \
    --ignore_cache \
2>&1 | tee ${RESULT_FILE}.log
# --fcn_testing