#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=$(pwd)

CONFIG_FILE=config_files/sthv2/tsm_tpn.py
CHECKPOINT_FILE=pretrained/sthv2_tpn.pth
RESULT_FILE=pretrained/sthv2_tpn.pkl
EVAL_METRICS=top_k_accuracy
NUM_GPU=1

./tools/dist_test_recognizer.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} \
    --out ${RESULT_FILE} \
    --eval ${EVAL_METRICS} \
    --ignore_cache \
2>&1 | tee ${RESULT_FILE}.log
# --fcn_testing