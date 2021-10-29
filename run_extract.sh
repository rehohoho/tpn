#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

export PYTHONPATH=$(pwd)

CONFIG_FILE=config_files/sthv2/tsm_tpn_hdf5.py
CHECKPOINT_FILE=pretrained/sthv2_tpn.pth
RESULT_FILE=pretrained/sthv2_tpn_backbone_features_val
NUM_GPU=1

./tools/dist_extract_features.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPU} \
    --out ${RESULT_FILE} \
    --fcn_testing \
2>&1 | tee ${RESULT_FILE}.log
