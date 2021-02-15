#!/usr/bin/env bash

PYTHON="/data/anaconda/envs/pytorch1.6.0/bin/python"
$PYTHON -m pip install mmcv-full==1.1.5
$PYTHON -m pip install -e .
$PYTHON -m pip install cityscapesscripts

CONFIG=$1 # fcn_hr18_512x1024_40k_cityscapes_baseline
GPUS=$2
# PRETRAINED_WEIGHTS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --seed 0 --work-dir "$(dirname $0)/../../mmsegmentation-logs" \
    ${@:3}

# ./tools/local_train.sh configs/ssl/mt_ocrnet_hr48_512x1024_80k_b16_cityscapes_u3k_w20_soft_sharpen_aux_t2.py 4
# ./tools/local_train.sh configs/ssl/mt_ocrnet_hr48_512x1024_80k_b16_cityscapes_u3k_w20_soft_sharpen_aux_t2.py 4