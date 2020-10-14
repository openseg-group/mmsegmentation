#!/usr/bin/env bash

PYTHON="/data/anaconda/envs/pytorch1.6.0/bin/python"
# PYTHON="/opt/conda/bin/python"

$PYTHON -m pip install mmcv-full==1.1.5
$PYTHON -m pip install -e .
$PYTHON -m pip install cityscapesscripts

CONFIG=$1
GPUS=$2
# PRETRAINED_WEIGHTS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --seed 0 --work-dir "$(dirname $0)/../../mmsegmentation-logs" \
    ${@:3}

# ./tools/local_train.sh configs/ocrnet/deeplabv3plus_r101-d8_512x1024_80k_b2_cityscapes.py 4