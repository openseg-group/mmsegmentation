#!/usr/bin/env bash

PYTHON="/data/anaconda/envs/pytorch1.6.0/bin/python"
$PYTHON -m pip install mmcv-full==1.1.5
$PYTHON -m pip install -e .
$PYTHON -m pip install cityscapesscripts

# ${PYTHON} $(dirname "$0")/benchmark.py $CONFIG "open-mmlab://resnet101_v1c" 
# ./tools/local_test.sh configs/ocrnet/ocrnetplus_r50-d8_512x1024_40k_cityscapes.py 4

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

# ./tools/local_test.sh configs/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes.py 
# checkpoints/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth 8 --out results.pkl --eval mIoU cityscapes
