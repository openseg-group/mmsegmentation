#!/usr/bin/env bash

PYTHON="/data/anaconda/envs/pytorch1.5.1/bin/python"

$PYTHON -m pip install mmcv

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --seed 0 --work_dir "$(dirname $0)/../../mmsegmentation-logs"

# ${@:3}
# ./tools/local_train.sh configs/ocrnet/fcn_hr48_512x1024_40k_b16_rmi_cityscapes.py 4
# ./tools/local_train.sh configs/ocrnet/ocrnet_hr48_1024x1024_320k_b8_rmi_mapillary.py 4