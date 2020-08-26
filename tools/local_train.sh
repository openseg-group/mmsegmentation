#!/usr/bin/env bash

PYTHON="/data/anaconda/envs/pytorch1.6.0/bin/python"
# PYTHON="/opt/conda/bin/python"

$PYTHON -m pip install mmcv==1.0.5
$PYTHON -m pip install -e .

CONFIG=$1
GPUS=$2
# PRETRAINED_WEIGHTS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --seed 0 --work_dir "$(dirname $0)/../../mmsegmentation-logs" \
    ${@:3}
    # --load-from $3

# ${@:3}
# ./tools/local_train.sh configs/isanet/isanet_r101-d8_512x512_160k_ade20k.py 4
# ./tools/local_train.sh configs/ocrnet/ocrnet_hr48_1024x1024_320k_b8_rmi_mapillary.py 4
# ./tools/local_train.sh configs/ocrnet/ocrnet_hr48_1024x1024_320k_b8_rmi_mapillary.py 4
# ./tools/local_train.sh configs/ocrnet/ocrnet_hr48_512x1024_40k_b16_rmi_cityscapes_pretrain_mapillary.py 4 --load-from=pretrained_models/hrnet_ocr_mapillary_524.pth

# ./tools/local_train.sh configs/ganet/ganet_r50_d8_512x1024_40k_cityscapes.py 4