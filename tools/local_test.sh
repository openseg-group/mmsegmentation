#!/usr/bin/env bash

PYTHON="/data/anaconda/envs/pytorch1.5.1/bin/python"
$PYTHON -m pip install -e .

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
WORK_DIR="./logs/${CONFIG}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 

${PYTHON} $(dirname "$0")/benchmark.py $CONFIG "open-mmlab://resnet101_v1c" 

# ${@:3}
# ./tools/local_test.sh configs/ocrnet/ocrnetplus_r50-d8_512x1024_40k_cityscapes.py 4
