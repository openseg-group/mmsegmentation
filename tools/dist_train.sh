#!/usr/bin/env bash
nvidia-smi

PYTHON="/opt/conda/bin/python"

$PYTHON -m pip install mmcv==1.0.5
$PYTHON -m pip install -e .

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --seed 0 --work_dir "$(dirname $0)/../../mmsegmentation-logs" \
    ${@:3}