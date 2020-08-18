#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install jq -y

all_containers=`cat $PHILLY_RUNTIME_CONFIG | jq ".containers[].id"`
for container in $all_containers
do
        #echo $container
        container=${container//\"/}
        #echo "cat $PHILLY_RUNTIME_CONFIG | jq --arg name $container -r '.containers[\"container_e125_1515549374026_2107_01_000002\"].index'"
        index=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].index'`
        if [ $index -eq 0 ]
        then
                master_ip=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].ip'`
                master_port_start=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].portRangeStart'`
                master_port_end=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].portRangeEnd'`
                DIFF=$((master_port_end-master_port_start+1))
        fi
done

this_container_index=$PHILLY_CONTAINER_INDEX

export NODE_RANK=$this_container_index
export MASTER_IP=$master_ip
# export MASTER_PORT=$(($(($RANDOM%$DIFF))+master_port_start))
export MASTER_PORT=$((master_port_start+1))
echo '*************'
echo $MASTER_IP
echo $MASTER_PORT
echo $NODE_RANK
echo '*************'

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
NNODES=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

PYTHON="/opt/conda/bin/python"

$PYTHON -m pip install mmcv
$PYTHON -m pip install -e .

${PYTHON} -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_IP \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/train.py $CONFIG \
    --launcher pytorch \
    --seed 0 --work_dir "$(dirname $0)/../../mmsegmentation-logs"
