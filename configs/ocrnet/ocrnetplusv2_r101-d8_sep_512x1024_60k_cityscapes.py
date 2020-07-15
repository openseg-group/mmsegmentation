_base_ = [
    '../_base_/models/ocrnetplusv2_r50-d8_sep.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_60k.py'
]
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
