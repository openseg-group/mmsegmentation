_base_ = [
    '../_base_/models/deeplabv3plus_r50-d16.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
evaluation = dict(interval=200, metric='mIoU')
