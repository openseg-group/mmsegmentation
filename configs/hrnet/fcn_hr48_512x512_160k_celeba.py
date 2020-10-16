_base_ = [
    '../_base_/models/fcn_hr18.py',
    '../_base_/datasets/celeba.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            num_classes=19,
))
evaluation = dict(interval=200, metric='mIoU')