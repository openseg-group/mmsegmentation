_base_ = [
    '../_base_/models/fcn_hr18.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], 
        channels=sum([48, 96, 192, 384])
    )
)
optimizer = dict(lr=0.02)
lr_config = dict(min_lr=2e-4)
data = dict(samples_per_gpu=2, workers_per_gpu=2)
