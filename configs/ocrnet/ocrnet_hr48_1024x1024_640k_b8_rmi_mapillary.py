_base_ = [
    '../_base_/models/ocrnet_hr18.py',
    '../_base_/datasets/mapillary_1024x1024.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_640k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=66,
            norm_cfg=norm_cfg,
            align_corners=True,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=0.1,
            num_classes=66,
            norm_cfg=norm_cfg,
            align_corners=True,
            loss_decode=dict(
                type='RMILoss', num_classes=66, loss_weight=1.0)),
    ]
)
optimizer = dict(lr=0.01)
lr_config = dict(min_lr=1e-4)
data = dict(samples_per_gpu=1, workers_per_gpu=1)
# fp16 settings
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)