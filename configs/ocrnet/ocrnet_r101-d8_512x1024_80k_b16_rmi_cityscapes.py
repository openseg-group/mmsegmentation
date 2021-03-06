_base_ = [
    '../_base_/models/ocrnet_r50-d8.py', 
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(pretrained='open-mmlab://resnet101_v1c', 
            backbone=dict(depth=101),
            decode_head=[
                dict(
                    type='FCNHead',
                    in_channels=1024,
                    in_index=2,
                    channels=256,
                    num_convs=1,
                    concat_input=False,
                    dropout_ratio=0.1,
                    num_classes=19,
                    norm_cfg=norm_cfg,
                    align_corners=False,
                    loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
                dict(
                    type='OCRHead',
                    in_channels=2048,
                    in_index=3,
                    channels=512,
                    ocr_channels=256,
                    dropout_ratio=0.1,
                    num_classes=19,
                    norm_cfg=norm_cfg,
                    align_corners=False,
                    loss_decode=dict(
                        type='RMILoss', num_classes=19, loss_weight=1.0))
            ]
        )
optimizer = dict(lr=0.02)
lr_config = dict(min_lr=2e-4)
data = dict(samples_per_gpu=2, workers_per_gpu=2)
