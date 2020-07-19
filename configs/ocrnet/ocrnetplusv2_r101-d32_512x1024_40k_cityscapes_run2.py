_base_ = [
    '../_base_/models/ocrnet_r50-d32.py', 
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(depth=101),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=1024,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            drop_out_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRPlusHeadV2',
            in_channels=[256, 512, 1024, 2048],
            in_index=(0, 1, 2, 3),
            input_transform='multiple_select',
            feature_key=3,
            low_level_key=(2, 1, 0),
            low_level_channels=(1024, 512, 256),
            low_level_channels_project=(128, 64, 32),
            decoder_channels=256,
            channels=512,
            ocr_channels=256,
            drop_out_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])
