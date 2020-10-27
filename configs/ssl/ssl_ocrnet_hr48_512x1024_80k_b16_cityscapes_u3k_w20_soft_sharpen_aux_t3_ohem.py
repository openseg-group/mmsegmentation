_base_ = [
    '../_base_/models/ocrnet_hr18.py',
    '../_base_/datasets/cityscapes_custom.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CascadeParallelEncoderDecoder',
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
            num_classes=19,
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
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=True,
            sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=100000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ]
)

# model training and testing settings
# consistency loss hyper-parameters
train_cfg = dict(consistency_loss_weight=20,
                 consistency_w_hard_label=False,
                 auxiliary_consistency=True,
                 temperature=3) # set the weight for the consistency loss
test_cfg = dict(mode='whole')

optimizer = dict(lr=0.01)
lr_config = dict(min_lr=1e-4)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)

# ensure the ann_dir for coarse set doesn't contain any labels
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_dir=['../../../../dataset/original_cityscapes/leftImg8bit/train',
                 '../../../../dataset/cityscapes/coarse/image'],
        ann_dir=['../../../../dataset/original_cityscapes/gtFine/train',
                 '../../../../dataset/cityscapes/coarse/nolabel'],
        split = ['../../../../dataset/original_cityscapes/train.txt',
                 '../../../../dataset/cityscapes/uniform_coarse3k.txt']))

find_unused_parameters=True