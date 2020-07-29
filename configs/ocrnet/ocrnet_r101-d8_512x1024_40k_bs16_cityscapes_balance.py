_base_ = [
    '../_base_/models/ocrnet_r50-d8.py', 
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(depth=101),
    decode_head=[
            dict(
                type='FCNHead',
                loss_decode=dict(
                    class_weight=[
                        0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                        1.0865, 1.0955, 1.0865, 1.1529, 1.0507
                    ]
                )
            ),
            dict(
                type='OCRHead',
                loss_decode=dict(
                    class_weight=[
                        0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                        1.0865, 1.0955, 1.0865, 1.1529, 1.0507
                    ]
                )
            )
    ]
)
optimizer = dict(lr=0.02)
lr_config = dict(min_lr=2e-4)
data = dict(samples_per_gpu=2, workers_per_gpu=2)

