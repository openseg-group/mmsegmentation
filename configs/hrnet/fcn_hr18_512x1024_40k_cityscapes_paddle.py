_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(pretrained='pretrained_models/HRNet_W18_C_ssld_pretrained.pth')
optimizer = dict(lr=0.02)
lr_config = dict(min_lr=1e-4)
data = dict(samples_per_gpu=2, workers_per_gpu=2)