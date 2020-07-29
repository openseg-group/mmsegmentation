_base_ = '../ocrnet/ocrnet_r101-d8_512x1024_40k_b16_cityscapes.py'
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
