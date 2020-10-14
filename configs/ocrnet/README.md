# Object-Contextual Representations for Semantic Segmentation

## Introduction

```latex
@article{YuanW18,
  title={Ocnet: Object context network for scene parsing},
  author={Yuhui Yuan and Jingdong Wang},
  booktitle={arXiv preprint arXiv:1809.00916},
  year={2018}
}

@article{YuanCW20,
  title={Object-Contextual Representations for Semantic Segmentation},
  author={Yuhui Yuan and Xilin Chen and Jingdong Wang},
  booktitle={ECCV},
  year={2020}
}
```

## Results and models

### Cityscapes

#### HRNet backbone
| Method |      Backbone      | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                               download                                                                                                                                                                                               |
|--------|--------------------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OCRNet | HRNetV2p-W18-Small | 512x1024  |   40000 |      3.5 |          10.45 | 74.30 |         75.95 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x1024_40k_cityscapes/ocrnet_hr18s_512x1024_40k_cityscapes_20200601_033304-fa2436c2.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x1024_40k_cityscapes/ocrnet_hr18s_512x1024_40k_cityscapes_20200601_033304.log.json)     |
| OCRNet | HRNetV2p-W18       | 512x1024  |   40000 |      4.7 |           7.50 | 77.72 |         79.49 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_40k_cityscapes/ocrnet_hr18_512x1024_40k_cityscapes_20200601_033320-401c5bdd.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_40k_cityscapes/ocrnet_hr18_512x1024_40k_cityscapes_20200601_033320.log.json)         |
| OCRNet | HRNetV2p-W48       | 512x1024  |   40000 |        8 |           4.22 | 80.58 |         81.79 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_40k_cityscapes/ocrnet_hr48_512x1024_40k_cityscapes_20200601_033336-55b32491.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_40k_cityscapes/ocrnet_hr48_512x1024_40k_cityscapes_20200601_033336.log.json)         |
| OCRNet | HRNetV2p-W18-Small | 512x1024  |   80000 | -        | -              | 77.16 |         78.66 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x1024_80k_cityscapes/ocrnet_hr18s_512x1024_80k_cityscapes_20200601_222735-55979e63.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x1024_80k_cityscapes/ocrnet_hr18s_512x1024_80k_cityscapes_20200601_222735.log.json)     |
| OCRNet | HRNetV2p-W18       | 512x1024  |   80000 | -        | -              | 78.57 |         80.46 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_80k_cityscapes/ocrnet_hr18_512x1024_80k_cityscapes_20200614_230521-c2e1dd4a.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_80k_cityscapes/ocrnet_hr18_512x1024_80k_cityscapes_20200614_230521.log.json)         |
| OCRNet | HRNetV2p-W48       | 512x1024  |   80000 | -        | -              | 80.70 |         81.87 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_80k_cityscapes/ocrnet_hr48_512x1024_80k_cityscapes_20200601_222752-9076bcdf.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_80k_cityscapes/ocrnet_hr48_512x1024_80k_cityscapes_20200601_222752.log.json)         |
| OCRNet (w/ dropout) | HRNetV2p-W48       | 512x1024  |   80000 | -        | -              | 81.15 |         - | |
| OCRNet | HRNetV2p-W18-Small | 512x1024  |  160000 | -        | -              | 78.45 |         79.97 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x1024_160k_cityscapes/ocrnet_hr18s_512x1024_160k_cityscapes_20200602_191005-f4a7af28.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x1024_160k_cityscapes/ocrnet_hr18s_512x1024_160k_cityscapes_20200602_191005.log.json) |
| OCRNet | HRNetV2p-W18       | 512x1024  |  160000 | -        | -              | 79.47 |         80.91 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes/ocrnet_hr18_512x1024_160k_cityscapes_20200602_191001-b9172d0c.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes/ocrnet_hr18_512x1024_160k_cityscapes_20200602_191001.log.json)     |
| OCRNet | HRNetV2p-W48       | 512x1024  |  160000 | -        | -              | 81.35 |         82.70 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037.log.json)     |
| OCRNet (w/ dropout) | HRNetV2p-W48       | 512x1024  |   160000 | -        | -              | 81.15 |         - | |
<!-- | OCRNet (w/ dropout) | HRNetV2p-W48  |  512x1024  | 40000 | -        | -     |  80.72  |        - | | -->

#### ResNet backbone

| Method |      Backbone      | Crop Size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip)|
|--------|--------------------|-----------|-----------|--------:|----------|----------------|------:|--------------:|
| DeepLabV3 | R-101-D8 | 769x769   |   8   |   40000 |     10.9 |           0.83 | 79.27 |         80.11 | 
| DeepLabV3 | R-101-D8 | 512x1024  |   8   |   80000 | -        | -              | 80.20 |         81.21 |
| DeepLabV3 | R-101-D8 | 512x1024   |   16   |   40000 |     10.9 |           0.83 | 79.13 |        - | 
| DeepLabv3  | R-101-D8 | 512x1024  | 16 |   60000 |   -  |    -  |   80.10,79.96  |        - |  
| DeepLabv3 (w/ SepDepthWiseConv)  | R-101-D8 | 512x1024  | 16 |   40000 |   -  |    -  |   80.48  |
| DeepLabV3+ | R-101-D8 | 512x1024  |  8   |   40000 |     11 |        2.60 | 80.21 |         81.82 | 
| DeepLabV3+ | R-101-D8 | 512x1024  |  16   |   40000 |     11 |        2.60 | -  |        - |
| DeepLabV3+ | R-101-D8 | 512x1024  |  8   | 80000 | -        | -              | 80.97 |         82.03 |
| DeepLabV3+ | R-101-D8 | 769x769   |  8   | 80000 | -        | -              | 80.98 |         82.18 |
| DeepLabv3+  | R-101-D8 | 512x1024  | 16 |   40000 |   -  |    -  |   80.31  |        - |   
| DeepLabv3+  | R-101-D8 | 512x1024  | 16 |   60000 |   -  |    -  |   80.91,80.60  |        - |   
| DeepLabv3+  | R-101-D16 | 512x1024  | 16 |   80000 |   -  |    -  |   80.38,80.82  |        - |   
| OCRNet  | R-101-D8 | 769x769 | 8 | 40000 |   -  |    -  |   79.15  | 
| OCRNet  | R-50-D8 | 512x1024  | 8 |   40000 |   -  |    -  |   78.65  |
| OCRNet  | R-101-D8 | 512x1024  | 8 |   40000 |   -  |    -  |   79.88  |
| OCRNet  | R-101-D8 | 512x1024  | 8 |   60000 |   -  |    -  |   80.27  |
| OCRNet  | R-101-D8 | 512x1024  | 8  |   80000 |   -  |    -  |   79.66  |   
| OCRNet  | R-101-D8 | 512x1024  | 16 |   40000 |   -  |    -  |   80.29,80.65  |
| OCRNet  | R-101-D8 | 512x1024  | 16 |   60000 |   -  |    -  |   -  | 
| OCRNet  | R-101-D8 | 512x1024  | 16 |   80000 |   -  |    -  |   80.63,80.41,80.71  | 
| OCRNetPlus  | R-101-D16 | 512x1024  | 16 |   80000 |   -  |    -  |   80.39,80.30  |        - | 
| OCRNet (replace the 1x1 Conv (fusion) with 3x3 SepDepthWiseConv) | R-101-D8 | 512x1024  | 8 |   40000 |   -  |    -  |   80.01 | 
| OCRNetPlus | R-101-D8 | 512x1024  | 8 |   40000 |   -  |    -  |   79.90  | 
| OCRNetPlus (Decoder w/ SepDepthWiseConv)   | R-101-D8 | 512x1024  | 8 |   40000 |   -  |    -  |   80.33  | -  |
| OCRNetPlus (Decoder w/ SepDepthWiseConv)   | R-101-D8 | 512x1024  | 8 |   60000 |   -  |    -  |   80.17  | -  |
| OCRNetPlus (Decoder w/ SepDepthWiseConv)   | R-101-D8 | 512x1024  | 8 |   80000 |   -  |    -  |   80.77  |
| OCRNetPlus (Decoder w/ SepDepthWiseConv)   | R-101-D8 | 512x1024  | 16 |   40000 |   -  |   -  |  80.44, 80.73  |  -  |
| OCRNetPlus (Decoder w/ SepDepthWiseConv)   | R-101-D8 | 512x1024  | 16 |   60000 |   -  |   -  |  80.77  |  -  |
| OCRNetPlus (OCR & Decoder w/ SepDepthWiseConv)   | R-101-D8 | 512x1024  | 8 |   40000 |   -  |  -  |  79.99  |   -  |  
| OCRNetPlus (OCR & Decoder w/ SepDepthWiseConv)   | R-101-D8 | 512x1024  | 8 |   60000 |   -  |  -  |  80.15  |   -  |  
| OCRNetPlus (OCR & Decoder w/ SepDepthWiseConv)   | R-101-D8 | 512x1024  | 16 |   40000 |   -  |  -  |  80.91,80.36,80.08  |   -  |
| OCRNetPlus (OCR & Decoder w/ SepDepthWiseConv)   | R-101-D8 | 512x1024  | 8 |   60000 |   -  |  -  |  80.15  |   -  |     
| OCRNetPlus (OCR & Decoder w/ SepDepthWiseConv)   | R-101-D8 | 512x1024  | 16 |   60000 |   -  |  -  |  80.82,81.03,80.82  |   -  |
| FCN | HRNetV2p-W48   | 512x1024  |  16  | 80000 | -        | -     | 81.40 | -  |
| FCN+Decoder | HRNetV2p-W48   | 512x1024  |  16  | 80000 | -        | -     | 79.69 | -  |
| OCRNet+   | R-101-D8 | 512x1024  | 16 |   40000 |   10.3  |  3  |  80.91,80.36,80.08  |   -  |    
| OCRNet+   | R-101-D8 | 512x1024  | 16 |   80000 |   10.3  |  3  |  80.50  |   -  | 

| Method |      Backbone      | Crop Size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip)|
|--------|--------------------|-----------|-----------|--------:|----------|----------------|------:|--------------:|
| OCRNet  | R-101-D8 | 512x1024  | 8 |   40000 |  -   |   -  |   79.88,80.09  |  -  | 
| OCRNet  | R-101-D8 | 512x1024  | 16 |   40000 |  8.8   |   3.02  |   80.29,80.30,80.65  |  -  | 
| OCRNet + RMI | R-101-D8 | 512x1024  | 16 |   40000 |  8.8   |   3.02  |   81.28  |  -  | 
| OCRNet + RMI | HRNetV2p-W18 | 512x1024  | 16 |   40000 |  -   |   -  |   82.43  |  -  | 
| OCRNet + RMI + Mapillary | HRNetV2p-W18 | 512x1024  | 16 |   40000 |  -   |   -  |   82.91  |  -  | 
| OCRNet + RMI + Mapillary + Small-LR | HRNetV2p-W18 | 512x1024  | 16 |   40000 |  -   |   -  |   -  |  -  | 
| OCRNet  | R-101-D8 | 512x1024  | 16 |   80000 |  8.8   |   3.02  |   80.40,80.54,80.81  |  -  | 
| DeepLabv3  | R-101-D8 | 512x1024  | 16 |  40000 |  9.6  |   2    |  79.69 | -  | 
| DeepLabv3  | R-101-D8 | 512x1024  | 16 |  80000 |  9.6  |   2    |  80.43 | -  | 
| DeepLabv3+  | R-101-D8 | 512x1024  | 16 |  40000 |  11  |   2.64    |  80.13 | -  | 
| DeepLabv3+  | R-101-D8 | 512x1024  | 16 |  80000 |  11  |   2.64    |  80.86 | -  | 

### ADE20K

#### HRNet backbone

| Method |      Backbone      | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                     download                                                                                                                                                                                     |
|--------|--------------------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OCRNet | HRNetV2p-W18-Small | 512x512   |   80000 |      6.7 |          28.98 | 35.06 |         35.80 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x512_80k_ade20k/ocrnet_hr18s_512x512_80k_ade20k_20200615_055600-e80b62af.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x512_80k_ade20k/ocrnet_hr18s_512x512_80k_ade20k_20200615_055600.log.json)     |
| OCRNet | HRNetV2p-W18       | 512x512   |   80000 |      7.9 |          18.93 | 37.79 |         39.16 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x512_80k_ade20k/ocrnet_hr18_512x512_80k_ade20k_20200615_053157-d173d83b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x512_80k_ade20k/ocrnet_hr18_512x512_80k_ade20k_20200615_053157.log.json)         |
| OCRNet | HRNetV2p-W48       | 512x512   |   80000 |     11.2 |          16.99 | 43.00 |         44.30 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x512_80k_ade20k/ocrnet_hr48_512x512_80k_ade20k_20200615_021518-d168c2d1.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x512_80k_ade20k/ocrnet_hr48_512x512_80k_ade20k_20200615_021518.log.json)         |
| OCRNet | HRNetV2p-W18-Small | 512x512   |  160000 | -        | -              | 37.19 |         38.40 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x512_160k_ade20k/ocrnet_hr18s_512x512_160k_ade20k_20200615_184505-8e913058.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x512_160k_ade20k/ocrnet_hr18s_512x512_160k_ade20k_20200615_184505.log.json) |
| OCRNet | HRNetV2p-W18       | 512x512   |  160000 | -        | -              | 39.32 |         40.80 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x512_160k_ade20k/ocrnet_hr18_512x512_160k_ade20k_20200615_200940-d8fcd9d1.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x512_160k_ade20k/ocrnet_hr18_512x512_160k_ade20k_20200615_200940.log.json)     |
| OCRNet | HRNetV2p-W48       | 512x512   |  160000 | -        | -              | 43.25 |         44.88 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x512_160k_ade20k/ocrnet_hr48_512x512_160k_ade20k_20200615_184705-a073726d.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x512_160k_ade20k/ocrnet_hr48_512x512_160k_ade20k_20200615_184705.log.json)     |


#### ResNet backbone

| Method |      Backbone      | Crop Size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip)|
|--------|--------------------|-----------|-----------|--------:|----------|----------------|------:|--------------:|
| DeepLabV3+ | R-101-D8 | 512x512   |  16 | 160000        | -     | -         | 45.47 |         46.35 |
| DeepLabV3 | R-101-D8 | 512x512   |  16 | 160000      | -    | -          | 45.00 |         46.66 |
| OCRNet  | R-50-D8  | 512x512   |  16  |    160000  |    -  | -  |   41.31  | - |
| OCRNet  | R-101-D8 | 512x512   |  16  |    160000  |    -  | -  |   43.64  | - |
| OCRNet  | R-101-D8  | 512x512   |  16  |    160000  |    -  | -  |   43.64  | - |
| OCRNet  | R-101-D8  | 512x512   |  16  |    160000  |    -  | -  |   44.38  | - |
| OCRNet  | R-101-D8  | 512x512   |  32  |    80000  |    -  | -  |   44.02  | - |
| OCRNetPlus (Decoder w/ SepDepthWiseConv)  | R-101-D8  | 512x512   |  16  |    160000 |    -  | -  |   44.33  |  -  |
| OCRNetPlus (OCR & Decoder w/ SepDepthWiseConv)   | R-101-D8  | 512x512   |  16  |    160000 |    -  | -  |  44.10,44.73  |  -  |

### Pascal VOC 2012 + Aug

| Method |      Backbone      | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                       download                                                                                                                                                                                       |
|--------|--------------------|-----------|--------:|----------|----------------|------:|--------------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OCRNet | HRNetV2p-W18-Small | 512x512   |   20000 |      3.5 |          31.55 | 71.70 |         73.84 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x512_20k_voc12aug/ocrnet_hr18s_512x512_20k_voc12aug_20200617_233913-02b04fcb.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x512_20k_voc12aug/ocrnet_hr18s_512x512_20k_voc12aug_20200617_233913.log.json) |
| OCRNet | HRNetV2p-W18       | 512x512   |   20000 |      4.7 |          19.91 | 74.75 |         77.11 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x512_20k_voc12aug/ocrnet_hr18_512x512_20k_voc12aug_20200617_233932-8954cbb7.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x512_20k_voc12aug/ocrnet_hr18_512x512_20k_voc12aug_20200617_233932.log.json)     |
| OCRNet | HRNetV2p-W48       | 512x512   |   20000 |      8.1 |          17.83 | 77.72 |         79.87 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x512_20k_voc12aug/ocrnet_hr48_512x512_20k_voc12aug_20200617_233932-9e82080a.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x512_20k_voc12aug/ocrnet_hr48_512x512_20k_voc12aug_20200617_233932.log.json)     |
| OCRNet | HRNetV2p-W18-Small | 512x512   |   40000 | -        | -              | 72.76 |         74.60 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x512_40k_voc12aug/ocrnet_hr18s_512x512_40k_voc12aug_20200614_002025-42b587ac.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x512_40k_voc12aug/ocrnet_hr18s_512x512_40k_voc12aug_20200614_002025.log.json) |
| OCRNet | HRNetV2p-W18       | 512x512   |   40000 | -        | -              | 74.98 |         77.40 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x512_40k_voc12aug/ocrnet_hr18_512x512_40k_voc12aug_20200614_015958-714302be.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x512_40k_voc12aug/ocrnet_hr18_512x512_40k_voc12aug_20200614_015958.log.json)     |
| OCRNet | HRNetV2p-W48       | 512x512   |   40000 | -        | -              | 77.14 |         79.71 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x512_40k_voc12aug/ocrnet_hr48_512x512_40k_voc12aug_20200614_015958-255bc5ce.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x512_40k_voc12aug/ocrnet_hr48_512x512_40k_voc12aug_20200614_015958.log.json)     |
