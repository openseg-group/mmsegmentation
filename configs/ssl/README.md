
## Results and models

### Cityscapes


#### DeepLabv3+ based results (all)
| Method |      Backbone      | Dataset | Two-branch model + Consistency Loss | Crop Size | Batch Size | Max Iters |  Base LR |  mIoU (single-scale) |
|--------|--------------------|-----------|-----------|--------:|----------|----------------|------:|--------------:|
| DeepLabV3+ | R-101-D16 | 2975 Labeled | No | 512x1024 |  16 Labeled  |   40000 |    0.02   |   79.63  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | No | 512x1024 |  16 Labeled (need to recheck)  |   80000 |    0.01    |   80.07  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | Yes (loss-weight=10) | 512x1024 |  8 Labeled  |   80000 |    0.01    |   -  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | Yes (loss-weight=50) | 512x1024 |  8 Labeled  |   80000 |    0.01    |   -  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | Yes (loss-weight=10) | 512x1024 |  16 Labeled  |   40000 |    0.02   |   79.98  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | Yes (loss-weight=20) | 512x1024 |  16 Labeled  |   40000 |    0.02   |   79.52  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | Yes (loss-weight=50) | 512x1024 |  16 Labeled  |   40000 |    0.02   |   80.03  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | Yes (loss-weight=10) | 512x1024 |  16 Labeled  |    80000 |    0.02  |   79.92  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | Yes (loss-weight=50) | 512x1024 |  16 Labeled  |    80000 |    0.02  |   79.97  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | Yes (loss-weight=100) | 512x1024 |  16 Labeled  |   80000 |    0.02  |   79.95  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=0)  | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.01   |  79.46   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=10) | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.01   |  80.07   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=20) | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.01   |  80.38   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=30) | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.01   |  80.30   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=40) | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.01   |  80.20   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=50) | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.01   |  80.22   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=100)| 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.01   |  79.50   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=0)  | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.02   |  78.83   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=10) | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.02   |  79.52   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=50) | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.02   |  79.40   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=100)| 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.02   |  78.42   |



#### HRNet+OCR based results
| Method |      Backbone      | Dataset | Two-branch model + Consistency Loss | Crop Size | Batch Size | Max Iters |  Base LR |  mIoU (single-scale) |
|--------|--------------------|-----------|-----------|--------:|----------|----------------|------:|--------------:|
| OCRNet  | HRNetV2p-W48 | 2975 Labeled | No | 512x1024 |  8 Labeled  |   80000 |    0.01  |   81.21   |
| OCRNet  | HRNetV2p-W48 | 2975 Labeled | Yes (loss-weight=10) | 512x1024 |  8 Labeled  |   80000 |    0.01  |   81.62   |
| OCRNet  | HRNetV2p-W48 | 2975 Labeled | Yes (loss-weight=50) | 512x1024 |  8 Labeled  |   80000 |    0.01  |   81.31   |
| OCRNet  | HRNetV2p-W48 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=10) | 512x1024 |  8 Labeled + 8 Unlabeled  |  80000 |    0.01  |   81.27   |
| OCRNet  | HRNetV2p-W48 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=50) | 512x1024 |  8 Labeled + 8 Unlabeled  |  80000 |    0.01  |   81.36   |
| OCRNet  | HRNetV2p-W48 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=20) + Sharpening (T=2) | 512x1024 |  8 Labeled + 8 Unlabeled  |  80000 |    0.01  |   81.48   |
| OCRNet  | HRNetV2p-W48 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=20) + Sharpening (T=2) + Aux-Head | 512x1024 |  8 Labeled + 8 Unlabeled  |  80000 |    0.01  |   81.57   |
| OCRNet  | HRNetV2p-W48 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=20) + Sharpening (T=3) + Aux-Head | 512x1024 |  8 Labeled + 8 Unlabeled  |  80000 |    0.01  |   81.78   |
