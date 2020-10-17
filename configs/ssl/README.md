
## Results and models

### Cityscapes


#### DeepLabv3+ based results
| Method |      Backbone      | Dataset | Two-branch model + Consistency Loss | Crop Size | Batch Size | Max Iters |  Base LR |  mIoU (single-scale) |
|--------|--------------------|-----------|-----------|--------:|----------|----------------|------:|--------------:|
| DeepLabV3+ | R-101-D16 | 2975 Labeled | No | 512x1024 |  8 Labeled  |   80000 |    0.02   |     -    |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | No | 512x1024 |  16 Labeled  |   40000 |    0.02   |   79.63  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | Yes (loss-weight=10) | 512x1024 |  16 Labeled  |    40000 |    0.02  |   79.92  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | Yes (loss-weight=50) | 512x1024 |  16 Labeled  |    40000 |    0.02  |   79.97  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled | Yes (loss-weight=100) | 512x1024 |  16 Labeled  |   40000 |    0.02  |   79.95  |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=0)  | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.01   |  79.46   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=10) | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.01   |     -    |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=50) | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.01   |     -    |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=100)| 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.01   |  79.50   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=0)  | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.02   |  78.83   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=10) | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.02   |  79.52   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=50) | 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.02   |  79.40   |
| DeepLabV3+ | R-101-D16 | 2975 Labeled + 3000 Unlabeled | Yes (loss-weight=100)| 512x1024  |  8 Labeled + 8 Unlabeled | 80000 |   0.02   |  78.42   |


#### HRNet+OCR based results
| Method |      Backbone      | Dataset | Two-branch model + Consistency Loss | Crop Size | Batch Size | Max Iters |  Base LR |  mIoU (single-scale) |
|--------|--------------------|-----------|-----------|--------:|----------|----------------|------:|--------------:|
| OCRNet  | HRNetV2p-W48 | 2975 Labeled | No | 512x1024 |  16 Labeled  |   40000 |    0.02  |   -   |