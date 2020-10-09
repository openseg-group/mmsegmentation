from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CelebAMaskHQDataset(CustomDataset):
    """Mapillary dataset.

    In segmentation map annotation for Mapillary, which contains 65 categories.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.

    Currently, we include the "unlabeled" category, which is ignored in our previous
    experimental settings.
    Notably, the recent Panoptic-DeepLab trains the models with 65 categories while
    the recent Multi-Scale Attention trains the model with 66 categories. We use 66
    categories for simplicity.
    """

    CLASSES = ('skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 
               'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth')

    PALETTE = [[204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204],
               [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
               [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    def __init__(self, **kwargs):
        super(CelebAMaskHQDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)


if __name__ == '__main__':
    CLASSES = []
    PALETTE = []

    print(len(CLASSES))
