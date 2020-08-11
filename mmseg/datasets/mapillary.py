from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MapillaryDataset(CustomDataset):
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

    CLASSES = (
        "bird", "ground-animal", "curb", "fence", "guard-rail", "other-barrier", "wall", "bike-lane",
        "crosswalk-plain", "curb-cut", "parking", "pedestrian-area", "rail-track", "road", "service-lane",
        "sidewalk", "bridge", "building", "tunnel", "person", "bicyclist", "motorcyclist", "other-rider",
        "crosswalk-zebra", "general", "mountain", "sand", "sky", "snow", "terrain", "vegetation", "water",
        "banner", "bench", "bike-rack", "billboard", "catch-basin", "cctv-camera", "fire-hydrant", "junction-box",
        "mailbox", "manhole", "phone-booth", "pothole", "street-light", "pole", "traffic-sign-frame", "utility-pole",
        "traffic-light", "back", "front", "trash-can", "bicycle", "boat", "bus", "car",
        "caravan", "motorcycle", "on-rails", "other-vehicle", "trailer", "truck", "wheeled-slow", "car-mount",
        "ego-vehicle", "unlabeled")

    PALETTE = [ [165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
                [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255],
                [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
                [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
                [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
                [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
                [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
                [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
                [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
                [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
                [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
                [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
                [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
                [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142],
                [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64],
                [0, 0, 110], [0, 0, 70], [0, 0, 192], [32, 32, 32],
                [120, 10, 10], [0, 0, 0] ]

    def __init__(self, **kwargs):
        super(MapillaryDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)


if __name__ == '__main__':
    CLASSES = []
    PALETTE = []

    print(len(CLASSES))

    # import json
    # import pdb
    # with open('./map_config.json') as json_file:
    #     data = json.load(json_file)
    #     labels = data['labels']
    #     for label in labels:
    #         CLASSES.append(label['name'].split("--")[-1])
    #         PALETTE.append(label['color'])
    
    # num = 0
    # for _ in CLASSES:
    #     print(f'"{_}", ', end="")
    #     # print(",", end =" ")
    #     num += 1
    #     if num % 8 == 0:
    #         print("")

    # print("\n")
    # print("\n")

    # num = 0
    # for _ in PALETTE:
    #     print(_, end =", ")
    #     num += 1
    #     if num % 4 == 0:
    #         print("")
