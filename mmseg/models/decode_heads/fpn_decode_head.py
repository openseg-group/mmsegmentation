import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from functools import partial

from mmseg.ops import resize
from mmcv.cnn import DepthwiseSeparableConvModule
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FpnDecodeHead(BaseDecodeHead):
    """FPN decoder head following the Panoptic-DeepLab
    """
    def __init__(self,
                 feature_key, 
                 low_level_channels, 
                 low_level_key, 
                 low_level_channels_project,
                 decoder_channels,
                 **kargs):
        super(FpnDecodeHead, self).__init__(**kargs)
 
        self.feature_key = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        self.low_level_key = low_level_key

        self.bottleneck = DepthwiseSeparableConvModule(
            self.in_channels[feature_key],
            self.channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        fuse_bottleneck = partial(DepthwiseSeparableConvModule,
                                  kernel_size=5,
                                  padding=2,
                                  norm_cfg=self.norm_cfg,
                                  act_cfg=self.act_cfg)

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                nn.Sequential(
                    nn.Conv2d(low_level_channels[i], 
                              low_level_channels_project[i], 
                              1, bias=False),
                    nn.BatchNorm2d(low_level_channels_project[i]),
                    nn.ReLU(inplace=True)
                )
            )
            if i == 0:
                fuse_in_channels = self.channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(
                fuse_bottleneck(
                    fuse_in_channels,
                    decoder_channels,
                )
            )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)

        self.conv_seg = nn.Conv2d(decoder_channels, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        output = self.bottleneck(inputs[self.feature_key])
        # build decoder
        for i in range(self.decoder_stage):
            low_level_key = self.low_level_key[i]
            low_level_feats = self.project[i](inputs[low_level_key])
            output = resize(output,
                       size=low_level_feats.shape[2:],
                       mode='bilinear',
                       align_corners=self.align_corners)
            output = torch.cat([output, low_level_feats], dim=1)
            output = self.fuse[i](output)

        output = self.cls_seg(output)
        return output
