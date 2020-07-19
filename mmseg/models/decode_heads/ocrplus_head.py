import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from functools import partial

from mmseg.ops import DepthwiseSeparableConvModule, resize
from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .cascade_decode_head import BaseCascadeDecodeHead
from .ocr_head import SpatialGatherModule, ObjectAttentionBlock


@HEADS.register_module()
class OCRPlusHead(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is augment the original `OCRNet
    <https://arxiv.org/abs/1909.11065>` with a decoder head.

    We make 3 modifications based on the OCRHead:
    -1- apply a decoder head to combine the 2x-resolution feature maps
        from Res-2 stage following the DeepLabv3+
    -2- replace the 3x3 conv -> separable 3x3 conv that is used decrease
        the channel from 2048->512 (self.bottleneck)
    # -3- replace the ObjectAttentionBlock ->
    #     DepthwiseSeparableObjectAttentionBlock

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        c1_in_channels (int): The input channels of c1 decoder.
                              If is 0, the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
        scale (int): The scale of probability map in SpatialGatherModule.
    """

    def __init__(self,
                 ocr_channels,
                 c1_in_channels,
                 c1_channels,
                 scale=1,
                 use_sep_conv=False,
                 **kwargs):
        super(OCRPlusHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0

        self.ocr_channels = ocr_channels
        self.scale = scale
        self.use_sep_conv = use_sep_conv

        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            use_sep_conv=self.use_sep_conv)

        self.spatial_gather_module = SpatialGatherModule(self.scale)
        
        self.bottleneck = DepthwiseSeparableConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.fuse_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None

    def forward(self, inputs, prev_output):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        output = self.object_context_block(feats, context)

        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)

        output = self.fuse_bottleneck(output)
        output = self.cls_seg(output)

        return output


@HEADS.register_module()
class OCRPlusHeadV2(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is augment the original `OCRNet
    <https://arxiv.org/abs/1909.11065>` with multiple decoder heads
    following the panotpic deeplab.

    Panoptic-DeepLab: A Simple, Strong, and Fast Baseline 
    for Bottom-Up Panoptic Segmentation, CVPR2020

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        feature_key (int): specify to apply the context module on 
            which feature maps.  
        low_level_key (int): specify a group of low-level features maps 
            as inputs to the decoder.
        low_level_channels_project (int): specify the output channels of 
            the transformed low-level features.
        decoder_channels: the output channels after each decoder.
    """
    def __init__(self, 
                 ocr_channels, 
                 feature_key, 
                 low_level_channels, 
                 low_level_key, 
                 low_level_channels_project,
                 decoder_channels,
                 scale=1,
                 use_sep_conv=False,
                 **kargs):
        super(OCRPlusHeadV2, self).__init__(**kargs)

        self.ocr_channels = ocr_channels
        self.scale=scale
        self.use_sep_conv = use_sep_conv

        self.feature_key = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        self.low_level_key = low_level_key
    
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            use_sep_conv=self.use_sep_conv)

        self.spatial_gather_module = SpatialGatherModule(self.scale)

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
                fuse_in_channels =decoder_channels + low_level_channels_project[i]
            fuse.append(
                fuse_bottleneck(
                    fuse_in_channels,
                    decoder_channels,
                )
            )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)

        self.conv_seg = nn.Conv2d(decoder_channels, self.num_classes, kernel_size=1)

    def forward(self, inputs, prev_output):
        """Forward function."""
        feats = self.bottleneck(inputs[self.feature_key])
        cur_prob = resize(
                input=prev_output,
                size=feats.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        print("probability map shape within ocr modeule {}".format(cur_prob.shape))

        context = self.spatial_gather_module(feats, cur_prob)
        output = self.object_context_block(feats, context)

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

            # apply the same coarse map in a coarse to fine manner.
            # cur_prob = resize(
            #         input=prev_output,
            #         size=output.shape[2:],
            #         mode='bilinear',
            #         align_corners=self.align_corners)
            # context = self.spatial_gather_module(output, cur_prob)
            # output = self.object_context_block(output, context)

        output = self.cls_seg(output)
        return output
