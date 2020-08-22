import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .fcn_head import FCNHead

class ConditionalFilterLayer(nn.Module):
    def __init__(self, ichn, ochn):
        super(ConditionalFilterLayer, self).__init__()
        self.ichn = ichn
        self.ochn = ochn
        self.mask_conv = nn.Conv2d(ichn, ochn, kernel_size=1)
        self.filter_conv = nn.Conv2d(ochn * ichn, ochn * ichn, kernel_size=1,
                                     groups=ochn)

    def forward(self, x):
        feat = x
        aux_pred = torch.sigmoid(self.mask_conv(x))

        # b k h w
        b, k, h, w = aux_pred.size()
        mask = aux_pred.view(b, k, -1)

        feat = feat.view(b, self.ichn, -1)
        feat = feat.permute(0, 2, 1)
        # b, k, ichn
        class_feat = torch.bmm(mask, feat) / (h * w)
        class_feat = class_feat.view(b, k * self.ichn, 1, 1)

        # b, k*ichn, 1, 1
        filters = self.filter_conv(class_feat)
        filters = filters.view(b * k, self.ichn, 1, 1)

        x = x.view(-1, h, w).unsqueeze(0)
        pred = F.conv2d(x, filters, groups=b).view(b, k, h, w)

        return pred, aux_pred


@HEADS.register_module()
class DFHead(FCNHead):
    """Dynamic Filter Head.

    This head is the implementation of `NLNet
    <https://arxiv.org/abs/1711.07971>`_.

    Args:
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): 
    """

    def __init__(self,
                 **kwargs):
        super(DFHead, self).__init__(**kwargs)
        self.dynamic_conv_seg = ConditionalFilterLayer(self.channels,
                                                       self.num_classes)
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.dynamic_conv_seg(feat)
        return output
        
    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
