import torch
import torch.nn.functional as F

from ..builder import HEADS
from ..utils import GeneralizedAttentionBlock as _GeneralizedAttentionBlock
from .fcn_head import FCNHead

@HEADS.register_module()
class GAHead(FCNHead):
    """Generalized Attention Head.

    Args:
        attention_type: specify the settings of the usages of the appearance/geometric features
        num_heads: specify the number of heads
    """
    def __init__(self, num_heads, attention_type, **kwargs):
        super(GAHead, self).__init__(**kwargs)
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.ga_block = _GeneralizedAttentionBlock(
            in_channels=self.channels,
            spatial_range=-1,
            num_heads=self.num_heads,
            position_embedding_dim=-1,
            position_magnitude=1,
            kv_stride=1,
            q_stride=1,
            attention_type=self.attention_type
        )

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.ga_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output