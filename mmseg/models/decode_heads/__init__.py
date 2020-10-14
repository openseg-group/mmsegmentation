from .ann_head import ANNHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .dnl_head import DNLHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .fpn_decode_head import FpnDecodeHead
from .gc_head import GCHead
from .isa_head import ISAHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .ocrplus_head import OCRPlusHead, OCRPlusHeadV2
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .uper_head import UPerHead
from .df_head import DFHead
from .ga_head import GAHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'PointHead', 'OCRPlusHead', 'OCRPlusHeadV2', 'FpnDecodeHead', 'ISAHead',
    'DFHead', 'GAHead'
]