import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import pdb

@SEGMENTORS.register_module()
class ParallelEncoderDecoder(BaseSegmentor):
    """Parallel (Two Branches) Encoder Decoder segmentors.

    ParallelEncoderDecoder typically consists of parallelized backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    We constrain the predictions of the two parallelized encoder-decoder to be close with MSE loss.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(ParallelEncoderDecoder, self).__init__()
        self.backbone_l = builder.build_backbone(backbone)
        self.backbone_r = builder.build_backbone(backbone)
        if neck is not None:
            self.neck_l = builder.build_neck(neck)
            self.neck_r = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        # assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head_l = builder.build_head(decode_head)
        self.decode_head_r = builder.build_head(decode_head)
        self.align_corners = self.decode_head_l.align_corners
        self.num_classes = self.decode_head_l.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head_l = nn.ModuleList()
                self.auxiliary_head_r = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head_l.append(builder.build_head(head_cfg))
                    self.auxiliary_head_r.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head_l = builder.build_head(auxiliary_head)
                self.auxiliary_head_r = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(ParallelEncoderDecoder, self).init_weights(pretrained)
        self.backbone_l.init_weights(pretrained=pretrained)
        self.backbone_r.init_weights(pretrained=pretrained)
        self.decode_head_l.init_weights()
        self.decode_head_r.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head_l, nn.ModuleList):
                for aux_head in self.auxiliary_head_l:
                    aux_head.init_weights()
            else:
                self.auxiliary_head_l.init_weights()

            if isinstance(self.auxiliary_head_r, nn.ModuleList):
                for aux_head in self.auxiliary_head_r:
                    aux_head.init_weights()
            else:
                self.auxiliary_head_r.init_weights()

    def extract_feat_l(self, img):
        """Extract features from images."""
        x = self.backbone_l(img)
        if self.with_neck:
            x = self.neck_l(x)
        return x

    def extract_feat_r(self, img):
        """Extract features from images."""
        x = self.backbone_r(img)
        if self.with_neck:
            x = self.neck_r(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat_l(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train_l(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head_l.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode_l'))
        return losses

    def _decode_head_forward_train_r(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head_r.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode_r'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head_l.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train_l(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head_l, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head_l):
                loss_aux_l = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux_l, f'aux_l_{idx}'))
        else:
            loss_aux_l = self.auxiliary_head_l.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux_l, 'aux_l'))
        
        return losses

    def _auxiliary_head_forward_train_r(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head_r, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head_r):
                loss_aux_r = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux_r, f'aux_r_{idx}'))
        else:
            loss_aux_r = self.auxiliary_head_r.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux_r, 'aux_r'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x_l = self.extract_feat_l(img)
        x_r = self.extract_feat_r(img)

        losses = dict()

        loss_decode_l = self._decode_head_forward_train_l(x_l, img_metas,
                                                      gt_semantic_seg)
        loss_decode_r = self._decode_head_forward_train_r(x_r, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode_l)
        losses.update(loss_decode_r)

        if self.with_auxiliary_head:
            loss_aux_l = self._auxiliary_head_forward_train_l(
                x_l, img_metas, gt_semantic_seg)
            loss_aux_r = self._auxiliary_head_forward_train_r(
                x_r, img_metas, gt_semantic_seg)
            losses.update(loss_aux_l)
            losses.update(loss_aux_r)

        #TODO add the mse loss between the parallized predictions...
        pred_l = F.softmax(self.decode_head_l.forward(x_l), 1)
        pred_r = F.softmax(self.decode_head_r.forward(x_r), 1)
        losses['consistency_loss'] = F.mse_loss(pred_l, pred_r.detach()) + F.mse_loss(pred_r, pred_l.detach())

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                pad_img = crop_img.new_zeros(
                    (crop_img.size(0), crop_img.size(1), h_crop, w_crop))
                pad_img[:, :, :y2 - y1, :x2 - x1] = crop_img
                pad_seg_logit = self.encode_decode(pad_img, img_meta)
                preds[:, :, y1:y2,
                      x1:x2] += pad_seg_logit[:, :, :y2 - y1, :x2 - x1]
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        flip_direction = img_meta[0]['flip_direction']
        if flip:
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


@SEGMENTORS.register_module()
class CascadeParallelEncoderDecoder(ParallelEncoderDecoder):
    """Cascade Parallel Encoder Decoder segmentors.

    CascadeParallelEncoderDecoder almost the same as ParallelEncoderDecoder, 
    while decoders of CascadeEncoderDecoder are cascaded. The output of previous
    decoder_head will be the input of next decoder_head.
    """
    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        self.num_stages = num_stages
        super(CascadeParallelEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head_l = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head_l.append(builder.build_head(decode_head[i]))
        self.decode_head_r = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head_r.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head_l[-1].align_corners
        self.num_classes = self.decode_head_l[-1].num_classes

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone_l.init_weights(pretrained=pretrained)
        self.backbone_r.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            self.decode_head_l[i].init_weights()
            self.decode_head_r[i].init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head_l, nn.ModuleList):
                for aux_head in self.auxiliary_head_l:
                    aux_head.init_weights()
            else:
                self.auxiliary_head_l.init_weights()
            if isinstance(self.auxiliary_head_r, nn.ModuleList):
                for aux_head in self.auxiliary_head_r:
                    aux_head.init_weights()
            else:
                self.auxiliary_head_r.init_weights()

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self.decode_head_l[0].forward_test(x, img_metas, self.test_cfg)
        for i in range(1, self.num_stages):
            out = self.decode_head_l[i].forward_test(x, out, img_metas,
                                                   self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train_l(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode = self.decode_head_l[0].forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_l_0'))

        for i in range(1, self.num_stages):
            # forward test again, maybe unnecessary for most methods.
            prev_outputs = self.decode_head_l[i - 1].forward_test(
                x, img_metas, self.test_cfg)
            loss_decode = self.decode_head_l[i].forward_train(
                x, prev_outputs, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_decode, f'decode_l_{i}'))

        return losses

    def _decode_head_forward_train_r(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode = self.decode_head_r[0].forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_r_0'))

        for i in range(1, self.num_stages):
            # forward test again, maybe unnecessary for most methods.
            prev_outputs = self.decode_head_r[i - 1].forward_test(
                x, img_metas, self.test_cfg)
            loss_decode = self.decode_head_r[i].forward_train(
                x, prev_outputs, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_decode, f'decode_r_{i}'))

        return losses
