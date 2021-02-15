import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from mmseg.core import add_prefix
from mmseg.ops import resize, TensorColorJitter
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import pdb

@SEGMENTORS.register_module()
class MeanTeacherEncoderDecoder(BaseSegmentor):
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
        super(MeanTeacherEncoderDecoder, self).__init__()
        self.backbone_student = builder.build_backbone(backbone)
        self.backbone_teacher = builder.build_backbone(backbone)
        if neck is not None:
            self.neck_student = builder.build_neck(neck)
            self.neck_teacher = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        self.cur_iter = 0
        self.ema_decay = 0.999

        #TODO: implement a ColorJitter augmentation that supports tensors
        self.strong_aug = TensorColorJitter(0.5, 0.5, 0.5, 0.25)

        # https://github.com/pytorch/vision/issues/528
        self.mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        self.std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
        self.normalize = transforms.Normalize(self.mean.tolist(), self.std.tolist())
        self.unnormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head_student = builder.build_head(decode_head)
        self.decode_head_teacher = builder.build_head(decode_head)
        self.align_corners = self.decode_head_student.align_corners
        self.num_classes = self.decode_head_student.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head_student = nn.ModuleList()
                self.auxiliary_head_teacher = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head_student.append(builder.build_head(head_cfg))
                    self.auxiliary_head_r.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head_student = builder.build_head(auxiliary_head)
                self.auxiliary_head_teacher = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(MeanTeacherEncoderDecoder, self).init_weights(pretrained)
        self.backbone_student.init_weights(pretrained=pretrained)
        self.backbone_teacher.init_weights(pretrained=pretrained)
        self.decode_head_student.init_weights()
        self.decode_head_teacher.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head_student, nn.ModuleList):
                for aux_head in self.auxiliary_head_student:
                    aux_head.init_weights()
            else:
                self.auxiliary_head_student.init_weights()

            if isinstance(self.auxiliary_head_r, nn.ModuleList):
                for aux_head in self.auxiliary_head_r:
                    aux_head.init_weights()
            else:
                self.auxiliary_head_r.init_weights()

    def _update_ema_variables(self):
        # update the teacher model by exponential moving average
        ema_decay = min(1 - 1 / (self.cur_iter + 1), self.ema_decay)
        # update the backbone parameters
        for t_param, s_param in zip(self.backbone_teacher.parameters(), self.backbone_student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
        # update the decode head parameters
        for t_param, s_param in zip(self.decode_head_teacher.parameters(), self.decode_head_student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
        # update the neck head parameters
        if self.with_neck:
            for t_param, s_param in zip(self.neck_teacher.parameters(), self.neck_student.parameters()):
                t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
        # update the auxiliary head parameters
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head_student, nn.ModuleList):
                for aux_head_student, aux_head_teacher in zip(self.auxiliary_head_student, self.auxiliary_head_r):
                    for t_param, s_param in zip(self.aux_head_r.parameters(), self.aux_head_student.parameters()):
                        t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
            else:
                for t_param, s_param in zip(self.auxiliary_head_r.parameters(), self.auxiliary_head_student.parameters()):
                    t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)

    def extract_feat_student(self, img):
        """Extract features from images."""
        x = self.backbone_student(img)
        if self.with_neck:
            x = self.neck_student(x)
        return x

    def extract_feat_teacher(self, img):
        """Extract features from images."""
        x = self.backbone_teacher(img)
        if self.with_neck:
            x = self.neck_teacher(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat_student(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train_student(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head_student.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode_student'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head_student.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train_student(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head_student, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head_student):
                loss_aux_student = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux_student, f'aux_student_{idx}'))
        else:
            loss_aux_student = self.auxiliary_head_student.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux_student, 'aux_student'))
        
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
        self._update_ema_variables()
        self.cur_iter += 1

        batch_size, _, _, _ = img.size()
        strong_aug_img = torch.cat([self.unnormalize(img[_, :, :, :]).unsqueeze(0) for _ in range(batch_size)], dim=0)
        strong_aug_img = torch.cat([self.strong_aug(strong_aug_img[_, :, :, :]).unsqueeze(0) for _ in range(batch_size)], dim=0)
        strong_aug_img = torch.cat([self.normalize(strong_aug_img[_, :, :, :]).unsqueeze(0) for _ in range(batch_size)], dim=0)
        x_student= self.extract_feat_student(strong_aug_img)

        with torch.no_grad():
            x_teacher = self.extract_feat_teacher(img)

        losses = dict()

        loss_decode_student= self._decode_head_forward_train_student(x_student, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode_student)

        if self.with_auxiliary_head:
            loss_aux_student = self._auxiliary_head_forward_train_student(
                x_student, img_metas, gt_semantic_seg)
            losses.update(loss_aux_student)

        pred_student = F.softmax(self.decode_head_student.forward(x_student), 1)
        pred_teacher = F.softmax(self.decode_head_teacher.forward(x_teacher), 1)
        losses['consistency_loss'] = self.train_cfg.consistency_loss_weight * \
            F.mse_loss(pred_student, pred_teacher.detach())

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
class CascadeMeanTeacherEncoderDecoder(MeanTeacherEncoderDecoder):
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
        super(CascadeMeanTeacherEncoderDecoder, self).__init__(
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
        self.decode_head_student = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head_student.append(builder.build_head(decode_head[i]))
        self.decode_head_teacher = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head_teacher.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head_student[-1].align_corners
        self.num_classes = self.decode_head_student[-1].num_classes

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone_student.init_weights(pretrained=pretrained)
        self.backbone_teacher.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            self.decode_head_student[i].init_weights()
            self.decode_head_teacher[i].init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head_student, nn.ModuleList):
                for aux_head in self.auxiliary_head_student:
                    aux_head.init_weights()
            else:
                self.auxiliary_head_student.init_weights()
            if isinstance(self.auxiliary_head_teacher, nn.ModuleList):
                for aux_head in self.auxiliary_head_teacher:
                    aux_head.init_weights()
            else:
                self.auxiliary_head_teacher.init_weights()

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat_student(img)
        out = self.decode_head_student[0].forward_test(x, img_metas, self.test_cfg)
        for i in range(1, self.num_stages):
            out = self.decode_head_student[i].forward_test(x, out, img_metas,
                                                   self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train_student(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode = self.decode_head_student[0].forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_student_0'))

        for i in range(1, self.num_stages):
            # forward test again, maybe unnecessary for most methods.
            prev_outputs = self.decode_head_student[i - 1].forward_test(
                x, img_metas, self.test_cfg)
            loss_decode = self.decode_head_student[i].forward_train(
                x, prev_outputs, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_decode, f'decode_student_{i}'))

        return losses


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
        self._update_ema_variables()
        self.cur_iter += 1

        # strong_aug_img = mmcv.imdenormalize(img, self.mean, self.std, self.to_bgr)
        # strong_aug_img = self.strong_aug(strong_aug_img)
        # strong_aug_img = mmcv.imnormalize(strong_aug_img, self.mean, self.std, self.to_rgb)
        # x_student= self.extract_feat_student(strong_aug_img)
        # strong_aug_img = self.unnormalize(img)
        batch_size, _, _, _ = img.size()
        strong_aug_img = torch.cat([self.unnormalize(img[_, :, :, :]).unsqueeze(0) for _ in range(batch_size)], dim=0)
        strong_aug_img = torch.cat([self.strong_aug(strong_aug_img[_, :, :, :]).unsqueeze(0) for _ in range(batch_size)], dim=0)
        print(self.cur_iter)
        strong_aug_img = torch.cat([self.normalize(strong_aug_img[_, :, :, :]).unsqueeze(0) for _ in range(batch_size)], dim=0)
        x_student= self.extract_feat_student(strong_aug_img)

        with torch.no_grad():
            x_teacher = self.extract_feat_teacher(img)

        losses = dict()

        loss_decode_student = self._decode_head_forward_train_student(x_student, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode_student)

        if self.with_auxiliary_head:
            loss_aux_student = self._auxiliary_head_forward_train_student(
                x_student, img_metas, gt_semantic_seg)
            losses.update(loss_aux_student)

        pred_student_list = []
        prev_output = self.decode_head_student[0].forward_test(
            x_student, img_metas, self.test_cfg)
        pred_student_list.append(prev_output)
        for i in range(1, self.num_stages):
            cur_output = self.decode_head_student[i].forward(x_student, prev_output)
            pred_student_list.append(cur_output)
            prev_output = cur_output

        pred_teacher_list = []
        with torch.no_grad():
            prev_output = self.decode_head_teacher[0].forward_test(
                x_teacher, img_metas, self.test_cfg)
            pred_teacher_list.append(prev_output)
            for i in range(1, self.num_stages):
                cur_output = self.decode_head_teacher[i].forward(x_teacher, prev_output)
                pred_teacher_list.append(cur_output)
                prev_output = cur_output

        confidense_threshold = self.train_cfg.get('confidense_threshold', 0.5)
        ignore_index = self.train_cfg.get('ignore_index', 255)
        # compute the consistency loss across the two models.
        if self.train_cfg.get('consistency_w_hard_label', False):
            max_pred_teacher, one_hot_label_teacher = torch.max(pred_teacher_list[-1], dim=1) # n x h x w
            one_hot_label_teacher = one_hot_label_teacher.long()
            one_hot_label_teacher[max_pred_teacher < confidense_threshold] = ignore_index

            if self.train_cfg.get('auxiliary_consistency', False):
                aux_max_pred_teacher, aux_one_hot_label_teacher = torch.max(pred_teacher_list[-2], dim=1)
                aux_one_hot_label_teacher = aux_one_hot_label_teacher.long()
                aux_one_hot_label_teacher[aux_max_pred_teacher < confidense_threshold] = ignore_index
                losses['consistency_loss'] = self.train_cfg.consistency_loss_weight * \
                    (F.cross_entropy(pred_student_list[-1], one_hot_label_teacher, ignore_index=ignore_index, reduction='none').mean() + \
                        0.4 * F.cross_entropy(pred_student_list[-2], aux_one_hot_label_teacher, ignore_index=ignore_index, reduction='none').mean())
            else:
                losses['consistency_loss'] = self.train_cfg.consistency_loss_weight * \
                    F.cross_entropy(pred_student_list[-1], one_hot_label_teacher, ignore_index=ignore_index, reduction='none').mean()
        else:
            T = self.train_cfg.get('temperature', 2) # we ensure that T > 1
            pred_student = F.softmax(pred_student_list[-1], 1)
            pred_teacher = F.softmax(pred_teacher_list[-1], 1)
            sharpe_label_teacher = F.normalize(torch.pow(pred_teacher, T), dim=1, p=1)
            if self.train_cfg.get('auxiliary_consistency', False):
                aux_pred_student = F.softmax(pred_student_list[-2], 1)
                aux_pred_teacher = F.softmax(pred_teacher_list[-2], 1)
                sharpe_aux_teacher = F.normalize(torch.pow(aux_pred_teacher, T), dim=1, p=1)
                losses['consistency_loss'] = self.train_cfg.consistency_loss_weight * \
                    (F.mse_loss(pred_student, sharpe_label_teacher.detach()) + \
                        0.4 * F.mse_loss(aux_pred_student, sharpe_aux_teacher.detach()))
            else:
                losses['consistency_loss'] = self.train_cfg.consistency_loss_weight * \
                    F.mse_loss(pred_student, sharpe_label_teacher.detach())
        return losses