# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import gt_instances_preprocess
from .yolov5_head import YOLOv5Head


@MODELS.register_module()
class YOLOv9HeadModule(BaseModule):
    """YOLOv9Head head module used in YOLOv9."""

    def __init__(
            self,
            num_classes: int,
            in_channels: Union[int, Sequence],
            aux_in_channels: Union[int, Sequence],
            featmap_strides: Sequence[int] = (8, 16, 32),
            groups: int = 4,
            reg_max: int = 16,
            reg_feat_channels: int = 64,
            cls_feat_channels: int = 64,
            aux_reg_feat_channels: int = 128,
            aux_cls_feat_channels: int = 256,
            train_use_auxiliary: bool = False,
            test_use_auxiliary: bool = False,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='ReLU'),
            **kwargs,
    ):
        if not train_use_auxiliary and test_use_auxiliary:
            raise ValueError('train_use_auxiliarytest_use_auxiliary must be \
                    True if test_use_auxiliary is True')
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.aux_in_channels = aux_in_channels
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.train_use_auxiliary = train_use_auxiliary
        self.test_use_auxiliary = test_use_auxiliary
        self.groups = groups
        self.reg_feat_channels = reg_feat_channels
        self.cls_feat_channels = cls_feat_channels
        self.aux_reg_feat_channels = aux_reg_feat_channels
        self.aux_cls_feat_channels = aux_cls_feat_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.reg_max = reg_max
        self.no = self.num_classes + reg_max * 4
        self._init_layers()

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds,
                                              self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (640 / stride)**2)

    def _init_layers(self):
        """initialize conv layers in YOLOv9 head."""
        self.reg_preds = nn.ModuleList()  # reg
        self.cls_preds = nn.ModuleList()  # cls
        for i in range(self.num_levels):
            reg_pred = nn.Sequential(
                ConvModule(
                    self.in_channels[i],
                    self.reg_feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                ConvModule(
                    self.reg_feat_channels,
                    self.reg_feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=self.groups,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                nn.Conv2d(
                    self.reg_feat_channels,
                    4 * self.reg_max,
                    kernel_size=1,
                    stride=1,
                    groups=self.groups,
                ),
            )
            cls_pred = nn.Sequential(
                ConvModule(
                    self.in_channels[i],
                    self.cls_feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                ConvModule(
                    self.cls_feat_channels,
                    self.cls_feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                nn.Conv2d(self.cls_feat_channels, self.num_classes, 1, 1),
            )
            self.reg_preds.append(reg_pred)
            self.cls_preds.append(cls_pred)
        if self.train_use_auxiliary or self.test_use_auxiliary:
            for i in range(self.num_levels):
                aux_reg_pred = nn.Sequential(
                    ConvModule(
                        self.aux_in_channels[i],
                        self.aux_reg_feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    ConvModule(
                        self.aux_reg_feat_channels,
                        self.aux_reg_feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=self.groups,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    nn.Conv2d(
                        self.aux_reg_feat_channels,
                        4 * self.reg_max,
                        kernel_size=1,
                        stride=1,
                        groups=self.groups,
                    ),
                )
                aux_cls_pred = nn.Sequential(
                    ConvModule(
                        self.aux_in_channels[i],
                        self.aux_cls_feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    ConvModule(
                        self.aux_cls_feat_channels,
                        self.aux_cls_feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    nn.Conv2d(self.aux_cls_feat_channels, self.num_classes, 1,
                              1),
                )
                self.reg_preds.append(aux_reg_pred)
                self.cls_preds.append(aux_cls_pred)

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        if (self.training
                and self.train_use_auxiliary) or (not self.training
                                                  and self.test_use_auxiliary):
            assert len(x) == self.num_levels * 2
        else:
            assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.reg_preds,
                           self.cls_preds)

    def forward_single(self, x: Tensor, reg_pred: nn.Module,
                       cls_pred: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        b, _, h, w = x.shape
        bbox_dist_preds = reg_pred(x)
        cls_logit = cls_pred(x)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = (
                bbox_dist_preds.softmax(3).matmul(self.proj.view(
                    [-1, 1])).squeeze(-1))
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


# Training mode is currently not supported
@MODELS.register_module()
class YOLOv9Head(YOLOv5Head):
    """YOLOv9Head
    Args:
        head_module(nn.Module): Base module used for YOLOv6Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_obj (:obj:`ConfigDict` or dict): Config of objectness loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        head_module: nn.Module,
        prior_generator: ConfigType = dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
        loss_cls: ConfigType = dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0,
        ),
        loss_bbox: ConfigType = dict(
            type='mmdet.GIoULoss', reduction='sum', loss_weight=5.0),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=1.5 / 4,
        ),
        loss_obj: ConfigType = dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0,
        ),
        aux_weight: int = 0.25,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_obj=loss_obj,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )
        self.train_use_auxiliary = self.head_module.train_use_auxiliary
        self.test_use_auxiliary = self.head_module.test_use_auxiliary
        self.loss_dfl = MODELS.build(loss_dfl)
        # YOLOv9 doesn't need loss_obj
        self.loss_obj = None
        self.aux_weight = aux_weight

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.aux_assigner = TASK_UTILS.build(self.train_cfg.assigner)

            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        objectnesses: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = True,
        with_nms: bool = True,
    ) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        if (self.training
                and self.train_use_auxiliary) or (not self.training
                                                  and self.test_use_auxiliary):
            featmap_sizes = featmap_sizes[:len(featmap_sizes) // 2]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.featmap_sizes = featmap_sizes
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
            )
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)
        if (self.training
                and self.train_use_auxiliary) or (not self.training
                                                  and self.test_use_auxiliary):
            flatten_priors = torch.concat([flatten_priors, flatten_priors], 0)
            flatten_stride = torch.concat([flatten_stride, flatten_stride], 0)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for bboxes, cls_confs, obj_confs, img_meta in zip(
                flatten_decoded_bboxes,
                flatten_cls_scores,
                flatten_objectness,
                batch_img_metas,
        ):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if (obj_confs is not None and score_thr > 0
                    and not cfg.get('yolox_style', False)):
                conf_inds = obj_confs > score_thr
                bboxes = bboxes[conf_inds, :]
                cls_confs = cls_confs[conf_inds, :]
                obj_confs = obj_confs[conf_inds]

            if obj_confs is not None:
                # conf = obj_conf * cls_conf
                cls_confs *= obj_confs[:, None]

            if cls_confs.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = cls_confs[:, 0]
                empty_results.labels = cls_confs[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                cls_confs, labels = cls_confs.max(1, keepdim=True)
                cls_confs, _, keep_idxs, results = filter_scores_and_topk(
                    cls_confs,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]),
                )
                labels = results['labels']
            else:
                cls_confs, labels, keep_idxs, _ = filter_scores_and_topk(
                    cls_confs, score_thr, nms_pre)

            results = InstanceData(
                scores=cls_confs, labels=labels, bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2],
                        pad_param[0],
                        pad_param[2],
                        pad_param[0],
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta,
            )
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list

    def loss_by_feat_with_aux(
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        bbox_dist_preds: Sequence[Tensor],
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.
        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        pred_cls_scores, aux_cls_scores = (
            cls_scores[:self.num_levels],
            cls_scores[self.num_levels:],
        )
        pred_bbox_preds, aux_bbox_preds = (
            bbox_preds[:self.num_levels],
            bbox_preds[self.num_levels:],
        )
        pred_bbox_dist_preds, aux_bbox_dist_preds = (
            bbox_dist_preds[:self.num_levels],
            bbox_dist_preds[self.num_levels:],
        )

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in pred_cls_scores
        ]

        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=pred_cls_scores[0].dtype,
                device=pred_cls_scores[0].device,
                with_stride=True,
            )

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        pred_flatten_clss = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in pred_cls_scores
        ]
        pred_flatten_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in pred_bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        pred_flatten_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in pred_bbox_dist_preds
        ]

        # aux info
        aux_flatten_clss = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in aux_cls_scores
        ]
        aux_flatten_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in aux_bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        aux_flatten_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in aux_bbox_dist_preds
        ]

        pred_flatten_dists = torch.cat(pred_flatten_dists, dim=1)
        pred_flatten_clss = torch.cat(pred_flatten_clss, dim=1)
        pred_flatten_bboxes = torch.cat(pred_flatten_bboxes, dim=1)
        pred_flatten_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2],
            pred_flatten_bboxes,
            self.stride_tensor[..., 0],
        )

        aux_flatten_dists = torch.cat(aux_flatten_dists, dim=1)
        aux_flatten_clss = torch.cat(aux_flatten_clss, dim=1)
        aux_flatten_bboxes = torch.cat(aux_flatten_bboxes, dim=1)
        aux_flatten_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2],
            aux_flatten_bboxes,
            self.stride_tensor[..., 0],
        )

        assigned_result = self.assigner(
            (pred_flatten_bboxes.detach()).type(gt_bboxes.dtype),
            pred_flatten_clss.detach().sigmoid(),
            self.flatten_priors_train,
            gt_labels,
            gt_bboxes,
            pad_bbox_flag,
        )

        aux_assigned_result = self.aux_assigner(
            (aux_flatten_bboxes.detach()).type(gt_bboxes.dtype),
            aux_flatten_clss.detach().sigmoid(),
            self.flatten_priors_train,
            gt_labels,
            gt_bboxes,
            pad_bbox_flag,
        )

        aux_assigned_bboxes = aux_assigned_result['assigned_bboxes']
        aux_assigned_scores = aux_assigned_result['assigned_scores']
        aux_fg_mask_pre_prior = aux_assigned_result['fg_mask_pre_prior']
        aux_assigned_scores_sum = aux_assigned_scores.sum().clamp(min=1)
        loss_cls = (self.loss_cls(aux_flatten_clss, aux_assigned_scores).sum()
                    / aux_assigned_scores_sum) * self.aux_weight

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']
        assigned_scores_sum = assigned_scores.sum().clamp(min=1)
        loss_cls += (
            self.loss_cls(pred_flatten_clss, assigned_scores).sum() /
            assigned_scores_sum)

        aux_flatten_bboxes /= self.stride_tensor
        aux_num_pos = aux_fg_mask_pre_prior.sum()
        if aux_num_pos > 0:
            prior_bbox_mask = aux_fg_mask_pre_prior.unsqueeze(-1).repeat(
                [1, 1, 4])
            aux_bboxes_pos = torch.masked_select(
                aux_flatten_bboxes, prior_bbox_mask).reshape([-1, 4])
            aux_assigned_bboxes_pos = torch.masked_select(
                aux_assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            aux_bbox_weight = torch.masked_select(
                aux_assigned_scores.sum(-1),
                aux_fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = (self.loss_bbox(
                aux_bboxes_pos,
                aux_assigned_bboxes_pos,
                weight=aux_bbox_weight,
            ) / aux_assigned_scores_sum) * self.aux_weight

            aux_dist_pos = aux_flatten_dists[aux_fg_mask_pre_prior]
            aux_assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                aux_assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01,
            )
            aux_assigned_ltrb_pos = torch.masked_select(
                aux_assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = (
                self.loss_dfl(
                    aux_dist_pos.reshape(-1, self.head_module.reg_max),
                    aux_assigned_ltrb_pos.reshape(-1),
                    weight=aux_bbox_weight.expand(-1, 4).reshape(-1),
                    avg_factor=aux_assigned_scores_sum,
                ) * self.aux_weight)
        else:
            loss_bbox = aux_flatten_bboxes.sum() * 0
            loss_dfl = aux_flatten_bboxes.sum() * 0

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        pred_flatten_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        # print(f"aux_num_pos: {aux_num_pos}, num_pos: {num_pos}")
        if num_pos > 0:
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                pred_flatten_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox += (
                self.loss_bbox(
                    pred_bboxes_pos, assigned_bboxes_pos, weight=bbox_weight) /
                assigned_scores_sum)

            # dfl loss
            pred_dist_pos = pred_flatten_dists[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01,
            )
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl += self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum,
            )
        else:
            loss_bbox += pred_flatten_bboxes.sum() * 0
            loss_dfl += pred_flatten_bboxes.sum() * 0

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size,
        )

    def loss_by_feat(
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        bbox_dist_preds: Sequence[Tensor],
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.
        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        if self.training and self.train_use_auxiliary:
            return self.loss_by_feat_with_aux(
                cls_scores,
                bbox_preds,
                bbox_dist_preds,
                batch_gt_instances,
                batch_img_metas,
                batch_gt_instances_ignore,
            )
        num_imgs = len(batch_img_metas)

        pred_cls_scores = cls_scores
        pred_bbox_preds = bbox_preds
        pred_bbox_dist_preds = bbox_dist_preds

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in pred_cls_scores
        ]

        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=pred_cls_scores[0].dtype,
                device=pred_cls_scores[0].device,
                with_stride=True,
            )

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        pred_flatten_clss = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in pred_cls_scores
        ]
        pred_flatten_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in pred_bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        pred_flatten_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in pred_bbox_dist_preds
        ]

        pred_flatten_dists = torch.cat(pred_flatten_dists, dim=1)
        pred_flatten_clss = torch.cat(pred_flatten_clss, dim=1)
        pred_flatten_bboxes = torch.cat(pred_flatten_bboxes, dim=1)
        pred_flatten_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2],
            pred_flatten_bboxes,
            self.stride_tensor[..., 0],
        )

        assigned_result = self.assigner(
            (pred_flatten_bboxes.detach()).type(gt_bboxes.dtype),
            pred_flatten_clss.detach().sigmoid(),
            self.flatten_priors_train,
            gt_labels,
            gt_bboxes,
            pad_bbox_flag,
        )

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']
        assigned_scores_sum = assigned_scores.sum().clamp(min=1)
        loss_cls = (
            self.loss_cls(pred_flatten_clss, assigned_scores).sum() /
            assigned_scores_sum)

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        pred_flatten_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                pred_flatten_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = (
                self.loss_bbox(
                    pred_bboxes_pos, assigned_bboxes_pos, weight=bbox_weight) /
                assigned_scores_sum)

            # dfl loss
            pred_dist_pos = pred_flatten_dists[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01,
            )
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum,
            )
        else:
            loss_bbox = pred_flatten_bboxes.sum() * 0
            loss_dfl = pred_flatten_bboxes.sum() * 0

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size,
        )
