# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from .yolov5_head import YOLOv5Head, YOLOv5HeadModule


@MODELS.register_module()
class YOLOv9HeadModule(YOLOv5HeadModule):
    """YOLOv9Head head module used in YOLOv9."""

    def __init__(self,
                 groups: int = 4,
                 reg_max: int = 16,
                 feat_channels=64,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        self.groups = groups
        self.feat_channels = feat_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.reg_max = reg_max
        super().__init__(**kwargs)
        self.no = self.num_classes + reg_max * 4

    def _init_layers(self):
        """initialize conv layers in YOLOv9 head."""
        self.reg_preds = nn.ModuleList()  # reg
        self.cls_preds = nn.ModuleList()  # cls
        for i in range(self.num_levels):
            reg_pred = nn.Sequential(
                ConvModule(
                    self.in_channels[i],
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=self.groups,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                nn.Conv2d(
                    self.feat_channels,
                    4 * self.reg_max,
                    kernel_size=1,
                    stride=1,
                    groups=self.groups),
            )
            cls_pred = nn.Sequential(
                ConvModule(
                    self.in_channels[i],
                    self.feat_channels * 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                ConvModule(
                    self.feat_channels * 4,
                    self.feat_channels * 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                nn.Conv2d(self.feat_channels * 4, self.num_classes, 1, 1),
            )
            self.reg_preds.append(reg_pred)
            self.cls_preds.append(cls_pred)

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
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
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
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
            loss_weight=1.0),
        loss_bbox: ConfigType = dict(
            type='mmdet.GIoULoss', reduction='sum', loss_weight=5.0),
        loss_obj: ConfigType = dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
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

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        pass

    def loss_by_feat(
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
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
        raise NotImplementedError('Not implemented yet!')
