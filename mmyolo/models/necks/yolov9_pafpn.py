# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import SPPELAN, ADown
from .base_yolo_neck import BaseYOLONeck


@MODELS.register_module()
class YOLOv9PAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLOv9."""

    def __init__(
        self,
        in_channels: List[int] = [512, 512, 512],
        out_channels: List[int] = [512, 512, 512],
        block_cfg: dict = dict(type='RepNCSPELAN4'),
        spp_expand_ratio: float = 0.5,
        is_tiny_version: bool = False,
        use_adown: bool = True,
        use_in_channels_in_downsample: bool = False,
        use_repconv_outs: bool = True,
        upsample_feats_cat_first: bool = True,
        freeze_all: bool = False,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        init_cfg: OptMultiConfig = None,
    ):

        self.is_tiny_version = is_tiny_version
        self.use_adown = use_adown
        self.use_in_channels_in_downsample = use_in_channels_in_downsample
        self.spp_expand_ratio = spp_expand_ratio
        self.use_repconv_outs = use_repconv_outs
        self.block_cfg = block_cfg
        self.block_cfg.setdefault('norm_cfg', norm_cfg)
        self.block_cfg.setdefault('act_cfg', act_cfg)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            upsample_feats_cat_first=upsample_feats_cat_first,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
        )

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == len(self.in_channels) - 1:
            layer = SPPELAN(
                in_channels=self.in_channels[idx],
                out_channels=self.in_channels[idx],
                mid_channels=self.in_channels[idx] // 2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            layer = nn.Identity()
        return layer

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        block_cfg = self.block_cfg.copy()
        if idx == 2:
            block_cfg[
                'in_channels'] = self.in_channels[2] + self.in_channels[0]
            block_cfg['out_channels'] = self.in_channels[2]
            block_cfg['mid_channels1'] = block_cfg['out_channels']
            block_cfg['mid_channels2'] = block_cfg['out_channels'] // 2
        else:
            block_cfg[
                'in_channels'] = self.in_channels[1] + self.in_channels[1]
            block_cfg['out_channels'] = self.in_channels[2] // 2
            block_cfg['mid_channels1'] = block_cfg['out_channels']
            block_cfg['mid_channels2'] = block_cfg['out_channels'] // 2
        return MODELS.build(block_cfg)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        if self.use_adown:
            return ADown(
                self.out_channels[idx],
                self.out_channels[idx],
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            return ConvModule(
                self.out_channels[idx],
                self.out_channels[idx + 1],
                3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        block_cfg = self.block_cfg.copy()
        if idx == 0:
            block_cfg[
                'in_channels'] = self.in_channels[1] + self.out_channels[0]
        else:
            block_cfg[
                'in_channels'] = self.in_channels[2] + self.in_channels[1]
        block_cfg['out_channels'] = self.out_channels[1]
        block_cfg['mid_channels1'] = block_cfg['out_channels']
        block_cfg['mid_channels2'] = block_cfg['out_channels'] // 2
        return MODELS.build(block_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return nn.Identity()
