# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import ADown, SPPFBottleneck
from .base_yolo_neck import BaseYOLONeck


@MODELS.register_module()
class YOLOv9PAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLOv9."""

    def __init__(
        self,
        in_channels: List[int] = [512, 512, 512],
        out_channels: List[int] = [256, 512, 512],
        mid_channels=512,
        down_module: str = 'ADown',
        sppf_mid_channels_scale=0.5,
        block_cfg: dict = dict(type='RepNCSPELAN4'),
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        init_cfg: OptMultiConfig = None,
    ):

        self.down_module = down_module
        self.block_cfg = block_cfg
        self.mid_channels = mid_channels
        self.sppf_mid_channels_scale = sppf_mid_channels_scale
        self.block_cfg.setdefault('norm_cfg', norm_cfg)
        self.block_cfg.setdefault('act_cfg', act_cfg)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
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
            layer = SPPFBottleneck(
                in_channels=self.in_channels[idx],
                out_channels=self.mid_channels,
                mid_channels_scale=self.sppf_mid_channels_scale,
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
            block_cfg['in_channels'] = self.in_channels[1] + self.mid_channels
            block_cfg['out_channels'] = self.out_channels[1]
            block_cfg['mid_channels'] = self.out_channels[1]
        else:
            block_cfg['in_channels'] = self.in_channels[0] + self.mid_channels
            block_cfg['out_channels'] = self.out_channels[0]
            block_cfg['mid_channels'] = self.out_channels[0]
        return MODELS.build(block_cfg)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        if self.down_module == 'ConvModule':
            return ConvModule(
                self.out_channels[idx],
                self.out_channels[idx + 1],
                3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        elif self.down_module == 'ADown':
            return ADown(
                self.out_channels[idx],
                self.out_channels[idx],
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            raise ValueError(
                '`down_module` must be in ["ConvModule", "ADown"]')

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        block_cfg = self.block_cfg.copy()
        block_cfg['out_channels'] = self.out_channels[1]
        block_cfg['mid_channels'] = self.out_channels[1]
        if idx == 0:
            block_cfg['in_channels'] = self.mid_channels + self.out_channels[0]
        else:
            block_cfg['in_channels'] = self.mid_channels + self.out_channels[1]
            if self.in_channels[-1] == 1024:  # support e model
                block_cfg['mid_channels'] = self.out_channels[1] * 2

        return MODELS.build(block_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return nn.Identity()
