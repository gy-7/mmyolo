# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import ADown, RepNCSPELAN4
from .base_backbone import BaseBackbone


@MODELS.register_module()
class YOLOv9Backbone(BaseBackbone):
    arch_settings = {
        'c': [  # c1,c2,c3,num_blocks
            [128, 256, 128, 1],
            [256, 512, 256, 1],
            [512, 512, 512, 1],
            [512, 512, 512, 1],
        ],
    }

    def __init__(
            self,
            arch: str = 'c',
            input_channels: int = 3,
            stem_channels: Tuple[int] = (64, 128),
            down_module: str = 'ConvModule',
            frozen_stages: int = -1,
            out_indices: Tuple[int] = (2, 3, 4),
            plugins: Union[dict, List[dict]] = None,
            norm_eval: bool = False,
            init_cfg: OptMultiConfig = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
    ):
        assert arch in self.arch_settings.keys()
        self.arch = arch
        self.down_module = down_module
        self.stem_channels = stem_channels
        super().__init__(
            self.arch_settings[arch],
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg,
        )

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        stem = nn.Sequential(
            ConvModule(
                self.input_channels,
                self.stem_channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            ConvModule(
                self.stem_channels[0],
                self.stem_channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        )
        return stem

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        in_channels, out_channels, mid_channels, num_blocks = setting
        stage = []
        stage.append(
            RepNCSPELAN4(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=mid_channels,
                num_blocks=num_blocks,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ))
        if stage_idx != 0:
            downsample_layer = self._build_downsample_layer(
                in_channels, in_channels)
            stage.insert(0, downsample_layer)
        return stage

    def _build_downsample_layer(self, in_channel: int,
                                out_channel: int) -> Optional[nn.Module]:
        if self.down_module == 'ConvModule':
            downsample_layer = ConvModule(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        elif self.down_module == 'ADown':
            downsample_layer = ADown(
                in_channel,
                out_channel,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            raise ValueError(
                '`down_module` must be in ["ConvModule", "ADown"]')
        return downsample_layer
