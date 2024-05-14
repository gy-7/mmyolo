# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.registry import MODELS
from ..layers import ADown, CBFuse, CBLinear, RepNCSPELAN4
from .base_backbone import BaseBackbone


@MODELS.register_module()
class YOLOv9Backbone(BaseBackbone):
    arch_settings = {
        # input_channels, out_channels, mid_channels, num_blocks
        'c': [
            [128, 256, 128, 1],
            [256, 512, 256, 1],
            [512, 512, 512, 1],
            [512, 512, 512, 1],
        ],
        'e1': [
            [128, 256, 128, 2],
            [256, 512, 256, 2],
            [512, 1024, 512, 2],
            [1024, 1024, 512, 2],
        ],
        'e2': [
            [64, 128, 0, 0],
            [128, 256, 128, 2],
            [256, 512, 256, 2],
            [512, 1024, 512, 2],
            [1024, 1024, 512, 2],
        ],
    }

    def __init__(
            self,
            arch: str = 'c',
            input_channels: int = 3,
            stem_channels: int = 64,
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
        stem = ConvModule(
            self.input_channels,
            self.stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        return stem

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        in_channels, out_channels, mid_channels, num_blocks = setting
        stage = []
        if self.arch == 'e2':
            stage.append(CBFuse())
            if stage_idx == 0:
                stage.append(
                    self._build_downsample_layer(in_channels, out_channels,
                                                 'ConvModule'))
            else:
                stage.append(
                    RepNCSPELAN4(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        mid_channels=mid_channels,
                        num_blocks=num_blocks,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ))
        else:
            if stage_idx == 0:
                downsample_layer = self._build_downsample_layer(
                    self.stem_channels, in_channels, 'ConvModule')
            else:
                downsample_layer = self._build_downsample_layer(
                    in_channels, in_channels, 'ADown')
            stage.append(downsample_layer)
            stage.append(
                RepNCSPELAN4(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    mid_channels=mid_channels,
                    num_blocks=num_blocks,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ))
        return stage

    def _build_downsample_layer(self, in_channel: int, out_channel: int,
                                down_module: str) -> Optional[nn.Module]:
        if down_module == 'ConvModule':
            downsample_layer = ConvModule(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        elif down_module == 'ADown':
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


@MODELS.register_module()
class CB_YOLOv9Backbone(BaseModule):

    def __init__(
        self,
        arch='e',
        mid_indices=(0, 1, 2, 3, 4),
        out_indices=(3, 4, 5),
        down_start_stage=2,
        cb_channels=[64, 256, 512, 1024, 1024],
        **kwargs,
    ):
        super().__init__()
        if arch == 'e':
            archs = ('e1', 'e2')
        else:
            raise ValueError(f'arch {arch} is not supported')
        self.mid_indices = mid_indices
        self.out_indices = out_indices
        self.down_start_stage = down_start_stage
        self.backbone1 = YOLOv9Backbone(
            arch=archs[0], out_indices=mid_indices, **kwargs)
        self.backbone2 = YOLOv9Backbone(
            arch=archs[1], out_indices=out_indices, **kwargs)

        # init cb_linears
        self.cb_linears = nn.ModuleList()
        for i in self.mid_indices:
            channels = [cb_channels[i], [64 * 2**j for j in range(i + 1)]]
            self.cb_linears.append(CBLinear(*channels))

        # init downmodules
        self.down_modules = nn.ModuleList()
        for i in range(down_start_stage, len(self.mid_indices)):
            self.down_modules.append(
                ADown(self.backbone2.arch_setting[i][0],
                      self.backbone2.arch_setting[i][0]))

    def forward(self, x):
        cb_feats = self.backbone1(x)
        cb_feats = [self.cb_linears[i](cb_feats[i]) for i in self.mid_indices]

        outs = []
        for i, layer_name in enumerate(self.backbone2.layers):
            layer = getattr(self.backbone2, layer_name)
            if i > self.down_start_stage:
                x = self.down_modules[i - self.down_start_stage - 1](x)
            if i > 0:
                x = [
                    cb_feats[j][i - 1]
                    for j in range(i - 1, len(self.mid_indices))
                ] + [x]
            x = layer(x)
            if i in self.backbone2.out_indices:
                outs.append(x)

        return tuple(outs)
