# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.registry import MODELS
from ..layers import CBFuse, CBLinear
from .base_backbone import BaseBackbone


@MODELS.register_module()
class YOLOv9Backbone(BaseBackbone):
    arch_settings = {
        # input_channels, out_channels, mid_channels, num_blocks/pass
        # -1: default value, not used
        # --: not used
        't': [
            [
                'ConvModule',
                'ELAN1',
                32,
                32,
                32,
                -1,
            ],  # out_indices=1, stage_idx=0
            [
                'AConv',
                'RepNCSPELAN4',
                64,
                64,
                64,
                3,
            ],  # out_indices=2, stage_idx=1
            [
                'AConv',
                'RepNCSPELAN4',
                96,
                96,
                96,
                3,
            ],  # out_indices=3, stage_idx=2
            [
                'AConv',
                'RepNCSPELAN4',
                128,
                128,
                128,
                3,
            ],  # out_indices=4, stage_idx=3
        ],
        's': [
            ['ConvModule', 'ELAN1', 64, 64, 64, -1],
            ['AConv', 'RepNCSPELAN4', 128, 128, 128, 3],
            ['AConv', 'RepNCSPELAN4', 192, 192, 192, 3],
            ['AConv', 'RepNCSPELAN4', 256, 256, 256, 3],
        ],
        'm1': [
            ['ConvModule', 'RepNCSPELAN4', 64, 128, 128, 1],
            ['AConv', 'RepNCSPELAN4', 240, 240, 240, 1],
            ['AConv', 'RepNCSPELAN4', 360, 360, 360, 1],
            ['AConv', 'RepNCSPELAN4', 480, 480, 480, 1],
        ],
        'm2': [
            ['ConvModule', 'RepNCSPELAN4', 64, 128, 128, 1],
            ['CBFuse', 'RepNCSPELAN4', 240, 240, 240, 1],
            ['CBFuse', 'RepNCSPELAN4', 360, 360, 360, 1],
            ['CBFuse', 'RepNCSPELAN4', 480, 480, 480, 1],
        ],
        'c1': [
            ['ConvModule', 'RepNCSPELAN4', 128, 256, 128, 1],
            ['ADown', 'RepNCSPELAN4', 256, 512, 256, 1],
            ['ADown', 'RepNCSPELAN4', 512, 512, 512, 1],
            ['ADown', 'RepNCSPELAN4', 512, 512, 512, 1],
        ],
        'c2': [
            ['ConvModule', 'RepNCSPELAN4', 128, 256, 128, 1],
            ['CBFuse', 'RepNCSPELAN4', 256, 512, 256, 1],
            ['CBFuse', 'RepNCSPELAN4', 512, 512, 512, 1],
            ['CBFuse', 'RepNCSPELAN4', 512, 512, 512, 1],
        ],
        'e1': [
            ['ConvModule', 'RepNCSPELAN4', 128, 256, 128, 2],
            ['ADown', 'RepNCSPELAN4', 256, 512, 256, 2],
            ['ADown', 'RepNCSPELAN4', 512, 1024, 512, 2],
            ['ADown', 'RepNCSPELAN4', 1024, 1024, 512, 2],
        ],
        'e2': [
            ['CBFuse', 'ConvModule', 64, 128, -1, -1],
            ['CBFuse', 'RepNCSPELAN4', 128, 256, 128, 2],
            ['CBFuse', 'RepNCSPELAN4', 256, 512, 256, 2],
            ['CBFuse', 'RepNCSPELAN4', 512, 1024, 512, 2],
            ['CBFuse', 'RepNCSPELAN4', 1024, 1024, 512, 2],
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

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()

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
        (
            downsample_type,
            block_type,
            in_channels,
            out_channels,
            mid_channels,
            num_blocks,
        ) = setting
        stage = []
        if downsample_type == 'CBFuse':
            stage.append(CBFuse())
        else:
            # build downsample layer
            down_out_channels = in_channels
            if stage_idx == 0:
                down_in_channels = self.stem_channels
            else:
                down_in_channels = self.arch_setting[stage_idx - 1][3]
            downsample_layer = self._build_downsample_layer(
                downsample_type, down_in_channels, down_out_channels)
            stage.append(downsample_layer)

        # build module layer
        block_cfg = dict(
            type=block_type,
            in_channels=in_channels,
            out_channels=out_channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        if block_type == 'ELAN1':
            block_cfg['mid_channels'] = mid_channels
        elif block_type == 'RepNCSPELAN4':
            block_cfg['mid_channels'] = mid_channels
            block_cfg['num_blocks'] = num_blocks
        elif block_type == 'ConvModule':
            block_cfg['kernel_size'] = 3
            block_cfg['stride'] = 2
            block_cfg['padding'] = 1
        else:
            raise ValueError(f'block_type {block_type} is not supported')
        stage.append(MODELS.build(block_cfg))
        return stage

    def _build_downsample_layer(self, downsample_type: str, in_channel: int,
                                out_channel: int) -> Optional[nn.Module]:
        assert downsample_type in (
            'ConvModule',
            'AConv',
            'ADown',
        ), "downsample_type must be in ['ConvModule', 'AConv', 'ADown']"
        downsample_layer_cfg = dict(
            type=downsample_type,
            in_channels=in_channel,
            out_channels=out_channel,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        if downsample_type == 'ConvModule':
            downsample_layer_cfg['kernel_size'] = 3
            downsample_layer_cfg['stride'] = 2
            downsample_layer_cfg['padding'] = 1
        downsample_layer = MODELS.build(downsample_layer_cfg)
        return downsample_layer


@MODELS.register_module()
class CB_YOLOv9Backbone(BaseModule):
    # support yolov9-m/c/e arch
    def __init__(
            self,
            arch: str = 'm',
            train_use_auxiliary: bool = False,
            test_use_auxiliary: bool = False,
            reduce_indices: Tuple[int] = (2, 3, 4),
            pred_out_indices: Tuple[int] = (2, 3, 4),
            aux_out_indices: Tuple[int] = (2, 3, 4),
            down_start_index: int = 2,
            down_module_cfg: dict = dict(type='AConv', ),
            **kwargs,
    ):
        super().__init__()
        if arch not in ('m', 'c', 'e'):
            raise ValueError(
                f'arch {arch} is not supported, only support m/c/e')
        if not train_use_auxiliary and test_use_auxiliary:
            raise ValueError('train_use_auxiliarytest_use_auxiliary must be \
                True if test_use_auxiliary is True')
        archs = (arch + '1', arch + '2')
        self.train_use_auxiliary = train_use_auxiliary
        self.test_use_auxiliary = test_use_auxiliary
        self.reduce_indices = reduce_indices
        self.reduce_start_index = reduce_indices[0]
        self.pred_out_indices = pred_out_indices
        self.aux_out_indices = aux_out_indices
        self.down_start_index = down_start_index
        self.down_module_cfg = down_module_cfg
        if 'out_indices' in kwargs:
            kwargs.pop('out_indices')

        self._init_backbones(archs, pred_out_indices, reduce_indices, **kwargs)
        self.pre_layers_num = len(self.backbone2.layers) - len(
            self.backbone1.layers)
        self._init_reduce_layers()
        self._init_down_layers()

    def _init_backbones(self, archs, pred_out_indices, reduce_indices,
                        **kwargs):
        self.backbone1 = YOLOv9Backbone(
            arch=archs[0], out_indices=pred_out_indices, **kwargs)
        if (self.training
                and self.train_use_auxiliary) or (not self.training
                                                  and self.test_use_auxiliary):
            self.backbone2 = YOLOv9Backbone(
                arch=archs[1], out_indices=reduce_indices, **kwargs)

    def _init_reduce_layers(self):
        self.reduce_layers = nn.ModuleList()
        start_index = max(self.reduce_start_index - 1, 0)
        all_out_channels = [
            i[2] for i in self.backbone2.arch_setting[start_index:]
        ]
        for i in range(self.reduce_start_index, len(self.backbone1.layers)):
            if i > 0:
                in_channels = self.backbone1.arch_setting[i - 1][3]
            else:
                in_channels = self.backbone1.stem_channels
            channels = [
                in_channels,
                all_out_channels[:i - self.reduce_start_index + 1],
            ]
            self.reduce_layers.append(CBLinear(*channels))

    def _init_down_layers(self):
        self.down_layers = nn.ModuleList()
        for i in range(self.down_start_index, len(self.backbone2.layers)):
            self.down_module_cfg['in_channels'] = self.backbone2.arch_setting[
                i - 2][3]
            self.down_module_cfg['out_channels'] = self.backbone2.arch_setting[
                i - 1][2]
            self.down_layers.append(MODELS.build(self.down_module_cfg))

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()

    def forward(self, x):
        cb_feats = self.backbone1(x)
        pred_outs = []
        aux_outs = []
        for i, reduce_index in enumerate(self.reduce_indices):
            if reduce_index in self.pred_out_indices:
                pred_outs.append(cb_feats[i])

        if (self.training
                and self.train_use_auxiliary) or (not self.training
                                                  and self.test_use_auxiliary):
            cb_feats = [
                self.reduce_layers[i](cb_feat)
                for i, cb_feat in enumerate(cb_feats)
            ]
            cb_feats = [[val[i] for val in cb_feats[i:]]
                        for i in range(len(cb_feats))]

            for i, layer_name in enumerate(self.backbone2.layers):
                layer = getattr(self.backbone2, layer_name)
                if i >= self.down_start_index:
                    x = self.down_layers[i - self.down_start_index](x)
                if i >= self.reduce_start_index + self.pre_layers_num:
                    x = cb_feats[i - self.reduce_start_index -
                                 self.pre_layers_num] + [x]
                x = layer(x)
                if i in self.aux_out_indices:
                    aux_outs.append(x)
        return pred_outs + aux_outs


@MODELS.register_module()
class CB_YOLOv9Backbone2(CB_YOLOv9Backbone):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs, )

    def _init_backbones(self, archs, pred_out_indices, reduce_indices,
                        **kwargs):
        self.backbone1 = YOLOv9Backbone(
            arch=archs[0], out_indices=reduce_indices, **kwargs)
        self.backbone2 = YOLOv9Backbone(
            arch=archs[1], out_indices=pred_out_indices, **kwargs)

    def forward(self, x):
        cb_feats = self.backbone1(x)
        pred_outs = []
        aux_outs = []
        if (self.training
                and self.train_use_auxiliary) or (not self.training
                                                  and self.test_use_auxiliary):
            for i, reduce_index in enumerate(self.reduce_indices):
                if reduce_index in self.aux_out_indices:
                    aux_outs.append(cb_feats[i])

        cb_feats = [
            self.reduce_layers[i](cb_feat)
            for i, cb_feat in enumerate(cb_feats)
        ]
        cb_feats = [[val[i] for val in cb_feats[i:]]
                    for i in range(len(cb_feats))]

        for i, layer_name in enumerate(self.backbone2.layers):
            layer = getattr(self.backbone2, layer_name)
            if i >= self.down_start_index:
                x = self.down_layers[i - self.down_start_index](x)
            if i >= self.reduce_start_index + self.pre_layers_num:
                x = cb_feats[i - self.reduce_start_index -
                             self.pre_layers_num] + [x]
            x = layer(x)
            if i in self.pred_out_indices:
                pred_outs.append(x)

        return pred_outs + aux_outs
