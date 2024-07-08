# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from .base_yolo_neck import BaseYOLONeck


@MODELS.register_module()
class YOLOv9PAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLOv9."""

    arch_settings = {
        # input_channels, out_channels, mid_channels, num_blocks
        # -1 / -- : not used
        't': [
            ['SPPFBottleneck', 128, 128, 64, -1],  # reduce layer2
            ['RepNCSPELAN4', 224, 96, 96, 3],  # top_down layer2
            ['RepNCSPELAN4', 160, 64, 64, 3],  # top_down layer1
            ['AConv', 64, 48, -1, -1],  # downsample layer0
            ['AConv', 96, 64, -1, -1],  # downsample layer1
            ['RepNCSPELAN4', 144, 96, 96, 3],  # bottom_up layer0
            ['RepNCSPELAN4', 192, 128, 128, 3],  # bottom_up layer1
        ],
        's': [
            ['SPPFBottleneck', 256, 256, 128, -1],  # reduce layer2
            ['RepNCSPELAN4', 448, 192, 192, 3],  # top_down layer2
            ['RepNCSPELAN4', 320, 128, 128, 3],  # top_down layer1
            ['AConv', 128, 96, -1, -1],  # downsample layer0
            ['AConv', 192, 128, -1, -1],  # downsample layer1
            ['RepNCSPELAN4', 288, 192, 192, 3],  # bottom_up layer0
            ['RepNCSPELAN4', 384, 256, 256, 3],  # bottom_up layer1
        ],
        'm': [
            ['SPPFBottleneck', 480, 480, 240, -1],  # reduce layer2
            ['RepNCSPELAN4', 840, 360, 360, 1],  # top_down layer2
            ['RepNCSPELAN4', 600, 240, 240, 1],  # top_down layer1
            ['AConv', 240, 184, -1, -1],  # downsample layer0
            ['AConv', 360, 240, -1, -1],  # downsample layer1
            ['RepNCSPELAN4', 544, 360, 360, 1],  # bottom_up layer0
            ['RepNCSPELAN4', 720, 480, 480, 1],  # bottom_up layer1
        ],
        'c': [
            ['SPPFBottleneck', 512, 512, 256, -1],  # reduce layer2
            ['RepNCSPELAN4', 1024, 512, 512, 1],  # top_down layer2
            ['RepNCSPELAN4', 1024, 256, 256, 1],  # top_down layer1
            ['ADown', 256, 256, -1, -1],  # downsample layer0
            ['ADown', 512, 512, -1, -1],  # downsample layer1
            ['RepNCSPELAN4', 768, 512, 512, 1],  # bottom_up layer0
            ['RepNCSPELAN4', 1024, 512, 512, 1],  # bottom_up layer1
        ],
        'e': [
            ['SPPFBottleneck', 1024, 512, 256, -1],  # reduce layer2
            ['RepNCSPELAN4', 1536, 512, 512, 2],  # top_down layer2
            ['RepNCSPELAN4', 1024, 256, 256, 2],  # top_down layer1
            ['ADown', 256, 256, -1, -1],  # downsample layer0
            ['ADown', 512, 512, -1, -1],  # downsample layer1
            ['RepNCSPELAN4', 768, 512, 512, 2],  # bottom_up layer0
            ['RepNCSPELAN4', 1024, 512, 1024, 2],  # bottom_up layer1
        ],
    }

    def __init__(
        self,
        arch: str = 't',
        in_channels: List[int] = [128, 192, 256],
        out_channels: List[int] = [128, 192, 256],
        train_use_auxiliary: bool = False,
        test_use_auxiliary: bool = False,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        if arch not in self.arch_settings.keys():
            raise ValueError(f'arch must be in {self.arch_settings.keys()}')
        if not train_use_auxiliary and test_use_auxiliary:
            raise ValueError('train_use_auxiliarytest_use_auxiliary must be \
                    True if test_use_auxiliary is True')
        self.arch = arch
        self.arch_setting = self.arch_settings[arch]
        self.train_use_auxiliary = train_use_auxiliary
        self.test_use_auxiliary = test_use_auxiliary
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
        )

        if train_use_auxiliary or test_use_auxiliary:
            self.aux_reduce_layers = nn.ModuleList()
            for idx in range(len(in_channels)):
                self.aux_reduce_layers.append(self.build_reduce_layer(idx))

            # build top-down blocks
            self.aux_upsample_layers = nn.ModuleList()
            self.aux_top_down_layers = nn.ModuleList()
            for idx in range(len(in_channels) - 1, 0, -1):
                self.aux_upsample_layers.append(self.build_upsample_layer(idx))
                self.aux_top_down_layers.append(self.build_top_down_layer(idx))

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The reduce layer.
        """
        if idx == len(self.in_channels) - 1:
            block_type, in_channels, out_channels, mid_channels, _ = (
                self.arch_setting[0])
            mid_channels_scale = mid_channels / in_channels
            block_cfg = dict(
                type=block_type,
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels_scale=mid_channels_scale,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
            layer = MODELS.build(block_cfg)
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
        if idx == 2:
            block_type, in_channels, out_channels, mid_channels, num_blocks = (
                self.arch_setting[1])
        else:
            block_type, in_channels, out_channels, mid_channels, num_blocks = (
                self.arch_setting[2])
        block_cfg = dict(
            type=block_type,
            in_channels=in_channels,
            out_channels=out_channels,
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        return MODELS.build(block_cfg)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The downsample layer.
        """
        if idx == 0:
            block_type, in_channels, out_channels, _, _ = self.arch_setting[3]
        else:
            block_type, in_channels, out_channels, _, _ = self.arch_setting[4]
        block_cfg = dict(
            type=block_type,
            in_channels=in_channels,
            out_channels=out_channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        return MODELS.build(block_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The bottom up layer.
        """
        if idx == 0:
            block_type, in_channels, out_channels, mid_channels, num_blocks = (
                self.arch_setting[5])
        else:
            block_type, in_channels, out_channels, mid_channels, num_blocks = (
                self.arch_setting[6])
        block_cfg = dict(
            type=block_type,
            in_channels=in_channels,
            out_channels=out_channels,
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        return MODELS.build(block_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The out layer.
        """
        return nn.Identity()

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        if (self.training
                and self.train_use_auxiliary) or (not self.training
                                                  and self.test_use_auxiliary):
            aux_reduce_outs = []
            for idx in range(len(self.in_channels)):
                aux_reduce_outs.append(self.aux_reduce_layers[idx](
                    inputs[idx]))

            # top-down path
            aux_inner_outs = [aux_reduce_outs[-1]]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_high = aux_inner_outs[0]
                feat_low = aux_reduce_outs[idx - 1]
                upsample_feat = self.aux_upsample_layers[len(self.in_channels)
                                                         - 1 - idx](
                                                             feat_high)
                if self.upsample_feats_cat_first:
                    top_down_layer_inputs = torch.cat(
                        [upsample_feat, feat_low], 1)
                else:
                    top_down_layer_inputs = torch.cat(
                        [feat_low, upsample_feat], 1)
                inner_out = self.aux_top_down_layers[len(self.in_channels) -
                                                     1 - idx](
                                                         top_down_layer_inputs)
                aux_inner_outs.insert(0, inner_out)

            for idx in range(len(self.in_channels)):
                results.append(aux_inner_outs[idx])
        else:
            if len(inputs) > len(self.in_channels):
                results.extend(inputs[len(self.in_channels):])
            else:
                pass

        return tuple(results)
