_base_ = './yolov9_c.py'

model = dict(
    type='YOLODetector',
    backbone=dict(
        type='CB_YOLOv9Backbone',
        arch='e',
        mid_indices=(0, 1, 2, 3, 4),
        out_indices=(3, 4, 5),
        down_start_stage=2,
        cb_channels=[64, 256, 512, 1024, 1024],
    ),
    neck=dict(
        type='YOLOv9PAFPN',
        sppf_mid_channels_scale=0.25,
        in_channels=[512, 1024, 1024],
        out_channels=[256, 512, 512],
        block_cfg=dict(type='RepNCSPELAN4', num_blocks=2),
    ),
)
