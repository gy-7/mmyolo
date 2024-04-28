_base_ = '../_base_/default_runtime.py'

# dataset settings
data_root = 'data/coco/'
dataset_type = 'YOLOv5CocoDataset'

# parameters that often need to be modified
num_classes = 80
img_scale = (640, 640)  # height, width
strides = [8, 16, 32]

last_stage_out_channels = 1024
env_cfg = dict(cudnn_benchmark=True)

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv9Backbone',
        arch='c',
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    neck=dict(
        type='YOLOv9PAFPN',
        in_channels=[512, 512, 512],
        out_channels=[256, 512, 512],
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    bbox_head=dict(
        type='YOLOv9Head',
        head_module=dict(
            type='YOLOv9HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 512],
            feat_channels=64,
            featmap_strides=strides,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU'),
        ),
    ),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=300,
    ),
)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    # dict(type="YOLOv5KeepRatioResize", scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        use_mini_pad=True,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'),
    ),
]
