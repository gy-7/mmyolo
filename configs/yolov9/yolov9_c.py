_base_ = '../_base_/default_runtime.py'

# -----train val related-----
model_test_cfg = dict(
    multi_label=False,
    nms_pre=30000,
    score_thr=0.25,  # inf
    nms=dict(type='nms', iou_threshold=0.45),  # inf
    max_per_img=1000,  # inf
    # score_thr=0.5,    # val
    # nms=dict(type="nms", iou_threshold=0.7),   # val
    # max_per_img=300,   # val
)

# -----data related-----
dataset_type = 'YOLOv5CocoDataset'
data_root = 'data/coco/'
num_classes = 80
img_scale = (640, 640)  # height, width
val_batch_size_per_gpu = 32
val_num_workers = 8
persistent_workers = True
val_ann_file = 'annotations/instances_val2017.json'
val_data_prefix = 'val2017/'  # Prefix of val image path
batch_shapes_cfg = None

# -----model related-----
strides = (8, 16, 32)
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
act_cfg = dict(type='SiLU', inplace=True)

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
        down_module='ADown',
        out_indices=(2, 3, 4),
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
    ),
    neck=dict(
        type='YOLOv9PAFPN',
        in_channels=[512, 512, 512],
        out_channels=[256, 512, 512],
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
    ),
    bbox_head=dict(
        type='YOLOv9Head',
        head_module=dict(
            type='YOLOv9HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 512],
            reg_max=16,
            feat_channels=64,
            featmap_strides=strides,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        ),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
    ),
    test_cfg=model_test_cfg,
)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
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

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
