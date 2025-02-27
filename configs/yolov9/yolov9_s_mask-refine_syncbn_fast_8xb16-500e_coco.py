_base_ = './yolov9_s_syncbn_fast_8xb16-500e_coco.py'

# This config will refine bbox by mask while loading annotations and
# transforming after `YOLOv5RandomAffine`

# ========================modified parameters======================
use_mask2refine = True
min_area_ratio = 0.01
copypaste_prob = 0.3

# ===============================Unmodified in most cases====================
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        mask2bbox=use_mask2refine,
    ),
]

last_transform = [
    # Delete gt_masks to avoid more computation
    dict(type='RemoveDataElement', keys=['gt_masks']),
    dict(
        type='mmdet.Albu',
        transforms=_base_.albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
        ),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
    ),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
    ),
]

mosiac_pipeline = [
    dict(
        type='Mosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=pre_transform,
    ),
    dict(type='YOLOv5CopyPaste', prob=copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=_base_.max_translate_ratio,  # note
        scaling_ratio_range=_base_.scaling_ratio_range,  # note
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=min_area_ratio,
        use_mask_refine=use_mask2refine,
    ),
]

train_pipeline = [
    *pre_transform,
    *mosiac_pipeline,
    dict(
        type='YOLOv5MixUp',
        alpha=_base_.mixup_alpha,  # note
        beta=_base_.mixup_beta,  # note
        prob=_base_.mixup_prob,
        pre_transform=[*pre_transform, *mosiac_pipeline],
    ),
    *last_transform,
]
train_pipeline2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0),
    ),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=_base_.max_translate_ratio,  # note
        scaling_ratio_range=_base_.scaling_ratio_range,  # note
        border_val=(114, 114, 114),
        min_area_ratio=min_area_ratio,
        use_mask_refine=use_mask2refine,
    ),
    *last_transform,
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline, ), )
_base_.custom_hooks[1].switch_pipeline = train_pipeline2
