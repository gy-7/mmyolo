_base_ = './yolov9_s_mask-refine_syncbn_fast_8xb16-500e_coco.py'

train_use_auxiliary = True
test_use_auxiliary = False

model = dict(
    type='YOLODetector',
    backbone=dict(
        type='YOLOv9Backbone',
        arch='t',
        stem_channels=16,
    ),
    neck=dict(
        type='YOLOv9PAFPN',
        arch='t',
        in_channels=[64, 96, 128],
        out_channels=[64, 96, 128],
        train_use_auxiliary=train_use_auxiliary,
        test_use_auxiliary=test_use_auxiliary,
    ),
    bbox_head=dict(
        type='YOLOv9Head',
        head_module=dict(
            type='YOLOv9HeadModule',
            in_channels=[64, 96, 128],
            aux_in_channels=[64, 96, 128],
            reg_feat_channels=64,
            cls_feat_channels=80,
            aux_reg_feat_channels=64,
            aux_cls_feat_channels=80,
            train_use_auxiliary=train_use_auxiliary,
            test_use_auxiliary=test_use_auxiliary,
        ),
    ),
)
