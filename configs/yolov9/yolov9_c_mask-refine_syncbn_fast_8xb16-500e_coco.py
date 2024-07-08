_base_ = './yolov9_s_mask-refine_syncbn_fast_8xb16-500e_coco.py'

train_use_auxiliary = True
test_use_auxiliary = False

model = dict(
    type='YOLODetector',
    backbone=dict(
        type='CB_YOLOv9Backbone',
        arch='c',
        train_use_auxiliary=train_use_auxiliary,
        test_use_auxiliary=test_use_auxiliary,
        stem_channels=64,
        down_module_cfg=dict(type='ADown', ),
    ),
    neck=dict(
        type='YOLOv9PAFPN',
        arch='c',
        in_channels=[512, 512, 512],
        out_channels=[256, 512, 512],
    ),
    bbox_head=dict(
        type='YOLOv9Head',
        head_module=dict(
            type='YOLOv9HeadModule',
            in_channels=[256, 512, 512],
            aux_in_channels=[512, 512, 512],
            reg_feat_channels=64,
            cls_feat_channels=256,
            aux_reg_feat_channels=128,
            aux_cls_feat_channels=512,
            train_use_auxiliary=train_use_auxiliary,
            test_use_auxiliary=test_use_auxiliary,
        ),
    ),
)
