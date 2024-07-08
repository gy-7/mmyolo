_base_ = './yolov9_s_syncbn_fast_8xb16-500e_coco.py'

train_use_auxiliary = True
test_use_auxiliary = False

model = dict(
    type='YOLODetector',
    backbone=dict(
        type='CB_YOLOv9Backbone',
        arch='m',
        train_use_auxiliary=train_use_auxiliary,
        test_use_auxiliary=test_use_auxiliary,
        stem_channels=32,
        reduce_indices=(2, 3, 4),
        pred_out_indices=(2, 3, 4),
        aux_out_indices=(2, 3, 4),
        down_start_index=2,
        down_module_cfg=dict(type='AConv', ),
    ),
    neck=dict(
        type='YOLOv9PAFPN',
        arch='m',
        in_channels=[240, 360, 480],
        out_channels=[240, 360, 480],
        train_use_auxiliary=False,
        test_use_auxiliary=False,
    ),
    bbox_head=dict(
        type='YOLOv9Head',
        head_module=dict(
            type='YOLOv9HeadModule',
            in_channels=[240, 360, 480],
            aux_in_channels=[240, 360, 480],
            reg_feat_channels=64,
            cls_feat_channels=240,
            aux_reg_feat_channels=64,
            aux_cls_feat_channels=240,
            train_use_auxiliary=train_use_auxiliary,
            test_use_auxiliary=test_use_auxiliary,
        ),
    ),
)
