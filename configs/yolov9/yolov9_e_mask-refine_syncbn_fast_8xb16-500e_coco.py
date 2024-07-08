_base_ = './yolov9_s_mask-refine_syncbn_fast_8xb16-500e_coco.py'

train_use_auxiliary = True
test_use_auxiliary = False

model = dict(
    type='YOLODetector',
    backbone=dict(
        type='CB_YOLOv9Backbone2',
        arch='e',
        train_use_auxiliary=train_use_auxiliary,
        test_use_auxiliary=test_use_auxiliary,
        stem_channels=64,
        reduce_indices=(0, 1, 2, 3, 4),  # backbone1
        pred_out_indices=(3, 4, 5),
        aux_out_indices=(2, 3, 4),
        down_start_index=3,
        down_module_cfg=dict(type='ADown', ),
    ),
    neck=dict(
        type='YOLOv9PAFPN',
        arch='e',
        in_channels=[512, 1024, 1024],
        out_channels=[240, 360, 480],
        train_use_auxiliary=train_use_auxiliary,
        test_use_auxiliary=test_use_auxiliary,
    ),
    bbox_head=dict(
        type='YOLOv9Head',
        head_module=dict(
            type='YOLOv9HeadModule',
            in_channels=[256, 512, 512],
            aux_in_channels=[256, 512, 512],
            reg_feat_channels=64,
            cls_feat_channels=256,
            aux_reg_feat_channels=64,
            aux_cls_feat_channels=256,
            train_use_auxiliary=train_use_auxiliary,
            test_use_auxiliary=test_use_auxiliary,
        ),
    ),
)
