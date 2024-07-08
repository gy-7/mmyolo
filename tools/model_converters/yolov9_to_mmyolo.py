# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch
from terminaltables import AsciiTable

yolov9_convert_dict = {
    't': {
        # backbone
        'model.0': 'backbone.stem',
        'model.1': 'backbone.stage1.0',
        'model.2': 'backbone.stage1.1',
        'model.3': 'backbone.stage2.0',
        'model.4': 'backbone.stage2.1',
        'model.5': 'backbone.stage3.0',
        'model.6': 'backbone.stage3.1',
        'model.7': 'backbone.stage4.0',
        'model.8': 'backbone.stage4.1',
        # neck
        'model.9': 'neck.reduce_layers.2',
        'model.10': 'neck.upsample_layers.0',
        'model.12': 'neck.top_down_layers.0',
        'model.13': 'neck.upsample_layers.1',
        'model.15': 'neck.top_down_layers.1',
        'model.16': 'neck.downsample_layers.0',
        'model.18': 'neck.bottom_up_layers.0',
        'model.19': 'neck.downsample_layers.1',
        'model.21': 'neck.bottom_up_layers.1',
        # aux neck
        'model.22': 'neck.aux_reduce_layers.2',
        'model.23': 'neck.aux_upsample_layers.0',
        'model.25': 'neck.aux_top_down_layers.0',
        'model.26': 'neck.aux_upsample_layers.1',
        'model.28': 'neck.aux_top_down_layers.1',
        # head
        'model.29': 'bbox_head.head_module',
    },
    'm': {
        # backbone1
        'model.1': 'backbone.backbone1.stem',
        'model.2': 'backbone.backbone1.stage1.0',
        'model.3': 'backbone.backbone1.stage1.1',
        'model.4': 'backbone.backbone1.stage2.0',
        'model.5': 'backbone.backbone1.stage2.1',
        'model.6': 'backbone.backbone1.stage3.0',
        'model.7': 'backbone.backbone1.stage3.1',
        'model.8': 'backbone.backbone1.stage4.0',
        'model.9': 'backbone.backbone1.stage4.1',
        # neck
        'model.10': 'neck.reduce_layers.2',
        'model.11': 'neck.upsample_layers.0',
        'model.13': 'neck.top_down_layers.0',
        'model.14': 'neck.upsample_layers.1',
        'model.16': 'neck.top_down_layers.1',
        'model.17': 'neck.downsample_layers.0',
        'model.19': 'neck.bottom_up_layers.0',
        'model.20': 'neck.downsample_layers.1',
        'model.22': 'neck.bottom_up_layers.1',
        # aux backbone2
        'model.23': 'backbone.reduce_layers.0',
        'model.24': 'backbone.reduce_layers.1',
        'model.25': 'backbone.reduce_layers.2',
        'model.26': 'backbone.backbone2.stem',
        'model.27': 'backbone.backbone2.stage1.0',
        'model.28': 'backbone.backbone2.stage1.1',
        'model.29': 'backbone.down_layers.0',
        'model.30': 'backbone.backbone2.stage2.0',
        'model.31': 'backbone.backbone2.stage2.1',
        'model.32': 'backbone.down_layers.1',
        'model.33': 'backbone.backbone2.stage3.0',
        'model.34': 'backbone.backbone2.stage3.1',
        'model.35': 'backbone.down_layers.2',
        'model.36': 'backbone.backbone2.stage4.0',
        'model.37': 'backbone.backbone2.stage4.1',
        # head
        'model.38': 'bbox_head.head_module',
    },
    'e': {
        # backbone1
        'model.1': 'backbone.backbone1.stem',
        'model.2': 'backbone.backbone1.stage1.0',
        'model.3': 'backbone.backbone1.stage1.1',
        'model.4': 'backbone.backbone1.stage2.0',
        'model.5': 'backbone.backbone1.stage2.1',
        'model.6': 'backbone.backbone1.stage3.0',
        'model.7': 'backbone.backbone1.stage3.1',
        'model.8': 'backbone.backbone1.stage4.0',
        'model.9': 'backbone.backbone1.stage4.1',
        # backbone2
        'model.10': 'backbone.reduce_layers.0',
        'model.11': 'backbone.reduce_layers.1',
        'model.12': 'backbone.reduce_layers.2',
        'model.13': 'backbone.reduce_layers.3',
        'model.14': 'backbone.reduce_layers.4',
        'model.15': 'backbone.backbone2.stem',
        'model.16': 'backbone.backbone2.stage1.0',
        'model.17': 'backbone.backbone2.stage1.1',
        'model.18': 'backbone.backbone2.stage2.0',
        'model.19': 'backbone.backbone2.stage2.1',
        'model.20': 'backbone.down_layers.0',
        'model.21': 'backbone.backbone2.stage3.0',
        'model.22': 'backbone.backbone2.stage3.1',
        'model.23': 'backbone.down_layers.1',
        'model.24': 'backbone.backbone2.stage4.0',
        'model.25': 'backbone.backbone2.stage4.1',
        'model.26': 'backbone.down_layers.2',
        'model.27': 'backbone.backbone2.stage5.0',
        'model.28': 'backbone.backbone2.stage5.1',
        # aux neck
        'model.29': 'neck.aux_reduce_layers.2',
        'model.30': 'neck.aux_upsample_layers.0',
        'model.32': 'neck.aux_top_down_layers.0',
        'model.33': 'neck.aux_upsample_layers.1',
        'model.35': 'neck.aux_top_down_layers.1',
        # neck
        'model.36': 'neck.reduce_layers.2',
        'model.37': 'neck.upsample_layers.0',
        'model.39': 'neck.top_down_layers.0',
        'model.40': 'neck.upsample_layers.1',
        'model.42': 'neck.top_down_layers.1',
        'model.43': 'neck.downsample_layers.0',
        'model.45': 'neck.bottom_up_layers.0',
        'model.46': 'neck.downsample_layers.1',
        'model.48': 'neck.bottom_up_layers.1',
        # head
        'model.49': 'bbox_head.head_module',
    },
}
yolov9_convert_dict['s'] = yolov9_convert_dict['t']
yolov9_convert_dict['c'] = yolov9_convert_dict['m']

yolov9_convert_dict['t-converted'] = yolov9_convert_dict['t'].copy()
yolov9_convert_dict['t-converted']['model.22'] = 'bbox_head.head_module'
yolov9_convert_dict['s-converted'] = yolov9_convert_dict['t-converted']


def convert(src, dst, arch='t'):
    """Convert keys in pretrained YOLOv9 models to mmyolo style."""
    convert_dict = yolov9_convert_dict[arch]
    try:
        yolov9_model = torch.load(src)['model']
        blobs = yolov9_model.state_dict()
    except ModuleNotFoundError:
        raise RuntimeError(
            'This script must be placed under the yolov9 repo,'
            ' because loading the official pretrained model need'
            ' `model.py` to build model.'
            'Also need to install hydra-core>=1.2.0 and thop>=0.1.1')
    state_dict = OrderedDict()

    table_data = [['YOLOv9 key', 'MMYOLO key']]
    for key, weight in blobs.items():
        num, module = key.split('.')[1:3]
        prefix = f'model.{num}'
        if prefix not in convert_dict:
            continue
        new_key = key.replace(prefix, convert_dict[prefix])

        if '.m.' in new_key:
            new_key = new_key.replace('.m.', '.block.')
        if 'bbox_head.head_module' in new_key:
            if 'converted' in arch:
                # official head [aux, pred] -> mmyolo head [pred, aux]
                new_key = new_key.replace('.cv2.', '.reg_preds.')
                new_key = new_key.replace('.cv3.', '.cls_preds.')
            else:
                # official head [aux, pred] -> mmyolo head [pred, aux]
                new_key = new_key.replace('.cv4.', '.reg_preds.')
                new_key = new_key.replace('.cv5.', '.cls_preds.')
                # aux
                if '.cv2.' in new_key:
                    new_key = new_key.replace('.cv2.0', '.reg_preds.3')
                    new_key = new_key.replace('.cv2.1', '.reg_preds.4')
                    new_key = new_key.replace('.cv2.2', '.reg_preds.5')
                if '.cv3.' in new_key:
                    new_key = new_key.replace('.cv3.0', '.cls_preds.3')
                    new_key = new_key.replace('.cv3.1', '.cls_preds.4')
                    new_key = new_key.replace('.cv3.2', '.cls_preds.5')
        if 'reduce_layers' in new_key:
            new_key = new_key.replace('.cv5', '.conv2')
        if '.cv' in new_key:
            new_key = new_key.replace('.cv1.', '.conv1.')
            new_key = new_key.replace('.cv2.', '.conv2.')
            new_key = new_key.replace('.cv3.', '.conv3.')
            new_key = new_key.replace('.cv4.', '.conv4.')
            new_key = new_key.replace('.cv5.', '.conv5.')
        if '.conv1.conv' in new_key:
            new_key = new_key.replace('.conv1.conv1', '.conv1.rbr_dense')
            new_key = new_key.replace('.conv1.conv2', '.conv1.rbr_1x1')

        if new_key in (
                'bbox_head.head_module.dfl.conv.weight',
                'bbox_head.head_module.dfl2.conv.weight',
        ):
            print('Drop "bbox_head.head_module.dfl", '
                  'because it is useless')
            continue
        state_dict[new_key] = weight
        table_data.append([key, new_key])

    table = AsciiTable(table_data)
    print(table.table)

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)
    print(f'save checkpoint path: {dst}')


# Note: This script must be placed under the YOLOv9 repo to run.
def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument(
        '--src',
        default='./yolov9-s.pt',
        help='src YOLOv9 model path',
    )
    parser.add_argument(
        '--dst',
        default='./mmyolo_yolov9-s.pth',
        help='save path',
    )
    parser.add_argument(
        '--arch',
        default='s',
        help='model architecture, t/s/m/c/e, default: s',
    )
    args = parser.parse_args()
    convert(args.src, args.dst, args.arch)


if __name__ == '__main__':
    main()
