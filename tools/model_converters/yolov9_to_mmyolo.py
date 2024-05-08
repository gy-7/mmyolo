# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch

convert_dict_yolov9_c = {
    # backbone
    'model.1': 'backbone.stem.0',
    'model.2': 'backbone.stem.1',
    'model.3': 'backbone.stage1.0',
    'model.4': 'backbone.stage2.0',
    'model.5': 'backbone.stage2.1',
    'model.6': 'backbone.stage3.0',
    'model.7': 'backbone.stage3.1',
    'model.8': 'backbone.stage4.0',
    'model.9': 'backbone.stage4.1',
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
    # head
    'model.38': 'bbox_head.head_module',
}


def convert(src, dst):
    """Convert keys in pretrained YOLOv9 models to mmyolo style."""
    convert_dict = convert_dict_yolov9_c

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

    for key, weight in blobs.items():
        num, module = key.split('.')[1:3]
        prefix = f'model.{num}'
        if prefix not in convert_dict:
            continue
        new_key = key.replace(prefix, convert_dict[prefix])
        if new_key[:25] in [
                'bbox_head.head_module.cv1',
                'bbox_head.head_module.cv2',
                'bbox_head.head_module.cv3',
        ]:
            continue

        if '.m.' in new_key:
            new_key = new_key.replace('.m.', '.block.')
        if 'bbox_head.head_module' in new_key:
            new_key = new_key.replace('.cv4', '.reg_preds')
            new_key = new_key.replace('.cv5', '.cls_preds')
        if '.cv' in new_key:
            new_key = new_key.replace('.cv1.', '.conv1.')
            new_key = new_key.replace('.cv2.', '.conv2.')
            new_key = new_key.replace('.cv3.', '.conv3.')
            new_key = new_key.replace('.cv4.', '.conv4.')
            new_key = new_key.replace('.cv5.', '.conv5.')
        if '.conv1.conv' in new_key:
            new_key = new_key.replace('.conv1.conv1', '.conv1.rbr_dense')
            new_key = new_key.replace('.conv1.conv2', '.conv1.rbr_1x1')
        if '.reduce_layers.2.conv5' in new_key:
            new_key = new_key.replace('.reduce_layers.2.conv5',
                                      '.reduce_layers.2.conv2')

        if new_key in (
                'bbox_head.head_module.dfl.conv.weight',
                'bbox_head.head_module.dfl2.conv.weight',
        ):
            print('Drop "bbox_head.head_module.dfl", '
                  'because it is useless')
            continue
        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

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
        default='yolov9-c.pt',
        help='src YOLOv9 model path',
    )
    parser.add_argument(
        '--dst',
        default='mmyolo_yolov9-c.pth',
        help='save path',
    )
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
