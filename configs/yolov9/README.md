# YOLOv9

Implementation of paper - [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)

## Abstract

...

<div align="center">
    <img src="https://github.com/WongKinYiu/yolov9/raw/main/figure/performance.png"/>
    YOLOv9 performance
</div>

<!-- [ALGORITHM] -->

## Structures

## Results and models

### COCO

| Backbone | Arch | size | Mask Refine | SyncBN | AMP | Mem (GB) |   box AP    | TTA box AP |                             Config                              |                                                                                                                                                                                   Download                                                                                                                                                                                   |
| :------: | :--: | :--: | :---------: | :----: | :-: | :------: | :---------: | :--------: | :-------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv8-n |  P5  | 640  |     No      |  Yes   | Yes |   2.8    |    37.2     |            |       [config](./yolov8_n_syncbn_fast_8xb16-500e_coco.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804.log.json)                         |
| YOLOv8-n |  P5  | 640  |     Yes     |  Yes   | Yes |   2.5    | 37.4 (+0.2) |    39.9    | [config](./yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_101206-b975b1cd.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_101206.log.json) |
| YOLOv8-s |  P5  | 640  |     No      |  Yes   | Yes |   4.0    |    44.2     |            |       [config](./yolov8_s_syncbn_fast_8xb16-500e_coco.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101.log.json)                         |
| YOLOv8-s |  P5  | 640  |     Yes     |  Yes   | Yes |   4.0    | 45.1 (+0.9) |    46.8    | [config](./yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_095938-ce3c1b3f.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_095938.log.json) |
| YOLOv8-m |  P5  | 640  |     No      |  Yes   | Yes |   7.2    |    49.8     |            |       [config](./yolov8_m_syncbn_fast_8xb16-500e_coco.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200.log.json)                         |
| YOLOv8-m |  P5  | 640  |     Yes     |  Yes   | Yes |   7.0    | 50.6 (+0.8) |    52.3    | [config](./yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_223400-f40abfcd.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_223400.log.json) |
| YOLOv8-l |  P5  | 640  |     No      |  Yes   | Yes |   9.8    |    52.1     |            |       [config](./yolov8_l_syncbn_fast_8xb16-500e_coco.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526.log.json)                         |
| YOLOv8-l |  P5  | 640  |     Yes     |  Yes   | Yes |   9.1    | 53.0 (+0.9) |    54.4    | [config](./yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120100-5881dec4.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120100.log.json) |
| YOLOv8-x |  P5  | 640  |     No      |  Yes   | Yes |   12.2   |    52.7     |            |       [config](./yolov8_x_syncbn_fast_8xb16-500e_coco.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338.log.json)                         |
| YOLOv8-x |  P5  | 640  |     Yes     |  Yes   | Yes |   12.4   | 54.0 (+1.3) |    55.0    | [config](./yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120411-079ca8d1.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120411.log.json) |

**Note**

1.

## Citation
