# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .base_video_dataset import BaseVideoDataset


@DATASETS.register_module()
class BDD100KDataset(BaseVideoDataset):
    """Dataset for BDD100K Multi-Object Tracking.

    BDD100K MOT uses 8 tracking categories. Annotations should be in
    CocoVID format (converted from Scalabel format using the ``bdd100k``
    package's ``to_coco -m box_track`` command).

    The ``BaseVideoDataset.parse_data_info()`` already handles the
    CocoVID fields (``instance_id``, ``video_id``, ``bbox``,
    ``category_id``) that the converter produces, so no override is
    needed here.
    """

    METAINFO = {
        'classes':
        ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
         'bicycle')
    }
