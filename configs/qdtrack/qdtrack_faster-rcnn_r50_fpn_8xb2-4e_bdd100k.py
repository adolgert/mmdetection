_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_4e_base.py',
    '../_base_/datasets/bdd100k_track.py',
]

# Override detector head for 8-class BDD100K tracking
model = dict(
    detector=dict(
        roi_head=dict(bbox_head=dict(num_classes=8)),
        # Use full COCO-pretrained model instead of person-only
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
            'faster_rcnn_r50_fpn_1x_coco/'
            'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
            # noqa: E501
        )))

# evaluator
val_evaluator = [
    dict(type='CocoVideoMetric', metric=['bbox'], classwise=True),
    dict(type='MOTChallengeMetric', metric=['HOTA', 'CLEAR', 'Identity'],
         seq_path_component=-2)
]

test_evaluator = val_evaluator
