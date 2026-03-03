# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmdet.datasets import BDD100KDataset


class TestBDD100KDataset(unittest.TestCase):

    def test_bdd100k_dataset(self):
        metainfo = dict(
            classes=('pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
                     'motorcycle', 'bicycle'))
        dataset = BDD100KDataset(
            data_prefix=dict(img_path='imgs'),
            ann_file='tests/data/bdd100k_sample.json',
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[],
            serialize_data=False,
            lazy_init=False)
        self.assertEqual(len(dataset.metainfo['classes']), 8)
        self.assertEqual(dataset.metainfo['classes'][0], 'pedestrian')
        self.assertEqual(dataset.metainfo['classes'][2], 'car')
        # 2 videos
        self.assertEqual(len(dataset), 2)
        # 5 images total across both videos
        self.assertEqual(dataset.num_all_imgs, 5)
        # Video 1: 3 frames, video 2: 2 frames
        self.assertEqual(dataset.get_len_per_video(0), 3)
        self.assertEqual(dataset.get_len_per_video(1), 2)

    def test_bdd100k_dataset_categories(self):
        # Test that only requested categories are loaded
        metainfo = dict(classes=('car', 'bus'))
        dataset = BDD100KDataset(
            data_prefix=dict(img_path='imgs'),
            ann_file='tests/data/bdd100k_sample.json',
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[],
            serialize_data=False,
            lazy_init=False)
        self.assertEqual(len(dataset.metainfo['classes']), 2)
        # Only frames with car or bus annotations should pass the filter
        self.assertEqual(dataset.num_all_imgs, 5)

    def test_bdd100k_instance_ids(self):
        metainfo = dict(
            classes=('pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
                     'motorcycle', 'bicycle'))
        dataset = BDD100KDataset(
            data_prefix=dict(img_path='imgs'),
            ann_file='tests/data/bdd100k_sample.json',
            metainfo=metainfo,
            pipeline=[],
            serialize_data=False,
            lazy_init=False)
        # Check first video, first frame has instance_ids
        video_0 = dataset[0]
        frame_0_instances = video_0['images'][0]['instances']
        self.assertTrue(len(frame_0_instances) > 0)
        for inst in frame_0_instances:
            self.assertIn('instance_id', inst)
            self.assertIn('bbox', inst)
            self.assertIn('bbox_label', inst)

    def test_bdd100k_multiclass_labels(self):
        metainfo = dict(
            classes=('pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
                     'motorcycle', 'bicycle'))
        dataset = BDD100KDataset(
            data_prefix=dict(img_path='imgs'),
            ann_file='tests/data/bdd100k_sample.json',
            metainfo=metainfo,
            pipeline=[],
            serialize_data=False,
            lazy_init=False)
        # Collect all bbox_labels across all videos and frames
        all_labels = set()
        for video_idx in range(len(dataset)):
            video_data = dataset[video_idx]
            for frame in video_data['images']:
                for inst in frame['instances']:
                    all_labels.add(inst['bbox_label'])
        # Sample has pedestrian (cat 1->label 0), car (cat 3->label 2),
        # bus (cat 5->label 4), motorcycle (cat 7->label 6)
        self.assertTrue(len(all_labels) > 1,
                        'Expected multiple classes in BDD100K sample')
