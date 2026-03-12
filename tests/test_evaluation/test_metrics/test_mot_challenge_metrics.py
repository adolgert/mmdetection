# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile
from unittest import TestCase

import torch
from mmengine.structures import BaseDataElement, InstanceData

from mmdet.evaluation import MOTChallengeMetric
from mmdet.structures import DetDataSample, TrackDataSample


class TestMOTChallengeMetric(TestCase):

    def test_init(self):
        with self.assertRaisesRegex(KeyError, 'metric unknown is not'):
            MOTChallengeMetric(metric='unknown')
        with self.assertRaises(AssertionError):
            MOTChallengeMetric(benchmark='MOT21')

    def __del__(self):
        self.tmp_dir.cleanup()

    @staticmethod
    def _get_predictions_demo():
        instances = [{
            'bbox_label': 0,
            'bbox': [0, 0, 100, 100],
            'ignore_flag': 0,
            'instance_id': 1,
            'mot_conf': 1.0,
            'category_id': 1,
            'visibility': 1.0
        }, {
            'bbox_label': 0,
            'bbox': [0, 0, 100, 100],
            'ignore_flag': 0,
            'instance_id': 2,
            'mot_conf': 1.0,
            'category_id': 1,
            'visibility': 1.0
        }]
        instances_2 = copy.deepcopy(instances)
        sep = os.sep
        pred_instances_data = dict(
            bboxes=torch.tensor([
                [0, 0, 100, 100],
                [0, 0, 100, 40],
            ]),
            instances_id=torch.tensor([1, 2]),
            scores=torch.tensor([1.0, 1.0]))
        pred_instances_data_2 = copy.deepcopy(pred_instances_data)
        pred_instances = InstanceData(**pred_instances_data)
        pred_instances_2 = InstanceData(**pred_instances_data_2)
        img_data_sample = DetDataSample()
        img_data_sample.pred_track_instances = pred_instances
        img_data_sample.instances = instances
        img_data_sample.set_metainfo(
            dict(
                frame_id=0,
                ori_video_length=2,
                video_length=2,
                img_id=1,
                img_path=f'xxx{sep}MOT17-09-DPM{sep}img1{sep}000001.jpg',
            ))
        img_data_sample_2 = DetDataSample()
        img_data_sample_2.pred_track_instances = pred_instances_2
        img_data_sample_2.instances = instances_2
        img_data_sample_2.set_metainfo(
            dict(
                frame_id=1,
                ori_video_length=2,
                video_length=2,
                img_id=2,
                img_path=f'xxx{sep}MOT17-09-DPM{sep}img1{sep}000002.jpg',
            ))
        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = [
            img_data_sample, img_data_sample_2
        ]
        # [TrackDataSample]
        predictions = []
        if isinstance(track_data_sample, BaseDataElement):
            predictions.append(track_data_sample.to_dict())
        return predictions

    def _test_evaluate(self, format_only, outfile_predix=None):
        """Test using the metric in the same way as Evaluator."""
        metric = MOTChallengeMetric(
            metric=['HOTA', 'CLEAR', 'Identity'],
            format_only=format_only,
            outfile_prefix=outfile_predix)
        metric.dataset_meta = {'classes': ('pedestrian', )}
        data_batch = dict(input=None, data_samples=None)
        predictions = self._get_predictions_demo()
        metric.process(data_batch, predictions)
        eval_results = metric.evaluate()
        return eval_results

    def test_evaluate(self):
        eval_results = self._test_evaluate(False)
        target = {
            'motchallenge-metric/IDF1': 0.5,
            'motchallenge-metric/MOTA': 0,
            'motchallenge-metric/HOTA': 0.755,
            'motchallenge-metric/IDSW': 0,
        }
        for key in target:
            assert eval_results[key] - target[key] < 1e-3

    def test_evaluate_format_only(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        eval_results = self._test_evaluate(
            True, outfile_predix=self.tmp_dir.name)
        assert eval_results == dict()

    @staticmethod
    def _get_bdd100k_predictions_demo():
        """Create mock BDD100K tracking data with two videos.

        BDD100K differences from MOT17:
        - Image paths: .../track/val/<video_id>/<frame>.jpg (seq at index -2)
        - Instances lack mot_conf, category_id, visibility (use .get defaults)
        - Multiple video sequences in a single evaluation
        """
        sep = os.sep

        def _make_frame(video_id, frame_id, video_length, img_id,
                        instance_ids):
            """Create one frame's DetDataSample for a BDD100K video."""
            # GT instances without mot_conf, category_id, visibility
            # (these are MOTChallenge-specific fields absent in BDD100K)
            instances = [{
                'bbox_label': 0,
                'bbox': [10 * i, 10 * i, 100 + 10 * i, 100 + 10 * i],
                'ignore_flag': 0,
                'instance_id': iid,
            } for i, iid in enumerate(instance_ids)]

            pred_instances = InstanceData(
                bboxes=torch.tensor([
                    [10 * i, 10 * i, 100 + 10 * i, 100 + 10 * i]
                    for i in range(len(instance_ids))
                ], dtype=torch.float32),
                instances_id=torch.tensor(instance_ids),
                scores=torch.ones(len(instance_ids)))

            sample = DetDataSample()
            sample.pred_track_instances = pred_instances
            sample.instances = instances
            sample.set_metainfo(dict(
                frame_id=frame_id,
                ori_video_length=video_length,
                video_length=video_length,
                img_id=img_id,
                img_path=(f'data{sep}bdd100k{sep}images{sep}track{sep}val'
                          f'{sep}{video_id}{sep}{frame_id:07d}.jpg'),
            ))
            return sample

        # Video 1: 2 frames, 2 tracked objects
        v1_frames = [
            _make_frame('b1c66a42-6f7d68ca', 0, 2, 1, [1, 2]),
            _make_frame('b1c66a42-6f7d68ca', 1, 2, 2, [1, 2]),
        ]
        track_sample_1 = TrackDataSample()
        track_sample_1.video_data_samples = v1_frames

        # Video 2: 2 frames, 1 tracked object
        v2_frames = [
            _make_frame('b1c81faa-3df17267', 0, 2, 3, [3]),
            _make_frame('b1c81faa-3df17267', 1, 2, 4, [3]),
        ]
        track_sample_2 = TrackDataSample()
        track_sample_2.video_data_samples = v2_frames

        predictions = [
            track_sample_1.to_dict(),
            track_sample_2.to_dict(),
        ]
        return predictions

    def test_evaluate_bdd100k(self):
        """Test MOTChallengeMetric with BDD100K-style data.

        Exercises BDD100K-specific code paths:
        - seq_path_component=-2 (extract video_id from path)
        - Missing mot_conf/category_id/visibility (use .get() defaults)
        - Multiple video sequences
        """
        metric = MOTChallengeMetric(
            metric=['HOTA', 'CLEAR', 'Identity'],
            seq_path_component=-2)
        metric.dataset_meta = {'classes': ('pedestrian', )}
        data_batch = dict(input=None, data_samples=None)
        predictions = self._get_bdd100k_predictions_demo()
        metric.process(data_batch, predictions)
        eval_results = metric.evaluate()

        # With perfect predictions (GT == pred), we expect high scores
        assert 'motchallenge-metric/HOTA' in eval_results
        assert 'motchallenge-metric/MOTA' in eval_results
        assert 'motchallenge-metric/IDF1' in eval_results
        # MOTA should be 1.0 for perfect tracking
        assert eval_results['motchallenge-metric/MOTA'] == 1.0

    def test_bdd100k_config_has_seq_path_component(self):
        """Verify the BDD100K config sets seq_path_component=-2.

        This catches config inheritance issues where a child config
        overrides val_evaluator without preserving seq_path_component.
        """
        from mmengine.config import Config
        cfg = Config.fromfile(
            'configs/qdtrack/'
            'qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py')
        mot_cfgs = [e for e in cfg.val_evaluator
                    if e['type'] == 'MOTChallengeMetric']
        assert len(mot_cfgs) == 1, \
            'Expected exactly one MOTChallengeMetric in val_evaluator'
        assert mot_cfgs[0].get('seq_path_component') == -2, \
            ('MOTChallengeMetric must have seq_path_component=-2 for '
             'BDD100K path structure')
