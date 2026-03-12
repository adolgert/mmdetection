"""Diagnose model predictions without running full evaluation.

Loads a checkpoint, runs inference on N validation images, and prints
statistics about the predictions to help identify why AP might be zero.

Usage:
    python tools/analysis_tools/check_predictions.py \
        configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
        --checkpoint work_dirs/.../epoch_4.pth \
        [--num-images 20]
"""
import argparse
import sys

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check model prediction statistics')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument(
        '--num-images', type=int, default=20,
        help='Number of images to run inference on')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Override to process fewer images
    cfg.work_dir = '/tmp/check_predictions'

    # Build runner with test config
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)
    model = runner.model
    model.eval()

    # Get the val dataloader
    dataloader = runner.val_dataloader

    classes = cfg.get('val_dataloader', {}).get('dataset', {}).get(
        'metainfo', {}).get('classes', None)
    if classes is None:
        classes = getattr(
            dataloader.dataset, 'METAINFO', {}).get('classes', ())

    print(f'\nClasses: {classes}')
    print(f'Num classes: {len(classes)}')
    print(f'Checking {args.num_images} images...\n')

    all_num_dets = []
    all_scores = []
    label_counts = {}
    bbox_stats = []

    with torch.no_grad():
        for i, data_batch in enumerate(dataloader):
            if i >= args.num_images:
                break

            # Run inference
            results = model.val_step(data_batch)

            for track_sample in results:
                for frame in track_sample:
                    # Check pred_instances (used by CocoVideoMetric)
                    if hasattr(frame, 'pred_instances'):
                        pi = frame.pred_instances
                        n = len(pi.bboxes) if hasattr(pi, 'bboxes') else 0
                        all_num_dets.append(n)
                        if n > 0:
                            scores = pi.scores.cpu().numpy()
                            labels = pi.labels.cpu().numpy()
                            bboxes = pi.bboxes.cpu().numpy()
                            all_scores.extend(scores.tolist())
                            for lbl in labels:
                                label_counts[int(lbl)] = \
                                    label_counts.get(int(lbl), 0) + 1
                            # Check bbox scale
                            widths = bboxes[:, 2] - bboxes[:, 0]
                            heights = bboxes[:, 3] - bboxes[:, 1]
                            bbox_stats.extend(zip(
                                widths.tolist(), heights.tolist()))
                    else:
                        print(f'  WARNING: frame {i} has no pred_instances!')
                        all_num_dets.append(0)

                    # Also check pred_track_instances
                    if hasattr(frame, 'pred_track_instances'):
                        pti = frame.pred_track_instances
                        n_track = len(pti.bboxes) if hasattr(
                            pti, 'bboxes') else 0
                    else:
                        n_track = 0

            if (i + 1) % 5 == 0:
                print(f'  Processed {i + 1}/{args.num_images} images...')

    # Print summary
    print('\n' + '=' * 60)
    print('PREDICTION DIAGNOSTICS')
    print('=' * 60)

    num_dets = np.array(all_num_dets)
    print(f'\nDetections per frame:')
    print(f'  Mean: {num_dets.mean():.1f}')
    print(f'  Min:  {num_dets.min()}')
    print(f'  Max:  {num_dets.max()}')
    print(f'  Zeros: {(num_dets == 0).sum()}/{len(num_dets)} frames')

    if len(all_scores) > 0:
        scores = np.array(all_scores)
        print(f'\nScore distribution ({len(scores)} total detections):')
        for thr in [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
            pct = (scores >= thr).mean() * 100
            print(f'  >= {thr}: {pct:.1f}%')
        print(f'  Mean: {scores.mean():.4f}')
        print(f'  Median: {np.median(scores):.4f}')

        print(f'\nLabel distribution:')
        for lbl in sorted(label_counts.keys()):
            name = classes[lbl] if lbl < len(classes) else f'class_{lbl}'
            print(f'  {lbl} ({name}): {label_counts[lbl]}')

        if bbox_stats:
            widths, heights = zip(*bbox_stats)
            widths = np.array(widths)
            heights = np.array(heights)
            print(f'\nBbox size (pixels):')
            print(f'  Width  - mean: {widths.mean():.1f}, '
                  f'min: {widths.min():.1f}, max: {widths.max():.1f}')
            print(f'  Height - mean: {heights.mean():.1f}, '
                  f'min: {heights.min():.1f}, max: {heights.max():.1f}')
            zero_area = ((widths <= 0) | (heights <= 0)).sum()
            if zero_area > 0:
                print(f'  WARNING: {zero_area} bboxes with zero/negative area')
    else:
        print('\nNo detections at all! The model produces no output.')
        print('Likely causes:')
        print('  - Classification head randomly initialized '
              '(checkpoint shape mismatch)')
        print('  - All scores below score_thr=0.05')
        print('  - Training did not converge')

    print('\n' + '=' * 60)

    if len(all_scores) == 0:
        sys.exit(1)  # Signal failure


if __name__ == '__main__':
    main()
