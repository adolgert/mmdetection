# Copyright (c) OpenMMLab. All rights reserved.
"""Validate BDD100K data for QDTrack training.

Checks five layers of data correctness and reports actionable diagnostics:
  Layer 1: File/directory existence
  Layer 2: Annotation JSON structure (required fields)
  Layer 3: Category consistency (JSON vs config)
  Layer 4: Image file accessibility (optional, --check-images)
  Layer 5: Full pipeline smoke test (dataset class + transforms)

Usage:
    python tools/analysis_tools/mot/validate_bdd100k_data.py \\
        configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \\
        [--split train|val] [--check-images] [--max-samples 10]
"""
import argparse
import json
import os
import os.path as osp
import sys
from collections import Counter

from mmengine import Config
from mmengine.registry import init_default_scope


def _green(s):
    return f'\033[92m{s}\033[0m'


def _red(s):
    return f'\033[91m{s}\033[0m'


def _yellow(s):
    return f'\033[93m{s}\033[0m'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate BDD100K data for QDTrack training')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val'],
        help='which split to validate (default: train)')
    parser.add_argument(
        '--check-images',
        action='store_true',
        help='check that image files exist and can be opened')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10,
        help='max images to check in --check-images (default: 10)')
    return parser.parse_args()


def validate_layer1_files(cfg, split):
    """Layer 1: Check that files and directories exist."""
    print('\n' + '=' * 60)
    print('Layer 1: File and directory existence')
    print('=' * 60)
    errors = []

    if split == 'train':
        ds_cfg = cfg.train_dataloader.dataset
    else:
        ds_cfg = cfg.val_dataloader.dataset

    data_root = ds_cfg.get('data_root', '')
    ann_file = ds_cfg.get('ann_file', '')
    img_prefix = ds_cfg.get('data_prefix', {}).get('img_path', '')

    ann_path = osp.join(data_root, ann_file)
    img_path = osp.join(data_root, img_prefix)

    # Check data_root
    if not osp.isdir(data_root):
        errors.append(
            f'data_root directory does not exist: {data_root}\n'
            f'  Create it with: mkdir -p {data_root}')
    else:
        print(f'  {_green("OK")} data_root exists: {data_root}')

    # Check annotation file
    if not osp.isfile(ann_path):
        errors.append(
            f'Annotation file not found: {ann_path}\n'
            f'  Expected from config: data_root={data_root} + '
            f'ann_file={ann_file}\n'
            f'  Run the converter:\n'
            f'    python -m bdd100k.label.to_coco -m box_track \\\n'
            f'      -i {data_root}/labels/box_track_20/{split}/ \\\n'
            f'      -o {ann_path}')
    else:
        size_mb = osp.getsize(ann_path) / (1024 * 1024)
        print(f'  {_green("OK")} annotation file exists: {ann_path} '
              f'({size_mb:.1f} MB)')

    # Check image directory
    if not osp.isdir(img_path):
        errors.append(
            f'Image directory not found: {img_path}\n'
            f'  Expected from config: data_root={data_root} + '
            f'data_prefix.img_path={img_prefix}\n'
            f'  Download images and unpack to: {img_path}')
    else:
        # Count items inside
        contents = os.listdir(img_path)
        n_dirs = sum(1 for c in contents if osp.isdir(osp.join(img_path, c)))
        n_files = len(contents) - n_dirs
        print(f'  {_green("OK")} image directory exists: {img_path} '
              f'({n_dirs} subdirs, {n_files} files)')
        if n_dirs == 0 and n_files == 0:
            errors.append(
                f'Image directory is empty: {img_path}\n'
                f'  Unpack the BDD100K MOT images into this directory.')

    for e in errors:
        print(f'  {_red("FAIL")} {e}')

    return len(errors) == 0, ann_path, img_path


def validate_layer2_json(ann_path):
    """Layer 2: Validate annotation JSON structure."""
    print('\n' + '=' * 60)
    print('Layer 2: Annotation JSON structure')
    print('=' * 60)
    errors = []
    warnings = []

    # Parse JSON
    try:
        with open(ann_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f'  {_red("FAIL")} Invalid JSON: {e}')
        return False, None
    except Exception as e:
        print(f'  {_red("FAIL")} Cannot read file: {e}')
        return False, None

    print(f'  {_green("OK")} Valid JSON file')

    # Check top-level keys
    for key in ('images', 'annotations', 'categories'):
        if key not in data:
            errors.append(f'Missing top-level key: "{key}"')
        else:
            print(f'  {_green("OK")} Has "{key}" '
                  f'({len(data[key])} entries)')

    if errors:
        for e in errors:
            print(f'  {_red("FAIL")} {e}')
        return False, data

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    # Check image fields
    image_ids = set()
    images_missing_video_id = 0
    images_missing_frame_id = 0
    for img in images:
        image_ids.add(img['id'])
        if 'video_id' not in img:
            images_missing_video_id += 1
        if 'frame_id' not in img:
            images_missing_frame_id += 1

    if images_missing_video_id > 0:
        errors.append(
            f'{images_missing_video_id}/{len(images)} images missing '
            f'"video_id" field.\n'
            f'  BaseVideoDataset needs video_id to group frames into videos.\n'
            f'  The bdd100k to_coco converter should add this field.')
    else:
        n_videos = len(set(img['video_id'] for img in images))
        print(f'  {_green("OK")} All images have video_id '
              f'({n_videos} videos)')

    if images_missing_frame_id > 0:
        warnings.append(
            f'{images_missing_frame_id}/{len(images)} images missing '
            f'"frame_id" (optional but useful for debugging)')

    # Check annotation fields
    ann_ids = []
    missing_fields = Counter()
    category_ids_seen = set()
    image_ids_referenced = set()
    required_ann_fields = [
        'id', 'image_id', 'category_id', 'bbox', 'area', 'instance_id'
    ]

    for ann in annotations:
        ann_ids.append(ann.get('id'))
        image_ids_referenced.add(ann.get('image_id'))
        category_ids_seen.add(ann.get('category_id'))
        for field in required_ann_fields:
            if field not in ann:
                missing_fields[field] += 1

    # Check annotation ID uniqueness
    if len(set(ann_ids)) != len(ann_ids):
        n_dupes = len(ann_ids) - len(set(ann_ids))
        errors.append(
            f'{n_dupes} duplicate annotation IDs found.\n'
            f'  BaseVideoDataset requires unique annotation IDs.')
    else:
        print(f'  {_green("OK")} All {len(ann_ids)} annotation IDs '
              f'are unique')

    # Report missing fields
    for field, count in missing_fields.items():
        if field == 'instance_id':
            errors.append(
                f'{count}/{len(annotations)} annotations missing '
                f'"instance_id".\n'
                f'  Tracking requires instance_id to associate objects '
                f'across frames.\n'
                f'  If missing, BaseVideoDataset falls back to '
                f'per-frame index (tracking will not work).')
        elif field == 'bbox':
            errors.append(
                f'{count}/{len(annotations)} annotations missing "bbox".')
        elif field == 'area':
            errors.append(
                f'{count}/{len(annotations)} annotations missing "area".\n'
                f'  BaseVideoDataset skips annotations with area <= 0.')
        else:
            errors.append(
                f'{count}/{len(annotations)} annotations missing '
                f'"{field}".')

    if not missing_fields:
        print(f'  {_green("OK")} All annotations have required fields: '
              f'{", ".join(required_ann_fields)}')

    # Check referential integrity: image_id -> images
    orphan_anns = image_ids_referenced - image_ids
    if orphan_anns:
        errors.append(
            f'{len(orphan_anns)} annotations reference image_ids that '
            f'do not exist in "images": {list(orphan_anns)[:5]}...')
    else:
        print(f'  {_green("OK")} All annotation image_ids reference '
              f'valid images')

    # Check referential integrity: category_id -> categories
    declared_cat_ids = {c['id'] for c in categories}
    unknown_cats = category_ids_seen - declared_cat_ids - {None}
    if unknown_cats:
        errors.append(
            f'Annotations reference undeclared category_ids: '
            f'{unknown_cats}')
    else:
        print(f'  {_green("OK")} All annotation category_ids match '
              f'declared categories')

    # Check bbox format
    sample_anns = annotations[:100]
    bad_bboxes = 0
    for ann in sample_anns:
        bbox = ann.get('bbox', [])
        if len(bbox) != 4:
            bad_bboxes += 1
        elif bbox[2] <= 0 or bbox[3] <= 0:
            bad_bboxes += 1
    if bad_bboxes > 0:
        errors.append(
            f'{bad_bboxes}/{len(sample_anns)} sampled annotations have '
            f'invalid bbox (expected [x, y, w, h] with w>0, h>0)')
    else:
        print(f'  {_green("OK")} bbox format looks correct '
              f'([x, y, w, h], sampled {len(sample_anns)})')

    for w in warnings:
        print(f'  {_yellow("WARN")} {w}')
    for e in errors:
        print(f'  {_red("FAIL")} {e}')

    return len(errors) == 0, data


def validate_layer3_categories(data, cfg, split):
    """Layer 3: Validate categories match config METAINFO."""
    print('\n' + '=' * 60)
    print('Layer 3: Category consistency')
    print('=' * 60)
    errors = []

    if split == 'train':
        ds_cfg = cfg.train_dataloader.dataset
    else:
        ds_cfg = cfg.val_dataloader.dataset

    # Get expected classes from config
    config_classes = ds_cfg.get('metainfo', {}).get('classes', None)
    if config_classes is None:
        print(f'  {_yellow("WARN")} No metainfo.classes in config; '
              f'will use dataset default')
        return True

    json_categories = data.get('categories', [])
    json_cat_names = [c['name'] for c in json_categories]
    json_cat_ids = [c['id'] for c in json_categories]

    print(f'  Config expects classes: {config_classes}')
    print(f'  JSON has categories: {json_cat_names} '
          f'(ids: {json_cat_ids})')

    # Check all config classes exist in JSON
    missing_in_json = set(config_classes) - set(json_cat_names)
    if missing_in_json:
        errors.append(
            f'Config expects classes not in JSON: {missing_in_json}\n'
            f'  Either update the config metainfo.classes or check the '
            f'annotation conversion.')

    extra_in_json = set(json_cat_names) - set(config_classes)
    if extra_in_json:
        print(f'  {_yellow("WARN")} JSON has extra categories not in '
              f'config: {extra_in_json} (they will be ignored)')

    if not missing_in_json:
        print(f'  {_green("OK")} All {len(config_classes)} config '
              f'classes found in JSON')

    # Check category IDs are positive integers
    if any(cid < 1 for cid in json_cat_ids):
        errors.append(
            f'Category IDs should start from 1, but found: {json_cat_ids}')
    else:
        print(f'  {_green("OK")} Category IDs are positive integers '
              f'starting from {min(json_cat_ids)}')

    for e in errors:
        print(f'  {_red("FAIL")} {e}')

    return len(errors) == 0


def validate_layer4_images(data, img_path, max_samples):
    """Layer 4: Check that image files exist and can be opened."""
    print('\n' + '=' * 60)
    print(f'Layer 4: Image file accessibility (sampling {max_samples})')
    print('=' * 60)

    images = data.get('images', [])
    if not images:
        print(f'  {_red("FAIL")} No images in annotation file')
        return False

    # Sample evenly across the dataset
    step = max(1, len(images) // max_samples)
    sampled = images[::step][:max_samples]

    missing = []
    unreadable = []
    size_mismatch = []

    for img_info in sampled:
        fpath = osp.join(img_path, img_info['file_name'])
        if not osp.isfile(fpath):
            missing.append(img_info['file_name'])
            continue
        try:
            # Try to read image header to verify it's valid
            import cv2
            img = cv2.imread(fpath)
            if img is None:
                unreadable.append(img_info['file_name'])
                continue
            h, w = img.shape[:2]
            if h != img_info.get('height', h) or w != img_info.get(
                    'width', w):
                size_mismatch.append(
                    f'{img_info["file_name"]}: annotation says '
                    f'{img_info["width"]}x{img_info["height"]}, '
                    f'actual is {w}x{h}')
        except ImportError:
            # cv2 not available, just check file exists
            pass

    n_ok = len(sampled) - len(missing) - len(unreadable)
    if missing:
        print(f'  {_red("FAIL")} {len(missing)}/{len(sampled)} sampled '
              f'images not found:')
        for m in missing[:5]:
            print(f'    Expected: {osp.join(img_path, m)}')
        if len(missing) > 5:
            print(f'    ... and {len(missing) - 5} more')
        # Show what the directory actually contains
        if osp.isdir(img_path):
            actual = os.listdir(img_path)[:5]
            print(f'    Directory {img_path} contains: {actual}')
    if unreadable:
        print(f'  {_red("FAIL")} {len(unreadable)}/{len(sampled)} images '
              f'could not be decoded')
    if size_mismatch:
        print(f'  {_yellow("WARN")} {len(size_mismatch)} image size '
              f'mismatches:')
        for s in size_mismatch[:3]:
            print(f'    {s}')
    if n_ok > 0:
        print(f'  {_green("OK")} {n_ok}/{len(sampled)} sampled images '
              f'are accessible')

    return len(missing) == 0 and len(unreadable) == 0


def validate_layer5_pipeline(cfg, split):
    """Layer 5: Smoke test the full dataset class + pipeline."""
    print('\n' + '=' * 60)
    print('Layer 5: Dataset class smoke test')
    print('=' * 60)
    errors = []

    init_default_scope(cfg.get('default_scope', 'mmdet'))
    from mmdet.registry import DATASETS

    if split == 'train':
        ds_cfg = cfg.train_dataloader.dataset
    else:
        ds_cfg = cfg.val_dataloader.dataset

    # Step 5a: Load with empty pipeline (annotation-only)
    print('  5a. Loading dataset with pipeline=[] (annotations only)...')
    try:
        empty_cfg = ds_cfg.copy()
        empty_cfg['pipeline'] = []
        empty_cfg['serialize_data'] = False
        empty_cfg['lazy_init'] = False
        dataset = DATASETS.build(empty_cfg)
        n_videos = len(dataset)
        n_imgs = dataset.num_all_imgs
        print(f'  {_green("OK")} Loaded: {n_videos} videos, '
              f'{n_imgs} images')
    except Exception as e:
        errors.append(
            f'Failed to build dataset with empty pipeline:\n'
            f'  {type(e).__name__}: {e}\n'
            f'  This usually means the annotation JSON structure is '
            f'incompatible with the dataset class.')
        for err in errors:
            print(f'  {_red("FAIL")} {err}')
        return False

    # Step 5b: Check data structure of first video
    print('  5b. Checking data structure of first video...')
    try:
        video_0 = dataset[0]
        if 'images' not in video_0:
            errors.append(
                'First video entry has no "images" key.\n'
                f'  Keys found: {list(video_0.keys())}')
        else:
            n_frames = len(video_0['images'])
            print(f'  {_green("OK")} First video has {n_frames} frames')

            # Check instances in first frame
            frame_0 = video_0['images'][0]
            instances = frame_0.get('instances', [])
            print(f'  {_green("OK")} First frame has '
                  f'{len(instances)} instances')

            if instances:
                inst = instances[0]
                for field in ['bbox', 'bbox_label', 'instance_id']:
                    if field not in inst:
                        errors.append(
                            f'Instance missing "{field}" field.\n'
                            f'  Instance keys: {list(inst.keys())}\n'
                            f'  This field is needed by '
                            f'LoadTrackAnnotations.')
                    else:
                        print(f'  {_green("OK")} Instance has '
                              f'"{field}": {inst[field]}')

                # Check multi-class labels
                all_labels = set()
                for frame in video_0['images']:
                    for i in frame.get('instances', []):
                        all_labels.add(i.get('bbox_label'))
                if len(all_labels) > 1:
                    print(f'  {_green("OK")} Multiple class labels '
                          f'present: {all_labels}')
                elif len(all_labels) == 1:
                    print(f'  {_yellow("WARN")} Only one class label in '
                          f'first video: {all_labels} '
                          f'(may be fine for pedestrian-heavy scenes)')
    except Exception as e:
        errors.append(
            f'Failed reading first video data:\n'
            f'  {type(e).__name__}: {e}')

    # Step 5c: Try loading with full pipeline (if images exist)
    print('  5c. Trying full pipeline (first sample)...')
    try:
        full_dataset = DATASETS.build(ds_cfg)
        sample = full_dataset[0]
        print(f'  {_green("OK")} Full pipeline produced a sample '
              f'with keys: {list(sample.keys())}')
    except FileNotFoundError as e:
        print(f'  {_yellow("WARN")} Full pipeline failed — images '
              f'not found: {e}\n'
              f'    (Layers 1-4 results above show whether images '
              f'are available)')
    except Exception as e:
        errors.append(
            f'Full pipeline failed:\n'
            f'  {type(e).__name__}: {e}\n'
            f'  Check that images are in the right place and the '
            f'config pipeline is correct.')

    for err in errors:
        print(f'  {_red("FAIL")} {err}')

    return len(errors) == 0


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    print(f'Validating BDD100K data for: {args.config}')
    print(f'Split: {args.split}')

    # Layer 1: Files
    ok1, ann_path, img_path = validate_layer1_files(cfg, args.split)
    if not ok1:
        print(f'\n{_red("STOPPED")} Fix Layer 1 errors before continuing.')
        if not osp.isfile(ann_path):
            sys.exit(1)

    # Layer 2: JSON structure
    ok2, data = validate_layer2_json(ann_path)
    if data is None:
        print(f'\n{_red("STOPPED")} Cannot parse annotation file.')
        sys.exit(1)

    # Layer 3: Categories
    ok3 = validate_layer3_categories(data, cfg, args.split)

    # Layer 4: Images (optional)
    ok4 = True
    if args.check_images:
        ok4 = validate_layer4_images(data, img_path, args.max_samples)
    else:
        print(f'\n  (Layer 4 skipped — use --check-images to enable)')

    # Layer 5: Pipeline
    ok5 = validate_layer5_pipeline(cfg, args.split)

    # Summary
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    results = [
        ('Layer 1: File existence', ok1),
        ('Layer 2: JSON structure', ok2),
        ('Layer 3: Category consistency', ok3),
        ('Layer 4: Image accessibility',
         ok4 if args.check_images else None),
        ('Layer 5: Pipeline smoke test', ok5),
    ]
    all_ok = True
    for name, ok in results:
        if ok is None:
            print(f'  {_yellow("SKIP")} {name}')
        elif ok:
            print(f'  {_green("PASS")} {name}')
        else:
            print(f'  {_red("FAIL")} {name}')
            all_ok = False

    if all_ok:
        print(f'\n{_green("All checks passed.")}')
    else:
        print(f'\n{_red("Some checks failed.")} '
              f'See details above for how to fix.')
        sys.exit(1)


if __name__ == '__main__':
    main()
