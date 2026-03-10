# Copyright (c) OpenMMLab. All rights reserved.
"""Fix doubled directory paths in BDD100K COCO annotation file_names.

Some annotation files produced by the bdd100k converter contain file_name
entries like ``vid/vid/vid-frame.jpg`` instead of ``vid/vid-frame.jpg``.
This script normalises all file_names to exactly one directory level:
``<video_id>/<image_file>``.

Usage:
    python tools/analysis_tools/mot/fix_bdd100k_filenames.py \
        data/bdd100k/annotations/box_track_train_cocoformat.json
"""
import argparse
import json
import os.path as osp
import shutil


def main():
    parser = argparse.ArgumentParser(
        description='Fix doubled directory paths in BDD100K annotations')
    parser.add_argument('ann_file', help='path to COCO annotation JSON')
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='skip creating a .bak backup file')
    args = parser.parse_args()

    with open(args.ann_file) as f:
        data = json.load(f)

    fixed = 0
    for img in data['images']:
        parts = img['file_name'].split('/')
        if len(parts) > 2:
            # Keep only last two components: video_id/filename.jpg
            img['file_name'] = '/'.join(parts[-2:])
            fixed += 1

    if fixed == 0:
        print(f'All {len(data["images"])} file_names already have '
              f'the correct format. Nothing to do.')
        return

    if not args.no_backup:
        bak = args.ann_file + '.bak'
        shutil.copy2(args.ann_file, bak)
        print(f'Backup saved to {bak}')

    with open(args.ann_file, 'w') as f:
        json.dump(data, f)

    print(f'Fixed {fixed}/{len(data["images"])} file_names in {args.ann_file}')


if __name__ == '__main__':
    main()
