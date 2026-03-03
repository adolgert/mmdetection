# QDTrack on BDD100K: Detailed List of Changes

This document itemizes every code change, new file, and configuration
that was added to support training and evaluating QDTrack on the BDD100K
Multi-Object Tracking benchmark using this mmdetection v3.3.0 repository.

---

## BDD100K Label Format Notes

BDD100K labels were revised in 2020 from the old 2018 format to the
**Scalabel format**. The tracking labels downloaded today (`box_track_20`)
use this newer format where:

- Label `id` is a **string** (not int32 as in the old format)
- Each frame has `frameIndex` and `videoName` fields
- `box2d` uses `{x1, y1, x2, y2}` coordinates

The `bdd100k` Python package's `to_coco -m box_track` command converts
Scalabel-format labels to **CocoVID-style** JSON with:
- `instance_id` on each annotation (cross-frame object identity)
- `video_id` on each image (linking frames to parent video)
- Standard COCO fields: `bbox` [x,y,w,h], `area`, `category_id`, `iscrowd`

This output is directly compatible with mmdetection's `BaseVideoDataset`.

**Important**: Install `bdd100k` from GitHub (not PyPI) to avoid bugs with
empty frames. See the setup guide for details.

---

## Version Compatibility Summary

This repository (v3.3.0, last updated May 2024) was tested with:

| Component   | Tested Range       | Profile A Target  |
|-------------|--------------------|-------------------|
| Python      | 3.7 - 3.9         | 3.10              |
| PyTorch     | 1.8.0 - 2.0.0     | 2.1.2             |
| CUDA        | 11.1 - 11.7       | 12.1              |
| mmcv        | >=2.0.0rc4, <2.2.0| 2.1.0             |
| mmengine    | >=0.7.1, <1.0.0   | 0.10.4            |
| numpy       | <1.24.0 (tracking) | 1.23.x            |

Profile A covers: V100, A100, Ada RTX 4500.

RTX 5080 (Blackwell) requires a separate Profile B with PyTorch 2.6+ / CUDA 12.8
and is not covered in these changes (see "Conditional Changes" section at the end).

---

## Change 1: New BDD100K Dataset Class

### File created: `mmdet/datasets/bdd100k_dataset.py`

A thin ~25-line subclass of `BaseVideoDataset` that sets the correct
`METAINFO` for BDD100K's 8 tracking categories. No `parse_data_info()`
override is needed because `BaseVideoDataset` already handles the CocoVID
fields (`instance_id`, `video_id`, `bbox`, `category_id`) produced by
the `bdd100k.label.to_coco` converter.

**BDD100K MOT categories (8 classes)**:
1. pedestrian
2. rider
3. car
4. truck
5. bus
6. train
7. motorcycle
8. bicycle

**Why not reuse `MOTChallengeDataset`?** It requires `visibility` and
`mot_conf` fields (lines 80-81 of `mot_challenge_dataset.py`) that don't
exist in BDD100K annotations. It also has a 13-class `METAINFO`.

---

## Change 2: Register BDD100K Dataset in `__init__.py`

### File modified: `mmdet/datasets/__init__.py`

Added `from .bdd100k_dataset import BDD100KDataset` import and added
`'BDD100KDataset'` to the `__all__` list.

---

## Change 3: Fix Multi-Class Label Handling in QDTrack Model

### File modified: `mmdet/models/mot/qdtrack.py`

**Problem**: The `loss()` method zeroed out ALL ground truth class labels
before passing them to both the RPN and the roi_head. For MOT17 (1 class)
this was harmless, but for BDD100K (8 classes) it destroyed all class
information.

**Fix**: Save original labels before zeroing for the RPN, then restore
them before the roi_head forward pass.

The key changes:

1. Added `saved_labels = []` list before the data sample loop
2. In the loop, save original labels with `.clone()` before zeroing
3. After the RPN forward pass, restore original labels before roi_head:

```python
for kds, orig_labels in zip(key_data_samples, saved_labels):
    kds.gt_instances.labels = orig_labels
```

**Backward compatibility**: This change is safe for MOT17 — saving and
restoring all-zero labels is a no-op.

---

## Change 4: BDD100K Base Dataset Config

### File created: `configs/_base_/datasets/bdd100k_track.py`

Follows the pattern of `configs/_base_/datasets/mot_challenge.py`.

**Key differences from MOT Challenge config**:

| Setting | MOT Challenge | BDD100K |
|---------|--------------|---------|
| `dataset_type` | `'MOTChallengeDataset'` | `'BDD100KDataset'` |
| `data_root` | `'data/MOT17/'` | `'data/bdd100k/'` |
| `ann_file` (train) | `'annotations/half-train_cocoformat.json'` | `'annotations/box_track_train_cocoformat.json'` |
| `ann_file` (val) | `'annotations/half-val_cocoformat.json'` | `'annotations/box_track_val_cocoformat.json'` |
| `data_prefix` (train) | `dict(img_path='train')` | `dict(img_path='images/track/train')` |
| `data_prefix` (val) | `dict(img_path='train')` | `dict(img_path='images/track/val')` |
| `metainfo.classes` | `('pedestrian',)` | All 8 BDD100K classes |
| `visibility_thr` | `-1` | N/A (not used) |
| `img_scale` | `(1088, 1088)` | `(1280, 736)` |

**Note on image scale**: BDD100K native resolution is 1280x720. We use
1280x736 for divisibility by 32 (required by FPN stride).

---

## Change 5: BDD100K QDTrack Training Config

### File created: `configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py`

Inherits from the base QDTrack config and the BDD100K dataset config.

**Key overrides**:

1. **`num_classes=8`** in `detector.roi_head.bbox_head` (overrides the
   base config's `num_classes=1`)

2. **Full COCO-pretrained Faster R-CNN** instead of person-only model
   (`faster_rcnn_r50_fpn_1x_coco` instead of `faster_rcnn_r50_fpn_1x_coco-person`)

3. **Evaluator**: `CocoVideoMetric` with `classwise=True` for per-class
   detection AP, plus `MOTChallengeMetric` for tracking metrics (HOTA,
   CLEAR, Identity)

---

## Change 6: Data Validation Script

### File created: `tools/analysis_tools/mot/validate_bdd100k_data.py`

A standalone script that checks BDD100K data correctness at five layers,
reporting actionable error messages at each stage:

1. **File existence** — checks data_root, annotation file, and image directory
2. **JSON structure** — validates required fields (`video_id`, `frame_id`,
   `instance_id`, `bbox`, `category_id`), ID uniqueness, referential integrity
3. **Category consistency** — compares JSON categories against config METAINFO
4. **Image accessibility** (optional `--check-images`) — file existence,
   decodability, annotation/actual size match
5. **Pipeline smoke test** — instantiates `BDD100KDataset` with `pipeline=[]`,
   then with the full pipeline, checking data structure at each stage

Usage:
```bash
python tools/analysis_tools/mot/validate_bdd100k_data.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    [--split train|val] [--check-images] [--max-samples 10]
```

---

## Change 7: BDD100K Unit Test and Sample Data

### File created: `tests/data/bdd100k_sample.json`

Minimal mock annotation file (2 videos, 5 frames, 11 annotations) with 4
of the 8 BDD100K categories (pedestrian, car, bus, motorcycle). Follows the
same structure as `tests/data/mot_sample.json` but adapted for BDD100K.

### File created: `tests/test_datasets/test_bdd100k_dataset.py`

Unit test with 4 test methods following the `test_mot_challenge_dataset.py`
pattern:

- `test_bdd100k_dataset` — loads all 8 classes, checks video/image counts
- `test_bdd100k_dataset_categories` — loads subset of classes (car, bus)
- `test_bdd100k_instance_ids` — verifies instance_id, bbox, bbox_label
- `test_bdd100k_multiclass_labels` — verifies multiple class labels exist

---

## Summary: All Changes

| # | Type | File | Status |
|---|------|------|--------|
| 1 | **New** | `mmdet/datasets/bdd100k_dataset.py` | Done |
| 2 | **Edit** | `mmdet/datasets/__init__.py` | Done |
| 3 | **Edit** | `mmdet/models/mot/qdtrack.py` | Done |
| 4 | **New** | `configs/_base_/datasets/bdd100k_track.py` | Done |
| 5 | **New** | `configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py` | Done |
| 6 | **New** | `tools/analysis_tools/mot/validate_bdd100k_data.py` | Done |
| 7 | **New** | `tests/data/bdd100k_sample.json` | Done |
| 8 | **New** | `tests/test_datasets/test_bdd100k_dataset.py` | Done |

### Not implemented (optional, for future work)

| # | Type | File | Purpose |
|---|------|------|---------|
| 9 | New | `mmdet/evaluation/metrics/bdd100k_mot_metric.py` | Official BDD100K evaluation (for leaderboard submission) |
| 10 | New | `tools/dataset_converters/bdd100k2coco.py` | Not needed — `bdd100k.label.to_coco` handles conversion |

---

## Conditional Changes (RTX 5080 / Profile B Only)

These changes are **not needed** for Profile A (V100/A100/Ada RTX 4500) and
are documented here for future reference.

### Patch mmcv Version Check

**File**: `mmdet/__init__.py` (line 9)

```python
# Current:
mmcv_maximum_version = '2.2.0'

# Patch for Profile B (if mmcv built from source reports >= 2.2.0):
mmcv_maximum_version = '2.3.0'
```

### Risk

mmcv APIs may have breaking changes beyond 2.2.0. All QDTrack code paths
must be tested after this patch, particularly:
- `mmcv.ops.RoIAlign` (used in track_head roi_extractor)
- `mmcv.ops.nms` (used in tracker and detector)
- Deformable convolutions (if backbone uses them)

---

## Files Not Changed (and Why)

| File | Reason |
|------|--------|
| `mmdet/models/trackers/quasi_dense_tracker.py` | Already supports multi-class via `with_cats=True` (line 37-38) |
| `mmdet/models/tracking_heads/quasi_dense_track_head.py` | Class-agnostic embedding — works for any number of classes |
| `mmdet/models/tracking_heads/quasi_dense_embed_head.py` | Embedding head is class-agnostic |
| `mmdet/__init__.py` version checks | No change needed for Profile A (mmcv 2.1.0 is within bounds) |
| `requirements/tracking.txt` | No change needed — `numpy<1.24.0` is compatible with Profile A |
