# QDTrack on BDD100K: Detailed List of Required Changes

This document itemizes every code change, new file, and configuration needed
to train and evaluate QDTrack on the BDD100K Multi-Object Tracking benchmark
using this mmdetection v3.3.0 repository.

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

### File to create: `mmdet/datasets/bdd100k_dataset.py`

**Why**: The existing `MOTChallengeDataset` (`mmdet/datasets/mot_challenge_dataset.py`)
has MOT-specific fields (`visibility`, `mot_conf` at lines 80-81) that don't exist
in BDD100K annotations. It also has a 13-class `METAINFO` that doesn't match BDD100K.

**Pattern to follow**: `MOTChallengeDataset` is a thin subclass of `BaseVideoDataset`
(~89 lines). The BDD100K version should be similarly small.

**BDD100K MOT categories (8 classes)**:
1. pedestrian
2. rider
3. car
4. truck
5. bus
6. train
7. motorcycle
8. bicycle

**Key implementation details**:
- Subclass `BaseVideoDataset` (`mmdet/datasets/base_video_dataset.py`)
- Define `METAINFO` with the 8 BDD100K categories
- Override `parse_data_info()` to:
  - Extract `instance_id` from each annotation (required for tracking)
  - Extract `bbox` and convert to xyxy format if needed
  - Extract `category_id` and map to `bbox_label`
  - Skip the `visibility` and `mot_conf` fields (not in BDD100K)
  - Handle BDD100K's `iscrowd` / `ignore` flags if present
- Register with `@DATASETS.register_module()`

**Estimated code**: ~60-80 lines.

---

## Change 2: Register BDD100K Dataset in `__init__.py`

### File to modify: `mmdet/datasets/__init__.py`

**Current state** (line 21):
```python
from .mot_challenge_dataset import MOTChallengeDataset
```

**Add** (after line 21):
```python
from .bdd100k_dataset import BDD100KDataset
```

**Also add** `'BDD100KDataset'` to the `__all__` list (line 46).

---

## Change 3: Fix Multi-Class Label Handling in QDTrack Model

### File to modify: `mmdet/models/mot/qdtrack.py`

**Problem** (lines 137-139):
```python
key_data_sample = track_data_sample.get_key_frames()[0]
key_data_sample.gt_instances.labels = \
    torch.zeros_like(key_data_sample.gt_instances.labels)
key_data_samples.append(key_data_sample)
```

This zeros out ALL ground truth class labels. The comment on line 133 says
"set cat_id of gt_labels to 0 in RPN", but the zeroed `key_data_samples` are
also passed to `roi_head.loss()` on line 176-177:

```python
losses_detect = self.detector.roi_head.loss(x, rpn_results_list,
                                            key_data_samples, **kwargs)
```

For MOT17 (1 class), this is harmless — all objects are class 0 anyway.
For BDD100K (8 classes), this destroys all class information, making the
detector treat every object as class 0.

**Fix** (replace lines 137-140):
```python
key_data_sample = track_data_sample.get_key_frames()[0]
# Save original labels for roi_head (multi-class detection)
original_labels = key_data_sample.gt_instances.labels.clone()
# Zero labels for RPN (class-agnostic proposal generation)
key_data_sample.gt_instances.labels = \
    torch.zeros_like(key_data_sample.gt_instances.labels)
key_data_samples.append(key_data_sample)
```

Then, between the RPN forward pass (line 166) and the roi_head loss (line 176),
restore the original labels:

```python
# Restore original class labels for roi_head
for key_data_sample, track_data_sample in zip(key_data_samples, data_samples):
    original_key = track_data_sample.get_key_frames()[0]
    key_data_sample.gt_instances.labels = original_key.gt_instances.labels
```

**Alternative approach**: Since the save/restore approach above creates
coupling with `get_key_frames()` being called twice, a cleaner approach is
to save the labels in a list before the loop:

```python
saved_labels = []
for track_data_sample in data_samples:
    ...
    key_data_sample = track_data_sample.get_key_frames()[0]
    saved_labels.append(key_data_sample.gt_instances.labels.clone())
    key_data_sample.gt_instances.labels = \
        torch.zeros_like(key_data_sample.gt_instances.labels)
    key_data_samples.append(key_data_sample)
    ...

# After RPN forward, before roi_head:
for kds, orig_labels in zip(key_data_samples, saved_labels):
    kds.gt_instances.labels = orig_labels
```

**Backward compatibility**: This change is safe for MOT17 too — saving and
restoring all-zero labels is a no-op.

---

## Change 4: BDD100K Base Dataset Config

### File to create: `configs/_base_/datasets/bdd100k_track.py`

**Pattern to follow**: `configs/_base_/datasets/mot_challenge.py` (91 lines).

**Key differences from MOT Challenge config**:

| Setting | MOT Challenge | BDD100K |
|---------|--------------|---------|
| `dataset_type` | `'MOTChallengeDataset'` | `'BDD100KDataset'` |
| `data_root` | `'data/MOT17/'` | `'data/bdd100k/'` |
| `ann_file` (train) | `'annotations/half-train_cocoformat.json'` | `'annotations/box_track_train_cocoformat.json'` |
| `ann_file` (val) | `'annotations/half-val_cocoformat.json'` | `'annotations/box_track_val_cocoformat.json'` |
| `data_prefix` | `dict(img_path='train')` | `dict(img_path='images/track/train')` |
| `metainfo.classes` | `('pedestrian',)` | All 8 BDD100K classes |
| `visibility_thr` | `-1` | N/A (not applicable to BDD100K) |
| `img_scale` | `(1088, 1088)` | `(1280, 720)` — BDD100K native resolution |

**Note on image scale**: BDD100K images are 1280x720 (dashboard camera).
The training pipeline can resize/crop these, but the base scale should
reflect the native resolution. Consider `(1296, 736)` for divisibility by 32.

---

## Change 5: BDD100K QDTrack Training Config

### File to create: `configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py`

**Pattern to follow**: `configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py`

**Key differences from MOT17 config**:

1. **Base configs**: Inherit from `bdd100k_track.py` instead of `mot_challenge.py`

2. **Detector num_classes**: Change from `1` to `8`
   - In `configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_base.py` line 18:
     ```python
     detector.roi_head.bbox_head.update(dict(num_classes=1))
     ```
   - Override in the BDD100K config:
     ```python
     model = dict(
         detector=dict(
             roi_head=dict(
                 bbox_head=dict(num_classes=8))))
     ```

3. **Pretrained weights**: The MOT17 config uses a COCO-person-only pretrained
   Faster R-CNN (`faster_rcnn_r50_fpn_1x_coco-person`). For BDD100K with
   8 diverse categories, use the standard COCO-pretrained model instead:
   ```python
   model = dict(
       detector=dict(
           init_cfg=dict(
               type='Pretrained',
               checkpoint='https://download.openmmlab.com/mmdetection/v2.0/'
               'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/'
               'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')))
   ```

4. **Evaluator**: Use `CocoVideoMetric` with `classwise=True` for per-class AP,
   plus `MOTChallengeMetric` for tracking metrics. Or use BDD100K-specific
   evaluation (see Change 7).

5. **Training schedule**: May increase epochs from 4 to 8-12 since BDD100K
   is larger (1400 train videos vs MOT17's 7 sequences). Adjust learning
   rate schedule accordingly:
   ```python
   param_scheduler = [
       dict(type='MultiStepLR', begin=0, end=8, by_epoch=True, milestones=[6])
   ]
   train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=4)
   ```

---

## Change 6: BDD100K Annotation Converter (Optional)

### File to create: `tools/dataset_converters/bdd100k2coco.py`

**Why**: The `bdd100k` pip package provides `bdd100k.label.to_coco` for
conversion. However, the output may need post-processing to match what
`BaseVideoDataset` expects (specifically `instance_id`, `video_id`, and
`frame_id` fields in the COCO JSON).

**If the bdd100k converter output is compatible**: No converter needed. Just
verify the JSON structure.

**If not compatible**: Write a script that:
1. Reads BDD100K Scalabel JSON
2. Outputs COCO-format JSON with `images[].video_id`, `images[].frame_id`,
   `annotations[].instance_id`, and `videos[]` list
3. Maps BDD100K category names to integer IDs (0-7)

**Estimated code**: 50-100 lines if needed.

---

## Change 7: BDD100K Evaluation Metric (Optional)

### File to create: `mmdet/evaluation/metrics/bdd100k_mot_metric.py`

**Why**: BDD100K has its own evaluation protocol that computes multi-class
MOTA, MOTP, and IDF1 differently from MOTChallenge.

**Options** (in order of effort):
1. **Lowest effort**: Use existing `MOTChallengeMetric` — gives approximate
   results but won't match BDD100K leaderboard exactly
2. **Medium effort**: Post-hoc evaluation — save results to BDD100K format
   and use `bdd100k` CLI tools to evaluate
3. **Highest effort**: Write a custom `BDD100KMOTMetric` that calls the
   `bdd100k.evaluation` API inside the mmdet evaluation loop

**Recommendation**: Start with option 1 for development/debugging, then
use option 2 for final numbers.

---

## Summary: All Changes at a Glance

| # | Type | File | Lines Changed | Effort |
|---|------|------|---------------|--------|
| 1 | **New** | `mmdet/datasets/bdd100k_dataset.py` | ~70 new | 2-4 hours |
| 2 | **Edit** | `mmdet/datasets/__init__.py` | 2 lines | 5 minutes |
| 3 | **Edit** | `mmdet/models/mot/qdtrack.py` | ~10 lines | 1 hour |
| 4 | **New** | `configs/_base_/datasets/bdd100k_track.py` | ~90 new | 1-2 hours |
| 5 | **New** | `configs/qdtrack/qdtrack_..._bdd100k.py` | ~30 new | 1-2 hours |
| 6 | **New** (optional) | `tools/dataset_converters/bdd100k2coco.py` | ~80 new | 1-2 hours |
| 7 | **New** (optional) | `mmdet/evaluation/metrics/bdd100k_mot_metric.py` | ~150 new | 2-4 hours |

**Total code changes**: ~5 files modified/created (required), +2 optional.
**Core engineering effort**: ~1-2 days for required changes + testing.

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
