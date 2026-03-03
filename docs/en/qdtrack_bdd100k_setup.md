# QDTrack on BDD100K: Environment Setup Commands

This document provides step-by-step commands for setting up the environment
to train and benchmark QDTrack on BDD100K using **Profile A** hardware
(V100, A100, Ada RTX 4500) with PyTorch 2.1.2 and CUDA 12.1.

> **Scope**: This guide covers V100, A100, and Ada RTX 4500 GPUs.
> RTX 5080 (Blackwell) requires a separate Profile B environment
> with PyTorch 2.6+ and CUDA 12.8 — see the changes document for details.

---

## 1. Prerequisites

### 1a. Verify NVIDIA Driver

```bash
nvidia-smi
# Required: driver version >= 525 (for CUDA 12.1 runtime support)
# Verify the "CUDA Version" field shows 12.1 or higher
```

### 1b. Verify conda is installed

```bash
conda --version
# If not installed: https://docs.conda.io/en/latest/miniconda.html
```

---

## 2. Create and Activate Conda Environment

```bash
conda create -n qdtrack python=3.10 -y
conda activate qdtrack
```

---

## 3. Install PyTorch 2.1.2 with CUDA 12.1

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### Verify PyTorch + CUDA

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## 4. Install mmcv and mmengine

```bash
pip install openmim
mim install mmcv==2.1.0
pip install mmengine==0.10.4
```

### Verify mmcv CUDA ops

```bash
python -c "
from mmcv.ops import nms, roi_align
print('mmcv CUDA ops: OK')
"
```

---

## 5. Install mmdetection (editable mode)

```bash
cd /home/user/mmdetection
pip install -e .
```

### Verify mmdetection

```bash
python -c "
import mmdet
print(f'mmdetection: {mmdet.__version__}')
"
# Expected output: mmdetection: 3.3.0
```

---

## 6. Install Tracking Dependencies

```bash
pip install motmetrics mmpretrain "numpy<1.24.0" scikit-learn seaborn
pip install "git+https://github.com/JonathonLuiten/TrackEval.git"
```

---

## 7. Install BDD100K Tools

Install from **GitHub** (not PyPI) to get bug fixes for empty-frame handling:

```bash
pip install "git+https://github.com/bdd100k/bdd100k.git"
# This also installs scalabel from GitHub master as a dependency
```

> **Note on BDD100K label format**: BDD100K labels were revised in 2020 from
> the old 2018 format to the "Scalabel format". The labels downloaded today
> (`box_track_20`) use the 2020 Scalabel format where label `id` is a string
> (not int32). The `bdd100k` Python package's `to_coco` command handles this
> format and produces CocoVID-style JSON compatible with mmdetection's
> `BaseVideoDataset`.
>
> **Why install from GitHub**: The PyPI release (`bdd100k==1.0.1`) has not
> been updated in over a year and crashes on training frames that lack a
> `labels` key (`KeyError: 'labels'`, see
> [bdd100k#209](https://github.com/bdd100k/bdd100k/issues/209),
> [bdd100k#225](https://github.com/bdd100k/bdd100k/discussions/225)).
> The GitHub master branch has this fix.

---

## 8. Download and Prepare BDD100K Dataset

### 8a. Download from BDD100K website

Register and download from https://bdd-data.berkeley.edu/:
- MOT 2020 images (track split)
- MOT 2020 labels (`box_track_20` — these are in the 2020 Scalabel format)

### 8b. Organize directory structure

```bash
mkdir -p data/bdd100k/images/track/{train,val,test}
mkdir -p data/bdd100k/labels/box_track_20/{train,val}
mkdir -p data/bdd100k/annotations

# Unpack downloaded archives into the above directories:
# - Images go into data/bdd100k/images/track/{train,val,test}/
# - Labels go into data/bdd100k/labels/box_track_20/{train,val}/
```

Expected structure:
```
data/bdd100k/
├── images/
│   └── track/
│       ├── train/    # 1400 video sequence folders
│       ├── val/      # 200 video sequence folders
│       └── test/     # 400 video sequence folders
└── labels/
    └── box_track_20/
        ├── train/    # Per-video JSON files in Scalabel format
        └── val/      # Per-video JSON files in Scalabel format
```

### 8c. Convert BDD100K annotations to COCO format

The `to_coco -m box_track` command converts Scalabel-format labels to
CocoVID-style JSON with `instance_id` (cross-frame identity), `video_id`
(linking frames to videos), and standard COCO fields.

```bash
python -m bdd100k.label.to_coco \
    -m box_track \
    -i data/bdd100k/labels/box_track_20/train/ \
    -o data/bdd100k/annotations/box_track_train_cocoformat.json

python -m bdd100k.label.to_coco \
    -m box_track \
    -i data/bdd100k/labels/box_track_20/val/ \
    -o data/bdd100k/annotations/box_track_val_cocoformat.json
```

### 8d. Verify annotations

Check that the converter output has the required tracking fields:

```bash
python -c "
import json
with open('data/bdd100k/annotations/box_track_train_cocoformat.json') as f:
    data = json.load(f)

# Verify required tracking fields
assert any('video_id' in img for img in data['images']), \
    'Missing video_id on images — converter output incompatible'
assert any('instance_id' in ann for ann in data['annotations'][:100]), \
    'Missing instance_id on annotations — tracking will not work'

print(f'Images: {len(data[\"images\"])}')
print(f'Annotations: {len(data[\"annotations\"])}')
print(f'Categories: {[c[\"name\"] for c in data[\"categories\"]]}')
# Expected categories: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle

# Check a sample annotation
sample = data['annotations'][0]
print(f'Sample annotation keys: {list(sample.keys())}')
# Should include: id, image_id, category_id, bbox, area, instance_id, ...
"
```

---

## 9. Training

### 9a. Single-GPU (for debugging)

```bash
python tools/train.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py
```

### 9b. Multi-GPU

```bash
bash tools/dist_train.sh \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    8
```

### 9c. Quick sanity check (1 epoch only)

```bash
python tools/train.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    --cfg-options train_cfg.max_epochs=1
```

---

## 10. Inference and Benchmarking

### 10a. Evaluate on validation set

```bash
python tools/test_tracking.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    --checkpoint work_dirs/qdtrack_bdd100k/latest.pth
```

### 10b. Benchmark FPS per GPU

Run on each target GPU separately:

```bash
# V100
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    --checkpoint work_dirs/qdtrack_bdd100k/latest.pth

# A100
CUDA_VISIBLE_DEVICES=1 python tools/analysis_tools/benchmark.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    --checkpoint work_dirs/qdtrack_bdd100k/latest.pth

# Ada RTX 4500
CUDA_VISIBLE_DEVICES=2 python tools/analysis_tools/benchmark.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    --checkpoint work_dirs/qdtrack_bdd100k/latest.pth
```

### 10c. Single video demo

```bash
python demo/mot_demo.py \
    path/to/bdd100k/video.mp4 \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    --checkpoint work_dirs/qdtrack_bdd100k/latest.pth \
    --out output.mp4
```

### 10d. Torch profiler (optional, for detailed hardware comparison)

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Run inference on one batch here
    pass

prof.export_chrome_trace("trace.json")
# Open chrome://tracing in Chrome browser to visualize
```

---

## 11. Validating Data Before Training

Before training, run the validation script to check that data files,
annotation structure, and the dataset class all work correctly:

```bash
# Basic validation (checks files, JSON structure, categories, pipeline)
python tools/analysis_tools/mot/validate_bdd100k_data.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py

# Also check a sample of images can be read
python tools/analysis_tools/mot/validate_bdd100k_data.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    --check-images --max-samples 20

# Validate the val split instead of train
python tools/analysis_tools/mot/validate_bdd100k_data.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    --split val
```

The script checks five layers and reports actionable error messages:
1. **File existence** — data_root, annotation file, image directory
2. **JSON structure** — required fields, ID uniqueness, referential integrity
3. **Category consistency** — JSON categories vs config METAINFO
4. **Image accessibility** (optional) — file existence, decodability, size match
5. **Pipeline smoke test** — dataset class instantiation, data structure

You can also run the BDD100K dataset unit test to verify the dataset class:

```bash
python -m pytest tests/test_datasets/test_bdd100k_dataset.py -v
```

---

## 12. Full Verification Checklist

```bash
# 1. Environment
python -c "import mmdet; print(mmdet.__version__); import torch; print(torch.cuda.get_device_name(0))"

# 2. CUDA ops
python -c "from mmcv.ops import nms, roi_align; print('OK')"

# 3. BDD100K dataset unit test
python -m pytest tests/test_datasets/test_bdd100k_dataset.py -v

# 4. Existing QDTrack tests still pass
python -m pytest tests/test_models/test_mot/test_qdtrack.py -v

# 5. Data validation (after downloading and converting data)
python tools/analysis_tools/mot/validate_bdd100k_data.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    --check-images

# 6. Training sanity check (1 epoch)
python tools/train.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
    --cfg-options train_cfg.max_epochs=1
```

---

## 13. Docker

A multi-stage Dockerfile is provided at `qdtrack.docker` in the repository
root. It uses `nvidia/cuda:12.1.0-devel-ubuntu22.04` to compile mmcv CUDA
extensions, then copies the result into a smaller
`nvidia/cuda:12.1.0-runtime-ubuntu22.04` image (~1.5 GB smaller than keeping
the compiler toolchain).

### 13a. Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- NVIDIA driver >= 525 (for CUDA 12.1 runtime support)

### 13b. Create a `.dockerignore`

Before building, create a `.dockerignore` next to `qdtrack.docker` to exclude
large directories from the build context:

```
.git
data/
work_dirs/
__pycache__/
*.pyc
*.egg-info
```

Without this, `COPY . /opt/mmdetection` sends the `.git` history and any
local datasets into the build, inflating the context by gigabytes.

### 13c. Build the image

```bash
docker build -f qdtrack.docker -t qdtrack:latest .
```

### 13d. Mount the BDD100K dataset at runtime

Use **bind mounts** (`-v`), not Docker named volumes, for the dataset.

On Linux, bind mounts and Docker volumes have identical I/O performance —
both are directories on the host filesystem with no virtualization layer.
(Docker volumes are faster only on macOS/Windows Docker Desktop, where bind
mounts cross the VM boundary via VirtioFS.) Bind mounts are preferred because:

- **No data duplication.** The 7–10 GB dataset stays in place on the host.
  Docker named volumes live under `/var/lib/docker/volumes/`, which is
  typically on the root partition — copying a large dataset there risks
  filling it.
- **Easier to manage.** You can browse, update, and share the data across
  containers without `docker cp` or `docker volume` commands.

Mount the dataset **read-only** (`:ro`) since training only reads images.
Mount `work_dirs` **read-write** for checkpoints, logs, and saved models.

Expected dataset layout on the host (before mounting):

```
/path/to/bdd100k/
├── images/track/{train,val,test}/     # video-sequence directories
├── labels/box_track_20/{train,val}/   # Scalabel JSON label files
└── annotations/                       # created by to_coco conversion
    ├── box_track_train_cocoformat.json
    └── box_track_val_cocoformat.json
```

### 13e. Run: convert BDD100K labels (one-time)

If you haven't already converted the labels on the host (section 8c), you
can run the conversion inside the container. Omit `:ro` on this run since
the conversion writes to the `annotations/` directory:

```bash
docker run --gpus all \
    -v /path/to/bdd100k:/opt/mmdetection/data/bdd100k \
    qdtrack:latest \
    bash -c '
        python -m bdd100k.label.to_coco -m box_track \
            -i data/bdd100k/labels/box_track_20/train/ \
            -o data/bdd100k/annotations/box_track_train_cocoformat.json && \
        python -m bdd100k.label.to_coco -m box_track \
            -i data/bdd100k/labels/box_track_20/val/ \
            -o data/bdd100k/annotations/box_track_val_cocoformat.json'
```

### 13f. Run: single-GPU training

```bash
docker run --gpus all --shm-size=8g \
    -v /path/to/bdd100k:/opt/mmdetection/data/bdd100k:ro \
    -v /path/to/work_dirs:/opt/mmdetection/work_dirs \
    qdtrack:latest \
    python tools/train.py \
        configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py
```

### 13g. Run: multi-GPU training

```bash
docker run --gpus all --shm-size=8g \
    -v /path/to/bdd100k:/opt/mmdetection/data/bdd100k:ro \
    -v /path/to/work_dirs:/opt/mmdetection/work_dirs \
    qdtrack:latest \
    bash tools/dist_train.sh \
        configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py 8
```

### 13h. Run: data validation

```bash
docker run --gpus all \
    -v /path/to/bdd100k:/opt/mmdetection/data/bdd100k:ro \
    qdtrack:latest \
    python tools/analysis_tools/mot/validate_bdd100k_data.py \
        configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_bdd100k.py \
        --check-images
```

### 13i. Note on `--shm-size`

`--shm-size=8g` is required. PyTorch DataLoader workers communicate through
`/dev/shm` (shared memory), and Docker's default of 64 MB causes "bus error"
crashes. 8 GB is generous enough for 2–4 workers per GPU. An alternative is
`--ipc=host`, which shares the host's `/dev/shm` directly.
