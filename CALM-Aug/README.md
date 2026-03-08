# CALM-Aug

Class-Aware Long-tail Mitigation Augmentation for YOLO-based Plant Disease Detection

## Introduction

CALM-Aug is a lightweight, class-aware data augmentation framework designed to alleviate long-tail distribution and domain shift problems in agricultural object detection datasets such as PlantDoc.

It is specifically designed for YOLO-based detection pipelines and focuses on:

- Class imbalance mitigation
- Rare class reinforcement
- Domain robustness enhancement
- Lightweight offline augmentation (no GAN required)

CALM-Aug generates a new balanced training dataset through a structured multi-stage augmentation pipeline.

## Key Features

- Class-aware copy-paste augmentation
- Long-tail adaptive repetition strategy
- Photometric robustness enhancement
- Weather simulation augmentation
- Occlusion simulation
- Fully offline preprocessing (safe and reproducible)
- Directly compatible with YOLO format

## Directory Structure

```
CALM-Aug/
│
├── run_calm_aug.sh                # Main execution script
├── stat_yolo_classes.py           # Dataset class statistics
├── calm_copy_paste_classaware.py  # Class-aware copy-paste module
├── calm_photometric.py            # Color & illumination augmentation
├── calm_weather.py                # Rain/fog simulation
├── calm_occlusion.py              # Random occlusion augmentation
├── make_data_mix.py               # Generate YOLO data.yaml
└── filter_by_teacher.py           # Advanced filtering module (optional)
```

## Environment Requirements

### Python Version

Recommended: Python >= 3.9

Compatible with your current YOLO environment (Python 3.10 works perfectly).

### Dependencies

Install:

```bash
pip install -r requirements.txt
```

If already in YOLO environment:

```bash
pip install opencv-python numpy tqdm pyyaml
```

## Method Overview

CALM-Aug contains 4 major stages:

### Stage 1 - Class Statistics

Scan YOLO labels and compute:
- Instance count per class
- Sorted long-tail distribution
- Automatic tail-class identification

### Stage 2 - Class-Aware Copy-Paste

Rare classes are amplified using adaptive repetition:

```python
repeat_factor = determined by class frequency thresholds
```

Classes are categorized into four tiers:
- **Extreme tail** (< extreme_th): `r_extreme` repetitions
- **Tail** (< tail_th): `r_tail` repetitions
- **Mid** (< mid_th): `r_mid` repetitions
- **Head** (>= mid_th): `r_head` repetitions (usually 0)

Rare objects are copy-pasted into:
- Random background positions
- Non-overlapping regions
- Multi-object mixing scenarios

### Stage 3 - Photometric Augmentation

Simulates real agricultural variations:
- Brightness/contrast adjustment
- Gamma correction (low-light/over-exposure)
- Gaussian blur
- JPEG compression artifacts

### Stage 4 - Weather & Occlusion Simulation

Simulated environmental noise:
- Haze layers - Fog simulation
- Shadow simulation - Random polygonal shadows
- Color shift - Temperature variation
- Cutout occlusion - Random rectangular masking

## Directory Structure Requirements

Input data structure:

```
PlantDoc/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

CALM-Aug output structure:

```
PlantDoc_CalmAug/
├── train/                          # Class-Aware CopyPaste
├── train_p1/                       # Photometric
├── train_p2/                       # Weather
├── train_final/                    # Occlusion
├── train_mix/                      # Original + Augmented Mix
└── data_final.yaml                 # YOLO training config
```

## Usage

### Step 1 - Modify Dataset Paths

Edit `run_calm_aug.sh`:

```bash
DATA=/root/autodl-tmp/PlantDoc.v4i.yolov11
OUT=/root/autodl-tmp/PlantDoc_CalmAug
CALM=/root/autodl-tmp/CALM-Aug
```

### Step 2 - Run Augmentation

**One-step execution (recommended):**

```bash
bash run_calm_aug.sh
```

**Pipeline flow:**

```
[0/6] Stats
[1/6] Class-Aware Copy-Paste
[2/6] Photometric Augmentation
[3/6] Weather Augmentation
[4/6] Occlusion Augmentation
[5/6] Generate data_final.yaml
```

## Data Generation Explanation

| Stage | Purpose |
|-------|---------|
| Class-Aware Copy-Paste | Tail class resampling |
| Photometric | Illumination perturbation |
| Weather | Simulate real field conditions |
| Occlusion | Simulate leaf occlusion |

## Training

After augmentation is complete:

**Command line:**

```bash
yolo detect train \
  data=/root/autodl-tmp/PlantDoc_CalmAug/data_final.yaml \
  model=yolo11.yaml \
  epochs=100 \
  imgsz=640 \
  batch=32
```

**Python script:**

```python
from ultralytics import YOLO

model = YOLO("yolo11.yaml")
model.load("yolo11n.pt")

model.train(
    data="/root/autodl-tmp/PlantDoc_CalmAug/data_final.yaml",
    imgsz=640,
    epochs=100,
    batch=32,
    optimizer="SGD",
    cos_lr=True
)
```

## Teacher Filtering (Optional, Advanced)

If using teacher-guided filtering:

```bash
python filter_by_teacher.py \
  --teacher best.pt \
  --images train_final/images \
  --labels train_final/labels \
  --out train_final_filt \
  --conf 0.25
```

## Configurable Parameters

Adjust in `run_calm_aug.sh` or script internals:

| Parameter | Meaning |
|-----------|---------|
| `tail_ratio` | Threshold for tail class identification |
| `repeat_map` | Copy multiplier for each class |
| `max_gen_total` | Maximum number of generated images |
| `conf` | Teacher confidence threshold |

## Resource Requirements

| Item | Requirement |
|------|-------------|
| CPU | 4 cores or more |
| Memory | >= 8GB |
| Disk | 10GB+ |
| GPU | Not required (augmentation is CPU-based) |

## Expected Benefits

Compared with baseline training:

| Improvement | Effect |
|-------------|--------|
| Long-tail balancing | ↑ Recall for rare classes |
| Domain simulation | ↑ Generalization |
| Occlusion robustness | ↑ Real-world detection |
| Class-aware repetition | ↑ mAP50-95 stability |

## Design Philosophy

CALM-Aug avoids:
- Heavy GAN training
- Expensive teacher-student frameworks
- Unstable synthetic generation

Instead, it uses:
- Structured distribution balancing
- Deterministic offline augmentation
- Lightweight reproducibility

## FAQ

**Q: "duplicate labels removed" warning?**

A: Normal behavior - YOLO automatically removes duplicates, does not affect training.

**Q: Data not generated?**

A: Check if `PlantDoc_CalmAug/train_final/images` exists and contains images.

**Q: Classes still imbalanced?**

A: Increase `max_gen_total` or adjust `repeat_map` values.

## Recommended Use Cases

- Agricultural disease detection
- Long-tail object detection datasets
- YOLO-based detection pipelines
- Low-resource domain adaptation