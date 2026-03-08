#!/bin/bash
set -e

DATA=/root/autodl-tmp/PlantDoc.v4i.yolov11
OUT=/root/autodl-tmp/PlantDoc_CalmAug
CALM=/root/autodl-tmp/CALM-Aug

rm -rf $OUT
mkdir -p $OUT

echo "[0/6] Stats..."
python $CALM/stat_yolo_classes.py \
  --labels $DATA/train/labels --nc 30 \
  --out $OUT/class_stats.json --topk 12

echo "[1/6] Class-Aware Copy-Paste..."
python $CALM/calm_copy_paste_classaware.py \
  --images $DATA/train/images \
  --labels $DATA/train/labels \
  --stats $OUT/class_stats.json \
  --out $OUT/train \
  --seed 0 \
  --extreme_th 10 --tail_th 160 --mid_th 260 \
  --r_extreme 10 --r_tail 5 --r_mid 3 --r_head 0 \
  --max_paste 2 \
  --max_gen_total 9000

echo "[2/6] Photometric..."
python $CALM/calm_photometric.py \
  --images $OUT/train/images \
  --labels $OUT/train/labels \
  --out $OUT/train_p1 \
  --seed 0

echo "[3/6] Weather..."
python $CALM/calm_weather.py \
  --images $OUT/train_p1/images \
  --labels $OUT/train_p1/labels \
  --out $OUT/train_p2 \
  --seed 0

echo "[4/6] Occlusion..."
python $CALM/calm_occlusion.py \
  --images $OUT/train_p2/images \
  --labels $OUT/train_p2/labels \
  --out $OUT/train_final \
  --p 0.6 \
  --seed 0

echo "[5/6] Generate data_mix.yaml..."
python $CALM/make_data_mix.py \
  --src $DATA/data.yaml \
  --train_images $OUT/train_final/images \
  --out $OUT/data_mix.yaml

echo ""
echo "DONE."
echo "Train images: $OUT/train_final/images"
echo "Train labels: $OUT/train_final/labels"
echo "data_mix.yaml: $OUT/data_mix.yaml"
