#!/usr/bin/env bash
python main.py \
  --dataset Quake_10x_Limb_Muscle \
  --train_mode full \
  --k_list 10 15 20 25 \
  --pretrain_epochs 400 \
  --maxiter 150 \
  --lr_fit 0.5 \
  --device cuda
