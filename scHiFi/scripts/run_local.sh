#!/usr/bin/env bash
python main.py \
  --dataset Bach \
  --train_mode local \
  --local_batch_size 256 \
  --k_list 10 15 20 25 \
  --pretrain_epochs 400 \
  --maxiter 150 \
  --lr_fit 0.5 \
  --device cuda
