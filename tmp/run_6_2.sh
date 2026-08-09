#!/usr/bin/env bash
set -euo pipefail

# WBS 6-2: multi-GPU training launcher
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29510 python train.py -C=r3d200 epochs=5 fold=999 save_weights=True
