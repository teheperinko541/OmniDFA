#!/bin/bash

GPU_NUM=8
DATA_PATH="/share/project/OmniFake"

PART_3_LIST="datasets/split/part3/trainlist.txt"
PART_2_LIST="datasets/split/part2/trainlist.txt"
PART_1_LIST="datasets/split/part1/trainlist.txt"
PART_ZERO_SHOT="datasets/split/zero-shot/trainlist.txt"
REAL_PATH="datasets/split/part1/reallist.txt"

# Model trained on parts 1 & 2; evaluation will be performed on part 3.
CKPT_PATH="your_ckpt_path" 
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM train.py --use_bf16 --total_epochs 20 --data_path "$DATA_PATH" --fake_file_path "$PART_3_LIST" --real_file_path "$REAL_PATH"

# Model trained on parts 1 & 3; evaluation will be performed on part 2.
# OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM train.py --use_bf16 --total_epochs 20 --data_path "$DATA_PATH" --fake_file_path "$PART_2_LIST" --real_file_path "$REAL_PATH"

# Model trained on parts 2 & 3; evaluation will be performed on part 1.
# OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM train.py --use_bf16 --total_epochs 20 --data_path "$DATA_PATH" --fake_file_path "$PART_1_LIST" --real_file_path "$REAL_PATH"

# for zero-shot evaluation
# OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM train.py --use_bf16 --total_epochs 20 --data_path "$DATA_PATH" --fake_file_path "$PART_1_LIST" --real_file_path "$REAL_PATH"
