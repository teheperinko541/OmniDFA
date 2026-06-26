#!/bin/bash

GPU_NUM=4
DATA_PATH="/share/project/OmniFake"

PART_3_LIST="datasets/split/part3/vallist.txt"
PART_2_LIST="datasets/split/part2/vallist.txt"
PART_1_LIST="datasets/split/part1/vallist.txt"
REAL_PATH="datasets/split/part1/reallist.txt"

# Model trained on parts 1 & 2; evaluation is performed on part 3.
CKPT_PATH="OmniDFA_part1_epoch[20].pth" 
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM eval_detection.py --use_bf16 --data_path "$DATA_PATH" --fake_file_path "$PART_3_LIST" --real_file_path "$REAL_PATH" --ckpt_path "$CKPT_PATH"

# Model trained on parts 1 & 3; evaluation is performed on part 2.
# CKPT_PATH="OmniDFA_part2_epoch[20].pth" 
# OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM eval_detection.py --use_bf16 --data_path "$DATA_PATH" --fake_file_path "$PART_2_LIST" --real_file_path "$REAL_PATH" --ckpt_path "$CKPT_PATH"

# Model trained on parts 2 & 3; evaluation is performed on part 1.
# CKPT_PATH="OmniDFA_part3_epoch[20].pth" 
# OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM eval_detection.py --use_bf16 --data_path "$DATA_PATH" --fake_file_path "$PART_1_LIST" --real_file_path "$REAL_PATH" --ckpt_path "$CKPT_PATH"
