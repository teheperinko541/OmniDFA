#!/bin/bash

GPU_NUM=4
N_WAY=5 # 5 or 15
K_SHOT=10
DATA_PATH="/share/project/OmniFake"

PART_3_LIST="datasets/split/part3/vallist.txt"
PART_2_LIST="datasets/split/part2/vallist.txt"
PART_1_LIST="datasets/split/part1/vallist.txt"

# Model trained on parts 1 & 2; evaluation is performed on part 3.
CKPT_PATH="ckpt/OmniDFA_part1_epoch[20].pth" 
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM eval_attribution.py --use_bf16 --n_way $N_WAY --k_shot $K_SHOT --data_path "$DATA_PATH" --fake_file_path "$PART_3_LIST" --ckpt_path "$CKPT_PATH"

# Model trained on parts 1 & 3; evaluation is performed on part 2.
# CKPT_PATH="ckpt/OmniDFA_part2_epoch[20].pth" 
# OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM eval_attribution.py --use_bf16 --n_way $N_WAY --k_shot $K_SHOT --data_path "$DATA_PATH" --fake_file_path "$PART_2_LIST" --ckpt_path "$CKPT_PATH"

# Model trained on parts 2 & 3; evaluation is performed on part 1.
# CKPT_PATH="ckpt/OmniDFA_part3_epoch[20].pth" 
# OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM eval_attribution.py --use_bf16 --n_way $N_WAY --k_shot $K_SHOT --data_path "$DATA_PATH" --fake_file_path "$PART_1_LIST" --ckpt_path "$CKPT_PATH"
