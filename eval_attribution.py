# -*- coding: utf-8 -*-
""" Few-shot attribution evaluation. 
"""

import torch
import argparse
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.omnifake_episode import OmniFakeEpisode
from model.model import TwinNeXt
from model.engine_val import eval_classification
from util.utils import setup_dist
import util.logger as logger
from util.utils import load_model


def get_args_parser():
    parser = argparse.ArgumentParser('Omni-AIGI-Detector evaluation for deepfake few-shot classification')
    # Base and distributed parameters
    parser.add_argument('--seed', default=0, type=int, help="random seed for the main process")
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to log')
    parser.add_argument('--n_way', default=5, type=int, help="N-way")
    parser.add_argument('--k_shot', default=10, type=int, help="K-shot")
    parser.add_argument('--n_query', default=5, type=int, help="number of queries of each class in one episode")
    parser.add_argument('--total_length', default=10000, type=int, help="total number of episodes in one epoch")
    parser.add_argument('--classification_rule', default="prototype", type=str, help='rule for few-shot flassification, choose in ["prototype", "nearest"]')

    # Arguments for distributed parallel which will be set with `setup_dist``
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes')
    parser.add_argument('--master_addr', type=str, default='env://', help='master address for multiprocessing')
    parser.add_argument('--master_port', type=str, default='23456', help='master port for multiprocessing')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed data parallel')
    parser.add_argument('--rank', type=int, default=-1, help='rank for distributed data parallel')
    parser.add_argument('--device', default='cuda', help='device to use for current process')

    # Dataset & dataloader parameters
    parser.add_argument('--data_path', default='data/OmniFake', type=str, help='dataset path')
    parser.add_argument('--fake_file_path', default='datasets/split/part1/reallist.txt', type=str, help='file path for fake class')

    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--batch_size', default=5, type=int, help='batch size per GPU')
    parser.add_argument('--num_workers', default=20, type=int, help='number of workers for dataloader')
    
    # calculation parameters
    parser.add_argument('--use_fp16', action='store_true', help='use fp16 during evaluation')
    parser.add_argument('--use_bf16', action='store_true', help='use bf16 during evaluation')

    # checkpoint path
    parser.add_argument('--ckpt_path', required=True, help='path of checkpoint')

    return parser.parse_args()


def main(args): 
    args.batch_size = args.n_way
    #################### prepare ####################
    setup_dist(args) # ddp setup
    # terminal writer and file writer
    logger.setup(log_dir=args.log_dir, device=args.device)
    #################################################

    
    ########## setup dataset and dataloader #########
    logger.info("Creating training dataloader and sampler...")
    fake_dataset = OmniFakeEpisode(
        data_root=args.data_path, 
        class_file_path=args.fake_file_path,  
        mode="val", 
        output_size=args.input_size, 
        n_way=args.n_way, 
        k_shot=args.k_shot, 
        n_query=args.n_query, 
        total_length=args.total_length,
        world_size=args.world_size, 
    )
    fake_sampler = DistributedSampler(fake_dataset, shuffle=False)
    fake_dataloader = DataLoader(
        fake_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=fake_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    #################################################

    
    ############ create model & criterion ###########
    logger.info("Creating model 'TwinNeXt'... ")
    model = TwinNeXt(
        backbone_name="convnext_small", 
        mlp_hidden_dims=512, 
        out_dim=128, 
    )

    load_model(args.ckpt_path, model=model)

    model.to(args.device)
    model = DDP(model, device_ids=[args.local_rank])

    # dtype
    if args.use_fp16: 
        dtype = torch.float16
    elif args.use_bf16: 
        dtype = torch.bfloat16
    else: 
        dtype = torch.float32
    #################################################

    ################### evaluation ##################
    logger.info("Start classification evaluation. ")
    eval_classification(
        model=model, 
        dataloaders=[fake_dataloader],
        n_way=args.n_way, 
        k_shot=args.k_shot, 
        n_query=args.n_query, 
        device=args.device, 
        dtype=dtype, 
        rule=args.classification_rule, 
    )

    dist.destroy_process_group()
    #################################################


if __name__ == '__main__': 
    args = get_args_parser()
    main(args)

