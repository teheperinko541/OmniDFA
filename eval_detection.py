# -*- coding: utf-8 -*-
""" Deepfake detection evaluation. 
"""

import torch
import argparse
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.omnifake import OmniFakeDataset
from model.model import TwinNeXt
from model.loss import SphereCenterLoss
from model.engine_val import eval_authenticity
from util.utils import setup_dist
import util.logger as logger
from util.utils import load_model


def get_args_parser():
    parser = argparse.ArgumentParser('Omni-AIGI-Detector training for deepfake detection')
    # Base and distributed training parameters
    parser.add_argument('--seed', default=0, type=int, help="random seed for the main process")
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, help='path where to log')

    # Arguments for distributed parallel which will be set with `setup_dist``
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes')
    parser.add_argument('--master_addr', type=str, default='env://', help='master address for multiprocessing')
    parser.add_argument('--master_port', type=str, default='23456', help='master port for multiprocessing')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed data parallel')
    parser.add_argument('--rank', type=int, default=-1, help='rank for distributed data parallel')
    parser.add_argument('--device', default='cuda', help='device to use for current process')

    # Dataset & dataloader parameters
    parser.add_argument('--data_path', default='data/OmniFake', type=str, help='dataset path')
    parser.add_argument('--fake_file_path', default='datasets/split/part1/vallist.txt', type=str, help='file path for fake class')
    parser.add_argument('--real_file_path', default='datasets/split/part1/reallist.txt', type=str, help='file path for real class')
    
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--batch_size_fake', default=64, type=int, help='batch size per GPU')
    parser.add_argument('--batch_size_real', default=64, type=int, help='batch size per GPU')
    parser.add_argument('--num_workers', default=12, type=int, help='number of workers for dataloader')
    
    # calculation parameters
    parser.add_argument('--use_fp16', action='store_true', help='use fp16 during training')
    parser.add_argument('--use_bf16', action='store_true', help='use bf16 during training')

    # checkpoint path
    parser.add_argument('--ckpt_path', required=True, help='path of checkpoint')

    return parser.parse_args()


def main(args): 
    #################### prepare ####################
    setup_dist(args) # ddp setup

    # terminal writer and file writer
    logger.setup(log_dir=args.log_dir, device=args.device)
    #################################################

    
    ########## setup dataset and dataloader #########
    logger.info("Creating training dataloader and sampler...")

    fake_dataset = OmniFakeDataset(
        data_root=args.data_path, 
        class_file_path=args.fake_file_path,  
        mode="val", 
        output_size=args.input_size
    )
    fake_sampler = DistributedSampler(fake_dataset)
    fake_dataloader = DataLoader(
        fake_dataset,
        batch_size=args.batch_size_fake,
        shuffle=fake_sampler is None,
        sampler=fake_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    real_dataset = OmniFakeDataset(
        data_root=args.data_path,
        class_file_path=args.real_file_path,  
        mode="val", 
        output_size=args.input_size
    )
    real_sampler = DistributedSampler(real_dataset)
    real_dataloader = DataLoader(
        real_dataset,
        batch_size=args.batch_size_real,
        shuffle=real_sampler is None,
        sampler=real_sampler,
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
    center_loss_model = SphereCenterLoss(feat_dim=128)
    load_model(args.ckpt_path, model=model, cl_model=center_loss_model)

    model.to(args.device)
    model = DDP(model, device_ids=[args.local_rank])
    
    center_loss_model = center_loss_model.to(args.device)
    center_loss_model = DDP(center_loss_model, device_ids=[args.local_rank])

    # dtype
    if args.use_fp16: 
        dtype = torch.float16
    elif args.use_bf16: 
        dtype = torch.bfloat16
    else: 
        dtype = torch.float32
    #################################################

    ################### evaluation ##################
    logger.info("Start authenticity evaluation. ")

    # starts looping
    eval_authenticity(
        model=model, 
        dataloaders=[fake_dataloader, real_dataloader], 
        criterions=[center_loss_model], 
        device=args.device, 
        dtype=dtype, 
    )

    dist.destroy_process_group()
    #################################################


if __name__ == '__main__': 
    args = get_args_parser()
    main(args)

