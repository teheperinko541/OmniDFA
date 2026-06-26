# -*- coding: utf-8 -*-
""" Train network for detection and attribution. 
"""

import torch
import argparse
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler

from datasets.omnifake import OmniFakeDataset
from model.model import TwinNeXt
from model.loss import SupervisedContrasiveLoss, SphereCenterLoss
from model.engine_train import train_one_epoch
from util.utils import setup_dist
import util.logger as logger
from util.lr_scheduler import CosineAnnealingWithWarmup


def get_args_parser():
    parser = argparse.ArgumentParser('Omni-AIGI-Detector training for deepfake detection and attribution')
    # Base and distributed training parameters
    parser.add_argument('--model', type=str, default='omni-aigi-detector', help='Name of model in this project')
    parser.add_argument('--seed', default=42, type=int, help="random seed for the main process")
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to log')

    # Arguments for distributed parallel which will be set with `setup_dist``
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes')
    parser.add_argument('--master_addr', type=str, default='env://', help='master address for multiprocessing')
    parser.add_argument('--master_port', type=str, default='23456', help='master port for multiprocessing')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed data parallel')
    parser.add_argument('--rank', type=int, default=-1, help='rank for distributed data parallel')
    parser.add_argument('--device', default='cuda', help='device to use for current process')

    # Dataset & dataloader parameters
    parser.add_argument('--data_path', default='data/OmniFake', type=str, help='dataset path')
    parser.add_argument('--fake_file_path', default='datasets/split/part1/trainlist.txt', type=str, help='file path for fake class')
    parser.add_argument('--real_file_path', default='datasets/split/part1/reallist.txt', type=str, help='file path for real class')

    # it's ok for A100 40G
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--batch_size_fake', default=128, type=int, help='batch size per GPU')
    parser.add_argument('--batch_size_real', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--num_workers', default=20, type=int, help='number of workers for dataloader')
    parser.add_argument('--total_epochs', default=20, type=int)
    
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0.0, help='minimum learning rate for scheduling')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay (default: 0.01)')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='epochs to warmup LR')
    parser.add_argument('--use_fp16', action='store_true', help='use fp16 during training')
    parser.add_argument('--use_bf16', action='store_true', help='use bf16 during training')

    # resume
    parser.add_argument('--resume', default=None, help='resume from checkpoint')

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
        mode="train", 
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
        drop_last=True
    )

    real_dataset = OmniFakeDataset(
        data_root=args.data_path,
        class_file_path=args.real_file_path,  
        mode="train", 
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
        drop_last=True
    )

    data_length = min(len(fake_dataloader), len(real_dataloader))
    #################################################

    
    ############ create model & criterion ###########
    logger.info("Creating model 'TwinNeXt'... ")
    model = TwinNeXt(
        backbone_name="convnext_small", 
        mlp_hidden_dims=512, 
        out_dim=128, 
    ).to(args.device)
    model = DDP(model, device_ids=[args.local_rank])

    # criterion
    center_loss_model = SphereCenterLoss(feat_dim=128).to(args.device)
    center_loss_model = DDP(center_loss_model, device_ids=[args.local_rank])
    contrasive_loss = SupervisedContrasiveLoss()

    # dtype
    if args.use_fp16: 
        dtype = torch.float16
    elif args.use_bf16: 
        dtype = torch.bfloat16
    else: 
        dtype = torch.float32
    #################################################
    

    ######### create optimizer and criterion ########
    logger.info("Creating optimizer and scheduler... ")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters()}, 
            {"params": center_loss_model.parameters()}
        ], 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    if args.use_fp16: 
        scaler = GradScaler(enabled=args.use_fp16)
    else:
        scaler = None

    # scheduler, we call it every step
    scheduler = CosineAnnealingWithWarmup(
        optimizer=optimizer, 
        warmup_steps=args.warmup_epochs * data_length, 
        cycle_steps=args.total_epochs * data_length, 
        lr_min=args.min_lr, 
        lr_max=args.lr,
    )
    #################################################


    #################### training ###################
    logger.info("Start training for %d epochs. " % args.total_epochs)

    # starts looping
    for epoch in range(1, args.total_epochs + 1): 
        real_sampler.set_epoch(epoch)
        fake_sampler.set_epoch(epoch)
        
        train_one_epoch(
            model=model, 
            dataloaders=[fake_dataloader, real_dataloader], 
            criterions=[contrasive_loss, center_loss_model], 
            optimizer=optimizer, 
            scheduler=scheduler, 
            scaler=scaler, 
            epoch=epoch, 
            device=args.device, 
            dtype=dtype, 
            args=args
        )
    
    dist.destroy_process_group()
    #################################################


if __name__ == '__main__': 
    args = get_args_parser()
    main(args)

