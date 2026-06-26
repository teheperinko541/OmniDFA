# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.amp import autocast

import util.logger as logger
from util.utils import save_model


def all_gather_features(features, keep_grad=False): 
    """
    Collect tensor from all ranks and concatenate.

    Args:
        features: tensors on GPU to be gathered
    """
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered, features)

    # keep grad on own rank
    if keep_grad: 
        gathered[dist.get_rank()] = features

    return torch.cat(gathered, dim=0).contiguous()


def train_one_epoch(
        model, 
        dataloaders, 
        criterions, 
        optimizer, 
        scheduler=None, 
        scaler=None, 
        epoch=0, 
        device="cuda",
        dtype=torch.float32, 
        args=None, 
        logger_interval=100, 
        contrasive_loss_weight=1.0,
        center_loss_weight=0.01,
): 
    """
    Run one training epoch.

    Args:
        model: DDP wrapped custom model
        dataloaders: [fake_dataloader, real_dataloader]
        criterions: [contrasive_criterion, center_criterion]
        optimizer, scheduler, scaler: as created in main
        epoch: int, current epoch
        device: cuda device for current process
        dtype: precision during training
        args: namespace for saving
    """
    model.train()

    fake_dataloader = dataloaders[0]
    real_dataloader = dataloaders[1]

    contrasive_criterion = criterions[0]
    center_criterion = criterions[1]

    min_len = min(len(fake_dataloader), len(real_dataloader))

    for step, ((fake_crop, fake_global, fake_label), (real_crop, real_global, real_label)) in enumerate(
        tqdm(zip(fake_dataloader, real_dataloader), total=min_len, desc=f"Train Epoch {epoch}") if dist.is_initialized() or dist.get_rank() == 0 else zip(fake_dataloader, real_dataloader)
    ):
        fake_crop, real_crop = fake_crop.to(device), real_crop.to(device)
        fake_global, real_global = fake_global.to(device), real_global.to(device)
        fake_label, real_label = fake_label.to(device), real_label.to(device)

        # concat two sources
        len_fake = len(fake_crop)
        batch_crop = torch.cat([fake_crop, real_crop], dim=0).contiguous()
        batch_global = torch.cat([fake_global, real_global], dim=0).contiguous()
        labels = torch.cat([fake_label, real_label], dim=0).contiguous()

        with autocast(enabled=(dtype != torch.float32), device_type="cuda"): 
            feat = model(batch_crop, batch_global)
            feat_real = feat[len_fake:]
            
            feat_gather = all_gather_features(feat, keep_grad=True)
            labels_gather = all_gather_features(labels.unsqueeze(-1)).squeeze(-1)

            loss_con = contrasive_criterion(feat_gather, labels_gather)
            loss_cen = center_criterion(feat_real)
            loss = contrasive_loss_weight * loss_con + center_loss_weight * loss_cen
        
        optimizer.zero_grad()
        if scaler is not None: # use fp16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:                  # bf16 or fp32
            loss.backward()
            optimizer.step()
        
        if scheduler is not None: 
            scheduler.step()
        
        # logging info
        logger.logkv("loss", loss.item())
        logger.logkv("contrasive loss", loss_con.item())
        logger.logkv("center loss", loss_cen.item())
        logger.logkv("real boundary", center_criterion.module.cosine_threshold.item())

        if (step + 1) % logger_interval == 0 or (step + 1) == min_len: 
            logger.logkv("step", step)
            logger.logkv("lr", scheduler.get_last_lr()[0])
            logger.dumpkvs()
    
    # save model
    kwargs = {
        'epoch': epoch, 
        'model': model.module, 
        'cl_model': center_criterion.module, 
        'optimizer': optimizer, 
        'args': args
    }

    if scheduler is not None: 
        kwargs['scheduler'] = scheduler

    if scaler is not None: 
        kwargs['scaler'] = scaler 

    save_model(os.path.join(args.output_dir, "ckpt"), args.model, **kwargs)
    torch.cuda.empty_cache()
    
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    
    # per-epoch evaluation code has been removed
