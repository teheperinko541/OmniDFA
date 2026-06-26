# -*- coding: utf-8 -*-
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision, MulticlassAccuracy, MulticlassF1Score
from einops import rearrange

from model.engine_train import all_gather_features
import util.logger as logger


@torch.no_grad()
def eval_authenticity(
    model,
    dataloaders,
    criterions,
    device="cuda",
    dtype=torch.float32
):
    """
    Evaluate authenticity: center_loss_model return 0/1, 1=fake.
    dataloaders = [fake_dataloader, real_dataloader]
    criterions  = [center_loss_model]
    """
    model.eval()

    center_loss_model = criterions[0]
    norm_center = F.normalize(center_loss_model.module.center_real, dim=0, p=2)
    threshold = center_loss_model.module.cosine_threshold.item()
    # print(threshold)
    
    preds_fake, preds_real = [], []

    # fake dataloader -------------------------------------------------
    idxx = 0
    for crop_img, global_img, lbl in (tqdm(dataloaders[0], desc="Eval fake") if not dist.is_initialized() or dist.get_rank() == 0 else dataloaders[0]):
        crop_img, global_img = crop_img.to(device), global_img.to(device)
        with autocast(enabled=(dtype != torch.float32), device_type="cuda"):
            feats = model(crop_img, global_img)
            feats = F.normalize(feats, dim=-1, p=2) # (b, c)
            pred = torch.matmul(feats, norm_center)
        preds_fake.append(pred.cpu())
    
    # real dataloader -------------------------------------------------
    for crop_img, global_img, _ in (tqdm(dataloaders[1], desc="Eval real") if not dist.is_initialized() or dist.get_rank() == 0 else dataloaders[1]):
        crop_img, global_img = crop_img.to(device), global_img.to(device)
        with autocast(enabled=(dtype != torch.float32), device_type="cuda"):
            feats = model(crop_img, global_img)
            feats = F.normalize(feats, dim=-1, p=2)
            pred = torch.matmul(feats, norm_center)
        preds_real.append(pred.cpu())

    # gather features -------------------------------------------------
    # print("ALL DONE")
    preds_fake = torch.cat(preds_fake, dim=0).to(device)
    preds_real = torch.cat(preds_real, dim=0).to(device)

    preds_fake = all_gather_features(preds_fake).cpu()
    preds_real = all_gather_features(preds_real).cpu()
    
    # calculating results ---------------------------------------------
    if not dist.is_initialized() or dist.get_rank() == 0:
        y_true_fake = torch.zeros_like(preds_fake, dtype=torch.int32)      # 0 = fake
        y_true_real = torch.ones_like(preds_real, dtype=torch.int32)       # 1 = real

        metric_acc = BinaryAccuracy(threshold=threshold)
        metric_ap = BinaryAveragePrecision(thresholds=20)

        f_acc = metric_acc(preds_fake, y_true_fake)
        r_acc = metric_acc(preds_real, y_true_real)

        y_pred = torch.cat([preds_fake, preds_real], dim=0)
        y_true = torch.cat([y_true_fake, y_true_real], dim=0)

        """
        We 'assume' that the two classes are balanced in quantity. 
        In most datasets, a unified binary classification accuracy 
        is typically computed from roughly equal numbers of real and 
        synthetic images. In our case, however, if only a subset of 
        categories is selected for testing, the real images will 
        greatly outnumber the synthetic ones, yielding a skewed metric.

        To avoid criticism, you are free to subsample the real images 
        so that the two classes remain balanced. 
        """
        acc = (f_acc + r_acc) / 2
        ap = metric_ap(y_pred, y_true)

        print("F-Acc", f_acc)
        print("R-Acc", r_acc)
        print("Acc", acc)
        print("AP", ap)

        logger.logkv("F-Acc", f_acc)
        logger.logkv("R-Acc", r_acc)
        logger.logkv("Acc", acc)
        logger.logkv("AP", ap)
        logger.dumpkvs(all_gather=False)

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


@torch.no_grad()
def eval_classification(
    model,
    dataloaders,
    n_way=5, 
    k_shot=5, 
    n_query=5,
    device="cuda",
    dtype=torch.float32, 
    rule="prototype"
):
    assert rule.lower() in ["prototype", "nearest"]

    model.eval()

    preds, labels = [], []

    # fake dataloader -------------------------------------------------
    for crop_img, global_img, lbl in (tqdm(dataloaders[0], desc="Eval fake") if not dist.is_initialized() or dist.get_rank() == 0 else dataloaders[0]):
        crop_img, global_img = crop_img.to(device), global_img.to(device)
        episode_batch_size = len(global_img) // n_way
        crop_img = rearrange(crop_img, 'b n c h w -> (b n) c h w')
        global_img = rearrange(global_img, 'b n c h w -> (b n) c h w')
        
        with autocast(enabled=(dtype != torch.float32), device_type="cuda"):
            feats = model(crop_img, global_img)
            feats = F.normalize(feats, dim=-1, p=2)

        feats = rearrange(feats, '(b n s) c -> b n s c', b=episode_batch_size, n=n_way)
        support_feats = feats[:, :, :k_shot, :]
        query_feats = feats[:, :, k_shot:, :] # （b, n, s, c)

        y_label = torch.arange(n_way, device=query_feats.device).view(1, n_way, 1).repeat(episode_batch_size, 1, query_feats.size(2)).reshape(-1)

        if rule.lower() == "prototype": 
            prototypes = support_feats.mean(dim=2, keepdim=True) # (b, n, 1, c)
            prototypes = F.normalize(prototypes, dim=-1, p=2)

            prototypes = rearrange(prototypes, 'b n 1 c -> b 1 n c')
            query_feats = rearrange(query_feats, 'b n s c -> b (n s) c 1')
            scores = torch.matmul(prototypes, query_feats).squeeze(-1)
            y_pred = scores.argmax(dim=-1).reshape(-1)

            preds.append(y_pred.cpu())
            labels.append(y_label.cpu())
        else: # nearest method has been removed
            raise ValueError(f"Unknown classification rule {rule}")
        
        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])

    
    # gather features -------------------------------------------------
    preds = torch.cat(preds, dim=0).to(device)
    labels = torch.cat(labels, dim=0).to(device)

    preds = all_gather_features(preds).cpu()
    labels = all_gather_features(labels).cpu()

    # calculating results ---------------------------------------------
    if not dist.is_initialized() or dist.get_rank() == 0:
        # acc
        metric_acc = MulticlassAccuracy(num_classes=n_way)
        acc = metric_acc(preds, labels)
        print("Class-Acc: ", acc)
        logger.logkv("Acc", acc)
        
        # macro-F1: 
        metric_f1 = MulticlassF1Score(num_classes=n_way)
        batch_sample_num = n_way * n_query
        assert len(preds) % batch_sample_num == 0
        batch_num = len(preds) // batch_sample_num

        result = []
        for i in range(0, batch_num): 
            batch_preds = preds[batch_sample_num * i: batch_sample_num * (i + 1)]
            batch_labels = labels[batch_sample_num * i: batch_sample_num * (i + 1)]
            result.append(metric_f1(batch_preds, batch_labels))
        
        macro_f1 = sum(result) / len(result) if len(result) > 0 else 0.0
        print("Macro-F1: ", macro_f1)
        logger.logkv("Macro-F1", macro_f1)


        # auroc for 15-way: 
        # if n_way == 15: 
        #     from torchmetrics.classification import MulticlassAUROC
        #     metric_auroc = MulticlassAUROC(num_classes=n_way, thresholds=20)
        #     auroc = metric_auroc(preds, labels)
        #     print("AUROC: ", auroc)
        #     logger.logkv("AUROC", auroc)
        
        logger.dumpkvs(all_gather=False)

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])

