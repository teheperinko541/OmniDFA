# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from itertools import combinations
import random

from datasets.custom_transforms import get_custom_timm_rand_augment, RandomJPEG, RandomGaussianBlur, CustomResizeKeepRatio


class OmniFakeEpisode(Dataset):
    """
    A minimal dataset that returns the episode data for N-way K-shot question. 
    You should set shuffle=False for the sampler. And the batch size of dataloader should be a multiple of n_way. 
    You should set different seed for different processes to load different data. 
    So a batch of n_way samples will form an episode. 

    Item data: 
        1. Random crop (train) / center crop (val) from the original image.
        2. A snap of the original image.
        3. Ground-truth class label (int).
    """
    def __init__(
        self,
        data_root: str,
        class_file_path: str,
        mode: str = "train",
        output_size: int = 224,
        n_way: int = 15, 
        k_shot: int = 5, 
        n_query: int = 20, 
        world_size: int = 4, # to compute the episode part
        total_length: int = 10000, 
    ):
        super().__init__()
        assert mode in {"train", "val"}, "mode must be 'train' or 'val'"
        self.mode = mode
        self.data_root = os.path.join(data_root, self.mode)
        self.output_size = output_size

        # Parse class file: "<int_id> <class_name>"
        self.idx_to_class = {}
        with open(class_file_path, "r") as f:
            for line in f:
                cls_id, cls_name = line.strip().split(" ")
                self.idx_to_class[int(cls_id)] = cls_name
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}

        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.world_size = world_size
        self.total_length = total_length * self.n_way

        assert self.n_way <= len(self.idx_to_class), f"Not enough classes ({len(self.idx_to_class)}) in dataset for N-way ({self.n_way}) task"

        # Scan image paths
        self.samples_by_idx = {}
        for cls_id, cls_name in self.idx_to_class.items():
            cls_dir = os.path.join(self.data_root, cls_name)
            if not os.path.isdir(cls_dir): 
                raise FileNotFoundError(f"{cls_dir} does not exist")
            
            class_samples = []
            for file in os.listdir(cls_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")): 
                    class_samples.append(os.path.join(cls_dir, file))
            
            self.samples_by_idx[cls_id] = class_samples
        
        # Build transforms
        if self.mode == "train":
            self.img_augment = self.get_train_transforms(self.output_size + 32) # with a little margin
        else:
            self.img_augment = self.get_val_transforms(self.output_size + 32)
        
        # for path 1
        self.transform_crop = transforms.RandomCrop(self.output_size)

        # for path 2
        self.transform_resize = transforms.Compose([
            transforms.Resize(self.output_size),  # resize short edge with bilinear
            transforms.RandomCrop(self.output_size),
        ])

        self.uniform_class_samples = [list(c) for c in combinations(list(self.samples_by_idx.keys()), self.n_way)]
        # self.uniform_class_samples = []
        # for i, c in enumerate(combinations(list(self.samples_by_idx.keys()), self.n_way)): 
        #     self.uniform_class_samples.append(list(c))
        #     if i > 10000: 
        #         break
        # print("Load done. ")

    def __len__(self) -> int:
        return self.total_length # fake length

    def __getitem__(self, idx): 
        block_id = ((idx // (self.n_way * self.world_size)) * self.world_size + idx % self.world_size) % len(self.uniform_class_samples)
        class_id = (idx % (self.n_way * self.world_size)) // self.world_size
        uniform_sampled_class = self.uniform_class_samples[block_id][class_id]

        crop_list = []
        resize_list = []
        label_list = []
        selected_samples = random.sample(self.samples_by_idx[uniform_sampled_class], self.k_shot + self.n_query)
        for img_path in selected_samples: 
            img = Image.open(img_path) # convert to RGB in RandomJPEG
            img = self.img_augment(img)
            img_crop = self.transform_crop(img)
            img_resize = self.transform_resize(img)
            label = torch.tensor(uniform_sampled_class)
            label = torch.tensor(idx)
            
            crop_list.append(img_crop)
            resize_list.append(img_resize)
            label_list.append(label)
        
        # size (n_query + k_shot, 3, 224, 224)
        return torch.stack(crop_list), torch.stack(resize_list), torch.stack(label_list)
    
    @staticmethod
    def get_train_transforms(min_size):
        return transforms.Compose([
            RandomJPEG(p=0.5, compress_module=["pil"]), 
            CustomResizeKeepRatio(min_size=min_size, scale_range=(0.5, 2.0)), 
            transforms.RandomHorizontalFlip(p=0.5), 
            get_custom_timm_rand_augment(), 
            RandomGaussianBlur(p=0.5, sigma=2.0), 
            transforms.ToTensor(), 
        ])
    
    @staticmethod
    def get_val_transforms(min_size):
        return transforms.Compose([
            RandomJPEG(p=0.0, quality=(50, 95)),             # change here for ablation studies
            CustomResizeKeepRatio(min_size=min_size, p=0.0), 
            RandomGaussianBlur(p=0.0, sigma=0),              # change here for ablation studies
            transforms.ToTensor(), 
        ])

