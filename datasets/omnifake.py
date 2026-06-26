# -*- coding: utf-8 -*-
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from datasets.custom_transforms import get_custom_timm_rand_augment, RandomJPEG, RandomGaussianBlur, CustomResizeKeepRatio


class OmniFakeDataset(Dataset):
    """
    A minimal dataset that returns three tensors per sample for our experiments: 
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
    ):
        super().__init__()
        assert mode in {"train", "val"}, "mode must be 'train' or 'val'"
        self.mode = mode
        self.data_root = os.path.join(data_root, self.mode)
        # self.data_root = data_root
        self.output_size = output_size

        # Parse class file: "<int_id> <class_name>"
        self.idx_to_class = {}
        with open(class_file_path, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                assert len(parts) == 2

                cls_id_str, cls_name = parts
                cls_id = int(cls_id_str)
                self.idx_to_class[cls_id] = cls_name
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}

        # Scan image paths
        self.samples = []
        for cls_id, cls_name in self.idx_to_class.items():
            cls_dir = os.path.join(self.data_root, cls_name)
            if not os.path.isdir(cls_dir): 
                raise FileNotFoundError(f"{cls_dir} does not exist")
            for file in os.listdir(cls_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")): 
                    self.samples.append((os.path.join(cls_dir, file), cls_id))
        
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
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx): 
        img_path, label = self.samples[idx]
        img = Image.open(img_path) # convert to RGB in RandomJPEG

        img = self.img_augment(img)
        img_crop = self.transform_crop(img)
        img_resize = self.transform_resize(img)

        return img_crop, img_resize, torch.tensor(label)
    
    @staticmethod
    def get_train_transforms(min_size):
        return transforms.Compose([
            RandomJPEG(p=0.5, quality=(50, 95), compress_module=["pil", "cv2"]), 
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

