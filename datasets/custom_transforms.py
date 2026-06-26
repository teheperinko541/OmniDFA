# -*- coding: utf-8 -*-
import io
import random
import numpy as np
from PIL import Image, ImageFilter
import cv2
from timm.data.auto_augment import RandAugment, rand_augment_ops


_CUSTOM_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    # 'ShearX',        # removed in our project
    # 'ShearY',        # removed in our project
    # 'TranslateXRel', # removed in our project
    # 'TranslateYRel', # removed in our project
    # 'Cutout'         # NOTE I've implement this as random erasing separately
]


def get_custom_timm_rand_augment(): 
    hparams = {"magnitude_std": 0.5, "magnitude_max": 10}
    ra_ops = rand_augment_ops(magnitude=9, prob=0.5, hparams=hparams, transforms=_CUSTOM_RAND_TRANSFORMS)
    return RandAugment(ra_ops, num_layers=2, choice_weights=None)


class RandomJPEG():
    """
    Randomly applies JPEG compression, or convert image to RGB. 
    Args:
        quality: Integer quality value or tuple of quality range for JPEG. 
        p: Probability of applying JPEG. 
        compress_module: Modules to compress PNG to JPEG. 
    """
    def __init__(self, quality=(75, 95), p=0.5, compress_module=["pil", "cv2"]): 
        if isinstance(quality, tuple): 
            self.quality = [quality[0], quality[1]]
        else: 
            self.quality = [quality, quality]
        self.p = p

        assert len(compress_module) > 0
        self.compress_module = compress_module
    
    def __call__(self, img): 
        is_jpeg = (img.format == 'JPEG')
        
        if img.mode == 'P': # prevent warning
            img = img.convert('RGBA')
        img = img.convert('RGB')

        if not is_jpeg and random.random() < self.p: 
            quality = random.randint(self.quality[0], self.quality[1])
            cm = random.choice(self.compress_module)
            if cm == "pil": 
                img = self._pil_jpeg(img, quality)
            elif cm == "cv2": 
                img = self._cv2_jpeg(img, quality)
            else:
                raise ValueError(f"Unknown compress module: {cm}")
        return img
    
    @staticmethod
    def _pil_jpeg(img, quality):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf)

    @staticmethod
    def _cv2_jpeg(img, quality):
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode('.jpg', img_bgr, encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB))


class RandomGaussianBlur():
    """
    Randomly applies Gaussian Blur
    Args:
        sigma: tuple of sigma values for Gaussian Blur
        p: probability of applying JPEG
    """
    def __init__(self, sigma=(0.1, 2.0), p=0.5): 
        if isinstance(sigma, tuple): 
            self.sigma = [sigma[0], sigma[1]]
        else: 
            self.sigma = [sigma, sigma]
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class CustomResizeKeepRatio():
    """
    Randomly scale an image while ensuring its shortest side is at least `min_size`.
    If the original image is smaller than the required size, it will be upsampled
    to meet the constraint. The actual resampling filter is chosen randomly from
    `resample_modes`.

    Parameters
    ----------
    min_size: Minimum allowed (height, width) after resizing.
    scale_range: Uniform sampling range for the relative scale factor (min_scale, max_scale).
    resample_modes: Candidate resampling filters; one is drawn at random for every call. 
    p: probability of application
    """

    def __init__(
        self,
        min_size=(256, 256), 
        scale_range=(0.5, 2.0),
        resample_modes=[
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        ],
        p=0.5
    ):
        if isinstance(min_size, tuple): 
            self.min_size = [min_size[0], min_size[1]]
        else:
            self.min_size = [min_size, min_size]
        
        if isinstance(scale_range, tuple): 
            self.scale_range = [scale_range[0], scale_range[1]]
        else: 
            self.scale_range = [scale_range, scale_range]
        
        assert len(resample_modes) > 0
        self.resample_modes = resample_modes
        self.p = p

    def __call__(self, img): 
        w, h = img.size

        # Minimum scale factor that guarantees both sides >= min_size
        scale_w = self.min_size[1] / w
        scale_h = self.min_size[0] / h
        min_scale_needed = max(scale_w, scale_h)

        if random.random() < self.p: 
            # Pick a random scale inside the user-defined range
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            scale = 1.0
        scale = max(scale, min_scale_needed)

        if scale != 1.0:
            # Randomly choose a resampling filter
            resample = random.choice(self.resample_modes)
            img = img.resize((round(w * scale), round(h * scale)), resample=resample)

        return img

