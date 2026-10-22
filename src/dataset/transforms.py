import io
import random
from typing import Callable

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class SimJPEGCompression:
    """Simulates realistic lossy JPEG compression artifacts."""

    def __init__(self, quality_range=(60, 90)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            q = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf)
        return img


def get_training_transforms(image_size: int = 256) -> Callable:
    """
    Realistic training augmentations:
    - Mild random horizontal flip
    - Color jitter (brightness, contrast)
    - Mild Gaussian blur
    - JPEG recompression
    - ImageNet normalization
    """
    return T.Compose(
        [
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            SimJPEGCompression(quality_range=(60, 90)),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_validation_transforms(image_size: int = 256) -> Callable:
    """
    Deterministic validation/testing transforms:
    - Resize to target canonical dimensions
    - Tensor conversion & standard normalization
    """
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
