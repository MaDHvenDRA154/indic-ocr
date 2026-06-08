import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
"""
augment.py
Albumentations-based augmentation pipeline for handwritten Devanagari images.
Applies realistic handwriting distortions to improve model robustness.
"""

import albumentations as A
import numpy as np
import cv2
from typing import Tuple


def get_train_augmentations(height: int = 32, max_width: int = 512) -> A.Compose:
    """
    Build the training augmentation pipeline.
    Applies realistic handwriting variations: rotation, noise, blur, distortion.

    Args:
        height: Target image height after augmentation.
        max_width: Maximum image width.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_REPLICATE),
        A.ElasticTransform(alpha=30, sigma=5, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Sharpen(alpha=(0.1, 0.3), p=0.2),
        A.Downscale(scale_min=0.75, scale_max=0.95, p=0.1),
    ])


def get_val_augmentations() -> A.Compose:
    """
    Build the validation/test augmentation pipeline.
    No random transforms — only normalization-compatible ops.

    Returns:
        Albumentations Compose pipeline (identity for val/test).
    """
    return A.Compose([])


def augment_image(image: np.ndarray, train: bool = True) -> np.ndarray:
    """
    Apply augmentation to a single image.

    Args:
        image: Input image as numpy array (H, W) or (H, W, C).
        train: If True, apply training augmentations; else validation.

    Returns:
        Augmented image as numpy array.
    """
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        squeeze = True
    else:
        squeeze = False

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    pipeline = get_train_augmentations() if train else get_val_augmentations()
    result = pipeline(image=image)["image"]

    if squeeze:
        result = result.squeeze(-1)

    return result


if __name__ == "__main__":
    dummy = np.ones((32, 128), dtype=np.uint8) * 200
    augmented = augment_image(dummy, train=True)
    print(f"Input shape: {dummy.shape}, Output shape: {augmented.shape}")
