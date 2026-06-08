"""
preprocess.py
Image cleaning pipeline for handwritten Devanagari OCR.
Handles grayscale conversion, binarization, deskewing, and noise removal.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        image_path: Path to the image file.

    Returns:
        BGR image as numpy array.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.

    Args:
        image: BGR image as numpy array.

    Returns:
        Grayscale image.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def binarize(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Binarize a grayscale image using Otsu's thresholding or adaptive thresholding.

    Args:
        image: Grayscale image.
        method: 'otsu' or 'adaptive'.

    Returns:
        Binary image.
    """
    gray = to_grayscale(image)

    if method == "otsu":
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return binary


def remove_noise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Remove salt-and-pepper noise using morphological operations.

    Args:
        image: Binary or grayscale image.
        kernel_size: Size of the morphological kernel.

    Returns:
        Denoised image.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct skew in a document image using Hough line transform.

    Args:
        image: Grayscale or binary image.

    Returns:
        Deskewed image.
    """
    gray = to_grayscale(image) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return image

    angles = []
    for line in lines[:20]:
        rho, theta = line[0]
        angle = (theta - np.pi / 2) * (180 / np.pi)
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def resize_for_model(image: np.ndarray, height: int = 32, max_width: int = 512) -> np.ndarray:
    """
    Resize image to a fixed height while maintaining aspect ratio, capped at max_width.

    Args:
        image: Input image.
        height: Target height in pixels.
        max_width: Maximum width in pixels.

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    aspect = w / h
    new_width = min(int(height * aspect), max_width)
    resized = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_AREA)
    return resized


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] range.

    Args:
        image: Input image (any dtype).

    Returns:
        Float32 image with values in [0, 1].
    """
    return image.astype(np.float32) / 255.0


def full_pipeline(image_path: str) -> np.ndarray:
    """
    Run the complete preprocessing pipeline on an image file.

    Args:
        image_path: Path to the input image.

    Returns:
        Preprocessed, normalized image ready for model input.
    """
    img = load_image(image_path)
    img = to_grayscale(img)
    img = binarize(img, method="otsu")
    img = remove_noise(img, kernel_size=3)
    img = deskew(img)
    img = resize_for_model(img, height=32)
    img = normalize(img)
    return img


def preprocess_from_array(image: np.ndarray) -> np.ndarray:
    """
    Run the preprocessing pipeline on an in-memory image array.

    Args:
        image: Input image as numpy array (BGR or grayscale).

    Returns:
        Preprocessed, normalized image.
    """
    img = to_grayscale(image)
    img = binarize(img, method="otsu")
    img = remove_noise(img, kernel_size=3)
    img = deskew(img)
    img = resize_for_model(img, height=32)
    img = normalize(img)
    return img


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = full_pipeline(sys.argv[1])
        print(f"Preprocessed image shape: {result.shape}, dtype: {result.dtype}")
        print(f"Value range: [{result.min():.3f}, {result.max():.3f}]")
