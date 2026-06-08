"""
load_kaggle_dataset.py
Loads the Kaggle Devanagari Handwritten Character Dataset from data/data.csv
1024 pixel columns (32x32) + 1 character label column.
"""

import sys, os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from PIL import Image as PILImage
from datasets import Dataset, DatasetDict
from pathlib import Path

CACHE_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Maps character_01_ka -> index 0, character_02_kha -> index 1, etc.
CHARACTERS = [
    "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व", "श",
    "ष", "स", "ह", "क्ष", "त्र",
    "ज्ञ", "०", "१", "२", "३",
    "४", "५", "६", "७", "८", "९"
]


def label_to_char(label_val: str) -> str:
    """Convert 'character_01_ka' style label to Devanagari character."""
    label_val = str(label_val).strip()
    if label_val.startswith("character_"):
        parts = label_val.split("_")
        try:
            idx = int(parts[1]) - 1  # character_01 -> index 0
            return CHARACTERS[idx] if idx < len(CHARACTERS) else label_val
        except (ValueError, IndexError):
            return label_val
    try:
        idx = int(label_val) - 1
        return CHARACTERS[idx] if idx < len(CHARACTERS) else label_val
    except ValueError:
        return label_val


def load_kaggle_devanagari(sample=None, csv_path=None):
    """
    Load Kaggle Devanagari dataset using fast vectorized numpy.
    Each row: 1024 pixel values (32x32 image) + 1 character label.
    """
    if csv_path is None:
        csv_path = Path(os.path.dirname(os.path.abspath(__file__))) / "data.csv"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"Loading Kaggle Devanagari dataset from {csv_path}...")
    print(f"  File size: {csv_path.stat().st_size / 1e6:.1f} MB")

    nrows = sample if sample else None
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Identify label column
    label_col = "character"
    if label_col not in df.columns:
        label_col = df.columns[-1]
    print(f"  Label column: '{label_col}'")

    # Extract pixel columns (all except label)
    pixel_cols = [c for c in df.columns if c != label_col]
    img_size = int(round(len(pixel_cols) ** 0.5))
    print(f"  Image size: {img_size}x{img_size}, pixel cols: {len(pixel_cols)}")

    # Vectorized extraction — much faster than iterrows
    pixels_array = df[pixel_cols].values.astype(np.uint8)  # (N, 1024)
    labels_series = df[label_col].values                    # (N,)

    # Reshape all images at once
    images_array = pixels_array.reshape(-1, img_size, img_size)  # (N, 32, 32)

    # Invert dark images (some datasets use 0=white, 255=black)
    mean_brightness = images_array.mean()
    if mean_brightness < 50:
        images_array = 255 - images_array
        print(f"  Inverted pixel values (mean was {mean_brightness:.1f})")

    # Convert to PIL images and map labels
    samples = []
    for i in range(len(images_array)):
        pil_img = PILImage.fromarray(images_array[i])
        text = label_to_char(labels_series[i])
        samples.append({"image": pil_img, "text": text})

    print(f"  Converted {len(samples):,} images successfully")
    print(f"  Sample labels: {[s['text'] for s in samples[:5]]}")

    if len(samples) == 0:
        raise ValueError("No samples loaded! Check your CSV file.")

    ds = Dataset.from_list(samples)
    split_1 = ds.train_test_split(test_size=0.2, seed=42)
    split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)
    result = DatasetDict({
        "train":      split_1["train"],
        "validation": split_2["train"],
        "test":       split_2["test"],
    })
    print(f"  kaggle_devanagari → train: {len(result['train']):,} | val: {len(result['validation']):,} | test: {len(result['test']):,}")
    return result


if __name__ == "__main__":
    ds = load_kaggle_devanagari(sample=200, csv_path="data/data.csv")
    print(f"\nSample: text='{ds['train'][0]['text']}', size={ds['train'][0]['image'].size}")
    print("All good!")