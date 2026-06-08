"""
load_datasets.py — loads Kaggle Devanagari dataset from data/data.csv
"""
import sys, os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset, DatasetDict
from pathlib import Path
import numpy as np

CACHE_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv")


def load_pralekha(sample=None):
    """Load Kaggle Devanagari dataset from data/data.csv"""
    from data.load_kaggle_dataset import load_kaggle_devanagari
    csv = CSV_PATH
    if not Path(csv).exists():
        print(f"  data.csv not found at {csv}, using synthetic fallback.")
        return _split_dataset(_make_synthetic_dataset(sample or 5000), "synthetic")
    return load_kaggle_devanagari(sample=sample, csv_path=csv)


def load_indic_dlp(sample=None):
    return load_pralekha(sample)


def load_samanantar(sample=None):
    return load_pralekha(sample)


def load_indic_nlp(sample=None):
    return load_pralekha(sample)


def _make_synthetic_dataset(n):
    from PIL import Image as PILImage
    import random
    hindi_texts = ["नमस्ते","धन्यवाद","भारत","हिंदी","क","ख","ग","घ","च","ज","त","द","न","प","ब","म","य","र","ल","व","श","स","ह"]
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    samples = []
    for i in range(n):
        text = rng.choice(hindi_texts)
        w = max(64, len(text) * 14 + rng.randint(10, 30))
        img = np.ones((32, w), dtype=np.uint8) * 255
        img = np.clip(img.astype(int) - np_rng.randint(0, 20, (32, w)), 200, 255).astype(np.uint8)
        for _ in range(rng.randint(4, 10)):
            x1 = rng.randint(2, max(3, w-15))
            y1 = rng.randint(6, 18)
            img[y1:min(32,y1+rng.randint(1,5)), x1:min(w,x1+rng.randint(8,20))] = rng.randint(0,60)
        samples.append({"image": PILImage.fromarray(img), "text": text})
    return Dataset.from_list(samples)


def _split_dataset(ds, name):
    s1 = ds.train_test_split(test_size=0.2, seed=42)
    s2 = s1["test"].train_test_split(test_size=0.5, seed=42)
    result = DatasetDict({"train": s1["train"], "validation": s2["train"], "test": s2["test"]})
    print(f"  {name} → train: {len(result['train']):,} | val: {len(result['validation']):,} | test: {len(result['test']):,}")
    return result


def load_all_datasets(sample=None):
    print("\n=== Loading datasets ===\n")
    ds = load_pralekha(sample)
    total = sum(len(ds[s]) for s in ds)
    print(f"\n  Total: {total:,} samples")
    return {"pralekha": ds}


if __name__ == "__main__":
    load_all_datasets(sample=100)