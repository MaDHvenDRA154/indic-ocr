"""
dataset.py
PyTorch Dataset and DataLoader for the Indic OCR pipeline.
Wraps HuggingFace datasets loaded via load_datasets.py.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Optional, Dict
from src.preprocess import preprocess_from_array
from src.augment import augment_image


class DevanagariOCRDataset(Dataset):
    """
    PyTorch Dataset for Devanagari OCR training.
    Wraps a HuggingFace dataset split and applies preprocessing + augmentation.
    """

    def __init__(
        self,
        hf_dataset,
        image_col: str = "image",
        text_col: str = "text",
        char2idx: Optional[Dict[str, int]] = None,
        train: bool = True,
        max_label_len: int = 64,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            hf_dataset: HuggingFace Dataset split (train/val/test).
            image_col: Column name for images in the dataset.
            text_col: Column name for text labels.
            char2idx: Character-to-index mapping. Built from data if None.
            train: If True, apply training augmentations.
            max_label_len: Maximum label sequence length (longer samples are skipped).
        """
        self.dataset = hf_dataset
        self.image_col = image_col
        self.text_col = text_col
        self.train = train
        self.max_label_len = max_label_len

        if char2idx is None:
            self.char2idx = self._build_vocab()
        else:
            self.char2idx = char2idx

        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def _build_vocab(self) -> Dict[str, int]:
        """
        Build character vocabulary from the dataset labels.

        Returns:
            Dictionary mapping characters to indices.
        """
        chars = set()
        for item in self.dataset:
            text = item.get(self.text_col, "")
            if isinstance(text, str):
                chars.update(text)

        vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        for i, ch in enumerate(sorted(chars), start=len(vocab)):
            vocab[ch] = i

        print(f"Vocabulary size: {len(vocab)} characters")
        return vocab

    def encode_label(self, text: str) -> torch.Tensor:
        """
        Encode a text string as a tensor of character indices.

        Args:
            text: Input string.

        Returns:
            Long tensor of character indices.
        """
        indices = [self.char2idx.get(ch, self.char2idx["<UNK>"]) for ch in text]
        return torch.tensor(indices, dtype=torch.long)

    def decode_label(self, indices: List[int]) -> str:
        """
        Decode a list of character indices back to a string.

        Args:
            indices: List of character indices.

        Returns:
            Decoded string (PAD/SOS/EOS tokens removed).
        """
        special = {"<PAD>", "<SOS>", "<EOS>"}
        return "".join(
            self.idx2char.get(i, "") for i in indices
            if self.idx2char.get(i, "") not in special
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Fetch a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_tensor, label_length).
        """
        item = self.dataset[idx]

        raw_image = item[self.image_col]
        if isinstance(raw_image, Image.Image):
            raw_image = np.array(raw_image.convert("L"))
        elif isinstance(raw_image, dict) and "bytes" in raw_image:
            raw_image = np.frombuffer(raw_image["bytes"], dtype=np.uint8)
            raw_image = cv2.imdecode(raw_image, cv2.IMREAD_GRAYSCALE)

        image = preprocess_from_array(raw_image)
        image = augment_image((image * 255).astype(np.uint8), train=self.train)
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        text = item.get(self.text_col, "")
        label = self.encode_label(text)
        label_len = len(label)

        return image_tensor, label, label_len


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function to pad images and labels to the same length.

    Args:
        batch: List of (image, label, label_len) tuples.

    Returns:
        Batched (images, labels, label_lengths) tensors.
    """
    images, labels, lengths = zip(*batch)

    max_w = max(img.shape[-1] for img in images)
    padded_images = torch.zeros(len(images), 1, 32, max_w)
    for i, img in enumerate(images):
        padded_images[i, :, :, :img.shape[-1]] = img

    max_len = max(len(l) for l in labels)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return padded_images, padded_labels, lengths_tensor


def get_dataloader(
    hf_dataset,
    batch_size: int = 32,
    train: bool = True,
    char2idx: Optional[Dict[str, int]] = None,
    num_workers: int = 2,
) -> Tuple[DataLoader, Dict[str, int]]:
    """
    Build a DataLoader from a HuggingFace dataset split.

    Args:
        hf_dataset: HuggingFace Dataset split.
        batch_size: Number of samples per batch.
        train: If True, apply training augmentations and shuffle.
        char2idx: Character vocabulary. Built from data if None.
        num_workers: Number of DataLoader worker processes.

    Returns:
        Tuple of (DataLoader, char2idx).
    """
    dataset = DevanagariOCRDataset(
        hf_dataset,
        char2idx=char2idx,
        train=train,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.char2idx
