"""
inference.py
End-to-end inference on a single image using trained CRNN or TrOCR models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Union
import argparse

from src.preprocess import full_pipeline, preprocess_from_array


def load_crnn(checkpoint_path: str, vocab_size: int, device: str = "cpu"):
    """Load a trained CRNN model from a checkpoint."""
    from src.model import get_crnn
    model = get_crnn(vocab_size=vocab_size, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def load_trocr(checkpoint_path: str, device: str = "cpu"):
    """Load a fine-tuned TrOCR model from a checkpoint directory."""
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from src.model import TrOCRWrapper
    wrapper = TrOCRWrapper.__new__(TrOCRWrapper)
    wrapper.processor = TrOCRProcessor.from_pretrained(checkpoint_path)
    wrapper.model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path).to(device)
    wrapper.model.eval()
    return wrapper


def predict_crnn(
    image: Union[str, np.ndarray],
    model,
    idx2char: dict,
    device: str = "cpu",
) -> str:
    """Run CRNN inference on a single image."""
    if isinstance(image, str):
        img = full_pipeline(image)
    else:
        img = preprocess_from_array(image)

    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(tensor)

    preds = log_probs.argmax(dim=2).squeeze(1)
    decoded = []
    prev = -1
    for p in preds:
        p = p.item()
        if p != 0 and p != prev:
            decoded.append(idx2char.get(p, ""))
        prev = p
    return "".join(decoded)


def predict_trocr(
    image: Union[str, np.ndarray, Image.Image],
    model,
    device: str = "cpu",
) -> str:
    """Run TrOCR inference on a single image."""
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    pixel_values = model.preprocess_images([pil_image]).to(device)
    with torch.no_grad():
        results = model.generate(pixel_values)
    return results[0] if results else ""


def predict_tesseract_baseline(image_path: str, lang: str = "hin") -> str:
    """Run Tesseract OCR as a baseline comparison."""
    import pytesseract
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(binary, lang=lang).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indic OCR Inference")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", choices=["trocr", "crnn", "tesseract"], default="tesseract")
    parser.add_argument("--checkpoint", default="./checkpoints/trocr_best")
    parser.add_argument("--lang", default="hin")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "trocr":
        model = load_trocr(args.checkpoint, device=device)
        text = predict_trocr(args.image, model, device=device)
    elif args.model == "tesseract":
        text = predict_tesseract_baseline(args.image, lang=args.lang)
    else:
        print("CRNN inference requires vocab. Use the web UI instead.")
        text = ""

    print(f"\nPredicted text:\n{text}")
