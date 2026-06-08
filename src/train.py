import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
"""
train.py
Full training loop for the Indic OCR pipeline.
Supports both CRNN (CTC) and TrOCR fine-tuning modes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import json, time, argparse

from src.model import CRNN, TrOCRWrapper, CTCLoss, get_crnn, get_trocr
from src.evaluate import compute_cer, compute_wer
from src.dataset import get_dataloader
from data.load_datasets import load_pralekha

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def train_crnn_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels, label_lengths in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        optimizer.zero_grad()
        log_probs = model(images)
        T, B, _ = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        flat_labels = labels[labels != 0]
        loss = criterion(log_probs, flat_labels, input_lengths, label_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def validate_crnn(model, loader, criterion, idx2char, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels, label_lengths in tqdm(loader, desc="  Val  ", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            log_probs = model(images)
            T, B, _ = log_probs.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            flat_labels = labels[labels != 0]
            loss = criterion(log_probs, flat_labels, input_lengths, label_lengths)
            total_loss += loss.item()
            preds = log_probs.argmax(dim=2).permute(1, 0)
            for i in range(B):
                pred_seq = []
                prev = -1
                for p in preds[i]:
                    p = p.item()
                    if p != 0 and p != prev:
                        pred_seq.append(idx2char.get(p, ""))
                    prev = p
                target_seq = "".join(
                    idx2char.get(l.item(), "") for l in labels[i, :label_lengths[i]]
                )
                all_preds.append("".join(pred_seq))
                all_targets.append(target_seq)
    return {
        "val_loss": total_loss / max(len(loader), 1),
        "cer": compute_cer(all_preds, all_targets),
        "wer": compute_wer(all_preds, all_targets),
    }


def train_crnn(sample=10000, epochs=50, batch_size=16, lr=5e-4, patience=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training CRNN on device: {device}")
    ds = load_pralekha(sample=sample)
    train_loader, char2idx = get_dataloader(ds["train"], batch_size=batch_size, train=True)
    val_loader, _ = get_dataloader(ds["validation"], batch_size=batch_size, train=False, char2idx=char2idx)
    idx2char = {v: k for k, v in char2idx.items()}
    model = get_crnn(vocab_size=len(char2idx), device=device)
    criterion = CTCLoss(blank=char2idx.get("<PAD>", 0))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_cer = float("inf")
    no_improve = 0
    history = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_crnn_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_crnn(model, val_loader, criterion, idx2char, device)
        elapsed = time.time() - start
        scheduler.step(val_metrics["val_loss"])
        print(f"Epoch {epoch:03d} | loss {train_loss:.4f} | val_loss {val_metrics['val_loss']:.4f} | CER {val_metrics['cer']:.4f} | WER {val_metrics['wer']:.4f} | {elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "crnn_best.pt")
            print(f"  New best CER: {best_cer:.4f} — saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
    with open(CHECKPOINT_DIR / "crnn_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best CER: {best_cer:.4f}")


def train_trocr(sample=5000, epochs=10, batch_size=8, lr=5e-5, patience=3, freeze_encoder=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fine-tuning TrOCR on device: {device}")
    ds = load_pralekha(sample=sample)
    model = get_trocr(freeze_encoder=freeze_encoder, device=device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    best_loss = float("inf")
    no_improve = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start = time.time()
        from PIL import Image as PILImage
        import numpy as np
        for item in tqdm(ds["train"], desc=f"Epoch {epoch:02d}", leave=False):
            try:
                raw = item.get("image")
                if raw is None:
                    continue
                if not isinstance(raw, PILImage.Image):
                    raw = PILImage.fromarray(np.array(raw)).convert("RGB")
                else:
                    raw = raw.convert("RGB")
                pixel_values = model.preprocess_images([raw]).to(device)
                text = item.get("text", "")
                labels = model.processor.tokenizer(text, return_tensors="pt", padding=True).input_ids.to(device)
                labels[labels == model.processor.tokenizer.pad_token_id] = -100
                optimizer.zero_grad()
                output = model(pixel_values=pixel_values, labels=labels)
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += output.loss.item()
            except Exception:
                continue
        avg_loss = total_loss / max(len(ds["train"]), 1)
        elapsed = time.time() - start
        scheduler.step(avg_loss)
        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | {elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": avg_loss})
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            model.model.save_pretrained(str(CHECKPOINT_DIR / "trocr_best"))
            model.processor.save_pretrained(str(CHECKPOINT_DIR / "trocr_best"))
            print(f"  New best loss: {best_loss:.4f} — saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
    with open(CHECKPOINT_DIR / "trocr_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["crnn", "trocr"], default="crnn")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.model == "crnn":
        train_crnn(sample=args.sample, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        train_trocr(sample=args.sample, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
