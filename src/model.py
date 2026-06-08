"""
model.py
Two model architectures for Indic OCR:
  1. CRNN — CNN + BiLSTM + CTC loss (lightweight, fast training)
  2. TrOCRWrapper — Fine-tuned microsoft/trocr-base-handwritten (highest accuracy)
"""

import torch
import torch.nn as nn
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# CRNN Model
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Single convolutional block: Conv → BN → ReLU → MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for sequence recognition.
    Architecture: CNN feature extractor → BiLSTM → Linear → CTC loss.

    Args:
        vocab_size: Number of output classes (characters + blank for CTC).
        hidden_size: LSTM hidden state dimension.
        num_lstm_layers: Number of LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(1, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=False),
            ConvBlock(256, 256, pool=True),
            ConvBlock(256, 512, pool=False),
            ConvBlock(512, 512, pool=False),
        )
        # Adaptive pool collapses height to 1 so output is always (B, 512, 1, W)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_lstm_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            Log-probabilities of shape (T, B, vocab_size) for CTC loss.
        """
        features = self.cnn(x)
        features = self.adaptive_pool(features)    # (B, 512, 1, W)
        features = features.squeeze(2)             # (B, 512, W)
        features = features.permute(0, 2, 1)       # (B, W, 512)

        rnn_out, _ = self.rnn(features)
        output = self.fc(rnn_out)
        output = self.log_softmax(output)
        return output.permute(1, 0, 2)


class CTCLoss(nn.Module):
    """CTC loss wrapper for CRNN training."""

    def __init__(self, blank: int = 0) -> None:
        super().__init__()
        self.loss = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(log_probs, targets, input_lengths, target_lengths)


# ─────────────────────────────────────────────
# TrOCR Wrapper
# ─────────────────────────────────────────────

class TrOCRWrapper(nn.Module):
    """
    Fine-tunable wrapper around microsoft/trocr-base-handwritten.
    Provides a simplified interface for training and inference.

    Args:
        model_name: HuggingFace model identifier.
        freeze_encoder: If True, freeze the encoder weights during fine-tuning.
    """

    MODEL_NAME = "microsoft/trocr-base-handwritten"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        print(f"Loading TrOCR model: {model_name}")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print("Encoder weights frozen.")

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            pixel_values: Image tensors of shape (B, 3, H, W).
            labels: Token ids of shape (B, T) for teacher-forcing.

        Returns:
            Model output (loss if labels provided, else logits).
        """
        return self.model(pixel_values=pixel_values, labels=labels)

    def generate(self, pixel_values: torch.Tensor, max_new_tokens: int = 64) -> list:
        """
        Generate text predictions for a batch of images.

        Args:
            pixel_values: Image tensors of shape (B, 3, H, W).
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            List of decoded strings.
        """
        generated_ids = self.model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def preprocess_images(self, images: list) -> torch.Tensor:
        """
        Preprocess a list of PIL images for TrOCR input.

        Args:
            images: List of PIL Image objects.

        Returns:
            Pixel values tensor ready for model input.
        """
        encoding = self.processor(images=images, return_tensors="pt")
        return encoding.pixel_values


def get_crnn(vocab_size: int, device: str = "cpu") -> CRNN:
    """
    Instantiate and return a CRNN model on the given device.

    Args:
        vocab_size: Number of output classes.
        device: Target device ('cpu' or 'cuda').

    Returns:
        CRNN model.
    """
    model = CRNN(vocab_size=vocab_size)
    return model.to(device)


def get_trocr(freeze_encoder: bool = False, device: str = "cpu") -> TrOCRWrapper:
    """
    Instantiate and return a TrOCR wrapper on the given device.

    Args:
        freeze_encoder: If True, freeze encoder during fine-tuning.
        device: Target device ('cpu' or 'cuda').

    Returns:
        TrOCRWrapper model.
    """
    model = TrOCRWrapper(freeze_encoder=freeze_encoder)
    return model.to(device)
