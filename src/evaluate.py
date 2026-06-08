"""
evaluate.py
Evaluation metrics for the Indic OCR pipeline.
Computes Character Error Rate (CER) and Word Error Rate (WER) using jiwer.
"""

from jiwer import cer, wer
from typing import List, Dict, Optional
import json
from pathlib import Path


def compute_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Compute Character Error Rate (CER).

    Args:
        predictions: List of predicted strings.
        targets: List of ground-truth strings.

    Returns:
        CER as a float between 0 and 1.
    """
    if not predictions or not targets:
        return 1.0
    preds_clean = [p if p.strip() else " " for p in predictions]
    tgts_clean  = [t if t.strip() else " " for t in targets]
    return cer(tgts_clean, preds_clean)


def compute_wer(predictions: List[str], targets: List[str]) -> float:
    """
    Compute Word Error Rate (WER).

    Args:
        predictions: List of predicted strings.
        targets: List of ground-truth strings.

    Returns:
        WER as a float between 0 and 1.
    """
    if not predictions or not targets:
        return 1.0
    preds_clean = [p if p.strip() else " " for p in predictions]
    tgts_clean  = [t if t.strip() else " " for t in targets]
    return wer(tgts_clean, preds_clean)


def evaluate_model(
    predictions: List[str],
    targets: List[str],
    model_name: str = "model",
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a model and print a formatted results table.

    Args:
        predictions: List of predicted strings.
        targets: List of ground-truth strings.
        model_name: Name used in the report.
        save_path: If provided, save results JSON here.

    Returns:
        Dictionary with cer, wer, exact_match, sample_count.
    """
    cer_score = compute_cer(predictions, targets)
    wer_score = compute_wer(predictions, targets)
    exact_match = sum(p == t for p, t in zip(predictions, targets)) / max(len(targets), 1)

    results = {
        "model":        model_name,
        "cer":          round(cer_score, 4),
        "wer":          round(wer_score, 4),
        "exact_match":  round(exact_match, 4),
        "sample_count": len(targets),
    }

    print(f"\n{'─' * 45}")
    print(f"  Model        : {model_name}")
    print(f"  Samples      : {len(targets):,}")
    print(f"  CER          : {cer_score*100:.1f}%")
    print(f"  WER          : {wer_score*100:.1f}%")
    print(f"  Exact Match  : {exact_match*100:.1f}%")
    print(f"{'─' * 45}\n")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def compare_models(results_list: List[Dict]) -> None:
    """Print a comparison table for multiple model results."""
    print(f"\n{'Model':<20} {'CER':>8} {'WER':>8} {'Exact Match':>12}")
    print("─" * 52)
    for r in results_list:
        print(
            f"{r['model']:<20} "
            f"{r['cer']*100:>7.1f}% "
            f"{r['wer']*100:>7.1f}% "
            f"{r['exact_match']*100:>11.1f}%"
        )
    print("─" * 52)


if __name__ == "__main__":
    sample_preds   = ["नमस्ते", "मैं ठीक हूँ", "धन्यवाद"]
    sample_targets = ["नमस्ते", "मैं ठीक हूं",  "धन्यवाद"]
    evaluate_model(sample_preds, sample_targets, model_name="smoke_test")
