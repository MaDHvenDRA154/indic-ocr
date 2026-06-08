"""
run.py — convenience launcher for the Indic OCR project.
Run from the project root: python run.py [command]

Commands:
  datasets   — download and cache all HuggingFace datasets
  train      — train the CRNN model (pass --sample N for quick test)
  app        — launch the web UI at http://localhost:5000
  test       — quick smoke test (no training needed)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def cmd_datasets():
    from data.load_datasets import load_all_datasets
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=100)
    args, _ = p.parse_known_args()
    load_all_datasets(sample=args.sample)

def cmd_train():
    from src.train import train_crnn, train_trocr
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["crnn", "trocr"], default="crnn")
    p.add_argument("--sample", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    args, _ = p.parse_known_args()
    if args.model == "crnn":
        train_crnn(sample=args.sample, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        train_trocr(sample=args.sample, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

def cmd_app():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from app.app import app
    print("\n=== Indic OCR Web UI ===")
    print("Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

def cmd_test():
    print("Running smoke tests...")
    from src.evaluate import compute_cer, compute_wer
    assert compute_cer(["नमस्ते"], ["नमस्ते"]) == 0.0
    assert compute_wer(["नमस्ते"], ["नमस्ते"]) == 0.0
    print("  evaluate.py ✓")
    from src.preprocess import preprocess_from_array
    import numpy as np
    img = np.ones((64, 256), dtype=np.uint8) * 200
    result = preprocess_from_array(img)
    assert result is not None
    print("  preprocess.py ✓")
    from data.load_datasets import _make_dummy_dataset, _split_dataset
    ds = _split_dataset(_make_dummy_dataset(50), "test")
    assert len(ds["train"]) > 0
    print("  load_datasets.py ✓")
    print("\nAll smoke tests passed!")

COMMANDS = {
    "datasets": cmd_datasets,
    "train":    cmd_train,
    "app":      cmd_app,
    "test":     cmd_test,
}

if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "test"
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS.keys())}")
        sys.exit(1)
    COMMANDS[cmd]()
