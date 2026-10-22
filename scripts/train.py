"""
Video-Level Temporal Deepfake Detector Training Pipeline.
Supports Mixed Precision, Cosine Annealing, ROC-AUC Checkpointing, and Dry-Run Verification.
"""
import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.dataset.dataset_builder import DeepfakeSequenceDataset
from src.dataset.transforms import get_training_transforms, get_validation_transforms
from src.models.video_classifier import TemporalDeepfakeDetector


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels, _ in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits, probs, _ = model(sequences)
            loss = criterion(logits, labels)
            total_loss += loss.item() * sequences.size(0)

            all_targets.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    n_samples = max(1, len(all_targets))
    avg_loss = total_loss / n_samples

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    return avg_loss, acc, prec, rec, f1, auc


def train(
    train_metadata: Path,
    val_metadata: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    dry_run: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Check if dataset files exist
    if not dry_run and (not train_metadata.exists() or not val_metadata.exists()):
        print("\n" + "=" * 65)
        print("[!] DATASET NOT DETECTED — TRAINING HALTED (Integrity Rule)")
        print("=" * 65)
        print("To train on real data, please prepare your dataset as follows:")
        print(f" 1. Place raw video files in: data/raw/ (e.g. FaceForensics++ C23)")
        print(f" 2. Run: python scripts/prepare_dataset.py --dataset faceforensics")
        print(f" 3. Ensure metadata files exist at: {train_metadata.resolve()}")
        print(f"                                   {val_metadata.resolve()}")
        print("=" * 65)
        print("[*] To verify model architecture forward/backward passes now, run:")
        print("    python scripts/train.py --dry-run\n")
        return

    # Build datasets and loaders
    if dry_run:
        print("[*] Executing dry-run verification with synthetic temporal sequences (8 batches)...")
        synth_x = torch.randn(16, 16, 3, 256, 256)
        synth_y = torch.tensor([[0.0], [1.0]] * 8)
        dummy_ds = TensorDataset(synth_x, synth_y, torch.arange(16))
        # Wrap dummy dataset for evaluate interface
        class DummySeqDS:
            def __init__(self, ds): self.ds = ds
            def __len__(self): return len(self.ds)
            def __getitem__(self, idx):
                x, y, i = self.ds[idx]
                return x, y, f"dummy_{i}"
        train_loader = DataLoader(DummySeqDS(dummy_ds), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(DummySeqDS(dummy_ds), batch_size=batch_size, shuffle=False)
        epochs = 2
    else:
        train_ds = DeepfakeSequenceDataset(str(train_metadata), sequence_length=16, transform=get_training_transforms())
        val_ds = DeepfakeSequenceDataset(str(val_metadata), sequence_length=16, transform=get_validation_transforms())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = TemporalDeepfakeDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_auc = -1.0

    print(f"[*] Starting training ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for sequences, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, _, _ = model(sequences)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * sequences.size(0)

        scheduler.step()
        train_loss = train_loss / len(train_loader.dataset)

        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc and not dry_run:
            best_val_auc = val_auc
            checkpoint_path = output_dir / "best_temporal_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[+] Saved new best model checkpoint (Val AUC: {val_auc:.4f}) -> {checkpoint_path}")

    if dry_run:
        print("[+] Dry-run verification complete: Forward/backward passes executed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Video-Level Temporal Deepfake Detector")
    parser.add_argument("--train-meta", type=Path, default=Path("data/metadata/train_metadata.jsonl"))
    parser.add_argument("--val-meta", type=Path, default=Path("data/metadata/val_metadata.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true", help="Verify training on synthetic tensor sequence")
    args = parser.parse_args()

    train(args.train_meta, args.val_meta, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, dry_run=args.dry_run)
