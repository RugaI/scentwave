# -*- coding: utf-8 -*-
"""
ScentWave training loop.
Trains on the synthetic dataset. Saves best checkpoint to checkpoints/best.pt.
"""

import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from src.models.scentwave   import ScentWave
from src.training.dataset   import SyntheticScentDataset
from src.training.losses    import ScentWaveLoss
from src.data.fragrance_db  import get_db

CKPT_DIR = Path(__file__).parent.parent.parent / "checkpoints"


def train(
    n_samples:  int   = 50_000,
    epochs:     int   = 60,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    device:     str   = None,
    resume:     bool  = False,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    CKPT_DIR.mkdir(exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    db      = get_db()
    dataset = SyntheticScentDataset(n_samples=n_samples, db=db)
    n_val   = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=(device == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    model = ScentWave().to(device)
    model.init_db(db)

    if resume and (CKPT_DIR / "best.pt").exists():
        ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print("Resumed from checkpoint.")

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = ScentWaveLoss()

    best_val_loss = float("inf")

    # ── Loop ─────────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_losses = {"total": 0, "vad": 0, "generation": 0, "diversity": 0}

        for features, target_vad, target_notes in train_dl:
            features     = features.to(device)
            target_vad   = target_vad.to(device)
            target_notes = target_notes.to(device)

            optimizer.zero_grad()
            out = model(features)
            losses = criterion(out["emotion"], out["gen_formula"], target_vad, target_notes)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in train_losses:
                train_losses[k] += losses[k].item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, target_vad, target_notes in val_dl:
                features     = features.to(device)
                target_vad   = target_vad.to(device)
                target_notes = target_notes.to(device)
                out = model(features)
                losses = criterion(out["emotion"], out["gen_formula"], target_vad, target_notes)
                val_loss += losses["total"].item()

        n_batches_train = len(train_dl)
        n_batches_val   = len(val_dl)
        avg_train = train_losses["total"] / n_batches_train
        avg_val   = val_loss / n_batches_val

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train={avg_train:.4f} "
            f"(vad={train_losses['vad']/n_batches_train:.4f} "
            f"gen={train_losses['generation']/n_batches_train:.4f} "
            f"div={train_losses['diversity']/n_batches_train:.4f}) | "
            f"val={avg_val:.4f} | {elapsed:.1f}s"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": avg_val}, CKPT_DIR / "best.pt")
            print(f"  -> Saved best checkpoint (val={avg_val:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CKPT_DIR / 'best.pt'}")
    return model
