#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ScentWave — training entry point.

Usage:
    python train.py
    python train.py --epochs 100 --batch 512 --samples 100000
    python train.py --resume
"""

import argparse
from src.training.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ScentWave")
    parser.add_argument("--epochs",  type=int,   default=60)
    parser.add_argument("--batch",   type=int,   default=256)
    parser.add_argument("--samples", type=int,   default=50_000)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--device",  type=str,   default=None)
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    train(
        n_samples  = args.samples,
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        device     = args.device,
        resume     = args.resume,
    )
