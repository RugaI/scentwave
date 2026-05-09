# -*- coding: utf-8 -*-
"""
Synthetic dataset generator for ScentWave training.

Since we lack ground-truth (song, perfume) pairs, we generate synthetic
training examples using the known emotion-scent mappings:

  1. Sample random audio feature vectors with realistic distributions.
  2. Compute ground-truth VAD from the features.
  3. Compute ground-truth note distribution from VAD via the emotion-note table.
  4. The model learns to predict both VAD and note distribution from raw features.

This is self-supervised — the labels come from domain knowledge, not human annotation.
Real paired data (from Spotify x Fragrantica) can be mixed in via the external CSV loader.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

from src.data.fragrance_db import FragranceDB, get_db, NUM_NOTES, CANONICAL_NOTES, NOTE_INDEX


# ── Feature sampling distributions ─────────────────────────────────────────
# Each tuple: (mean, std) for a Beta-like distribution via clipping normal samples.
FEATURE_DISTRIBUTIONS = {
    "danceability":      (0.55, 0.20),
    "energy":            (0.60, 0.22),
    "key_norm":          (0.50, 0.30),
    "loudness_norm":     (0.55, 0.18),
    "mode":              (0.70, 0.00),   # binary — handled separately
    "speechiness":       (0.08, 0.07),
    "acousticness":      (0.35, 0.30),
    "instrumentalness":  (0.15, 0.22),
    "liveness":          (0.18, 0.12),
    "valence":           (0.50, 0.25),
    "tempo_norm":        (0.55, 0.18),
    "time_sig_norm":     (0.50, 0.10),
}
FEATURE_KEYS = list(FEATURE_DISTRIBUTIONS.keys())


def _sample_features(n: int, rng: np.random.Generator) -> np.ndarray:
    features = []
    for key in FEATURE_KEYS:
        mu, sigma = FEATURE_DISTRIBUTIONS[key]
        if key == "mode":
            col = (rng.random(n) < 0.70).astype(np.float32)
        else:
            col = rng.normal(mu, sigma, n).astype(np.float32)
            col = np.clip(col, 0.0, 1.0)
        features.append(col)
    return np.stack(features, axis=1)   # (n, 12)


def _features_to_vad(features: np.ndarray) -> np.ndarray:
    valence   = features[:, 9]
    arousal   = (features[:, 1] + features[:, 10]) / 2
    dominance = (features[:, 3] + features[:, 4])  / 2
    return np.stack([valence, arousal, dominance], axis=1)   # (n, 3)


def _vad_to_note_weights(vad: np.ndarray, db: FragranceDB) -> np.ndarray:
    """
    Soft rule-based mapping from VAD to note weight distribution.
    High valence → citrus/floral.  Low valence → woody/dark.
    High arousal → spicy/fresh.    Low arousal → musk/vanilla.
    High dominance → leather/oud.  Low dominance → powdery/clean.
    """
    n = vad.shape[0]
    weights = np.zeros((n, NUM_NOTES), dtype=np.float32)

    note_emos = db.note_emotions
    for note_name, emo in note_emos.items():
        if note_name not in NOTE_INDEX:
            continue
        idx = NOTE_INDEX[note_name]
        note_vad = np.array([emo["valence"], emo["arousal"], emo["dominance"]])
        # Similarity = 1 - mean absolute difference in VAD space
        diff = np.abs(vad - note_vad).mean(axis=1)   # (n,)
        sim  = np.clip(1.0 - diff, 0.0, 1.0)
        weights[:, idx] = sim

    # Normalise to sum to 1
    row_sums = weights.sum(axis=1, keepdims=True) + 1e-8
    weights /= row_sums
    return weights


def _load_spotify_songs(csv_path: str) -> np.ndarray | None:
    """Load real Spotify audio features from songs.csv if it exists."""
    import csv as csv_mod
    from pathlib import Path
    path = Path(csv_path)
    if not path.exists():
        return None
    cols = ["danceability","energy","key_norm","loudness_norm","mode",
            "speechiness","acousticness","instrumentalness","liveness",
            "valence","tempo_norm","time_sig_norm"]
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv_mod.DictReader(f):
            try:
                vec = [float(row[c]) for c in cols]
                rows.append(vec)
            except (KeyError, ValueError):
                continue
    if not rows:
        return None
    return np.array(rows, dtype=np.float32)


class SyntheticScentDataset(Dataset):
    """
    Training samples = real Spotify songs (if songs.csv exists) + synthetic.
    Labels are computed from the emotion-note domain knowledge mapping.
    """
    def __init__(self,
                 n_samples:    int = 50_000,
                 seed:         int = 42,
                 db:           FragranceDB = None,
                 songs_csv:    str = None):
        from pathlib import Path
        self.db  = db or get_db()
        base_dir = Path(__file__).parent.parent.parent
        songs_csv = songs_csv or str(base_dir / "data" / "songs.csv")

        rng = np.random.default_rng(seed)

        # Try real Spotify data first
        real = _load_spotify_songs(songs_csv)
        if real is not None and len(real) > 0:
            print(f"  Loaded {len(real):,} real Spotify songs from {songs_csv}")
            # Repeat real songs to fill n_samples, then top up with synthetic
            repeats = max(1, n_samples // len(real))
            real_rep = np.tile(real, (repeats, 1))[:n_samples // 2]
            noise = rng.normal(0, 0.02, real_rep.shape).astype(np.float32)
            real_aug = np.clip(real_rep + noise, 0.0, 1.0)
            synth_n = n_samples - len(real_aug)
            synth   = _sample_features(synth_n, rng)
            self.features = np.concatenate([real_aug, synth], axis=0)
        else:
            self.features = _sample_features(n_samples, rng)

        self.target_vad   = _features_to_vad(self.features)
        self.target_notes = _vad_to_note_weights(self.target_vad, self.db)
        print(f"  Dataset: {len(self.features):,} samples total")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx],      dtype=torch.float32),
            torch.tensor(self.target_vad[idx],    dtype=torch.float32),
            torch.tensor(self.target_notes[idx],  dtype=torch.float32),
        )
