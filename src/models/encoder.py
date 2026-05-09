# -*- coding: utf-8 -*-
"""
Shared Audio Encoder + Emotion Bridge.

Input:  12-dim normalized audio feature vector
Output: (latent [512], emotion_vad [3])

Architecture:
  MLP with residual connections + LayerNorm + GELU
  → latent projection
  → emotion bridge (3-dim VAD with sigmoid, bounded [0,1])
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class AudioEncoder(nn.Module):
    """
    Maps 12-dim audio features → 512-dim latent + 3-dim emotion (VAD).
    """
    def __init__(self,
                 input_dim:   int = 12,
                 hidden_dim:  int = 256,
                 latent_dim:  int = 512,
                 emotion_dim: int = 3,
                 dropout:     float = 0.2):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
        )

        # Latent projection
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        # Emotion bridge: V, A, D all in [0, 1]
        self.emotion_bridge = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, emotion_dim),
            nn.Sigmoid(),
        )
        # Residual: add direct mapping from raw features → VAD
        # Feature indices: valence=9, energy=1, tempo=10, loudness=3, mode=4
        self._vad_from_features = nn.Linear(input_dim, emotion_dim, bias=False)
        with torch.no_grad():
            # Valence  ← features[9] (spotify valence)
            # Arousal  ← (features[1] + features[10]) / 2
            # Dominance← (features[3] + features[4]) / 2
            w = torch.zeros(emotion_dim, input_dim)
            w[0, 9] = 1.0
            w[1, 1] = 0.5; w[1, 10] = 0.5
            w[2, 3] = 0.5; w[2,  4] = 0.5
            self._vad_from_features.weight.copy_(w)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, 12) float32 audio features
        Returns:
            latent: (batch, 512)
            emotion: (batch, 3)  — [valence, arousal, dominance]
        """
        h = self.input_proj(x)
        h = self.blocks(h)
        latent = self.latent_proj(h)
        # Blend learned emotion with rule-based VAD for a strong inductive bias
        learned = self.emotion_bridge(latent)
        direct  = torch.sigmoid(self._vad_from_features(x))
        emotion = 0.5 * learned + 0.5 * direct
        return latent, emotion
