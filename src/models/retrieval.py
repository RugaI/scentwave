# -*- coding: utf-8 -*-
"""
Retrieval Head — finds the closest EXISTING perfumes from the database.

Takes the latent + emotion vectors from the encoder and retrieves top-k
perfumes using a learned projection into fragrance embedding space,
then cosine similarity against pre-computed perfume embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class RetrievalHead(nn.Module):
    """
    Learns a projection from audio latent → fragrance embedding space.
    At inference time this embedding is compared against all perfume
    embeddings (cosine similarity) to retrieve the top-k matches.
    """
    def __init__(self,
                 latent_dim:    int = 512,
                 emotion_dim:   int = 3,
                 embed_dim:     int = 128,
                 dropout:       float = 0.2):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(latent_dim + emotion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
        )

        # Perfume embedding table — populated at runtime by build_index() (VAD: dim=3)
        self.register_buffer("perfume_embeddings", torch.zeros(1, 3))
        self.perfume_ids: List[int] = []

    def forward(self, latent: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised query embedding."""
        x = torch.cat([latent, emotion], dim=-1)
        emb = self.proj(x)
        return F.normalize(emb, dim=-1)

    def build_index(self, vad_matrix: np.ndarray, device: torch.device = None):
        """
        Store the raw VAD matrix as the retrieval index.
        Before training the model projects emotion → VAD space directly,
        so we compare query emotion against perfume VAD via cosine similarity.
        After training, the learned projection in self.proj refines results.
        """
        if device is None:
            device = next(self.parameters()).device
        # Store VAD directly — embed_dim will match emotion_dim (3) for direct comparison
        vad = vad_matrix.astype(np.float32)
        norms = np.linalg.norm(vad, axis=1, keepdims=True) + 1e-8
        self.perfume_embeddings = torch.tensor(vad / norms, dtype=torch.float32, device=device)
        self.perfume_ids = list(range(len(vad_matrix)))

    @torch.no_grad()
    def retrieve(self,
                 latent: torch.Tensor,
                 emotion: torch.Tensor,
                 top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Returns [(perfume_idx, similarity_score), ...] sorted descending.
        Uses emotion vector directly for cosine similarity against VAD index.
        """
        # Normalise the emotion query (3-dim VAD) → always (3,) for matmul
        query = F.normalize(emotion.reshape(-1, 3), dim=-1)[0]  # (3,)
        sims  = self.perfume_embeddings @ query               # (N,)
        top_k = min(top_k, len(sims))
        vals, idxs = torch.topk(sims, top_k)
        return [(self.perfume_ids[i.item()], v.item()) for i, v in zip(idxs, vals)]
