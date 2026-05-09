# -*- coding: utf-8 -*-
"""
Generation Head — creates a NOVEL fragrance formula that has never existed.

Input:  latent (512) + emotion (3)
Output: note_weights (60) split into top/middle/base pyramids.

Each song produces a unique, continuous-valued formula.
The formula is novel because:
  1. The weights are exact floating-point values (not discrete)
  2. The combination is learned jointly from audio features
  3. The model can output blends that no existing perfume has
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

import json
from pathlib import Path
from src.data.fragrance_db import CANONICAL_NOTES, NUM_NOTES, NOTE_INDEX

# Build layer masks dynamically from note_families.json so they stay in sync
def _build_layer_masks():
    _base = Path(__file__).parent.parent.parent
    with open(_base / "data" / "note_families.json", encoding="utf-8") as _f:
        _d = json.load(_f)
    top, mid, base = [], [], []
    for layer, notes in _d["pyramid_layer"].items():
        for n in notes:
            if n in NOTE_INDEX:
                if layer == "top":
                    top.append(NOTE_INDEX[n])
                elif layer == "middle":
                    mid.append(NOTE_INDEX[n])
                else:
                    base.append(NOTE_INDEX[n])
    # Notes not assigned to any layer go to middle
    assigned = set(top + mid + base)
    for i in range(NUM_NOTES):
        if i not in assigned:
            mid.append(i)
    return sorted(set(top)), sorted(set(mid)), sorted(set(base))

TOP_LAYER, MIDDLE_LAYER, BASE_LAYER = _build_layer_masks()


class ConditionedGeneratorBlock(nn.Module):
    """MLP block conditioned on emotion vector via FiLM modulation."""
    def __init__(self, dim: int, emotion_dim: int = 3, dropout: float = 0.2):
        super().__init__()
        self.fc  = nn.Linear(dim, dim)
        self.ln  = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop= nn.Dropout(dropout)
        # FiLM: scale + shift conditioned on emotion
        self.film_scale = nn.Linear(emotion_dim, dim)
        self.film_shift = nn.Linear(emotion_dim, dim)

    def forward(self, x: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        h = self.ln(self.fc(x))
        scale = torch.sigmoid(self.film_scale(emotion))
        shift = self.film_shift(emotion)
        return x + self.drop(self.act(h * scale + shift))


class GenerationHead(nn.Module):
    """
    Generates a novel fragrance formula as a weighted distribution over 60 notes,
    split into top / middle / base note pyramids with physically-grounded constraints.
    """
    def __init__(self,
                 latent_dim:  int = 512,
                 emotion_dim: int = 3,
                 num_notes:   int = NUM_NOTES,
                 dropout:     float = 0.2):
        super().__init__()
        self.num_notes   = num_notes
        self.emotion_dim = emotion_dim

        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + emotion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            ConditionedGeneratorBlock(512, emotion_dim, dropout),
            ConditionedGeneratorBlock(512, emotion_dim, dropout),
            ConditionedGeneratorBlock(512, emotion_dim, dropout),
            ConditionedGeneratorBlock(512, emotion_dim, dropout),
        ])

        # Separate heads per layer — better specialisation
        self.top_head    = nn.Linear(512, len(TOP_LAYER))
        self.middle_head = nn.Linear(512, len(MIDDLE_LAYER))
        self.base_head   = nn.Linear(512, len(BASE_LAYER))

        # Concentration weights: how much of each layer (learnable priors)
        self.layer_logits = nn.Parameter(torch.tensor([-1.5, -1.0, -0.5]))

        # Note intensity bias (some notes should always be low, e.g. civet)
        self.note_bias = nn.Parameter(torch.zeros(num_notes))

        self.top_indices    = torch.tensor(TOP_LAYER,    dtype=torch.long)
        self.middle_indices = torch.tensor(MIDDLE_LAYER, dtype=torch.long)
        self.base_indices   = torch.tensor(BASE_LAYER,   dtype=torch.long)

    def forward(self, latent: torch.Tensor,
                emotion: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          'formula'  : (batch, NUM_NOTES) — full note weight distribution (sums to 1)
          'top'      : (batch, |TOP|)     softmax weights within top layer
          'middle'   : (batch, |MIDDLE|)  softmax weights within middle layer
          'base'     : (batch, |BASE|)    softmax weights within base layer
          'layer_pct': (batch, 3)  — fraction of formula in [top, middle, base]
        """
        h = self.input_proj(torch.cat([latent, emotion], dim=-1))
        for block in self.blocks:
            h = block(h, emotion)

        top_logits    = self.top_head(h)
        middle_logits = self.middle_head(h)
        base_logits   = self.base_head(h)

        top_w    = F.softmax(top_logits,    dim=-1)
        middle_w = F.softmax(middle_logits, dim=-1)
        base_w   = F.softmax(base_logits,   dim=-1)

        # Layer proportions (physics: top evaporates first → highest %)
        # Default prior: top ~20%, middle ~35%, base ~45%
        layer_w = F.softmax(self.layer_logits + torch.tensor([0.2, 0.0, -0.2],
                  device=latent.device), dim=-1)             # (3,)

        # Assemble full 60-dim formula
        batch = latent.shape[0]
        formula = torch.zeros(batch, self.num_notes, device=latent.device)

        top_idx    = self.top_indices.to(latent.device)
        mid_idx    = self.middle_indices.to(latent.device)
        base_idx   = self.base_indices.to(latent.device)

        formula[:, top_idx]  = top_w    * layer_w[0]
        formula[:, mid_idx]  = middle_w * layer_w[1]
        formula[:, base_idx] = base_w   * layer_w[2]

        # Add learned note bias (small perturbation)
        formula = formula + torch.sigmoid(self.note_bias) * 0.02
        formula = formula / (formula.sum(dim=-1, keepdim=True) + 1e-8)

        return {
            "formula":   formula,
            "top":       top_w,
            "middle":    middle_w,
            "base":      base_w,
            "layer_pct": layer_w.unsqueeze(0).expand(batch, -1),
        }

    @torch.no_grad()
    def generate(self, latent: torch.Tensor,
                 emotion: torch.Tensor,
                 threshold: float = 0.005,
                 top_k_per_layer: int = 5) -> Dict[str, list]:
        """
        Human-readable formula for display.
        Returns dict with 'top', 'middle', 'base' as sorted (note, pct) lists.
        Uses top-k selection per layer so formula always has meaningful notes.
        """
        out = self(latent, emotion)
        formula = out["formula"][0].cpu().numpy()

        def extract(indices):
            # Sort by weight desc, take top_k_per_layer then filter by threshold
            scored = [(i, float(formula[i])) for i in indices]
            scored.sort(key=lambda x: -x[1])
            top_k = scored[:top_k_per_layer]
            # Normalise these selected notes so they sum nicely for display
            total = sum(w for _, w in top_k) + 1e-8
            notes = []
            for i, w in top_k:
                pct = round(w / total * 100, 1)
                if pct >= 5.0:  # at least 5% share within layer
                    notes.append((CANONICAL_NOTES[i].replace("_", " ").title(), pct))
            if not notes:  # fallback: always show at least top-3
                for i, w in top_k[:3]:
                    pct = round(w / total * 100, 1)
                    notes.append((CANONICAL_NOTES[i].replace("_", " ").title(), pct))
            return notes

        return {
            "top":    extract(TOP_LAYER),
            "middle": extract(MIDDLE_LAYER),
            "base":   extract(BASE_LAYER),
            "full":   formula,
        }
